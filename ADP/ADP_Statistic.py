from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .engine import utils
from .gpu import require_cupy


@dataclass(frozen=True, slots=True)
class ADP_Statistics:
    _fields: ClassVar[tuple[str, ...]] = (
        "I",
        "U",
        "mass",
        "mean",
        "n_eff",
        "eta",
    )

    I: Any
    U: Any
    mass: np.ndarray
    mean: np.ndarray
    n_eff: np.ndarray
    eta: np.ndarray

    def __getitem__(self, name: str) -> Any:
        if name not in self._fields:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)


def calculate_statistics(X, Y, weights, directions, batch_size=32) -> ADP_Statistics:
    """Calculate stable ADP statistics from a matrix or iterator of weight blocks."""
    utils._check_batch_size(batch_size)
    X, Y, Phi = utils._prepare_data(X, Y, directions)
    J, P = Phi.shape[:2]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar

    I = np.empty((J, P))
    U = np.empty((J, P, X.shape[1]))
    mass = np.empty(J)
    mean = np.empty((J, X.shape[1]))
    n_eff = np.empty(J)
    eta = np.empty((J, P))

    expected_start = 0
    for start, W in utils._weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)

        W, mass_block = utils._prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]

        batch = slice(start, stop)
        Phib = Phi[batch]
        max_nonzero = int(np.count_nonzero(W, axis=1).max())
        # ponytail: 25% is an empirical dense/local crossover; benchmark-based
        # dispatch is only needed if substantially different kernels are added.
        if 4 * max_nonzero <= X.shape[0]:
            block_values = _local_block(
                Xc,
                Y,
                W,
                Phib,
                mass_block,
                max_nonzero,
            )
        else:
            A = W / mass_block[:, None]
            block_values = _dense_block(Xc, Y, A, Phib, mass_block)

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch] = block_values
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        expected_start = stop

    if expected_start != J:
        raise ValueError("weight blocks do not cover all directions")

    return ADP_Statistics(
        I=I,
        U=U,
        mass=mass,
        mean=mean,
        n_eff=n_eff,
        eta=eta,
    )


def calculate_statistics_gpu(
    X,
    Y,
    weights,
    directions,
    batch_size=32,
) -> ADP_Statistics:
    """Calculate ADP statistics on the requested CUDA device via CuPy."""
    cp = require_cupy()
    utils._check_batch_size(batch_size)
    X, Y, Phi = utils._prepare_data(X, Y, directions)
    J, P = Phi.shape[:2]
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)
    Phi_gpu = cp.asarray(Phi)
    x_bar = X_gpu.mean(axis=0)
    Xc = X_gpu - x_bar

    I = cp.empty((J, P))
    U = cp.empty((J, P, X.shape[1]))
    mass = np.empty(J)
    mean = cp.empty((J, X.shape[1]))
    n_eff = cp.empty(J)
    eta = cp.empty((J, P))

    expected_start = 0
    for start, W in utils._weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)
        W, mass_block = utils._prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]
        batch = slice(start, stop)

        W_gpu = cp.asarray(W)
        mass_gpu = cp.asarray(mass_block)
        Phib = Phi_gpu[batch]
        max_nonzero = int(np.count_nonzero(W, axis=1).max())
        if 4 * max_nonzero <= X.shape[0]:
            block_values = _local_block(
                Xc,
                Y_gpu,
                W_gpu,
                Phib,
                mass_gpu,
                max_nonzero,
                xp=cp,
            )
        else:
            A = W_gpu / mass_gpu[:, None]
            block_values = _dense_block(
                Xc,
                Y_gpu,
                A,
                Phib,
                mass_gpu,
                xp=cp,
            )

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch] = block_values
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        expected_start = stop

    if expected_start != J:
        raise ValueError("weight blocks do not cover all directions")

    return ADP_Statistics(
        I=I,
        U=U,
        mass=mass,
        mean=cp.asnumpy(mean),
        n_eff=cp.asnumpy(n_eff),
        eta=cp.asnumpy(eta),
    )


def _dense_block(Xc, Y, A, Phi, mass, *, xp=np):
    mean = A @ Xc
    y_bar = A @ Y
    Q = Phi @ Xc.T - Phi @ mean[..., None]
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual, xp=xp)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = mass[:, None] * (H @ Y - summed * y_bar[:, None])
    U = mass[:, None, None] * (
        H @ Xc - summed[..., None] * mean[:, None, :]
    )
    n_eff = 1.0 / xp.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta


def _local_block(Xc, Y, W, Phi, mass, width, *, xp=np):
    rows, columns = xp.nonzero(W)
    counts = xp.bincount(rows, minlength=len(W))
    offsets = xp.repeat(xp.cumsum(counts) - counts, counts)
    slots = xp.arange(len(rows)) - offsets

    indices = xp.zeros((len(W), width), dtype=xp.intp)
    local_weights = xp.zeros((len(W), width), dtype=W.dtype)
    indices[rows, slots] = columns
    local_weights[rows, slots] = W[rows, columns]

    A = local_weights / mass[:, None]
    local_X = Xc[indices]
    local_Y = Y[indices]
    mean = xp.einsum("bm,bmd->bd", A, local_X, optimize=True)
    y_bar = xp.einsum("bm,bm->b", A, local_Y, optimize=True)
    Q = Phi @ xp.swapaxes(local_X - mean[:, None, :], 1, 2)
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual, xp=xp)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = mass[:, None] * (
        (H @ local_Y[..., None]).squeeze(-1) - summed * y_bar[:, None]
    )
    U = mass[:, None, None] * (
        H @ local_X - summed[..., None] * mean[:, None, :]
    )
    n_eff = 1.0 / xp.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta


def _normalized_residual(A, Q, residual, *, xp=np):
    denominator = (xp.abs(Q) @ A[..., None]).squeeze(-1)
    nonzero = denominator != 0
    return xp.where(
        nonzero,
        xp.abs(residual) / xp.where(nonzero, denominator, 1.0),
        0.0,
    )
