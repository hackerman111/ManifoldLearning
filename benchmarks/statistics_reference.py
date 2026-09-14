"""Замороженный эталон статистик до оптимизации, 2026-09-14."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.engine import utils


def calculate_statistics(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray | Iterator[tuple[int, np.ndarray]],
    directions: np.ndarray,
    batch_size: int = 32,
    *,
    normalized: bool = False,
) -> ADP_Statistics:
    """Вычислить устойчивые локальные ADP-статистики.

    При ``normalized=True`` величины ``I`` и ``U`` делятся на локальную
    массу; саму массу затем следует передать внешним весом в решатель.
    """
    utils._check_batch_size(batch_size)
    if not isinstance(normalized, bool):
        raise TypeError("normalized must be boolean")
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
    S = np.empty(J)

    expected_start = 0
    for start, W in utils._weight_blocks(weights, J, X.shape[0], batch_size):
        utils._check_weight_block_order(start, expected_start, J)

        W, mass_block = utils._prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]

        batch = slice(start, stop)
        A = W / mass_block[:, None]
        Phib = Phi[batch]
        # ponytail: 25% is an empirical dense/local crossover; benchmark-based
        # dispatch is only needed if substantially different kernels are added.
        if 4 * int(np.count_nonzero(W, axis=1).max()) <= X.shape[0]:
            block_values = _local_block(
                Xc,
                Y,
                W,
                Phib,
                mass_block,
                normalized,
            )
        else:
            block_values = _dense_block(
                Xc,
                Y,
                A,
                Phib,
                mass_block,
                normalized,
            )

        I[batch], U[batch], Mcb, n_eff[batch], eta[batch], S[batch] = block_values
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
        S=S,
    )


def _dense_block(Xc, Y, A, Phi, mass, normalized):
    mean = A @ Xc
    y_bar = A @ Y
    Q = Phi @ Xc.T - Phi @ mean[..., None]
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = H @ Y - summed * y_bar[:, None]
    U = H @ Xc - summed[..., None] * mean[:, None, :]
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    n_eff = 1.0 / np.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta, y_bar


def _local_block(Xc, Y, W, Phi, mass, normalized):
    rows, columns = np.nonzero(W)
    counts = np.bincount(rows, minlength=len(W))
    width = int(counts.max())
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    slots = np.arange(len(rows)) - offsets

    indices = np.zeros((len(W), width), dtype=np.intp)
    local_weights = np.zeros((len(W), width), dtype=W.dtype)
    indices[rows, slots] = columns
    local_weights[rows, slots] = W[rows, columns]

    A = local_weights / mass[:, None]
    local_X = Xc[indices]
    local_Y = Y[indices]
    mean = np.einsum("bm,bmd->bd", A, local_X, optimize=True)
    y_bar = np.einsum("bm,bm->b", A, local_Y, optimize=True)
    Q = Phi @ np.swapaxes(local_X - mean[:, None, :], 1, 2)
    residual = (Q @ A[..., None]).squeeze(-1)
    eta = _normalized_residual(A, Q, residual)
    H = (Q - residual[..., None]) * A[:, None, :]
    summed = H.sum(axis=2)
    I = (H @ local_Y[..., None]).squeeze(-1) - summed * y_bar[:, None]
    U = H @ local_X - summed[..., None] * mean[:, None, :]
    if not normalized:
        I *= mass[:, None]
        U *= mass[:, None, None]
    n_eff = 1.0 / np.square(A).sum(axis=1)
    return I, U, mean, n_eff, eta, y_bar


def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )


def manifold_statistics(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Вычислить normalized ``I/U`` при рабочей памяти ``O(B*P*n)``."""
    J, P, d = directions.shape
    I = np.empty((J, P))
    U = np.empty((J, P, d))
    mass = np.empty(J)
    n_eff = np.empty(J)
    edges = 0
    for start in range(0, J, self.batch_size):
        weights = self._weight_block(
            X, centers, projectors, eigenvalues, h, alpha, start
        )
        stop = start + len(weights)
        mass_block = weights.sum(axis=1)
        if not np.all(np.isfinite(mass_block)) or np.any(mass_block <= 0):
            raise RuntimeError("function weights contain an empty neighborhood")
        normalized = weights / mass_block[:, None]
        mean = normalized @ X
        y_mean = normalized @ Y
        phi = directions[start:stop]
        projected = phi @ X.T - (phi @ mean[..., None])
        correction = (projected @ normalized[..., None]).squeeze(-1)
        projected -= correction[..., None]
        moments = projected * normalized[:, None, :]
        summed = moments.sum(axis=2)
        I[start:stop] = moments @ Y - summed * y_mean[:, None]
        U[start:stop] = moments @ X - summed[..., None] * mean[:, None, :]
        mass[start:stop] = mass_block
        n_eff[start:stop] = 1.0 / np.square(normalized).sum(axis=1)
        edges += int(np.count_nonzero(weights))
    if not np.all(np.isfinite(I)) or not np.all(np.isfinite(U)):
        raise RuntimeError("ADP statistics contain non-finite values")
    return I, U, mass, n_eff, edges
