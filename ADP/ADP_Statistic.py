from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from . import utils


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

    I: np.ndarray
    U: np.ndarray
    mass: np.ndarray
    mean: np.ndarray
    n_eff: np.ndarray
    eta: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._fields:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)


def calculate_statistics(X, Y, weights, directions, batch_size=32) -> ADP_Statistics:
    """Calculate stable ADP statistics from a matrix or iterator of weight blocks."""
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

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
        if (
            not isinstance(start, (int, np.integer))
            or isinstance(start, bool)
            or start != expected_start
        ):
            raise ValueError("weight blocks must cover centers in order")

        W, mass_block = utils._prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]

        if stop > J:
            raise ValueError("weight blocks exceed the number of directions")

        batch = slice(start, stop)
        A = W / mass_block[:, None]
        Phib = Phi[batch]
        Mcb = A @ Xc
        y_bar = A @ Y
        Q = Phib @ Xc.T - Phib @ Mcb[..., None]
        residual = (Q @ A[..., None]).squeeze(-1)
        eta[batch] = _normalized_residual(A, Q, residual)
        H = (Q - residual[..., None]) * A[:, None, :]
        s = H.sum(axis=2)
        I[batch] = mass_block[:, None] * (H @ Y - s * y_bar[:, None])
        U[batch] = mass_block[:, None, None] * (
            H @ Xc - s[..., None] * Mcb[:, None, :]
        )
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        n_eff[batch] = 1.0 / np.square(A).sum(axis=1)
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

def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )
