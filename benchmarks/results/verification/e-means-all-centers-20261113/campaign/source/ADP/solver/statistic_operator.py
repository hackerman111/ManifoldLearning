"""CPU-операторы конечного ADP-скетча: плотный U и точный CSR C @ X."""

# ruff: noqa: RUF002

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..engine.common import utils
from .grassman import _project


def _matrix(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    array = utils._finite_real_array(value, name)
    if array.ndim != 2 or array.shape[0] != rows or not array.shape[1]:
        raise ValueError(f"{name} must have shape ({rows}, k), k > 0")
    return array


class DenseOperator:
    """Статистики U=(J,p,d); действия с матрицами, без d*d intermediates."""

    def __init__(self, U: np.ndarray) -> None:
        self.U = np.ascontiguousarray(utils._finite_real_array(U, "U"))
        if self.U.ndim != 3 or 0 in self.U.shape:
            raise ValueError("U must have non-empty shape (J, p, d)")
        self.shape = self.U.shape

    @property
    def nbytes(self) -> int:
        return self.U.nbytes

    def project(self, V: np.ndarray) -> np.ndarray:
        return _project(self.U, _matrix(V, self.shape[2], "V"))

    def adjoint(self, P: np.ndarray) -> np.ndarray:
        P = utils._finite_real_array(P, "P")
        if P.ndim != 3 or P.shape[:2] != self.shape[:2] or not P.shape[2]:
            raise ValueError("P must have shape (J, p, k) matching U")
        return self.U.reshape(-1, self.shape[2]).T @ P.reshape(-1, P.shape[2])


class FactorizedOperator:
    """U_flat=C X; хранит CSR C=(J*p,n) и общий X=(n,d)."""

    def __init__(
        self,
        C: csr_matrix,
        X: np.ndarray,
        *,
        n_centers: int,
        n_directions: int,
    ) -> None:
        for name, value in (("n_centers", n_centers), ("n_directions", n_directions)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not issparse(C):
            raise TypeError("C must be a sparse matrix")
        if np.iscomplexobj(C.data) or not np.issubdtype(C.dtype, np.number):
            raise TypeError("C must have a real numeric dtype")
        self.X = utils._finite_real_array(X, "X")
        if self.X.ndim != 2 or 0 in self.X.shape:
            raise ValueError("X must have non-empty shape (n, d)")
        if C.shape != (n_centers * n_directions, len(self.X)):
            raise ValueError("C must have shape (J*p, n) matching X")
        self.C = csr_matrix(C, dtype=float)
        if not np.all(np.isfinite(self.C.data)):
            raise ValueError("C must contain only finite values")
        self.shape = (int(n_centers), int(n_directions), self.X.shape[1])

    @property
    def nbytes(self) -> int:
        return int(
            self.X.nbytes
            + self.C.data.nbytes
            + self.C.indices.nbytes
            + self.C.indptr.nbytes
        )

    def project(self, V: np.ndarray) -> np.ndarray:
        V = _matrix(V, self.shape[2], "V")
        result = self.C @ (self.X @ V)
        return np.ascontiguousarray(result).reshape(*self.shape[:2], V.shape[1])

    def adjoint(self, P: np.ndarray) -> np.ndarray:
        P = utils._finite_real_array(P, "P")
        if P.ndim != 3 or P.shape[:2] != self.shape[:2] or not P.shape[2]:
            raise ValueError("P must have shape (J, p, k) matching U")
        return self.X.T @ (self.C.T @ P.reshape(-1, P.shape[2]))


def factorized_statistics(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | Iterator[tuple[int, np.ndarray]],
    directions: np.ndarray,
    batch_size: int = 32,
    *,
    normalized: bool = False,
) -> tuple[FactorizedOperator, np.ndarray, np.ndarray]:
    """Построить C без U; вернуть (оператор, I, mass) для frozen solve.

    C_ji=gamma*(H_ji-a_ji*sum_i H_ji), gamma=1 либо mass_j.
    При normalized=False solver получает mass=None, иначе внешнюю mass.
    Плотные окна могут сделать CSR дороже U; backend выбирается явно.
    """
    utils._check_batch_size(batch_size)
    utils.require(isinstance(normalized, bool), "normalized must be boolean", TypeError)
    X, y, Phi = utils._prepare_data(X, y, directions)
    J, p = Phi.shape[:2]
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    data, indices = [], []
    indptr = [0]
    mass = np.empty(J)
    expected = 0
    for start, block in utils._weight_blocks(weights, J, len(X), batch_size):
        utils._check_weight_block_order(start, expected, J)
        block, local_mass = utils._prepare_weight_block(block, len(X))
        stop = start + len(block)
        utils.require(stop <= J, "weight blocks exceed directions")
        for offset, row in enumerate(block):
            j = start + offset
            support = np.flatnonzero(row)
            a = row[support] / local_mass[offset]
            local_X = Xc[support]
            Q = Phi[j] @ (local_X - a @ local_X).T
            H = (Q - (Q @ a)[:, None]) * a
            H -= H.sum(axis=1)[:, None] * a
            if not normalized:
                H *= local_mass[offset]
            data.append(H.ravel())
            indices.append(np.tile(support, p))
            indptr.extend(indptr[-1] + len(support) * np.arange(1, p + 1))
        mass[start:stop] = local_mass
        expected = stop
    utils.require(expected == J, "weight blocks do not cover all directions")
    C = csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(J * p, len(X)),
    )
    C.eliminate_zeros()
    operator = FactorizedOperator(C, Xc, n_centers=J, n_directions=p)
    I = np.asarray(C @ yc).reshape(J, p)
    if not np.all(np.isfinite(I)):
        raise RuntimeError("non-finite factorized statistics")
    return operator, I, mass
