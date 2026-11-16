from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, lsmr

_operator_type: Any = LinearOperator
_lsmr_method: Any = lsmr

# Пределы bounded dense SVD; значения сохранены из исходного HYBRID.
DENSE_MAX_UNKNOWNS = 256
DENSE_MAX_BYTES = 64 * 1024**2


def use_dense(rows: int, columns: int, max_unknowns: int, max_bytes: int) -> bool:
    """Проверить, помещается ли bounded dense SVD в заданный memory budget."""
    return (
        columns <= max_unknowns
        and 8 * (4 * rows * columns + 4 * columns**2) <= max_bytes
    )


def design_matrix(
    U: np.ndarray, coefficients: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """Явный малый joint design: строки (j,p), столбцы (a,d)."""
    J, P, d = U.shape
    design = np.empty((J * P, coefficients.shape[1] * d))
    for a in range(coefficients.shape[1]):
        design[:, a * d : (a + 1) * d] = (
            (root * coefficients[:, a])[:, None, None] * U
        ).reshape(J * P, d)
    return design


@dataclass(frozen=True, slots=True)
class LinearResult:
    solution: np.ndarray
    backend: str
    stop: int
    iterations: int
    relative_residual: float
    rank: int | None


def solve_augmented(
    operator: Any,
    rhs: np.ndarray,
    initial: np.ndarray,
    diagonal: np.ndarray,
    *,
    tol: float,
    maxiter: int | None,
    dense: np.ndarray | None = None,
) -> LinearResult:
    """Решить LS-коррекцию и проверить stationarity в исходных координатах."""
    residual = rhs - operator @ initial
    rank = None
    if not np.any(rhs):
        return LinearResult(np.zeros_like(initial), "zero-rhs", 0, 0, 0.0, None)
    if dense is not None:
        correction, _, rank, _ = linalg.lstsq(
            dense, residual, cond=None, lapack_driver="gelsd", check_finite=False
        )
        stop, iterations, backend = 0, 0, "dense-svd"
        if rank != operator.shape[1]:
            raise RuntimeError(
                f"hybrid augmented system is rank-deficient: rank={rank}"
            )
    else:
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise RuntimeError("hybrid preconditioner has non-positive diagonal")
        scale = 1.0 / np.sqrt(diagonal)
        scaled = _operator_type(
            operator.shape,
            matvec=lambda z: operator @ (scale * z),
            rmatvec=lambda v: scale * operator.rmatvec(v),
            dtype=float,
        )
        result = _lsmr_method(
            scaled,
            residual,
            atol=min(1e-12, tol * 0.01),
            btol=min(1e-12, tol * 0.01),
            maxiter=maxiter or max(50, 5 * operator.shape[1]),
        )
        correction = scale * result[0]
        stop, iterations, backend = int(result[1]), int(result[2]), "scaled-lsmr"
        if stop not in (0, 1, 2, 4, 5):
            raise RuntimeError(
                f"hybrid LSMR did not converge: stop={stop}, iterations={iterations}"
            )
    solution = initial + correction
    normal = operator.rmatvec(rhs - operator @ solution)
    normal_rhs = operator.rmatvec(rhs)
    relative = float(np.linalg.norm(normal)) / max(
        float(np.linalg.norm(normal_rhs)), np.finfo(float).tiny
    )
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative):
        raise RuntimeError("hybrid returned a non-finite solution or residual")
    if relative > max(10 * tol, 256 * np.finfo(float).eps):
        raise RuntimeError(f"hybrid residual certificate failed: {relative:.3e}")
    return LinearResult(solution, backend, stop, iterations, relative, rank)
