from __future__ import annotations

import math

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import cg

from .._multi_operator import adjoint, forward
from .HYBRID import (
    DENSE_MAX_BYTES,
    DENSE_MAX_UNKNOWNS,
    LinearResult,
    _operator_type,
    design_matrix,
    solve_augmented,
    use_dense,
)

_cg_method = cg


class PenaltyRoot:
    """NUMERICAL: sqrt(I-sum w_j P_j.T P_j) через малый спектральный фактор."""

    def __init__(self, projectors: np.ndarray, weights: np.ndarray) -> None:
        """Построить малый спектральный фактор ``sqrt(I-F*F)``."""
        factor = (np.sqrt(weights)[:, None, None] * projectors).reshape(
            -1, projectors.shape[-1]
        )
        _, singular, self.basis = linalg.svd(
            factor, full_matrices=False, check_finite=False
        )
        if singular[0] > 1 + 256 * np.finfo(float).eps * max(factor.shape):
            raise RuntimeError("manifold penalty is not positive semidefinite")
        squares = np.minimum(singular**2, 1.0)
        self.factors = squares / (1.0 + np.sqrt(1.0 - squares))

    def apply(self, B: np.ndarray) -> np.ndarray:
        """Применить корень manifold penalty к матрице ``B``."""
        return B - ((B @ self.basis.T) * self.factors) @ self.basis


def _manifold_pcg(
    U: np.ndarray,
    I: np.ndarray,
    gamma: np.ndarray,
    normalized: np.ndarray,
    projectors: np.ndarray,
    slopes: np.ndarray,
    initial: np.ndarray,
    ridge: float,
    tol: float,
    maxiter: int | None,
) -> LinearResult:
    """Joint PCG: блок (m,m) для каждого признака вместо scalar Jacobi.

    Normal equations выбраны по benchmark manifold150; исходный остаток
    обязателен, при отказе вызывающий код запускает augmented LSMR.
    """
    m, d = initial.shape
    weighted_slopes = gamma[:, None] * slopes
    adjoint_U = U.swapaxes(1, 2)
    factor = (np.sqrt(normalized)[:, None, None] * projectors).reshape(-1, d)
    local = np.empty((len(U), d))

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Применить block-PCG normal operator без dense ``d x d`` матриц."""
        B = vector.reshape(m, d)
        np.matmul(slopes, B, out=local)
        projected = U @ local[..., None]
        np.matmul(adjoint_U, projected, out=local[..., None])
        result = weighted_slopes.T @ local
        # EXACT: C=I-F*F; два GEMM без (J,m,m) и перестановок внутри PCG.
        penalty = B - (B @ factor.T) @ factor
        return (result + ridge * penalty).ravel()

    energy = np.einsum("jpd,jpd->jd", U, U)
    blocks = np.empty((d, m, m))
    for a in range(m):
        blocks[:, :, a] = (
            (gamma[:, None] * slopes).T @ (slopes[:, a, None] * energy)
        ).T
    penalty_diag = np.maximum(
        1 - np.einsum("j,jmd,jmd->d", normalized, projectors, projectors), 0
    )
    blocks[:, np.arange(m), np.arange(m)] += ridge * penalty_diag[:, None]
    values, vectors = np.linalg.eigh(blocks)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        return LinearResult(initial.ravel(), "block-pcg-failed", -1, 0, math.inf, None)

    def precondition(vector: np.ndarray) -> np.ndarray:
        """Применить block spectral preconditioner к flattened ``B``."""
        local = vector.reshape(m, d).T[..., None]
        coordinates = (vectors.swapaxes(1, 2) @ local) / values[..., None]
        return (vectors @ coordinates).squeeze(-1).T.ravel()

    operator = _operator_type(
        (m * d, m * d), matvec=matvec, rmatvec=matvec, dtype=float
    )
    preconditioner = _operator_type(
        operator.shape, matvec=precondition, rmatvec=precondition, dtype=float
    )
    rhs = adjoint(U, gamma[:, None] * I, slopes).ravel()
    iterations = 0

    def record(_: np.ndarray) -> None:
        """Считать итерации block-PCG для linear diagnostics."""
        nonlocal iterations
        iterations += 1

    solution, info = _cg_method(
        operator,
        rhs,
        x0=initial.ravel(),
        M=preconditioner,
        rtol=tol,
        atol=0.0,
        maxiter=maxiter or max(50, min(1000, 5 * m * d)),
        callback=record,
    )
    relative = float(np.linalg.norm(matvec(solution) - rhs)) / max(
        float(np.linalg.norm(rhs)), np.finfo(float).tiny
    )
    accepted = (
        info == 0
        and np.all(np.isfinite(solution))
        and np.isfinite(relative)
        and relative <= max(10 * tol, 256 * np.finfo(float).eps)
    )
    return LinearResult(
        solution,
        "block-pcg" if accepted else "block-pcg-failed",
        int(info),
        iterations,
        relative,
        None,
    )


def solve_manifold(
    U: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    weights: np.ndarray,
    projectors: np.ndarray,
    slopes: np.ndarray,
    initial: np.ndarray,
    *,
    ridge: float,
    tol: float,
    maxiter: int | None = None,
    dense_max_unknowns: int = DENSE_MAX_UNKNOWNS,
    dense_max_bytes: int = DENSE_MAX_BYTES,
) -> LinearResult:
    """Исходная manifold B-задача: bounded SVD либо block-PCG/augmented LSMR."""
    from ..LSMR import _validate_inputs

    initial, U, I, mass = _validate_inputs(initial, U, I, mass)
    if initial.ndim != 2:
        raise ValueError("manifold initial must have shape (m,d)")
    for name, value, shape in (
        ("weights", weights, (len(U),)),
        ("slopes", slopes, (len(U), initial.shape[0])),
        ("projectors", projectors, (len(U), *initial.shape)),
    ):
        if np.iscomplexobj(value) or not np.issubdtype(
            np.asarray(value).dtype, np.number
        ):
            raise TypeError(f"{name} must be real numeric")
        if np.shape(value) != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {shape}")
    weights = np.asarray(weights, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    projectors = np.asarray(projectors, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    if not np.allclose(
        projectors @ projectors.swapaxes(1, 2),
        np.eye(initial.shape[0]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("source projectors must have orthonormal rows")
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError(
            "ridge must be nonnegative and tol must be positive and finite"
        )
    for name, value in (
        ("maxiter", maxiter),
        ("dense_max_unknowns", dense_max_unknowns),
        ("dense_max_bytes", dense_max_bytes),
    ):
        if value is None and name == "maxiter":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < (1 if name == "maxiter" else 0):
            raise ValueError(f"{name} is out of range")
    root = np.sqrt(mass * weights)
    normalized = weights / weights.sum()
    rows, columns = I.size, initial.size
    dense_enabled = use_dense(
        rows + columns, columns, dense_max_unknowns, dense_max_bytes
    )
    attempted_iterations = 0
    if not dense_enabled:
        attempt = _manifold_pcg(
            U,
            I,
            mass * weights,
            normalized,
            projectors,
            slopes,
            initial,
            ridge,
            tol,
            maxiter,
        )
        if attempt.backend == "block-pcg":
            return attempt
        attempted_iterations = attempt.iterations
    penalty = PenaltyRoot(projectors, normalized)
    root_ridge = math.sqrt(ridge)

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Применить augmented manifold forward operator."""
        B = vector.reshape(initial.shape)
        data = root[:, None] * forward(U, B, slopes)
        return np.concatenate((data.ravel(), root_ridge * penalty.apply(B).ravel()))

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        """Применить adjoint augmented manifold operator."""
        data = root[:, None] * vector[:rows].reshape(I.shape)
        result = adjoint(U, data, slopes)
        result += root_ridge * penalty.apply(vector[rows:].reshape(initial.shape))
        return result.ravel()

    operator = _operator_type(
        (rows + columns, columns), matvec=matvec, rmatvec=rmatvec, dtype=float
    )
    data_diagonal = (mass[:, None] * weights[:, None] * slopes**2).T @ np.einsum(
        "jpd,jpd->jd", U, U
    )
    penalty_diagonal = 1 - np.einsum("j,jmd,jmd->d", normalized, projectors, projectors)
    diagonal = (
        data_diagonal + ridge * np.maximum(penalty_diagonal, 0)[None, :]
    ).ravel()
    dense = None
    if dense_enabled:
        dense = np.zeros((rows + columns, columns))
        dense[:rows] = design_matrix(U, slopes, root)
        # Только bounded dense ветка; в большой задаче этот (d,d) фактор отсутствует.
        d = initial.shape[1]
        block = root_ridge * penalty.apply(np.eye(d))
        for a in range(initial.shape[0]):
            dense[rows + a * d : rows + (a + 1) * d, a * d : (a + 1) * d] = block
    rhs = np.concatenate(((root[:, None] * I).ravel(), np.zeros(columns)))
    linear = solve_augmented(
        operator,
        rhs,
        initial.ravel(),
        diagonal,
        tol=tol,
        maxiter=maxiter,
        dense=dense,
    )
    # Сертификат проверяется повторно через исходный penalty, без sqrt-factor.
    B = linear.solution.reshape(initial.shape)
    projected = (B @ projectors.swapaxes(1, 2)) * normalized[:, None, None]
    penalty_B = B - projected.transpose(1, 0, 2).reshape(
        len(B), -1
    ) @ projectors.reshape(-1, U.shape[2])
    gradient = (
        adjoint(U, (mass * weights)[:, None] * (forward(U, B, slopes) - I), slopes)
        + ridge * penalty_B
    )
    normal_rhs = adjoint(U, (mass * weights)[:, None] * I, slopes)
    relative = float(np.linalg.norm(gradient)) / max(
        float(np.linalg.norm(normal_rhs)), np.finfo(float).tiny
    )
    if not np.isfinite(relative) or relative > max(10 * tol, 256 * np.finfo(float).eps):
        raise RuntimeError(f"hybrid original manifold residual failed: {relative:.3e}")
    return LinearResult(
        linear.solution,
        linear.backend if dense_enabled else "block-pcg->" + linear.backend,
        linear.stop,
        linear.iterations + attempted_iterations,
        relative,
        linear.rank,
    )
