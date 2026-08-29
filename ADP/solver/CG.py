from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .LSMR import (
    HPAOResult,
    _gauge_fix,
    _index_distance,
    _local_refit,
    _loss,
    _normalize_index,
    _stationarity,
    _validate_inputs,
)

# NUMERICAL: CG решает тот же фиксированный quadratic beta-step через A.T @ A.


def cg(
    beta_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    **settings: Any,
) -> np.ndarray:
    """Адаптер для ``ADP_Solver``, возвращающий только beta."""
    return solve(beta_init, U, I, **settings).index


def solve(
    beta_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 1.0,
    max_steps: int = 10,
    tol: float = 1e-6,
    cg_maxiter: int | None = None,
) -> HPAOResult:
    """Решить single- или multi-index AO-задачу matrix-free методом CG.

    ``U`` имеет форму ``(J, p, d)``, ``I`` — ``(J, p)``, ``beta_init`` —
    ``(d,)`` или ``(m, d)``. Каждая итерация AO точно вычисляет локальные
    коэффициенты; затем CG решает совместную регуляризованную глобальную
    подзадачу для beta.
    """
    beta, U, I, mass = _validate_inputs(beta_init, U, I, mass)
    _validate_settings(lambda_prox, max_steps, tol, cg_maxiter)
    beta = _normalize_index(beta)

    coefficients, _ = _local_refit(I, U, beta)
    loss = _loss(I, U, beta, coefficients, mass)
    loss_history = [loss]
    total_cg_iterations = 0
    accepted_steps = 0
    converged = False
    cg_info = 0
    cg_iterations = 0
    relative_linear_residual = math.inf

    for _ in range(max_steps):
        prior = beta
        old_loss = loss
        raw_beta, cg_info, cg_iterations, relative_linear_residual = _global_step(
            I,
            U,
            coefficients,
            mass,
            prior,
            lambda_prox,
            tol,
            cg_maxiter,
        )
        total_cg_iterations += cg_iterations

        raw_norm = float(np.linalg.norm(raw_beta))
        if not np.isfinite(raw_norm) or raw_norm == 0:
            raise RuntimeError("CG returned a zero or non-finite beta")
        candidate, _, global_rank = _gauge_fix(raw_beta, coefficients, prior)
        expected_rank = 1 if beta.ndim == 1 else beta.shape[0]
        if global_rank < expected_rank:
            raise RuntimeError("CG returned a rank-deficient multi-index")
        candidate_coefficients, _ = _local_refit(I, U, candidate)
        candidate_loss = _loss(I, U, candidate, candidate_coefficients, mass)
        rounding = 64 * np.finfo(float).eps * max(1.0, old_loss)
        if candidate_loss > old_loss + rounding:
            raise RuntimeError("certified CG step increased the ADP objective")

        beta = candidate
        coefficients = candidate_coefficients
        loss = candidate_loss
        accepted_steps += 1
        loss_history.append(loss)

        gradient, local_gradient, orthogonality = _stationarity(
            I, U, beta, coefficients, mass, loss
        )
        relative_change = abs(old_loss - loss) / max(1.0, old_loss)
        if (
            relative_change < tol
            and _index_distance(beta, prior) < tol
            and max(gradient, local_gradient, orthogonality) < tol
        ):
            converged = True
            break

    gradient, local_gradient, orthogonality = _stationarity(
        I, U, beta, coefficients, mass, loss
    )
    diagnostics = {
        "cg_info": cg_info,
        "cg_iterations": cg_iterations,
        "cg_iterations_total": total_cg_iterations,
        "relative_linear_residual": relative_linear_residual,
        "accepted_steps": accepted_steps,
        "converged": converged,
        "loss": loss,
        "loss_history": tuple(loss_history),
        "riemannian_gradient": gradient,
        "local_gradient": local_gradient,
        "orthogonality": orthogonality,
    }
    return HPAOResult(beta, coefficients, diagnostics)


def _validate_settings(
    lambda_prox: float,
    max_steps: int,
    tol: float,
    cg_maxiter: int | None,
) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if cg_maxiter is not None:
        if isinstance(cg_maxiter, bool) or not isinstance(
            cg_maxiter, (int, np.integer)
        ):
            raise TypeError("cg_maxiter must be an integer or None")
        if cg_maxiter < 1:
            raise ValueError("cg_maxiter must be positive")
    if not np.isfinite(lambda_prox) or lambda_prox <= 0:
        raise ValueError("lambda_prox must be finite and positive")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")


def _normal_operator(
    U: np.ndarray,
    coefficients: np.ndarray,
    mass: np.ndarray,
    lambda_prox: float,
) -> tuple[Any, Any]:
    """Построить joint normal operator без матрицы ``(m*d) x (m*d)``."""
    if coefficients.ndim == 1:
        weighted_squares = mass * np.square(coefficients)
        diagonal = np.einsum("j,jpd,jpd->d", weighted_squares, U, U, optimize=True)
        size = U.shape[2]

        def matvec(vector: np.ndarray) -> np.ndarray:
            projected = U @ vector  # (J, p)
            result = np.einsum(
                "j,jpd,jp->d", weighted_squares, U, projected, optimize=True
            )
            result += lambda_prox * vector
            return result

    else:
        feature_squares = np.einsum("jpd,jpd->jd", U, U, optimize=True)
        weighted_squares = mass[:, None] * np.square(coefficients)
        diagonal = weighted_squares.T @ feature_squares  # (m, d)
        index_shape = (coefficients.shape[1], U.shape[2])
        size = math.prod(index_shape)

        def matvec(vector: np.ndarray) -> np.ndarray:
            matrix = vector.reshape(index_shape)
            projected = np.einsum(
                "jpd,md,jm->jp", U, matrix, coefficients, optimize=True
            )
            result = np.einsum(
                "j,jm,jpd,jp->md",
                mass,
                coefficients,
                U,
                projected,
                optimize=True,
            )
            result += lambda_prox * matrix
            return result.ravel()

    diagonal += lambda_prox
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise RuntimeError(
            "CG normal operator has a non-positive Jacobi diagonal; "
            "use positive lambda_prox"
        )

    def precondition(vector: np.ndarray) -> np.ndarray:
        return vector / diagonal.ravel()

    linear_operator: Any = sparse_linalg.LinearOperator
    operator = linear_operator(
        (size, size),
        matvec=matvec,
        rmatvec=matvec,
        dtype=float,
    )
    preconditioner = linear_operator(
        operator.shape,
        matvec=precondition,
        rmatvec=precondition,
        dtype=float,
    )
    return operator, preconditioner


def _global_step(
    I: np.ndarray,
    U: np.ndarray,
    coefficients: np.ndarray,
    mass: np.ndarray,
    prior: np.ndarray,
    lambda_prox: float,
    tol: float,
    maxiter: int | None,
) -> tuple[np.ndarray, int, int, float]:
    operator, preconditioner = _normal_operator(U, coefficients, mass, lambda_prox)
    if coefficients.ndim == 1:
        rhs = np.einsum("j,jpd,jp->d", mass * coefficients, U, I, optimize=True)
        rhs += lambda_prox * prior
    else:
        rhs = np.einsum("j,jm,jpd,jp->md", mass, coefficients, U, I, optimize=True)
        rhs += lambda_prox * prior
        rhs = rhs.ravel()
    krylov_tol = min(1e-10, tol * 0.1)
    iterations = 0

    def record_iteration(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    cg_method: Any = sparse_linalg.cg
    beta, info = cg_method(
        operator,
        rhs,
        x0=prior.ravel(),
        M=preconditioner,
        rtol=krylov_tol,
        atol=0.0,
        maxiter=maxiter or max(50, min(500, 5 * prior.size)),
        callback=record_iteration,
    )
    beta = np.asarray(beta, dtype=float)
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("CG returned a non-finite beta")

    residual = operator @ beta - rhs
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual = residual_norm / max(rhs_norm, np.finfo(float).tiny)
    if info != 0:
        raise RuntimeError(
            "CG did not converge: "
            f"info={int(info)}, iterations={iterations}, "
            f"relative_residual={relative_residual:.3e}"
        )
    certificate_limit = max(10.0 * krylov_tol, 256 * np.finfo(float).eps)
    if not np.isfinite(relative_residual) or relative_residual > certificate_limit:
        raise RuntimeError(
            "CG residual certificate failed: "
            f"{relative_residual:.3e} > {certificate_limit:.3e}"
        )
    return beta.reshape(prior.shape), int(info), iterations, relative_residual
