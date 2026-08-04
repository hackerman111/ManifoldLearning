from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr


def solve(
    statistics: dict[str, np.ndarray],
    beta: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Solve the single-index alternating problem with matrix-free LSMR."""
    I, U = statistics["I"], statistics["U"]
    stop = iterations = 0
    delta = math.inf

    for inner in range(max_steps):
        prior = beta
        slopes = _slopes(I, U, prior, local_ridge)
        beta, stop, iterations = _solve_beta(
            I,
            U,
            slopes,
            prior,
            lambda_penalty=lambda_penalty,
            tol=tol,
        )
        delta = min(np.linalg.norm(beta - prior), np.linalg.norm(beta + prior))
        if delta < tol:
            break

    return beta, _slopes(I, U, beta, local_ridge), {
        "inner_iterations": inner + 1,
        "lsmr_stop": stop,
        "lsmr_iterations": iterations,
        "beta_delta": float(delta),
    }


def _slopes(
    I: np.ndarray,
    U: np.ndarray,
    beta: np.ndarray,
    ridge: float,
) -> np.ndarray:
    projected = U @ beta
    return np.sum(I * projected, axis=1) / (
        np.sum(projected * projected, axis=1) + ridge
    )


def _solve_beta(
    I: np.ndarray,
    U: np.ndarray,
    slopes: np.ndarray,
    beta_prior: np.ndarray,
    *,
    lambda_penalty: float,
    tol: float,
) -> tuple[np.ndarray, int, int]:
    rows = I.size
    d = U.shape[2]
    sqrt_lambda = math.sqrt(lambda_penalty)

    def matvec(vector):
        data = (slopes[:, None] * (U @ vector)).ravel()
        return np.concatenate((data, sqrt_lambda * vector))

    def rmatvec(vector):
        data = vector[:rows].reshape(I.shape)
        return (
            np.einsum("j,jpd,jp->d", slopes, U, data, optimize=True)
            + sqrt_lambda * vector[rows:]
        )

    operator = LinearOperator(
        (rows + d, d),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    rhs = np.concatenate((I.ravel(), sqrt_lambda * beta_prior))
    result = lsmr(
        operator,
        rhs,
        atol=min(tol, 1e-8),
        btol=min(tol, 1e-8),
        maxiter=max(50, 5 * d),
    )
    beta = result[0]
    norm = np.linalg.norm(beta)
    if not np.all(np.isfinite(beta)) or not np.isfinite(norm) or norm == 0:
        raise RuntimeError("LSMR returned an invalid beta")
    beta /= norm
    if np.dot(beta, beta_prior) < 0:
        beta = -beta
    return beta, int(result[1]), int(result[2])
