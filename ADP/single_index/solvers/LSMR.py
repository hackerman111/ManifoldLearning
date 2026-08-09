import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ...ADP_Solver import ADP_SolverResult
from ...ADP_Statistic import ADP_Statistics


def solve(
    statistics: ADP_Statistics,
    beta: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int = 2,
    tol: float = 1e-6,
) -> ADP_SolverResult:
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    for name, value in (
        ("lambda_penalty", lambda_penalty),
        ("local_ridge", local_ridge),
        ("tol", tol),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    initial_index = np.asarray(beta, dtype=float)
    if not np.all(np.isfinite(initial_index)):
        raise ValueError("initial index must contain only finite values")
    parameters = {
        "lambda_penalty": lambda_penalty,
        "local_ridge": local_ridge,
        "max_steps": max_steps,
        "tol": tol,
    }
    if initial_index.ndim == 1:
        return _solve_single(statistics, initial_index, **parameters)
    if initial_index.ndim == 2:
        return _solve_multi(statistics, initial_index, **parameters)
    raise ValueError("initial index must have shape (d,) or (d, m)")


def _solve_single(
    statistics: ADP_Statistics,
    beta: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int,
    tol: float,
) -> ADP_SolverResult:
    I, U = statistics.I, statistics.U
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

    return ADP_SolverResult(
        index=beta,
        coefficients=_slopes(I, U, beta, local_ridge),
        diagnostics={
            "inner_iterations": inner + 1,
            "lsmr_stop": stop,
            "lsmr_iterations": iterations,
            "beta_delta": float(delta),
        },
    )


def _solve_multi(
    statistics: ADP_Statistics,
    basis: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int,
    tol: float,
) -> ADP_SolverResult:
    I, U = statistics.I, statistics.U
    d = U.shape[2]
    if basis.shape[0] != d or not 1 <= basis.shape[1] <= d:
        raise ValueError("multi-index basis must have shape (d, m) with 1 <= m <= d")

    basis, _ = np.linalg.qr(basis, mode="reduced")
    basis = _orient_columns(basis)
    stop = iterations = 0
    delta = math.inf
    eigenvalues = np.ones(basis.shape[1])

    for inner in range(max_steps):
        prior = basis
        coefficients = _multi_coefficients(I, U, prior, local_ridge)
        raw_basis, stop, iterations = _solve_basis(
            I,
            U,
            coefficients,
            prior,
            lambda_penalty=lambda_penalty,
            tol=tol,
        )
        basis, eigenvalues = _canonical_basis(raw_basis, coefficients)
        delta = np.linalg.norm(
            basis @ basis.T - prior @ prior.T,
            ord="fro",
        ) / math.sqrt(2.0)
        if delta < tol:
            break

    return ADP_SolverResult(
        index=basis,
        coefficients=_multi_coefficients(I, U, basis, local_ridge),
        diagnostics={
            "inner_iterations": inner + 1,
            "lsmr_stop": stop,
            "lsmr_iterations": iterations,
            "beta_delta": float(delta),
            "eigenvalues": eigenvalues.copy(),
        },
    )


def _multi_coefficients(I, U, basis, ridge):
    projected = U @ basis
    gram = np.einsum("jpm,jpn->jmn", projected, projected, optimize=True)
    gram += ridge * np.eye(basis.shape[1])
    rhs = np.einsum("jpm,jp->jm", projected, I, optimize=True)
    return np.linalg.solve(gram, rhs[..., None])[..., 0]


def _solve_basis(
    I,
    U,
    coefficients,
    basis_prior,
    *,
    lambda_penalty,
    tol,
):
    rows = I.size
    d, m = basis_prior.shape
    size = d * m
    sqrt_lambda = math.sqrt(lambda_penalty)

    def matvec(vector):
        matrix = vector.reshape(d, m)
        data = np.einsum(
            "jpd,dm,jm->jp",
            U,
            matrix,
            coefficients,
            optimize=True,
        )
        return np.concatenate((data.ravel(), sqrt_lambda * vector))

    def rmatvec(vector):
        data = vector[:rows].reshape(I.shape)
        adjoint = np.einsum(
            "jpd,jp,jm->dm",
            U,
            data,
            coefficients,
            optimize=True,
        )
        return (adjoint + sqrt_lambda * vector[rows:].reshape(d, m)).ravel()

    operator = LinearOperator(
        (rows + size, size),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    rhs = np.concatenate((I.ravel(), sqrt_lambda * basis_prior.ravel()))
    result = lsmr(
        operator,
        rhs,
        atol=min(tol, 1e-8),
        btol=min(tol, 1e-8),
        maxiter=max(50, 5 * size),
    )
    raw_basis = result[0].reshape(d, m)
    if not np.all(np.isfinite(raw_basis)) or np.linalg.norm(raw_basis) == 0:
        raise RuntimeError("LSMR returned an invalid multi-index basis")
    return raw_basis, int(result[1]), int(result[2])


def _canonical_basis(raw_basis, coefficients):
    values, vectors = np.linalg.eigh(coefficients.T @ coefficients)
    factor = raw_basis @ (vectors * np.sqrt(np.maximum(values, 0.0)))
    basis, singular_values, _ = np.linalg.svd(factor, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        raise RuntimeError("LSMR coefficients do not identify a multi-index basis")
    return _orient_columns(basis), np.square(singular_values)


def _orient_columns(basis):
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


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
