import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ...ADP_Solver import ADP_SolverResult
from ...ADP_Statistic import ADP_Statistics
from ...gpu import require_cupy


def _backend(U):
    if isinstance(U, np.ndarray):
        return np, LinearOperator, lsmr
    cp = require_cupy()
    if not isinstance(U, cp.ndarray):
        raise TypeError("statistics.U must be a NumPy or CuPy array")
    from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator
    from cupyx.scipy.sparse.linalg import lsmr as cupy_lsmr

    return cp, CuPyLinearOperator, cupy_lsmr


def _item(value):
    return value.item() if hasattr(value, "item") else value


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

    backend = _backend(statistics.U)
    xp = backend[0]
    initial_index = xp.asarray(beta, dtype=float)
    if not bool(_item(xp.all(xp.isfinite(initial_index)))):
        raise ValueError("initial index must contain only finite values")
    parameters = {
        "lambda_penalty": lambda_penalty,
        "local_ridge": local_ridge,
        "max_steps": max_steps,
        "tol": tol,
        "backend": backend,
    }
    if initial_index.ndim == 1:
        result = _solve_single(statistics, initial_index, **parameters)
    elif initial_index.ndim == 2:
        result = _solve_multi(statistics, initial_index, **parameters)
    else:
        raise ValueError("initial index must have shape (d,) or (d, m)")

    if xp is not np:
        result.index = xp.asnumpy(result.index)
        if result.coefficients is not None:
            result.coefficients = xp.asnumpy(result.coefficients)
        if "eigenvalues" in result.diagnostics:
            result.diagnostics["eigenvalues"] = xp.asnumpy(
                result.diagnostics["eigenvalues"]
            )
    return result


def _solve_single(
    statistics: ADP_Statistics,
    beta: np.ndarray,
    *,
    lambda_penalty: float,
    local_ridge: float,
    max_steps: int,
    tol: float,
    backend,
) -> ADP_SolverResult:
    xp = backend[0]
    I, U = statistics.I, statistics.U
    stop = iterations = 0
    delta = math.inf

    for inner in range(max_steps):
        prior = beta
        slopes = _slopes(I, U, prior, local_ridge, xp)
        beta, stop, iterations = _solve_beta(
            I,
            U,
            slopes,
            prior,
            lambda_penalty=lambda_penalty,
            tol=tol,
            backend=backend,
        )
        delta = min(
            float(_item(xp.linalg.norm(beta - prior))),
            float(_item(xp.linalg.norm(beta + prior))),
        )
        if delta < tol:
            break

    return ADP_SolverResult(
        index=beta,
        coefficients=_slopes(I, U, beta, local_ridge, xp),
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
    backend,
) -> ADP_SolverResult:
    xp = backend[0]
    I, U = statistics.I, statistics.U
    d = U.shape[2]
    if basis.shape[0] != d or not 1 <= basis.shape[1] <= d:
        raise ValueError("multi-index basis must have shape (d, m) with 1 <= m <= d")

    basis, _ = xp.linalg.qr(basis, mode="reduced")
    basis = _orient_columns(basis, xp)
    stop = iterations = 0
    delta = math.inf
    eigenvalues = xp.ones(basis.shape[1])

    for inner in range(max_steps):
        prior = basis
        coefficients = _multi_coefficients(I, U, prior, local_ridge, xp)
        raw_basis, stop, iterations = _solve_basis(
            I,
            U,
            coefficients,
            prior,
            lambda_penalty=lambda_penalty,
            tol=tol,
            backend=backend,
        )
        basis, eigenvalues = _canonical_basis(raw_basis, coefficients, xp)
        delta = float(
            _item(
                xp.linalg.norm(
                    basis @ basis.T - prior @ prior.T,
                    ord="fro",
                )
            )
        ) / math.sqrt(2.0)
        if delta < tol:
            break

    return ADP_SolverResult(
        index=basis,
        coefficients=_multi_coefficients(I, U, basis, local_ridge, xp),
        diagnostics={
            "inner_iterations": inner + 1,
            "lsmr_stop": stop,
            "lsmr_iterations": iterations,
            "beta_delta": float(delta),
            "eigenvalues": eigenvalues.copy(),
        },
    )


def _multi_coefficients(I, U, basis, ridge, xp=np):
    projected = U @ basis
    gram = xp.einsum("jpm,jpn->jmn", projected, projected, optimize=True)
    gram += ridge * xp.eye(basis.shape[1])
    rhs = xp.einsum("jpm,jp->jm", projected, I, optimize=True)
    return xp.linalg.solve(gram, rhs[..., None])[..., 0]


def _solve_basis(
    I,
    U,
    coefficients,
    basis_prior,
    *,
    lambda_penalty,
    tol,
    backend=(np, LinearOperator, lsmr),
):
    xp, linear_operator, lsmr_method = backend
    rows = I.size
    d, m = basis_prior.shape
    size = d * m
    sqrt_lambda = math.sqrt(lambda_penalty)

    def matvec(vector):
        matrix = vector.reshape(d, m)
        data = xp.einsum(
            "jpd,dm,jm->jp",
            U,
            matrix,
            coefficients,
            optimize=True,
        )
        return xp.concatenate((data.ravel(), sqrt_lambda * vector))

    def rmatvec(vector):
        data = vector[:rows].reshape(I.shape)
        adjoint = xp.einsum(
            "jpd,jp,jm->dm",
            U,
            data,
            coefficients,
            optimize=True,
        )
        return (adjoint + sqrt_lambda * vector[rows:].reshape(d, m)).ravel()

    operator = linear_operator(
        (rows + size, size),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    rhs = xp.concatenate((I.ravel(), sqrt_lambda * basis_prior.ravel()))
    result = lsmr_method(
        operator,
        rhs,
        atol=min(tol, 1e-8),
        btol=min(tol, 1e-8),
        maxiter=max(50, 5 * size),
    )
    raw_basis = result[0].reshape(d, m)
    valid = bool(_item(xp.all(xp.isfinite(raw_basis))))
    if not valid or float(_item(xp.linalg.norm(raw_basis))) == 0:
        raise RuntimeError("LSMR returned an invalid multi-index basis")
    return raw_basis, int(_item(result[1])), int(_item(result[2]))


def _canonical_basis(raw_basis, coefficients, xp=np):
    values, vectors = xp.linalg.eigh(coefficients.T @ coefficients)
    factor = raw_basis @ (vectors * xp.sqrt(xp.maximum(values, 0.0)))
    basis, singular_values, _ = xp.linalg.svd(factor, full_matrices=False)
    if float(_item(singular_values[0])) <= xp.finfo(float).eps:
        raise RuntimeError("LSMR coefficients do not identify a multi-index basis")
    eigenvalues = xp.square(singular_values)
    eigenvalues /= eigenvalues.max()
    return _orient_columns(basis, xp), eigenvalues


def _orient_columns(basis, xp=np):
    columns = xp.arange(basis.shape[1])
    signs = xp.sign(basis[xp.argmax(xp.abs(basis), axis=0), columns])
    basis *= xp.where(signs == 0, 1.0, signs)
    return basis


def _slopes(
    I: np.ndarray,
    U: np.ndarray,
    beta: np.ndarray,
    ridge: float,
    xp=np,
) -> np.ndarray:
    projected = U @ beta
    return xp.sum(I * projected, axis=1) / (
        xp.sum(projected * projected, axis=1) + ridge
    )


def _solve_beta(
    I: np.ndarray,
    U: np.ndarray,
    slopes: np.ndarray,
    beta_prior: np.ndarray,
    *,
    lambda_penalty: float,
    tol: float,
    backend=(np, LinearOperator, lsmr),
) -> tuple[np.ndarray, int, int]:
    xp, linear_operator, lsmr_method = backend
    rows = I.size
    d = U.shape[2]
    sqrt_lambda = math.sqrt(lambda_penalty)

    def matvec(vector):
        data = (slopes[:, None] * (U @ vector)).ravel()
        return xp.concatenate((data, sqrt_lambda * vector))

    def rmatvec(vector):
        data = vector[:rows].reshape(I.shape)
        return (
            xp.einsum("j,jpd,jp->d", slopes, U, data, optimize=True)
            + sqrt_lambda * vector[rows:]
        )

    operator = linear_operator(
        (rows + d, d),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    rhs = xp.concatenate((I.ravel(), sqrt_lambda * beta_prior))
    result = lsmr_method(
        operator,
        rhs,
        atol=min(tol, 1e-8),
        btol=min(tol, 1e-8),
        maxiter=max(50, 5 * d),
    )
    beta = result[0]
    norm = float(_item(xp.linalg.norm(beta)))
    valid = bool(_item(xp.all(xp.isfinite(beta))))
    if not valid or not np.isfinite(norm) or norm == 0:
        raise RuntimeError("LSMR returned an invalid beta")
    beta /= norm
    if float(_item(xp.dot(beta, beta_prior))) < 0:
        beta = -beta
    return beta, int(_item(result[1])), int(_item(result[2]))
