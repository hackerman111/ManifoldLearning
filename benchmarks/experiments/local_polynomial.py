"""Experimental degree-1/2 local ridge-WLS on a Grassmann chart.

This estimator is isolated from ADP defaults; derivation and limits are in
benchmarks/results/verification/moment-quadratic-20261023/DERIVATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ADP.core.solver_common import HPAOResult


@dataclass
class LocalData:
    delta: np.ndarray
    response: np.ndarray
    weights: np.ndarray
    mass: float


@dataclass
class Neighborhoods:
    local: list[LocalData]
    mass: np.ndarray
    mean: np.ndarray
    response_mean: np.ndarray
    effective_n: np.ndarray
    support: np.ndarray
    scale: float


@dataclass
class Profile:
    value: float
    gradient: np.ndarray
    coefficients: np.ndarray
    hessians: np.ndarray
    penalty_value: float
    data_ranks: np.ndarray | None


def neighborhoods(X, y, weights) -> Neighborhoods:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (
        X.ndim != 2
        or y.shape != (len(X),)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid finite X/y shapes")
    blocks = [(0, weights)] if isinstance(weights, np.ndarray) else weights
    local, means, responses, masses, effective, support = [], [], [], [], [], []
    expected = 0
    for start, block in blocks:
        block = np.asarray(block, dtype=float)
        if start != expected or block.ndim != 2 or block.shape[1] != len(X):
            raise ValueError("invalid neighborhood block coverage")
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("weights must be finite and nonnegative")
        for row in block:
            indices = np.flatnonzero(row)
            mass = float(row.sum())
            if mass <= 0:
                raise ValueError("neighborhood mass must be positive")
            a = row[indices] / mass
            mean = a @ X[indices]
            response_mean = float(a @ y[indices])
            local.append(
                LocalData(X[indices] - mean, y[indices] - response_mean, a, mass)
            )
            means.append(mean)
            responses.append(response_mean)
            masses.append(mass)
            effective.append(1.0 / (a @ a))
            support.append(len(indices))
        expected += len(block)
    if not local:
        raise ValueError("at least one neighborhood is required")
    scale = max(1.0, sum(v.mass * np.sum(v.weights * v.response**2) for v in local))
    return Neighborhoods(
        local,
        np.array(masses),
        np.array(means),
        np.array(responses),
        np.array(effective),
        np.array(support),
        scale,
    )


def profile(
    basis, data: Neighborhoods, *, degree=2, ridge=1e-4, diagnostics=False
) -> Profile:
    basis = np.asarray(basis, dtype=float)
    d = data.mean.shape[1]
    if (
        basis.ndim != 2
        or basis.shape[0] != d
        or not 1 <= basis.shape[1] < d
        or not np.isfinite(basis).all()
    ):
        raise ValueError("basis must be finite with shape (d,m), 1<=m<d")
    if degree not in (1, 2) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("degree must be 1/2 and ridge nonnegative")
    m = basis.shape[1]
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    q = m if degree == 1 else m + m * (m + 1) // 2
    ridge_rows = np.sqrt(ridge) * np.eye(q)
    coefficients = np.empty((len(data.local), m))
    hessians = np.zeros((len(data.local), m, m))
    ranks = np.empty(len(data.local), dtype=int) if diagnostics else None
    value, penalty_value = 0.0, 0.0
    gradient = np.zeros_like(basis)
    for j, section in enumerate(data.local):
        z = section.delta @ basis
        features = np.empty((len(z), q))
        features[:, :m] = z
        if degree == 2:
            features[:, m : 2 * m] = z**2 / np.sqrt(2)
            for k, (a, b) in enumerate(pairs):
                features[:, 2 * m + k] = z[:, a] * z[:, b]
        features -= section.weights @ features
        root = np.sqrt(section.weights)
        design = root[:, None] * features
        target = root * section.response
        augmented = np.vstack((design, ridge_rows)) if ridge else design
        rhs = np.r_[target, np.zeros(q)] if ridge else target
        theta = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        if diagnostics:
            ranks[j] = np.linalg.matrix_rank(design)
        coefficients[j] = theta[:m]
        if degree == 2:
            hessians[j, np.arange(m), np.arange(m)] = np.sqrt(2) * theta[m : 2 * m]
            for k, (a, b) in enumerate(pairs):
                hessians[j, a, b] = hessians[j, b, a] = theta[2 * m + k]
        residual = section.response - features @ theta
        penalty = ridge * float(theta @ theta)
        value += section.mass * (np.sum(section.weights * residual**2) + penalty)
        penalty_value += section.mass * penalty
        local_gradient = theta[:m] + z @ hessians[j]
        gradient -= (
            2
            * section.mass
            * (
                section.delta.T
                @ ((section.weights * residual)[:, None] * local_gradient)
            )
        )
    if (
        not np.isfinite(value)
        or not np.isfinite(gradient).all()
        or not np.isfinite(coefficients).all()
    ):
        raise RuntimeError("nonfinite polynomial profile")
    return Profile(float(value), gradient, coefficients, hessians, penalty_value, ranks)


def chart_value_gradient(k, reference, complement, data, *, degree=2, ridge=1e-4):
    basis, triangular = np.linalg.qr(reference + complement @ k, mode="reduced")
    state = profile(basis, data, degree=degree, ridge=ridge)
    horizontal = state.gradient - basis @ (basis.T @ state.gradient)
    ambient_chart = np.linalg.solve(triangular, horizontal.T).T
    return state.value, complement.T @ ambient_chart


def solve(
    index, data: Neighborhoods, *, degree=2, ridge=1e-4, max_steps=40, tol=1e-6
) -> HPAOResult:
    initial = np.asarray(index, dtype=float).T
    d, m = initial.shape
    if not np.allclose(initial.T @ initial, np.eye(m), atol=1e-10):
        raise ValueError("initial index must have orthonormal rows")
    full = np.linalg.qr(initial, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    scale = data.scale
    first = profile(reference, data, degree=degree, ridge=ridge)
    history = [first.value]
    cache = {}
    evaluations = 1

    def evaluate(flat):
        nonlocal evaluations
        if "point" not in cache or not np.array_equal(cache["point"], flat):
            basis, triangular = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(basis, data, degree=degree, ridge=ridge)
            evaluations += 1
            horizontal = state.gradient - basis @ (basis.T @ state.gradient)
            chart_gradient = complement.T @ np.linalg.solve(triangular, horizontal.T).T
            cache.update(point=flat.copy(), state=state, gradient=chart_gradient)
        return cache["state"].value / scale, cache["gradient"].ravel() / scale

    def accepted(flat):
        value = evaluate(flat)[0] * scale
        if value > history[-1] + 1e-10 * max(1.0, history[-1]):
            raise RuntimeError("polynomial optimizer accepted an increasing objective")
        history.append(value)

    result = minimize(
        evaluate,
        np.zeros(np.prod(shape)),
        jac=True,
        method="L-BFGS-B",
        callback=accepted,
        options={"maxiter": max_steps, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    evaluate(result.x)
    basis = np.linalg.qr(
        reference + complement @ result.x.reshape(shape), mode="reduced"
    )[0]
    final = profile(basis, data, degree=degree, ridge=ridge, diagnostics=True)
    evaluations += 1
    if abs(history[-1] - final.value) > 1e-10 * max(1, final.value):
        raise RuntimeError("optimizer final point differs from accepted history")
    horizontal = final.gradient - basis @ (basis.T @ final.gradient)
    gradient_norm = float(np.linalg.norm(horizontal))
    q = m if degree == 1 else m + m * (m + 1) // 2
    return HPAOResult(
        basis.T,
        final.coefficients,
        {
            "method": f"local_wls_degree{degree}_chart_lbfgs",
            "profile_objective": final.value,
            "profile_history": tuple(history),
            "loss": final.value / 2,
            "loss_history": tuple(v / 2 for v in history),
            "converged": bool(gradient_norm / scale <= tol),
            "riemannian_gradient": gradient_norm / scale,
            "gradient_norm": gradient_norm,
            "stop_reason": "stationary"
            if gradient_norm / scale <= tol
            else str(result.message),
            "iterations": int(result.nit),
            "accepted_steps": int(result.nit),
            "profile_evaluations": evaluations,
            "scipy_nfev": int(result.nfev),
            "return_point": "scipy_final",
            "scipy_success": bool(result.success),
            "orthogonality_error": float(np.linalg.norm(basis.T @ basis - np.eye(m))),
            "local_ridge": ridge,
            "degree": degree,
            "parameter_count": q + 1,
            "support_below_parameters": int(np.count_nonzero(data.support < q + 1)),
            "effective_n_below_parameters": int(
                np.count_nonzero(data.effective_n < q + 1)
            ),
            "data_rank_histogram": np.bincount(
                final.data_ranks, minlength=q + 1
            ).tolist(),
            "ridge_objective_fraction": final.penalty_value
            / max(final.value, np.finfo(float).tiny),
            "effective_n_quantiles": np.quantile(
                data.effective_n, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
        },
    )
