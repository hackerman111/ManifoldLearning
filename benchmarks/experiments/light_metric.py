"""Frozen chart preconditioning for the experimental full-moment ridge."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ADP.core.solver_common import HPAOResult
from experiments import full_ridge_batched as backend

ARMS = ("identity", "right", "feature", "both")


def regularized_inverse_sqrt(matrix):
    """Dimensionless SPD coordinate factor; never changes statistical ridge."""
    matrix = np.asarray(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("coordinate metric must be finite")
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return np.eye(len(values))
    if values[0] < -1e-10 * scale:
        raise ValueError("coordinate metric must be positive semidefinite")
    values = np.maximum(values / scale, 0)
    mean = values.mean()
    factors = np.sqrt(1.1 * mean / (values + 0.1 * mean))
    return (vectors * factors) @ vectors.T


class Chart:
    def __init__(self, basis, problem, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.problem = problem
        d, m = basis.shape
        full = np.linalg.qr(basis, mode="complete")[0]
        self.reference, self.complement = full[:, :m], full[:, m:]
        self.shape = (d - m, m)
        self.left, self.right = np.eye(d - m), np.eye(m)
        self.factor_evaluations = 0
        if arm in ("right", "both"):
            state = backend.profile(self.reference, problem)
            gram = state.coefficients.T @ (problem.pi[:, None] * state.coefficients)
            self.right = regularized_inverse_sqrt(gram)
            self.factor_evaluations = 1
        if arm in ("feature", "both"):
            metric = np.einsum("j,jab->ab", problem.pi, problem.covariance)
            scale = np.trace(metric) / d
            if scale > 0:
                metric += 0.1 * scale * np.eye(d)
                mb = metric @ self.reference
                schur = metric - mb @ np.linalg.solve(self.reference.T @ mb, mb.T)
                self.left = regularized_inverse_sqrt(
                    self.complement.T @ schur @ self.complement
                )

    def evaluate(self, flat):
        coordinate = self.left @ flat.reshape(self.shape) @ self.right
        basis, triangular = np.linalg.qr(
            self.reference + self.complement @ coordinate, mode="reduced"
        )
        state = backend.profile(basis, self.problem)
        horizontal = state.gradient - basis @ (basis.T @ state.gradient)
        chart_gradient = self.complement.T @ np.linalg.solve(triangular, horizontal.T).T
        gradient = self.left.T @ chart_gradient @ self.right.T
        return (
            state.value / self.problem.scale,
            gradient.ravel() / self.problem.scale,
            basis,
            state,
        )


def solve(index, stats, arm, *, max_steps=20, tol=1e-6):
    started = time.perf_counter()
    incoming = np.asarray(index, dtype=float).T
    d, m = incoming.shape
    if not 1 <= m < d or not np.allclose(incoming.T @ incoming, np.eye(m), atol=1e-10):
        raise ValueError("index must have orthonormal rows and 1<=m<d")
    p = backend.build(stats, incoming)
    chart = Chart(incoming, p, arm)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    cache, history, residuals = {}, [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(flat, cache["point"]):
            value, gradient, basis, state = chart.evaluate(flat)
            cache.update(
                point=flat.copy(),
                value=value,
                gradient=gradient,
                basis=basis,
                state=state,
            )
            residuals.append(state.normal_residual)
        return cache["value"], cache["gradient"]

    initial = np.zeros(np.prod(chart.shape))
    history.append(evaluate(initial)[0] * p.scale)

    def accepted(flat):
        value = evaluate(flat)[0] * p.scale
        if value > history[-1] + 1e-10 * max(1.0, abs(history[-1])):
            raise RuntimeError("chart accepted an increasing objective")
        history.append(value)

    result = minimize(
        evaluate,
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=accepted,
        options={"maxiter": max_steps, "maxls": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    evaluate(result.x)
    basis, state = cache["basis"], cache["state"]
    norm = float(
        np.linalg.norm(state.gradient - basis @ (basis.T @ state.gradient)) / p.scale
    )
    return HPAOResult(
        basis.T,
        state.coefficients,
        {
            "method": "frozen_metric_chart_lbfgs",
            "metric": arm,
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "iterations": int(result.nit),
            "profile_evaluations": len(residuals) + chart.factor_evaluations,
            "factor_evaluations": chart.factor_evaluations,
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": "stationary" if norm <= tol else str(result.message),
            "scipy_success": bool(result.success),
            "components": state.components,
            "normal_residual_max": max(residuals),
            "penalty": {
                "arm": arm,
                "local_ridge": p.tau,
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "full_moments": p.info,
            },
            "coordinate_condition": {
                "left": float(np.linalg.cond(chart.left)),
                "right": float(np.linalg.cond(chart.right)),
            },
        },
    )
