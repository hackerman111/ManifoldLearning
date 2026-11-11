"""Batched full-moment ridge preparation and a direct profile; opt-in only."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from ADP.core.solver_common import HPAOResult
from experiments import coupled_penalties as reference_solver
from experiments.full_ridge_fast import Problem, calculate_statistics
from experiments.full_ridge_fast import profile as direct_profile

__all__ = ["build", "calculate_statistics", "profile", "solve"]


def build(stats, basis):
    """Batch the reference spectral mask, ridge scale and constant preparation."""
    covariance, cross = np.asarray(stats.U), np.asarray(stats.I)
    j, d, m = len(stats.mass), *basis.shape
    if covariance.shape != (j, d, d) or cross.shape != (j, d):
        raise ValueError("direct moments require U:(J,d,d), I:(J,d)")
    if (
        not np.isfinite(covariance).all()
        or not np.isfinite(cross).all()
        or not np.isfinite(stats.mass).all()
        or np.any(stats.mass <= 0)
    ):
        raise ValueError("moments must be finite with positive mass")
    if not np.allclose(covariance, covariance.swapaxes(1, 2), atol=1e-13, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values[:, 0] < -1e-10 * np.maximum(values[:, -1], 1)):
        raise ValueError("covariance must be positive semidefinite")
    keep = values > (1e-12 * np.maximum(values[:, -1], np.finfo(float).tiny))[:, None]
    roots = np.sqrt(np.maximum(values, 0)) * keep
    inverse = np.divide(1.0, roots, out=np.zeros_like(roots), where=keep)
    projected = (vectors.swapaxes(1, 2) @ cross[..., None]).squeeze(-1)
    target = (vectors @ (inverse * projected)[..., None]).squeeze(-1)
    root = (vectors * roots[:, None, :]) @ vectors.swapaxes(1, 2)
    incoming_scale = float(np.median(np.sum((root @ basis) ** 2, axis=(1, 2))) / m)
    if incoming_scale == 0 and (np.any(root) or np.any(target)):
        raise ValueError("zero median ridge scale cannot guarantee SPD")
    pi = stats.mass / stats.mass.sum()
    energy = float(np.sum(pi[:, None] * target**2))
    ranks = keep.sum(axis=1)
    truncated = ranks < d
    if np.any(truncated):
        covariance, cross = covariance.copy(), cross.copy()
        v, retained = vectors[truncated], values[truncated] * keep[truncated]
        covariance[truncated] = (v * retained[:, None, :]) @ v.swapaxes(1, 2)
        cross[truncated] = (
            v @ (projected[truncated] * keep[truncated])[..., None]
        ).squeeze(-1)
    info = {
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "rank_histogram": np.bincount(ranks, minlength=d + 1).tolist(),
        "spectral_fallbacks": int(truncated.sum()),
        "covariance_cutoff_relative": 1e-12,
        "discarded_cross_relative": float(
            np.linalg.norm(projected * (~keep)) / max(np.linalg.norm(stats.I), 1e-300)
        ),
    }
    p = Problem(
        covariance, cross, pi, 0.01 * incoming_scale, energy, max(1.0, energy), info
    )
    # Near a numerical nullspace, root-products preserve the reference PSD solve.
    fallback = p.tau <= 1e-8 * max(float(values[:, -1].max()), np.finfo(float).tiny)
    p.reference_problem = (
        reference_solver.build(replace(stats, U=root, I=target), basis, "full")
        if fallback
        else None
    )
    info["whitened_fallback"] = bool(fallback)
    return p


def profile(basis, problem):
    if problem.reference_problem is not None:
        return reference_solver.profile(basis, problem.reference_problem)
    return direct_profile(basis, problem)


# Keep the archived reference QR chart arithmetic and budgets unchanged.
def solve(index, stats, *, max_steps=20, tol=1e-6):
    """The reference QR chart and L-BFGS settings with the direct profile."""
    started = time.perf_counter()
    original = np.asarray(index, dtype=float).T
    d, m = original.shape
    if not 1 <= m < d:
        raise ValueError("index dimension must lie between 1 and d-1")
    if not np.allclose(original.T @ original, np.eye(m), atol=1e-10):
        raise ValueError("index must have orthonormal rows")
    p = build(stats, original)
    full = np.linalg.qr(original, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    cache, history, residuals = {}, [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(cache["point"], flat):
            b, r = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(b, p)
            horizontal = state.gradient - b @ (b.T @ state.gradient)
            gradient = complement.T @ np.linalg.solve(r, horizontal.T).T
            cache.update(point=flat.copy(), basis=b, state=state, gradient=gradient)
            residuals.append(state.normal_residual)
        return cache["state"].value / p.scale, cache["gradient"].ravel() / p.scale

    initial = np.zeros(np.prod(shape))
    history.append(evaluate(initial)[0] * p.scale)

    def accepted(flat):
        value = evaluate(flat)[0] * p.scale
        if value > history[-1] + 1e-10 * max(1.0, history[-1]):
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
    b, state = cache["basis"], cache["state"]
    norm = float(np.linalg.norm(state.gradient - b @ (b.T @ state.gradient)) / p.scale)
    return HPAOResult(
        b.T,
        state.coefficients,
        {
            "method": "batched_full_ridge_chart_lbfgs",
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "iterations": int(result.nit),
            "profile_evaluations": len(residuals),
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": "stationary" if norm <= tol else str(result.message),
            "scipy_success": bool(result.success),
            "components": state.components,
            "normal_residual_max": max(residuals),
            "penalty": {
                "arm": "batched_full",
                "local_ridge": p.tau,
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "full_moments": p.info,
            },
        },
    )
