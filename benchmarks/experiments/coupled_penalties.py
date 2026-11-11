"""Experimental graph-coupled moment profiles; production ADP is untouched."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist

from ADP.core.solver_common import HPAOResult

ARMS = (
    "base",
    "ridge",
    "chart",
    "mean0.01",
    "mean0.1",
    "smooth0.1",
    "joint0.1",
    "full",
    "full_joint0.1",
)


@dataclass
class Problem:
    stats: object
    pi: np.ndarray
    tau: float
    edges: np.ndarray
    weights: np.ndarray
    delta: np.ndarray
    dy: np.ndarray
    incidence: csr_matrix
    lmean: float
    lsmooth: float
    scale: float


@dataclass
class Profile:
    value: float
    gradient: np.ndarray
    coefficients: np.ndarray
    components: dict
    cg_iterations: int
    normal_residual: float


def full_moments(stats, covariance, cross):
    """Whiten full moments on retained covariance modes, including singular C."""
    values, vectors = np.linalg.eigh(covariance)
    cutoff = 1e-12 * np.maximum(values[:, -1], np.finfo(float).tiny)
    if np.any(values[:, 0] < -1e-10 * np.maximum(values[:, -1], 1)):
        raise ValueError("covariance must be positive semidefinite")
    keep = values > cutoff[:, None]
    root = np.sqrt(np.maximum(values, 0)) * keep
    inverse = np.divide(1.0, root, out=np.zeros_like(root), where=keep)
    u = (vectors * root[:, None, :]) @ vectors.swapaxes(1, 2)
    projected = (vectors.swapaxes(1, 2) @ cross[..., None]).squeeze(-1)
    target = (vectors @ (inverse * projected)[..., None]).squeeze(-1)
    ranks = keep.sum(axis=1)
    return replace(stats, U=u, I=target), {
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "rank_histogram": np.bincount(ranks, minlength=u.shape[2] + 1).tolist(),
        "covariance_cutoff_relative": 1e-12,
        "discarded_cross_relative": float(
            np.linalg.norm(projected * (~keep)) / max(np.linalg.norm(cross), 1e-300)
        ),
    }


def build(stats, basis, arm):
    """Freeze all graph and penalty scales at the incoming basis."""
    if arm not in ARMS or arm in ("base", "ridge"):
        raise ValueError(f"invalid coupled arm: {arm}")
    j, m = len(stats.mass), basis.shape[1]
    if j < 2 or stats.S.shape != (j,) or np.any(stats.mass <= 0):
        raise ValueError("graph requires at least two positive-mass centers and S")
    pi = stats.mass / stats.mass.sum()
    scale = float(np.median(np.sum((stats.U @ basis) ** 2, axis=(1, 2))) / m)
    if scale == 0 and (np.any(stats.U) or np.any(stats.I)):
        raise ValueError("zero median ridge scale cannot guarantee SPD")
    distances = cdist(stats.mean, stats.mean, metric="sqeuclidean")
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(4, j - 1)]
    edges = np.unique(
        np.sort(
            np.c_[np.repeat(np.arange(j), neighbors.shape[1]), neighbors.ravel()],
            axis=1,
        ),
        axis=0,
    )
    a, b = edges.T
    edge_distances = distances[a, b]
    bandwidth = max(float(np.median(edge_distances)), 1e-12)
    weights = np.exp(-edge_distances / bandwidth)
    weights /= weights.sum()
    root = np.sqrt(weights)
    incidence = csr_matrix(
        (
            np.c_[-root, root].ravel(),
            (np.repeat(np.arange(len(edges)), 2), edges.ravel()),
        ),
        shape=(len(edges), j),
    )
    dy = stats.S[b] - stats.S[a]
    energy = float(np.sum(pi[:, None] * stats.I**2))
    beta = (
        0.01
        if arm == "mean0.01"
        else (0.1 if arm in ("mean0.1", "joint0.1", "full_joint0.1") else 0.0)
    )
    gamma = 0.1 if arm in ("smooth0.1", "joint0.1", "full_joint0.1") else 0.0
    lmean = (
        beta
        * energy
        / max(float(weights @ dy**2), 1e-12 * max(1.0, float(pi @ stats.S**2)))
    )
    return Problem(
        stats,
        pi,
        0.01 * scale,
        edges,
        weights,
        stats.mean[b] - stats.mean[a],
        dy,
        incidence,
        lmean,
        gamma * scale,
        max(1.0, energy),
    )


def profile(basis, problem):
    """Joint SPD coefficient solve and the envelope gradient."""
    p, s = problem, problem.stats
    j, m = len(p.pi), basis.shape[1]
    design = s.U @ basis
    blocks = p.pi[:, None, None] * (design.swapaxes(1, 2) @ design + p.tau * np.eye(m))
    rhs = p.pi[:, None] * (design.swapaxes(1, 2) @ s.I[..., None]).squeeze(-1)
    a, b = p.edges.T
    z = p.delta @ basis / 2
    q = csr_matrix(
        (
            np.repeat((np.sqrt(p.weights)[:, None] * z)[:, None, :], 2, axis=1).ravel(),
            (
                np.repeat(np.arange(len(a)), 2 * m),
                (p.edges[..., None] * m + np.arange(m)).ravel(),
            ),
        ),
        shape=(len(a), j * m),
    )
    if p.lmean:
        rhs += (p.lmean * (q.T @ (np.sqrt(p.weights) * p.dy))).reshape(j, m)
    iterations = 0
    if not np.any(s.U) and not np.any(s.I):
        coefficients = np.zeros((j, m))
        # A constant response provides no coefficient information in this case.
        if p.lmean:
            raise ValueError("zero moment scale with nonzero mean penalty")
        normal_residual = 0.0
    elif not p.lmean and not p.lsmooth:
        coefficients = np.linalg.solve(blocks, rhs[..., None]).squeeze(-1)
        normal_residual = float(
            np.linalg.norm((blocks @ coefficients[..., None]).squeeze(-1) - rhs)
            / max(np.linalg.norm(rhs), 1e-300)
        )
    else:

        def multiply(flat):
            g = flat.reshape(j, m)
            output = (blocks @ g[..., None]).squeeze(-1)
            if p.lmean:
                output += (p.lmean * (q.T @ (q @ flat))).reshape(j, m)
            if p.lsmooth:
                output += p.lsmooth * (p.incidence.T @ (p.incidence @ g))
            return output.ravel()

        diagonal = blocks.copy()
        if p.lmean:
            for node in (a, b):
                np.add.at(
                    diagonal,
                    node,
                    p.lmean * p.weights[:, None, None] * z[..., None] * z[:, None, :],
                )
        if p.lsmooth:
            degrees = np.bincount(
                p.edges.ravel(), weights=np.repeat(p.weights, 2), minlength=j
            )
            diagonal += p.lsmooth * degrees[:, None, None] * np.eye(m)
        inverse = np.linalg.inv(diagonal)
        operator = LinearOperator((j * m, j * m), matvec=multiply, dtype=float)
        preconditioner = LinearOperator(
            operator.shape,
            matvec=lambda v: (inverse @ v.reshape(j, m, 1)).ravel(),
            dtype=float,
        )

        def count(_):
            nonlocal iterations
            iterations += 1

        flat, info = cg(
            operator,
            rhs.ravel(),
            M=preconditioner,
            rtol=1e-10,
            atol=0.0,
            maxiter=500,
            callback=count,
        )
        normal_residual = float(
            np.linalg.norm(multiply(flat) - rhs.ravel())
            / max(np.linalg.norm(rhs), 1e-300)
        )
        if info != 0 or normal_residual > 1e-8:
            raise RuntimeError(
                f"CG failed: info={info}, relative residual={normal_residual}"
            )
        coefficients = flat.reshape(j, m)
    residual = s.I - (design @ coefficients[..., None]).squeeze(-1)
    average = (coefficients[a] + coefficients[b]) / 2
    mean_residual = p.dy - np.sum((p.delta @ basis) * average, axis=1)
    components = {
        "data": float(np.sum(p.pi[:, None] * residual**2)),
        "ridge": float(p.tau * np.sum(p.pi[:, None] * coefficients**2)),
        "mean": float(p.lmean * np.sum(p.weights * mean_residual**2)),
        "smooth": float(p.lsmooth * np.sum((p.incidence @ coefficients) ** 2)),
    }
    gradient = -2 * np.einsum(
        "j,jpd,jp,jm->dm", p.pi, s.U, residual, coefficients, optimize=True
    )
    if p.lmean:
        gradient -= (
            2 * p.lmean * p.delta.T @ ((p.weights * mean_residual)[:, None] * average)
        )
    return Profile(
        sum(components.values()),
        gradient,
        coefficients,
        components,
        iterations,
        normal_residual,
    )


def solve(index, stats, arm, *, max_steps=20, tol=1e-6, moments=None):
    """Profile graph-coupled coefficients in a QR Grassmann chart."""
    started = time.perf_counter()
    original = np.asarray(index, dtype=float).T
    d, m = original.shape
    if not np.allclose(original.T @ original, np.eye(m), atol=1e-10):
        raise ValueError("index must have orthonormal rows")
    moment_info = {}
    if arm.startswith("full"):
        if moments is None:
            raise ValueError("full arms require full moments")
        stats, moment_info = full_moments(stats, *moments)
    p = build(stats, original, arm)
    full = np.linalg.qr(original, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    cache, history, cg_effort, normal_residuals = {}, [], [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(cache["point"], flat):
            b, r = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(b, p)
            horizontal = state.gradient - b @ (b.T @ state.gradient)
            gradient = complement.T @ np.linalg.solve(r, horizontal.T).T
            cache.update(point=flat.copy(), basis=b, state=state, gradient=gradient)
            cg_effort.append(state.cg_iterations)
            normal_residuals.append(state.normal_residual)
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
    horizontal = state.gradient - b @ (b.T @ state.gradient)
    norm = float(np.linalg.norm(horizontal) / p.scale)
    return HPAOResult(
        b.T,
        state.coefficients,
        {
            "method": "coupled_chart_lbfgs",
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "iterations": int(result.nit),
            "profile_evaluations": len(cg_effort),
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": "stationary" if norm <= tol else str(result.message),
            "scipy_success": bool(result.success),
            "components": state.components,
            "cg_iterations_total": int(sum(cg_effort)),
            "cg_iterations_max": max(cg_effort),
            "normal_residual_max": max(normal_residuals),
            "penalty": {
                "arm": arm,
                "local_ridge": p.tau,
                "mean_coefficient": p.lmean,
                "smooth_coefficient": p.lsmooth,
                "edges": len(p.edges),
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "full_moments": moment_info,
            },
        },
    )
