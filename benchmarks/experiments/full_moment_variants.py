"""Opt-in full0, correlated residuals, joint smoothness and tangent ablation."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.core.solver_common import HPAOResult
from experiments import full_ridge_fast
from experiments.coupled_penalties import Profile


@dataclass(frozen=True, slots=True)
class WindowStatistics(ADP_Statistics):
    windows: tuple = ()


def calculate_statistics(X, Y, weights, directions, batch_size=32, *, normalized=False):
    blocks = list(weights)
    stats = full_ridge_fast.calculate_statistics(
        X, Y, iter(blocks), directions, batch_size, normalized=normalized
    )
    windows = []
    for _, block in blocks:
        for row in block:
            indices = np.flatnonzero(row > 0)
            a = row[indices] / row.sum()
            windows.append((indices, a, X[indices] - a @ X[indices]))
    return WindowStatistics(
        **{field.name: getattr(stats, field.name) for field in fields(stats)},
        windows=tuple(windows),
    )


@dataclass
class Problem:
    root: np.ndarray
    inverse_root: np.ndarray
    target: np.ndarray
    covariance_ranks: np.ndarray
    pi: np.ndarray
    groups: list
    metric_roots: np.ndarray
    noise_covariances: list
    edges: np.ndarray
    edge_weights: np.ndarray
    thresholds: np.ndarray
    lsmooth: float
    scale: float
    info: dict


def build(stats, basis, arm, *, smoothness=2.0):
    if arm not in ("full0", "metric", "coherence", "tangent"):
        raise ValueError(arm)
    covariance, cross = np.asarray(stats.U), np.asarray(stats.I)
    j, d = cross.shape
    if covariance.shape != (j, d, d) or np.any(stats.mass <= 0):
        raise ValueError("invalid moment shapes or masses")
    if not all(np.isfinite(a).all() for a in (covariance, cross, stats.mass)):
        raise ValueError("moments must be finite")
    if not np.allclose(covariance, covariance.swapaxes(1, 2), atol=1e-13):
        raise ValueError("covariance must be symmetric")
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values[:, 0] < -1e-10 * np.maximum(values[:, -1], 1)):
        raise ValueError("covariance must be positive semidefinite")
    keep = values > 1e-12 * np.maximum(values[:, -1:], np.finfo(float).tiny)
    roots = np.sqrt(np.maximum(values, 0)) * keep
    inverse = np.divide(1, roots, out=np.zeros_like(roots), where=keep)
    root = (vectors * roots[:, None]) @ vectors.swapaxes(1, 2)
    inverse_root = (vectors * inverse[:, None]) @ vectors.swapaxes(1, 2)
    target = (inverse_root @ cross[..., None]).squeeze(-1)
    pi = stats.mass / stats.mass.sum()
    distances = cdist(stats.mean, stats.mean)
    np.fill_diagonal(distances, np.inf)
    groups = []
    correlated = arm in ("metric", "coherence")
    remaining = set(range(j))
    while remaining:
        a = min(remaining)
        remaining.remove(a)
        group = [a]
        if correlated and remaining:
            b = min(remaining, key=lambda k: (distances[a, k], k))
            group.append(b)
            remaining.remove(b)
        groups.append(group)
    kmax = 2 if correlated else 1
    noise = []
    if correlated:
        if len(stats.windows) != j:
            raise ValueError("metric requires the observed window weights")
        psi = [
            np.sqrt(pi[k]) * inverse_root[k] @ (z.T * a)
            for k, (_, a, z) in enumerate(stats.windows)
        ]
        for group in groups:
            sigma = np.zeros((kmax * d, kmax * d))
            for a, ka in enumerate(group):
                for b, kb in enumerate(group):
                    _, ia, ib = np.intersect1d(
                        stats.windows[ka][0], stats.windows[kb][0], return_indices=True
                    )
                    sigma[a * d : (a + 1) * d, b * d : (b + 1) * d] = (
                        psi[ka][:, ia] @ psi[kb][:, ib].T
                    )
            noise.append(sigma)
        levels = np.array(
            [np.trace(s) / (len(g) * d) for s, g in zip(noise, groups, strict=True)]
        )
        positive = levels[levels > 0]
        noise_scale = float(np.median(positive)) if len(positive) else 1.0
        matrices = np.array(
            [
                noise_scale * np.linalg.inv(s + 0.1 * noise_scale * np.eye(kmax * d))
                for s in noise
            ]
        )
        normalization = sum(
            np.trace(w[: len(g) * d, : len(g) * d])
            for g, w in zip(groups, matrices, strict=True)
        ) / (j * d)
        matrices /= normalization
        eigen, rotation = np.linalg.eigh(matrices)
        metric_roots = (rotation * np.sqrt(eigen)[:, None]) @ rotation.swapaxes(1, 2)
    else:
        noise_scale = 0.0
        metric_roots = np.broadcast_to(np.eye(d), (j, d, d)).copy()
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(4, j - 1)]
    edges = np.unique(
        np.sort(
            np.c_[np.repeat(np.arange(j), neighbors.shape[1]), neighbors.ravel()],
            axis=1,
        ),
        axis=0,
    )
    edge_distances = distances[edges[:, 0], edges[:, 1]]
    weights = np.exp(
        -(edge_distances**2) / max(float(np.median(edge_distances**2)), 1e-12)
    )
    if len(weights):
        weights /= weights.sum()
    incoming_scale = float(
        np.median(np.sum((root @ basis) ** 2, axis=(1, 2))) / basis.shape[1]
    )
    ranks = keep.sum(axis=1)
    design_singular_values = np.linalg.svd(root @ basis, compute_uv=False)
    design_ranks = np.sum(
        design_singular_values > 1e-12 * design_singular_values[:, :1], axis=1
    )
    return Problem(
        root,
        inverse_root,
        target,
        ranks,
        pi,
        groups,
        metric_roots,
        noise,
        edges,
        weights,
        smoothness * edge_distances,
        0.1 * incoming_scale if arm == "coherence" else 0.0,
        max(1.0, float(np.sum(pi[:, None] * target**2))),
        {
            "rank_histogram": np.bincount(ranks, minlength=d + 1).tolist(),
            "incoming_design_rank_histogram": np.bincount(
                design_ranks, minlength=basis.shape[1] + 1
            ).tolist(),
            "mass_min_median_max": [
                float(np.min(stats.mass)),
                float(np.median(stats.mass)),
                float(np.max(stats.mass)),
            ],
            "n_eff_min_median_max": [
                float(np.min(stats.n_eff)),
                float(np.median(stats.n_eff)),
                float(np.max(stats.n_eff)),
            ],
            "noise_metric_shift": 0.1 * noise_scale,
            "smoothness": smoothness,
            "support_min": min((len(w[0]) for w in stats.windows), default=0),
            "support_max": max((len(w[0]) for w in stats.windows), default=0),
        },
    )


def _design(basis, p):
    d, m = p.root.shape[1], basis.shape[1]
    kmax = p.metric_roots.shape[1] // d
    design = np.zeros((len(p.groups), kmax * d, kmax * m))
    target = np.zeros((len(p.groups), kmax * d))
    for group_index, group in enumerate(p.groups):
        for node, k in enumerate(group):
            design[
                group_index, node * d : (node + 1) * d, node * m : (node + 1) * m
            ] = np.sqrt(p.pi[k]) * p.root[k] @ basis
            target[group_index, node * d : (node + 1) * d] = (
                np.sqrt(p.pi[k]) * p.target[k]
            )
    return p.metric_roots @ design, (p.metric_roots @ target[..., None]).squeeze(-1)


def _svd_fit(design, target, expected_ranks):
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    retained = s > 1e-12 * s[:, :1]
    if np.any(retained.sum(axis=1) < expected_ranks):
        raise RuntimeError(
            "nonregular coefficient design: rank drop invalidates envelope stationarity"
        )
    inverse = np.divide(1, s, out=np.zeros_like(s), where=retained)
    return (
        vh.swapaxes(1, 2)
        @ (inverse * (u.swapaxes(1, 2) @ target[..., None]).squeeze(-1))[..., None]
    ).squeeze(-1)


def _pack(coefficients, p, m):
    kmax = p.metric_roots.shape[1] // p.root.shape[1]
    result = np.zeros((len(p.groups), kmax * m))
    for group_index, group in enumerate(p.groups):
        result[group_index, : len(group) * m] = coefficients[group].ravel()
    return result


def _unpack(packed, p, m):
    result = np.zeros((len(p.pi), m))
    for group_index, group in enumerate(p.groups):
        result[group] = packed[group_index, : len(group) * m].reshape(len(group), m)
    return result


def profile(basis, p):
    d, m = basis.shape
    design, target = _design(basis, p)
    expected_ranks = np.array(
        [np.minimum(p.covariance_ranks[group], m).sum() for group in p.groups]
    )
    packed = _svd_fit(design, target, expected_ranks)
    coefficients = _unpack(packed, p, m)
    a, b = p.edges.T
    iterations = 0

    def objective(g):
        packed_g = _pack(g, p, m)
        residual = (design @ packed_g[..., None]).squeeze(-1) - target
        gradient = _unpack(
            2 * (design.swapaxes(1, 2) @ residual[..., None]).squeeze(-1), p, m
        )
        data = float(np.sum(residual**2))
        norms = np.linalg.norm(g[a] - g[b], axis=1)
        excess = np.maximum(norms - p.thresholds, 0)
        smooth = float(p.lsmooth * np.sum(p.edge_weights * excess**2))
        if p.lsmooth:
            factor = 2 * p.lsmooth * p.edge_weights * excess / np.maximum(norms, 1e-300)
            edge_gradient = factor[:, None] * (g[a] - g[b])
            np.add.at(gradient, a, edge_gradient)
            np.add.at(gradient, b, -edge_gradient)
        return (
            data + smooth,
            gradient,
            residual,
            data,
            smooth,
            int(np.count_nonzero(excess)),
        )

    rhs_norm = max(
        float(np.linalg.norm(2 * (design.swapaxes(1, 2) @ target[..., None]))), 1.0
    )
    if p.lsmooth and objective(coefficients)[4] > 1e-24:
        hessian = design.swapaxes(1, 2) @ design
        diagonal = np.zeros((len(p.pi), m, m))
        for gi, group in enumerate(p.groups):
            for node, k in enumerate(group):
                diagonal[k] = hessian[
                    gi, node * m : (node + 1) * m, node * m : (node + 1) * m
                ]
        degrees = np.bincount(
            p.edges.ravel(), weights=np.repeat(p.edge_weights, 2), minlength=len(p.pi)
        )
        diagonal += p.lsmooth * degrees[:, None, None] * np.eye(m)
        eigen, rotation = np.linalg.eigh(diagonal)
        floor = 1e-10 * max(float(eigen.max()), 1e-12)
        preconditioner = (
            rotation * (1 / np.sqrt(np.maximum(eigen, floor)))[:, None]
        ) @ rotation.swapaxes(1, 2)
        origin = coefficients.copy()

        def inner(flat):
            g = origin + (preconditioner @ flat.reshape(-1, m, 1)).squeeze(-1)
            value, gradient, *_ = objective(g)
            return value, (preconditioner @ gradient[..., None]).ravel()

        result = minimize(
            inner,
            np.zeros_like(origin).ravel(),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": 600,
                "maxls": 30,
                "gtol": 1e-11,
                "ftol": 1e-15,
                "maxcor": 20,
            },
        )
        coefficients = origin + (preconditioner @ result.x.reshape(-1, m, 1)).squeeze(
            -1
        )
        iterations = int(result.nit)
    value, normal, residual, data, smooth, active = objective(coefficients)
    certificate = float(np.linalg.norm(normal) / rhs_norm)
    if certificate > 2e-7:
        raise RuntimeError(f"uncertified joint coefficient profile: {certificate:.3g}")
    raw_residual = (p.metric_roots.swapaxes(1, 2) @ residual[..., None]).squeeze(-1)
    residuals = np.zeros((len(p.pi), d))
    for gi, group in enumerate(p.groups):
        for node, k in enumerate(group):
            residuals[k] = raw_residual[gi, node * d : (node + 1) * d]
    gradient = 2 * np.sum(
        np.sqrt(p.pi)[:, None, None]
        * (p.root.swapaxes(1, 2) @ residuals[..., None])
        * coefficients[:, None, :],
        axis=0,
    )
    return Profile(
        value,
        gradient,
        coefficients,
        {"data": data, "ridge": 0.0, "smooth": smooth, "active_edges": active},
        iterations,
        certificate,
    )


def tangent_system(basis, state, p):
    d, m = basis.shape
    complement = np.linalg.qr(basis, mode="complete")[0][:, m:]
    design = p.root @ basis
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    u = u * (s > 1e-12 * s[:, :1])[:, None, :]
    projected = p.root @ complement
    projected -= u @ (u.swapaxes(1, 2) @ projected)
    t = (projected[:, :, :, None] * state.coefficients[:, None, None, :]).reshape(
        len(p.pi), d, -1
    )
    residual = p.target - (design @ state.coefficients[..., None]).squeeze(-1)
    hessian = np.einsum("j,jdq,jdr->qr", p.pi, t, t)
    score = np.einsum("j,jdq,jd->q", p.pi, t, residual)
    return hessian, score


def solve(index, stats, arm, *, max_steps=20, tol=1e-6):
    started = time.perf_counter()
    original = np.asarray(index, dtype=float).T
    d, m = original.shape
    if not 1 <= m < d or not np.allclose(original.T @ original, np.eye(m), atol=1e-10):
        raise ValueError("invalid orthonormal basis")
    p = build(stats, original, arm)
    full = np.linalg.qr(original, mode="complete")[0]
    reference, complement = full[:, :m], full[:, m:]
    shape = (d - m, m)
    preparation = time.perf_counter() - started
    started = time.perf_counter()
    cache, history, certificates, inner_iterations = {}, [], [], []

    def evaluate(flat):
        if "point" not in cache or not np.array_equal(cache["point"], flat):
            b, r = np.linalg.qr(
                reference + complement @ flat.reshape(shape), mode="reduced"
            )
            state = profile(b, p)
            horizontal = state.gradient - b @ (b.T @ state.gradient)
            gradient = complement.T @ np.linalg.solve(r, horizontal.T).T
            cache.update(point=flat.copy(), basis=b, state=state, gradient=gradient)
            certificates.append(state.normal_residual)
            inner_iterations.append(state.cg_iterations)
        return cache["state"].value / p.scale, cache["gradient"].ravel() / p.scale

    initial = np.zeros(np.prod(shape))
    history.append(evaluate(initial)[0] * p.scale)
    if arm == "tangent":
        b, state = cache["basis"], cache["state"]
        accepted_steps, trials = 0, 0
        for _ in range(max_steps):
            horizontal = state.gradient - b @ (b.T @ state.gradient)
            if np.linalg.norm(horizontal) / p.scale <= tol:
                break
            hessian, score = tangent_system(b, state, p)
            damping = 0.05 * max(float(np.trace(hessian) / len(hessian)), 1e-12)
            step = np.linalg.solve(
                hessian + damping * np.eye(len(hessian)), score
            ).reshape(shape)
            delta = np.linalg.qr(b, mode="complete")[0][:, m:] @ step
            slope = float(np.sum(state.gradient * delta))
            if slope >= 0:
                break
            for shrink in range(20):
                trials += 1
                rate = 0.5**shrink
                candidate = np.linalg.qr(b + rate * delta)[0]
                next_state = profile(candidate, p)
                certificates.append(next_state.normal_residual)
                inner_iterations.append(next_state.cg_iterations)
                if next_state.value <= state.value + 1e-4 * rate * slope:
                    b, state = candidate, next_state
                    history.append(state.value)
                    accepted_steps += 1
                    break
            else:
                break
        iterations, scipy_success, message = (
            accepted_steps,
            None,
            "tangent_budget_or_line_search",
        )
    else:

        def accepted(flat):
            value = evaluate(flat)[0] * p.scale
            if value > history[-1] + 1e-10 * max(1, history[-1]):
                raise RuntimeError("increasing chart objective")
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
        iterations, scipy_success, message = (
            int(result.nit),
            bool(result.success),
            str(result.message),
        )
    norm = float(np.linalg.norm(state.gradient - b @ (b.T @ state.gradient)) / p.scale)
    return HPAOResult(
        b.T,
        state.coefficients,
        {
            "method": f"experimental_{arm}",
            "profile_objective": state.value,
            "loss": state.value / 2,
            "profile_history": history,
            "iterations": iterations,
            "profile_evaluations": len(certificates),
            "converged": norm <= tol,
            "riemannian_gradient": norm,
            "stop_reason": "stationary" if norm <= tol else message,
            "scipy_success": scipy_success,
            "components": state.components,
            "normal_residual_max": max(certificates),
            "inner_iterations": int(sum(inner_iterations)),
            "penalty": {
                "arm": arm,
                "local_ridge": 0.0,
                "prepare_seconds": preparation,
                "solve_seconds": time.perf_counter() - started,
                "full_moments": p.info,
            },
        },
    )
