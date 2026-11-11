"""Opt-in compact ADP: normalized local projector and Gaussian score moments."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ADP import ADP_Config
from ADP.core.ADP_Config import epanechnikov
from ADP.core.solver_common import HPAOResult
from ADP.engine.common import utils
from ADP.engine.common.calculus import (
    calculate_alpha_k,
    generate_isotropic_proj,
    pairwise_distance2,
    search_bandwidth,
)
from ADP.engine.common.index_fit import (
    IndexFitResult,
    _initial_index,
    canonical_index,
    validate_sizes,
)
from ADP.engine.common.weights import (
    _projected_multi_components,
    calculate_multi_weight,
)
from ADP.solver import grassman
from experiments import anisotropic_metrics, full_ridge_batched

ARMS = ("full_ridge", "full0", "spectral", "score", "score_spectral")


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def score_block(X, y, centers, basis, eigenvalues, h, alpha, weights, distance2):
    """Unnormalized sum y*(w*x-grad(w)); known standard Gaussian score, nu=0."""
    origin = X[0]
    orthogonal, principal = _projected_multi_components(
        (X - origin) @ basis,
        (centers - origin) @ basis,
        eigenvalues,
        distance2,
    )
    u = (alpha**2 * orthogonal + principal) / h**2
    derivative = np.where(weights > 0, 4 * u / h**2, 0.0) * y[None, :]
    displacement_sum = derivative @ X - derivative.sum(axis=1)[:, None] * centers
    # L=alpha² I + B diag(lambda-alpha²) B^T; apply after summing displacements.
    correction = alpha**2 * displacement_sum
    correction += ((displacement_sum @ basis) * (eigenvalues - alpha**2)) @ basis.T
    return (weights * y[None, :]) @ X + correction


def moment_design(stats, score_vectors=None):
    """One covariance mask for roots, cross moments and conventional slopes."""
    covariance, cross = np.asarray(stats.U), np.asarray(stats.I)
    if not np.isfinite(covariance).all() or not np.isfinite(cross).all():
        raise ValueError("nonfinite local moments")
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values[:, 0] < -1e-10 * np.maximum(values[:, -1], 1)):
        raise ValueError("indefinite covariance")
    keep = values > 1e-12 * np.maximum(values[:, -1, None], np.finfo(float).tiny)
    roots = np.sqrt(np.maximum(values, 0)) * keep
    root = (vectors * roots[:, None, :]) @ vectors.swapaxes(1, 2)
    if score_vectors is None:
        coordinates = (vectors.swapaxes(1, 2) @ cross[..., None]).squeeze(-1)
        inverse = np.divide(1.0, roots, out=np.zeros_like(roots), where=keep)
        target = (vectors @ (inverse * coordinates)[..., None]).squeeze(-1)
        slopes = (vectors @ ((inverse**2) * coordinates)[..., None]).squeeze(-1)
    else:
        slopes = np.asarray(score_vectors, dtype=float)
        if slopes.shape != cross.shape or not np.isfinite(slopes).all():
            raise ValueError("invalid corrected local moments")
        target = (root @ slopes[..., None]).squeeze(-1)
    if not np.isfinite(target).all() or not np.isfinite(slopes).all():
        raise ValueError("nonfinite whitened moments")
    pi = np.asarray(stats.mass, dtype=float) / np.sum(stats.mass)
    if not np.isfinite(pi).all() or np.any(pi <= 0):
        raise ValueError("invalid center masses")
    info = {
        "covariance_cutoff_relative": 1e-12,
        "rank_min": int(keep.sum(axis=1).min()),
        "rank_max": int(keep.sum(axis=1).max()),
        "zero_windows": int(np.count_nonzero(~np.any(keep, axis=1))),
    }
    return root, target, slopes, pi, info


def solve_local(index, stats, arm, *, score_vectors=None, max_steps=20):
    if arm not in ARMS:
        raise ValueError("unknown experimental arm")
    if arm == "full_ridge":
        return anisotropic_metrics.solve(index, stats, "two_sided", max_steps=max_steps)
    root, target, slopes, pi, info = moment_design(stats, score_vectors)
    if arm in ("full0", "score"):
        result = grassman.solve(
            index,
            root,
            target,
            mass=pi,
            local_ridge=0.0,
            max_steps=max_steps,
            tol=1e-6,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            objective_kind="full_moment_zero_ridge",
            covariance=info,
            corrected_moments=arm == "score",
        )
        return HPAOResult(result.index, result.coefficients, diagnostics)

    weighted = slopes.T * np.sqrt(pi)[None, :]
    left, singular, _ = np.linalg.svd(weighted, full_matrices=False)
    m = np.shape(index)[0]
    cutoff = np.finfo(float).eps * max(weighted.shape) * singular[0]
    if len(singular) < m or singular[m - 1] <= cutoff:
        raise ValueError("local moment matrix does not identify rank m")
    basis = left[:, :m]
    design = grassman._project(root, basis)
    state = grassman._profile_svd(design, target, pi, 0.0)
    spectral_loss = float(
        np.linalg.norm(basis - left[:, :m] @ (left[:, :m].T @ basis)) ** 2 / m
    )
    return HPAOResult(
        basis.T,
        state.coefficients,
        {
            "method": "normalized_local_moment_projector",
            "objective_kind": "empirical_projector_edr_squared",
            "profile_objective": spectral_loss,
            "full_moment_objective": state.value,
            "converged": True,
            "stop_reason": "spectral_global_minimum",
            "iterations": 0,
            "covariance": info,
            "local_moment_singular_values": singular.tolist(),
            "local_moment_condition": float(singular[0] / singular[m - 1]),
            "corrected_moments": arm == "score_spectral",
            "normalization_preserves_pca_minimizer": True,
            "local_profile_smooth": state.smooth,
            "local_profile_rank_min": int(state.ranks.min()),
        },
    )


def fit(X, y, *, m, arm, config=None, max_steps=20):
    """Same compact ADP outer rules; new moments/subspace updates are opt-in."""
    started = time.perf_counter()
    if arm not in ARMS:
        raise ValueError("unknown experimental arm")
    cfg = config or ADP_Config(N_J=min(len(X), 300), N_lin=50, N_phi=12, N_loc=20)
    if (
        cfg.kernel is not epanechnikov
        or cfg.estimator != "new"
        or cfg.direction_mode not in ("auto", "isotropic")
        or not cfg.redraw_directions
        or cfg.multi_tensor != "orthogonal"
        or cfg.center_displacement != 0
        or cfg.training_set != "all"
        or cfg.gpu
        or cfg.gpu_solver
        or cfg.smart_weights
        or cfg.experimental_centers is not None
    ):
        raise ValueError("unsupported experimental configuration")
    X, y = utils._prepare_xy(X, y, require_overdetermined=cfg.index_init != "random")
    n, d = X.shape
    n_lin, j, n_phi = validate_sizes(cfg, "multi", n, d, m)
    if cfg.h_min is None:
        raise ValueError("h_min must be configured")
    center_seed, init_seed, direction_seed = np.random.SeedSequence(cfg.seed).spawn(3)
    center_rng = np.random.default_rng(center_seed)
    center_indices = (
        np.arange(n) if j == n else center_rng.choice(n, size=j, replace=False)
    )
    centers = X[center_indices].copy()
    distance2 = pairwise_distance2(X, centers)
    index, initial_spectrum = _initial_index(
        "multi",
        d,
        m,
        cfg,
        X,
        y,
        centers,
        distance2,
        n_lin,
        np.random.default_rng(init_seed),
    )
    initial = index.copy()
    h = search_bandwidth(distance2, cfg.N_loc, cfg.kernel, lower=cfg.h_min)
    direction_rng = np.random.default_rng(direction_seed)
    eigenvalues, alpha = np.ones(m), 1.0
    trace, best_error, selected, best = [], np.inf, 0, None
    stop = "h_min"
    use_score = arm in ("score", "score_spectral")
    while True:
        directions = generate_isotropic_proj(direction_rng, j, n_phi, d)
        weights = calculate_multi_weight(
            X,
            centers,
            index.T,
            eigenvalues,
            h,
            alpha,
            cfg.kernel,
            block_size=cfg.batch_size,
            distance2=distance2,
            tensor="orthogonal",
        )
        weight_hash = hashlib.sha256()
        numerator = np.zeros((j, d)) if use_score else None

        def observed_weights(
            weights=weights,
            weight_hash=weight_hash,
            numerator=numerator,
            index=index,
            eigenvalues=eigenvalues,
            h=h,
            alpha=alpha,
        ):
            for start, block in weights:
                weight_hash.update(np.ascontiguousarray(block).tobytes())
                if numerator is not None:
                    end = start + len(block)
                    numerator[start:end] = score_block(
                        X,
                        y,
                        centers[start:end],
                        index.T,
                        eigenvalues,
                        h,
                        alpha,
                        block,
                        distance2[start:end],
                    )
                yield start, block

        stats = full_ridge_batched.calculate_statistics(
            X, y, observed_weights(), directions, cfg.batch_size, normalized=True
        )
        score_vectors = None if numerator is None else numerator / stats.mass[:, None]
        result = solve_local(
            index, stats, arm, score_vectors=score_vectors, max_steps=max_steps
        )
        index, eigenvalues = canonical_index("multi", result, index, mass=stats.mass)
        rotation = index @ result.index.T
        coefficients = result.coefficients @ rotation.T
        error = float(np.sum((y[center_indices] - stats.S) ** 2))
        iteration = len(trace)
        trace.append(
            {
                "iteration": iteration,
                "h": float(h),
                "factor": float(alpha),
                "err": error,
                "eigenvalues": eigenvalues.copy(),
                "basis": index.T.copy(),
                "solver": result.diagnostics,
                "median_neff": float(np.median(stats.n_eff)),
                "min_neff": float(stats.n_eff.min()),
                "mean_mass": float(stats.mass.mean()),
                "weights_sha256": weight_hash.hexdigest(),
                "moments_sha256": digest(stats.U, stats.I, stats.mass),
                "directions_sha256": digest(directions),
                "score_sha256": None
                if score_vectors is None
                else digest(score_vectors),
            }
        )
        if error < best_error:
            best_error, selected = error, iteration
            best = (index.copy(), coefficients.copy(), eigenvalues.copy())
        del stats
        if cfg.outer_steps is not None and len(trace) >= cfg.outer_steps:
            stop = "outer_steps"
            break
        next_h = h / cfg.a
        if next_h < cfg.h_min:
            break
        next_alpha = calculate_alpha_k(
            X,
            centers,
            index.T,
            eigenvalues,
            next_h,
            cfg.N_loc,
            cfg.kernel,
            distance2=distance2,
            tensor="orthogonal",
            block_size=cfg.batch_size,
        )
        if next_alpha is None:
            stop = "local_mass_limit"
            break
        h, alpha = float(next_h), next_alpha
    if cfg.select_step == "best":
        index, coefficients, eigenvalues = best
    else:
        selected = len(trace) - 1
    metadata = {
        "trace": trace,
        "selected_iteration": selected,
        "stop_reason": stop,
        "outer_iterations": len(trace),
        "selected_error": trace[selected]["err"],
        "selected_eigenvalues": eigenvalues.copy(),
        "initial_eigenvalues": initial_spectrum,
        "diagnostics": trace[selected]["solver"],
        "fit_seconds": time.perf_counter() - started,
        "center_indices_sha256": digest(center_indices),
        "initial_sha256": digest(initial),
        "arm": arm,
        "max_inner_steps": max_steps,
        "known_score": "standard_gaussian_x",
    }
    return IndexFitResult(initial, index, coefficients, eigenvalues, metadata)
