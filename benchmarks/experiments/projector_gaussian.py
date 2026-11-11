"""Experimental ADP with Gaussian windows and equal localization of m indices.

The full-moment profile, ridge, solver and canonicalization are unchanged.
The compact-kernel initializer is shared with the controls. This isolated
CPU loop supports only the study's configuration; it changes no ADP defaults.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ADP import ADP_Config
from ADP.core.ADP_Config import epanechnikov
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
from ADP.engine.common.weights import calculate_multi_weight
from experiments import anisotropic_metrics, full_ridge_batched


def gaussian_kernel(argument):
    """Argument is squared anisotropic distance divided by h²."""
    return np.exp(-2 * np.asarray(argument))


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def fit(
    X,
    y,
    *,
    m,
    config=None,
    kernel=gaussian_kernel,
    floor=1.0,
    solver="two_sided",
    max_steps=20,
):
    """Fit the isolated current ADP loop; no truth or recovery metric is used.

    ``config.kernel`` controls initialization and must remain Epanechnikov.
    ``kernel`` controls outer windows. ``floor=0,kernel=epanechnikov`` is the
    parity control; the proposed modification uses the function defaults.
    """
    started = time.perf_counter()
    cfg = config or ADP_Config(
        N_J=min(len(X), 300),
        N_lin=max(50, 2 * np.shape(X)[1]),
        N_phi=max(m + 1, min(np.shape(X)[1], 12)),
        N_loc=20,
    )
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
    if not np.isfinite(floor) or not 0 <= floor <= 1 or not callable(kernel):
        raise ValueError("invalid floor or kernel")
    if solver not in ("two_sided", "lbfgs"):
        raise ValueError("unsupported solver")
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
    h = search_bandwidth(distance2, cfg.N_loc, kernel, lower=cfg.h_min)
    direction_rng = np.random.default_rng(direction_seed)
    eigenvalues, alpha = np.ones(m), 1.0
    trace, best_error, selected = [], np.inf, 0
    best = None
    stop = "h_min"
    while True:
        directions = generate_isotropic_proj(direction_rng, j, n_phi, d)
        local_spectrum = np.maximum(eigenvalues, floor)
        weights = calculate_multi_weight(
            X,
            centers,
            index.T,
            local_spectrum,
            h,
            alpha,
            kernel,
            block_size=cfg.batch_size,
            distance2=distance2,
            tensor="orthogonal",
        )
        weight_hash = hashlib.sha256()

        def observed_weights(weights=weights, weight_hash=weight_hash):
            for start, block in weights:
                weight_hash.update(np.ascontiguousarray(block).tobytes())
                yield start, block

        stats = full_ridge_batched.calculate_statistics(
            X,
            y,
            observed_weights(),
            directions,
            cfg.batch_size,
            normalized=True,
        )
        result = (
            full_ridge_batched.solve(index, stats, max_steps=max_steps)
            if solver == "lbfgs"
            else anisotropic_metrics.solve(
                index, stats, "two_sided", max_steps=max_steps
            )
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
                "localization_eigenvalues": local_spectrum,
                "basis": index.T.copy(),
                "solver": result.diagnostics,
                "median_neff": float(np.median(stats.n_eff)),
                "min_neff": float(stats.n_eff.min()),
                "mean_mass": float(stats.mass.mean()),
                "weights_sha256": weight_hash.hexdigest(),
                "moments_sha256": digest(stats.U, stats.I, stats.mass),
                "directions_sha256": digest(directions),
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
            np.maximum(eigenvalues, floor),
            next_h,
            cfg.N_loc,
            kernel,
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
        "kernel": "gaussian_exp_minus_2u"
        if kernel is gaussian_kernel
        else kernel.__name__,
        "localization_floor": float(floor),
        "initializer_kernel": "epanechnikov",
        "solver": solver,
        "max_inner_steps": max_steps,
    }
    return IndexFitResult(initial, index, coefficients, eigenvalues, metadata)
