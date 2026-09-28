"""Reproducible fixed-statistics benchmark for the truncated-SVD hot path."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

import ADP.solver.SVD as svd


def _problem(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    J, p, d, m = 300, 4, 400, 8
    U = np.ascontiguousarray(rng.normal(size=(J, p, d)))
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.5, 1.5, size=J)
    P = np.linalg.qr(rng.normal(size=(d, m)), mode="reduced")[0].T
    return P, U, I, g, mass


def _run(
    seed: int,
    rank: int,
    repetitions: int,
    *,
    warm_start: bool,
    adaptive_krylov: bool,
) -> dict[str, object]:
    P, U, I, g, mass = _problem(seed)
    # Load SciPy/BLAS paths outside the timed repetitions.
    warm_P = np.linalg.qr(P[:3, :20].T, mode="reduced")[0].T
    svd.solve_fixed_coefficients(
        warm_P,
        U[:8, :, :20],
        I[:8],
        g[:8, :3],
        mass[:8],
        rank=1,
        lambda_penalty=0.7,
        inner_maxiter=2,
        warm_start=warm_start,
        adaptive_krylov=adaptive_krylov,
        direct_max_dimension=None,
    )
    timings = []
    outputs = []
    for _ in range(repetitions):
        started = time.perf_counter()
        B, diagnostics = svd.solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=rank,
            lambda_penalty=0.7,
            warm_start=warm_start,
            adaptive_krylov=adaptive_krylov,
        )
        timings.append(time.perf_counter() - started)
        outputs.append((B, diagnostics))

    tracemalloc.start()
    B_trace, diagnostics_trace = svd.solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=0.7,
        warm_start=warm_start,
        adaptive_krylov=adaptive_krylov,
    )
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profiler = cProfile.Profile()
    profiler.enable()
    svd.solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=rank,
        lambda_penalty=0.7,
        warm_start=warm_start,
        adaptive_krylov=adaptive_krylov,
    )
    profiler.disable()

    B, diagnostics = outputs[-1]
    profile_stream = io.StringIO()
    pstats.Stats(profiler, stream=profile_stream).sort_stats("cumulative").print_stats(
        20
    )
    return {
        "shape": {"J": 300, "p": 4, "d": 400, "m": 8, "rank": rank},
        "seed": seed,
        "dtype": str(U.dtype),
        "lambda_penalty": 0.7,
        "inner_tol": 1e-6,
        "inner_maxiter": 20,
        "warm_start": warm_start,
        "adaptive_krylov": adaptive_krylov,
        "timings_sec": timings,
        "timing_median_sec": float(np.median(timings)),
        "tracemalloc_peak_bytes": traced_peak,
        "B_sha256": hashlib.sha256(np.ascontiguousarray(B).view(np.uint8)).hexdigest(),
        "B_frobenius": float(np.linalg.norm(B)),
        "trace_B_max_abs_delta": float(np.max(np.abs(B_trace - B))),
        "objective_history": diagnostics["rank_objective_history"],
        "trace_objective_history": diagnostics_trace["rank_objective_history"],
        "inner_iterations": diagnostics["inner_iterations"],
        "inner_converged": diagnostics["inner_converged"],
        "lsmr_iterations": diagnostics["lsmr_iterations"],
        "lsmr_iterations_total": diagnostics["lsmr_iterations_total"],
        "lsmr_stops": diagnostics["lsmr_stops"],
        "lsmr_tolerances": diagnostics["lsmr_tolerances"],
        "lsmr_refinements": diagnostics["lsmr_refinements"],
        "direct_solves": diagnostics["direct_solves"],
        "direct_fallbacks": diagnostics["direct_fallbacks"],
        "total_u_passes_estimate": diagnostics["u_vector_passes"],
        "u_flat_copy": diagnostics["u_flat_copy"],
        "factor_cache_fallbacks": diagnostics["factor_cache_fallbacks"],
        "factor_condition_max": diagnostics["factor_condition_max"],
        "v_normal_residual_max": diagnostics["v_normal_residual_max"],
        "profile_top20": profile_stream.getvalue(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--adaptive-krylov", action="store_true")
    args = parser.parse_args()
    result = _run(
        args.seed,
        args.rank,
        args.repetitions,
        warm_start=not args.no_warm_start,
        adaptive_krylov=args.adaptive_krylov,
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], capture_output=True, check=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
