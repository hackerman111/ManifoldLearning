"""Bounded fixed-g comparison for the two rank constraints in SVD_corr.tex.

Run from the repository root with one BLAS thread, for example:
    OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/adp-uv-cache \
        uv run --no-sync python -m benchmarks.svd_correction \
        --output experiments/svd_correction_2026-09-29/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

from ADP.solver._multi_operator import forward
from ADP.solver.SVD import _complete_basis, solve_fixed_coefficients

J, P_DIM, M = 40, 8, 3
RIDGE = 0.3
SEEDS = (11, 23, 37)
REFERENCE_MAX_DIMENSION = 128


def _projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    m = len(left)
    return float(
        np.linalg.norm(left.T @ left - right.T @ right, ord="fro") / np.sqrt(2 * m)
    )


def _run_seed(seed: int, d: int, repeats: int) -> dict[str, object]:
    p, m = P_DIM, M
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(J, p, d))
    g = rng.normal(size=(J, m))
    prior = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    truth = np.linalg.qr(prior.T + 0.25 * rng.normal(size=(d, m)))[0].T
    I = forward(U, truth, g) + 0.1 * rng.normal(size=(J, p))
    mass = rng.uniform(0.5, 1.5, size=J)
    residual = I - forward(U, prior, g)

    def objective(B: np.ndarray) -> float:
        return float(
            np.sum(mass[:, None] * (I - forward(U, B, g)) ** 2)
            + RIDGE * np.sum((B - prior) ** 2)
        )

    full_basis = None
    energy = None
    if d <= REFERENCE_MAX_DIMENSION:
        # Dense fixed-g reference is kept bounded to this small-d path.
        tracemalloc.start()
        started = time.perf_counter()
        design = np.vstack([np.kron(g[j], U[j]) * np.sqrt(mass[j]) for j in range(J)])
        target = (np.sqrt(mass)[:, None] * residual).ravel()
        delta_star = np.linalg.lstsq(
            np.vstack((design, np.sqrt(RIDGE) * np.eye(m * d))),
            np.concatenate((target, np.zeros(m * d))),
            rcond=None,
        )[0].reshape(m, d)
        full_time = time.perf_counter() - started
        _, full_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del design, target
        B_star = prior + delta_star
        full_basis = np.linalg.qr(B_star.T, mode="reduced")[0].T
        values = np.linalg.svd(delta_star, compute_uv=False)
        energy = (np.cumsum(values**2) / np.sum(values**2)).tolist()
        rows = [
            {
                "method": "full_fixed_g",
                "rank": m,
                "objective": objective(B_star),
                "wall_sec": full_time,
                "tracemalloc_peak_bytes": full_peak,
                "u_vector_passes": None,
                "effective_rank": m,
                "projector_distance_to_full": 0.0,
                "projector_distance_to_truth": _projector_distance(full_basis, truth),
            }
        ]
    else:
        values = None
        rows = []

    for repeat in range(repeats):
        methods = ("matrix", "correction")
        if (seed + repeat) % 2:
            methods = methods[::-1]
        for method in methods:
            tracemalloc.start()
            started = time.perf_counter()
            try:
                B, diagnostics = solve_fixed_coefficients(
                    prior,
                    U,
                    I,
                    g,
                    mass,
                    rank=2,
                    lambda_penalty=RIDGE,
                    inner_tol=1e-8,
                    rank_tol=0,
                    low_rank_target=method,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append(
                    {
                        "method": method,
                        "rank": 2,
                        "repeat": repeat,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_sec": elapsed,
                        "tracemalloc_peak_bytes": peak,
                    }
                )
                continue
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if method == "matrix":
                _, singular, right = np.linalg.svd(B, full_matrices=False)
                basis = _complete_basis(right.T, singular, prior)
            elif full_basis is not None:
                basis = np.linalg.qr(B.T, mode="reduced")[0].T
            else:
                basis = None
            rows.append(
                {
                    "method": method,
                    "rank": 2,
                    "repeat": repeat,
                    "status": "ok",
                    "objective": objective(B),
                    "wall_sec": elapsed,
                    "tracemalloc_peak_bytes": peak,
                    "u_vector_passes": diagnostics["u_vector_passes"],
                    "effective_rank": diagnostics["effective_rank"],
                    "projector_distance_to_full": (
                        _projector_distance(basis, full_basis)
                        if full_basis is not None
                        else None
                    ),
                    "projector_distance_to_truth": (
                        _projector_distance(basis, truth) if basis is not None else None
                    ),
                    "v_normal_residual_max": diagnostics["v_normal_residual_max"],
                    "lsmr_iterations_total": diagnostics["lsmr_iterations_total"],
                    "direct_solves": diagnostics["direct_solves"],
                }
            )
    return {
        "seed": seed,
        "dimension": d,
        "correction_singular_values": values.tolist() if values is not None else None,
        "correction_energy": energy,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[20])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or any(d < M for d in args.dimensions):
        parser.error("repeats must be positive and dimensions must be >= m")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    output = {
        "shape_J_p_d_m": [J, P_DIM, args.dimensions, M],
        "ridge": RIDGE,
        "rank": 2,
        "repeats": args.repeats,
        "dense_reference_max_dimension": REFERENCE_MAX_DIMENSION,
        "dtype": "float64",
        "seeds": SEEDS,
        "git_commit": git,
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads_env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "memory_method": "tracemalloc peak; excludes native allocations",
        "runs": [
            _run_seed(seed, d, args.repeats) for d in args.dimensions for seed in SEEDS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
