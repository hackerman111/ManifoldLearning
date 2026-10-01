"""Frozen paired joint rank-r versus current greedy SVD comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver._multi_operator import forward
from ADP.solver.SVD import solve_fixed_coefficients as solve_greedy
from experiments.svd_joint_2026_09_30.prototype import (
    solve as solve_joint,
    solve_fixed_coefficients as solve_joint_fixed,
)

OUT = Path(__file__).parent
SEEDS = (73001, 73002, 73003)
FIT_SEEDS = (73101, 73102, 73103)
RANK, LAMBDA, MAXITER = 2, 0.7, 100


def projector_distance(A, B):
    return float(np.sqrt(max(0.0, len(A) - np.sum((A @ B.T) ** 2)) / len(A)))


def measure(call):
    start = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - start
    tracemalloc.start()
    try:
        call()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


def fixed_data(seed, d):
    rng = np.random.default_rng(seed)
    J, p, m = 60, 8, 4
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    truth = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    U = rng.normal(size=(J, p, d)) * np.geomspace(0.2, 5.0, d)
    g = rng.normal(size=(J, m))
    mass = np.geomspace(0.2, 3.0, J)
    I = np.einsum("jpd,md,jm->jp", U, truth, g) + 0.05 * rng.normal(size=(J, p))
    return P, U, I, g, mass, truth


def fixed_case(seed, d, target):
    P, U, I, g, mass, truth = fixed_data(seed, d)
    outcomes = {}
    for name, solver in (("greedy", solve_greedy), ("joint", solve_joint_fixed)):
        kwargs = {"rank": RANK, "lambda_penalty": LAMBDA, "low_rank_target": target}
        if name == "joint":
            kwargs.update(maxiter=MAXITER, tol=1e-6)

        def call(solver=solver, kwargs=kwargs):
            return solver(P, U, I, g, mass, **kwargs)

        try:
            call()  # warm up, outside measured call
            (B, diag), seconds, peak = measure(call)
            objective = float(
                np.sum(mass[:, None] * (I - forward(U, B, g)) ** 2)
                + LAMBDA * np.sum((B - P) ** 2)
            )
            outcomes[name] = {
                "status": "ok",
                "objective": objective,
                "seconds": seconds,
                "traced_peak_bytes": peak,
                "projector_error": projector_distance(
                    np.linalg.qr(B.T, mode="reduced")[0].T, truth
                ),
                "diagnostics": diag,
            }
        except Exception as exc:
            outcomes[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if all(v["status"] == "ok" for v in outcomes.values()):
        a, b = outcomes["greedy"], outcomes["joint"]
        b["objective_delta"] = b["objective"] - a["objective"]
        b["relative_objective_delta"] = b["objective_delta"] / max(
            1.0, abs(a["objective"])
        )
        b["time_ratio"] = b["seconds"] / a["seconds"]
        b["traced_memory_ratio"] = b["traced_peak_bytes"] / max(
            1, a["traced_peak_bytes"]
        )
    return {
        "kind": "fixed",
        "seed": seed,
        "d": d,
        "target": target,
        "outcomes": outcomes,
    }


def fit_case(seed, d, target):
    X, y, truth = _synthetic_data(320, d, 4, 0.05, seed)
    outcomes = {}
    for name, solver in (("greedy", solve_greedy), ("joint", solve_joint)):
        settings = {"rank": RANK, "low_rank_target": target}
        if name == "joint":
            settings.update(maxiter=MAXITER, tol=1e-6)

        def fit(solver=solver, settings=settings):
            config = ADP_Config(
                seed=seed + 10000,
                N_J=40,
                N_phi=8,
                N_loc=24,
                outer_steps=3,
                h_min=1e-6,
                lambda_penalty=LAMBDA,
            )
            return ADP_multi_index(4, config, ADP_solver(solver, **settings)).fit(X, y)

        try:
            fit()
            model, seconds, peak = measure(fit)
            outcomes[name] = {
                "status": "ok",
                "seconds": seconds,
                "traced_peak_bytes": peak,
                "projector_error": projector_distance(model.basis_.T, truth),
                "loss": float(model.trace_[-1]["loss"]),
                "stop_reason": model.trace_[-1].get("stop_reason"),
                "solver_diagnostics": model.trace_[-1]["solver"],
            }
        except Exception as exc:
            outcomes[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if all(v["status"] == "ok" for v in outcomes.values()):
        a, b = outcomes["greedy"], outcomes["joint"]
        b["projector_delta"] = b["projector_error"] - a["projector_error"]
        b["loss_delta"] = b["loss"] - a["loss"]
        b["time_ratio"] = b["seconds"] / a["seconds"]
        b["traced_memory_ratio"] = b["traced_peak_bytes"] / max(
            1, a["traced_peak_bytes"]
        )
    return {
        "kind": "fit",
        "seed": seed,
        "d": d,
        "target": target,
        "outcomes": outcomes,
    }


def main():
    manifest = {
        "stage": "frozen selection",
        "seeds": SEEDS,
        "fit_seeds": FIT_SEEDS,
        "fixed_d": [100, 300, 600],
        "fit_d": [10, 50],
        "J": 60,
        "p": 8,
        "m": 4,
        "rank": RANK,
        "lambda_penalty": LAMBDA,
        "fixed_columns": np.geomspace(0.2, 5.0, 600).tolist(),
        "outer_steps": 3,
        "fit_n": 320,
        "fit_N_J": 40,
        "fit_N_phi": 8,
        "fit_N_loc": 24,
        "joint_maxiter": MAXITER,
        "joint_tol": 1e-6,
        "greedy_settings": (
            "rank_two, alternating, direct solve <=128, defaults otherwise"
        ),
        "dtype": "float64",
        "rng": "default_rng(seed), shared generated arrays within pair",
        "memory": (
            "separate traced_peak_bytes call; inputs excluded; native BLAS "
            "allocations may be omitted"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpools": threadpool_info(),
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "ADP/solver/SVD.py",
                "experiments/svd_joint_2026_09_30/prototype.py",
                __file__,
            )
        },
        "gates": {
            "fixed_relative_delta_max": 1e-6,
            "fit_projector_delta_max": 0.02,
            "failure_count": 0,
            "validation_only_if_all_selection_gates_pass": True,
        },
    }
    (OUT / "runs.jsonl").write_text("")
    (OUT / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with threadpool_limits(limits=1):
        for d in (100, 300, 600):
            for target in ("matrix", "correction"):
                for seed in SEEDS:
                    row = fixed_case(seed, d, target)
                    rows.append(row)
                    print(
                        f"fixed d={d} target={target} seed={seed}: "
                        f"{row['outcomes']['joint']['status']}",
                        flush=True,
                    )
                    (OUT / "runs.jsonl").open("a").write(json.dumps(row) + "\n")
        for d in (10, 50):
            for target in ("matrix", "correction"):
                for seed in FIT_SEEDS:
                    row = fit_case(seed, d, target)
                    rows.append(row)
                    print(
                        f"fit d={d} target={target} seed={seed}: "
                        f"{row['outcomes']['joint']['status']}",
                        flush=True,
                    )
                    with (OUT / "runs.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
    fixed = [r for r in rows if r["kind"] == "fixed"]
    fits = [r for r in rows if r["kind"] == "fit"]
    fixed_pass = all(
        r["outcomes"].get("joint", {}).get("relative_objective_delta", np.inf) <= 1e-6
        and all(v["status"] == "ok" for v in r["outcomes"].values())
        for r in fixed
    )
    fit_pass = all(
        r["outcomes"].get("joint", {}).get("projector_delta", np.inf) <= 0.02
        and all(v["status"] == "ok" for v in r["outcomes"].values())
        for r in fits
    )
    summary = {
        "rows": len(rows),
        "fixed_pass": fixed_pass,
        "fit_pass": fit_pass,
        "validation_authorized": fixed_pass and fit_pass,
        "fixed_median_objective_delta": float(
            np.median(
                [
                    r["outcomes"]["joint"].get("relative_objective_delta", np.nan)
                    for r in fixed
                ]
            )
        ),
        "fixed_median_time_ratio": float(
            np.median([r["outcomes"]["joint"].get("time_ratio", np.nan) for r in fixed])
        ),
        "fit_median_projector_delta": float(
            np.median(
                [r["outcomes"]["joint"].get("projector_delta", np.nan) for r in fits]
            )
        ),
        "fit_median_time_ratio": float(
            np.median([r["outcomes"]["joint"].get("time_ratio", np.nan) for r in fits])
        ),
    }
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
