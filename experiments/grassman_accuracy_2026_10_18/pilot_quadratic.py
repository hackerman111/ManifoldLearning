"""Exploratory 90800 full-fit pilot for quadratic rank-one trials."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import backtracking, quadratic
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.runner import adp_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 90800


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(d: dict) -> dict:
    keys = ("method", "angle_backend", "effective_max_steps", "accepted_steps", "stop_reason", "riemannian_gradient", "stationarity_applicable", "rank_guard_fallbacks", "gn_solves", "profile_evaluations", "local_rank_loss", "converged")
    return {k: d[k] for k in keys if k in d}


def fit(case: Case, x: np.ndarray, y: np.ndarray, truth: np.ndarray, seed: int, method, budget: int, smooth_max_steps=5) -> dict:
    def fixed(index, U, I, **kwargs):
        forwarded = kwargs | {"max_steps": budget}
        if method is quadratic.solve:
            forwarded["smooth_max_steps"] = smooth_max_steps
        return method(index, U, I, **forwarded)

    model = ADP_multi_index(2, config=ADP_Config(seed=seed, **adp_settings(case, {})), solver=ADP_solver(fixed))
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    inner = [short(d) for d in model.solver_diagnostics_]
    totals = {k: sum(int(d.get(k, 0)) for d in inner) for k in ("accepted_steps", "rank_guard_fallbacks", "gn_solves", "profile_evaluations", "local_rank_loss")}
    return {"fit_seconds": elapsed, "projector_distance": metrics(model.basis_, truth)["projector_distance"], "trace_score": metrics(model.basis_, truth)["trace_score"], "stop_reason": model.result_.stop_reason, "inner": inner, "inner_totals": totals}


def run(full_smooth10: bool = False) -> None:
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5), "quadratic10_cap5": fit(case, x, y, truth, method_seed, quadratic.solve, 10)}
        if full_smooth10:
            fits["quadratic10_full_smooth"] = fit(case, x, y, truth, method_seed, quadratic.solve, 10, smooth_max_steps=None)
        else:
            fits["armijo10"] = fit(case, x, y, truth, method_seed, backtracking.solve, 10)
        cases.append({"case": case.name, "shape": list(x.shape), "method_seed": method_seed, "fits": fits})
    files = ("ADP/solver/grassman.py", "experiments/grassman_accuracy_2026_10_18/backtracking.py", "experiments/grassman_accuracy_2026_10_18/quadratic.py", "experiments/grassman_accuracy_2026_10_18/pilot_quadratic.py", "experiments/r_sdr/design.py", "experiments/r_sdr/runner.py")
    result = {"seed": SEED, "method_seed": method_seed, "dtype": "float64", "fit_settings": {"budgets": {"grassman": 5, "quadratic": 10}, "quadratic_smooth_max_steps": None if full_smooth10 else 5, "tol": 1e-6, "local_ridge": 0.0, "quadratic_trials": 3, "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}, "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}, "source_sha256": {p: digest(ROOT / p) for p in files}, "cases": cases, "timer_note": "model.fit wall time; no profiling or tracemalloc"}
    output = "pilot_quadratic_full10.json" if full_smooth10 else "pilot_quadratic.json"
    (OUT / output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run("--full-smooth10" in sys.argv[1:])
