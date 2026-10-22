"""Exploratory 90800 fit pilot for constant rank-one angle curves."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import constant_angle
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.runner import adp_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 90800


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(d: dict) -> dict:
    keys = ("method", "accepted_steps", "stop_reason", "riemannian_gradient", "stationarity_applicable", "rank_guard_fallbacks", "gn_solves", "profile_evaluations", "local_rank_loss", "converged")
    return {k: d[k] for k in keys if k in d}


def fit(case: Case, x: np.ndarray, y: np.ndarray, truth: np.ndarray, seed: int, method, budget: int, timed: bool = True) -> dict:
    def fixed(index, U, I, **kwargs):
        return method(index, U, I, **(kwargs | {"max_steps": budget}))

    model = ADP_multi_index(2, config=ADP_Config(seed=seed, **adp_settings(case, {})), solver=ADP_solver(fixed))
    start = time.perf_counter() if timed else None
    model.fit(x, y)
    elapsed = time.perf_counter() - start if start is not None else None
    out = {"projector_distance": metrics(model.basis_, truth)["projector_distance"], "trace_score": metrics(model.basis_, truth)["trace_score"], "stop_reason": model.result_.stop_reason, "inner": [short(d) for d in model.solver_diagnostics_]}
    if elapsed is not None:
        out["fit_seconds"] = elapsed
    return out


def counter_replay(case: Case, x: np.ndarray, y: np.ndarray, truth: np.ndarray, seed: int, budget: int) -> dict:
    counts = {"curve_constructions": 0, "centers_seen": 0, "constant_centers": 0, "scalar_refit_centers": 0}

    class Counted(constant_angle.AngleProfile):
        def __init__(self, M, w, a, I, mass, tau):
            super().__init__(M, w, a, I, mass, tau)
            counts["curve_constructions"] += 1
            counts["centers_seen"] += M.shape[0]
            counts["constant_centers"] += self.constant_centers
            counts["scalar_refit_centers"] += M.shape[0] - self.constant_centers

    core = constant_angle._core
    original = core._AngleProfile
    core._AngleProfile = Counted
    try:
        fit(case, x, y, truth, seed, constant_angle.solve, budget, timed=False)
    finally:
        core._AngleProfile = original
    counts["baseline_refit_centers_if_unskipped"] = counts["centers_seen"]
    counts["refit_centers_skipped"] = counts["constant_centers"]
    return counts


def run() -> None:
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5), "constant_angle8": fit(case, x, y, truth, method_seed, constant_angle.solve, 8), "constant_angle10": fit(case, x, y, truth, method_seed, constant_angle.solve, 10)}
        counters = {str(b): counter_replay(case, x, y, truth, method_seed, b) for b in (8, 10)}
        cases.append({"case": case.name, "shape": list(x.shape), "method_seed": method_seed, "fits": fits, "unmeasured_counter_replays": counters})
    files = ("ADP/solver/grassman.py", "experiments/grassman_accuracy_2026_10_18/constant_angle.py", "experiments/grassman_accuracy_2026_10_18/pilot_constant_angle.py", "experiments/r_sdr/design.py", "experiments/r_sdr/runner.py")
    result = {"seed": SEED, "method_seed": method_seed, "dtype": "float64", "fit_settings": {"max_steps": {"grassman": 5, "constant_angle": [8, 10]}, "tol": 1e-6, "local_ridge": 0.0, "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}, "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}, "source_sha256": {p: digest(ROOT / p) for p in files}, "cases": cases, "timer_note": "fit timers exclude separate counter replays; no profiling or tracemalloc"}
    (OUT / "pilot_constant_angle.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
