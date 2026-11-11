"""Exploratory 90800 full-fit pilot for intrinsic rank-loss GN proposals."""

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
from experiments.grassman_accuracy_2026_10_18 import intrinsic
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.runner import adp_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 90800


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(case, x, y, truth, seed, method, budget=5):
    def fixed(index, U, I, **kwargs):
        return method(index, U, I, **(kwargs | {"max_steps": budget}))

    model = ADP_multi_index(2, config=ADP_Config(seed=seed, **adp_settings(case, {})), solver=ADP_solver(fixed))
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    inner = model.solver_diagnostics_
    keys = ("accepted_steps", "rank_guard_fallbacks", "workspace_fallbacks", "gn_solves", "profile_evaluations", "local_rank_loss", "intrinsic_gn_proposals")
    totals = {key: sum(int(d.get(key, 0)) for d in inner) for key in keys}
    detail_keys = ("method", "effective_max_steps", "accepted_steps", "stop_reason", "profile_objective", "riemannian_gradient", "stationarity_applicable", "local_rank_loss", "intrinsic_gn_proposals", "rank_guard_fallbacks", "gn_solves", "profile_evaluations", "orthogonality_error")
    compact_inner = [{key: d[key] for key in detail_keys if key in d} for d in inner]
    quality = metrics(model.basis_, truth)
    return {"fit_seconds": elapsed, "projector_distance": quality["projector_distance"], "trace_score": quality["trace_score"], "basis_orthogonality_error": float(np.linalg.norm(model.basis_.T @ model.basis_ - np.eye(2))), "stop_reason": model.result_.stop_reason, "inner": compact_inner, "inner_totals": totals}


def run(budget_comparison: bool = False) -> None:
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve), "intrinsic5": fit(case, x, y, truth, method_seed, intrinsic.solve)}
        if budget_comparison:
            fits["intrinsic10"] = fit(case, x, y, truth, method_seed, intrinsic.solve, 10)
        cases.append({"case": case.name, "shape": list(x.shape), "method_seed": method_seed, "fits": fits})
    files = ("ADP/solver/grassman.py", "experiments/grassman_accuracy_2026_10_18/backtracking.py", "experiments/grassman_accuracy_2026_10_18/intrinsic.py", "experiments/grassman_accuracy_2026_10_18/pilot_intrinsic.py", "experiments/r_sdr/design.py", "experiments/r_sdr/runner.py")
    result = {"seed": SEED, "method_seed": method_seed, "dtype": "float64", "fit_settings": {"max_steps": {"grassman": 5, "intrinsic": [5, 10] if budget_comparison else 5}, "smooth_max_steps": 5, "tol": 1e-6, "local_ridge": 0.0, "intrinsic_gn": True, "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}, "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}, "source_sha256": {p: digest(ROOT / p) for p in files}, "cases": cases, "timer_note": "model.fit wall time; no profiling or tracemalloc"}
    output = "pilot_intrinsic_budget.json" if budget_comparison else "pilot_intrinsic.json"
    (OUT / output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run("--budget-comparison" in sys.argv[1:])
