"""Exploratory 90800 fit and frozen-profile microtiming for selective QR/SVD."""

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
from experiments.grassman_accuracy_2026_10_18 import selective
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.runner import adp_settings

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 90800


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short(d: dict) -> dict:
    keys = ("method", "effective_max_steps", "accepted_steps", "stop_reason", "riemannian_gradient", "stationarity_applicable", "rank_guard_fallbacks", "gn_solves", "profile_evaluations", "local_rank_loss", "converged")
    return {k: d[k] for k in keys if k in d}


def fit(case: Case, x: np.ndarray, y: np.ndarray, truth: np.ndarray, seed: int, method, budget: int) -> dict:
    def fixed(index, U, I, **kwargs):
        return method(index, U, I, **(kwargs | {"max_steps": budget}))

    model = ADP_multi_index(2, config=ADP_Config(seed=seed, **adp_settings(case, {})), solver=ADP_solver(fixed))
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    return {"fit_seconds": elapsed, **metrics(model.basis_, truth), "stop_reason": model.result_.stop_reason, "inner": [short(d) for d in model.solver_diagnostics_]}


def micro(number: int) -> dict:
    with np.load(OUT / "frozen" / f"{number:03}.npz") as saved:
        index, U, I, mass = (saved[k] for k in ("index_init", "U", "I", "mass"))
    Y = grassman._normalize_index(index).T
    M = grassman._project(U, Y)
    original = grassman._profile(M, I, mass, 0.0)
    candidate = selective.profile(M, I, mass, 0.0)
    g0, g1 = grassman._gradient(Y, U, original, mass), grassman._gradient(Y, U, candidate, mass)
    def timed(fn):
        for _ in range(2):
            fn()
        start = time.perf_counter()
        for _ in range(100):
            fn()
        return time.perf_counter() - start

    t0, t1 = timed(lambda: grassman._profile(M, I, mass, 0.0)), timed(lambda: selective.profile(M, I, mass, 0.0))
    return {"frozen": f"frozen/{number:03}.npz", "shape": list(M.shape), "calls": 100, "original_seconds": t0, "selective_seconds": t1, "speedup": t0 / t1, "max_coefficient_abs_diff": float(np.max(np.abs(original.coefficients - candidate.coefficients))), "objective_abs_diff": abs(original.value - candidate.value), "max_gradient_abs_diff": float(np.max(np.abs(g0 - g1)))}


def run() -> None:
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        cases.append({"case": case.name, "shape": list(x.shape), "method_seed": method_seed, "fits": {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5), "selective5": fit(case, x, y, truth, method_seed, selective.solve, 5), "selective10": fit(case, x, y, truth, method_seed, selective.solve, 10)}})
    files = ("ADP/solver/grassman.py", "benchmarks/experiments/grassman_accuracy_2026_10_18/selective.py", "benchmarks/experiments/grassman_accuracy_2026_10_18/pilot_selective.py", "benchmarks/experiments/r_sdr/design.py", "benchmarks/experiments/r_sdr/runner.py")
    result = {"seed": SEED, "method_seed": method_seed, "dtype": "float64", "fit_settings": {"max_steps": [5, 10], "tol": 1e-6, "local_ridge": 0.0, "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}, "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}, "source_sha256": {p: digest(ROOT / p) for p in files}, "cases": cases, "frozen_profile_microtiming": [micro(n) for n in (1, 3, 8)]}
    (OUT / "pilot_selective.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
