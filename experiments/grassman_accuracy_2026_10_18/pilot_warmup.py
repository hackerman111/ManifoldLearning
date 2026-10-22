"""Exploratory 90800 full-fit pilot for one-step intrinsic warmup."""
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
from experiments.grassman_accuracy_2026_10_18 import warmup
from experiments.r_sdr.design import Case, generate, metrics
from experiments.r_sdr.runner import adp_settings

ROOT, SEED = Path(__file__).resolve().parents[2], 90800
SUM = ("accepted_steps", "rank_guard_fallbacks", "gn_solves",
       "profile_evaluations", "local_rank_loss", "intrinsic_gn_proposals")
DETAIL = ("method", "accepted_steps", "stop_reason", "profile_objective",
          "riemannian_gradient", "stationarity_applicable", "local_rank_loss",
          "intrinsic_gn_proposals", "rank_guard_fallbacks", "gn_solves",
          "profile_evaluations", "orthogonality_error")


# fmt: off
def fit(case, x, y, truth, seed, method, budget):
    def solve(index, U, I, **kw):
        opts = kw | {"max_steps": budget}
        if method is warmup.solve:
            opts["smooth_max_steps"] = 5
        return method(index, U, I, **opts)
    config = ADP_Config(seed=seed, **adp_settings(case, {}))
    model = ADP_multi_index(2, config=config, solver=ADP_solver(solve))
    start = time.perf_counter()
    model.fit(x, y)
    elapsed, inner = time.perf_counter() - start, model.solver_diagnostics_
    return {"fit_seconds": elapsed,
        "projector_distance": metrics(model.basis_, truth)["projector_distance"],
        "basis_orthogonality_error": float(
            np.linalg.norm(model.basis_.T @ model.basis_ - np.eye(2))),
        "outer_stop_reason": model.result_.stop_reason,
        "inner": [{k: d[k] for k in DETAIL if k in d} for d in inner],
        "inner_totals": {k: sum(int(d.get(k, 0)) for d in inner) for k in SUM}}


def run():
    seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (2**31 - 1)
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, seed, grassman.solve, 5),
                "warmup10": fit(case, x, y, truth, seed, warmup.solve, 10)}
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest()
                  for k, v in (("x", x), ("y", y), ("truth", truth))}
        cases.append({"case": case.name, "shape": list(x.shape),
                      "method_seed": seed, "input_sha256": hashes, "fits": fits})
    sources = ("ADP/solver/grassman.py",
        "experiments/grassman_accuracy_2026_10_18/backtracking.py",
        "experiments/grassman_accuracy_2026_10_18/intrinsic.py",
        "experiments/grassman_accuracy_2026_10_18/warmup.py",
        "experiments/grassman_accuracy_2026_10_18/pilot_warmup.py",
        "experiments/r_sdr/design.py", "experiments/r_sdr/runner.py")
    result = {"seed": SEED, "method_seed": seed, "dtype": "float64", "cases": cases,
        "fit_settings": {"steps": {"grassman": 5, "warmup": 10}, "smooth_cap": 5,
            "tol": 1e-6, "local_ridge": 0.0, "warmup_nonsmooth_steps": 1,
            "threads": {k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in sources},
        "timer_note": "model.fit wall time; no profiling or tracemalloc"}
    (Path(__file__).parent / "pilot_warmup.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
# fmt: on


if __name__ == "__main__":
    run()
