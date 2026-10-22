"""Exploratory 90800 budget pilot for bounded scalar polish."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import polish
from experiments.grassman_accuracy_2026_10_18.pilot_intrinsic import (
    ROOT,
    SEED,
    digest,
    fit,
)
from experiments.r_sdr.design import Case, generate

OUT = Path(__file__).resolve().parent
BUDGETS = (7, 8, 10)


def run():
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5)}
        fits.update(
            {
                f"polish{budget}": fit(
                    case, x, y, truth, method_seed, polish.solve, budget
                )
                for budget in BUDGETS
            }
        )
        cases.append(
            {
                "case": case.name,
                "shape": list(x.shape),
                "method_seed": method_seed,
                "input_sha256": {
                    k: hashlib.sha256(v.tobytes()).hexdigest()
                    for k, v in (("x", x), ("y", y), ("truth", truth))
                },
                "fits": fits,
            }
        )
    files = (
        "ADP/solver/grassman.py",
        "experiments/grassman_accuracy_2026_10_18/backtracking.py",
        "experiments/grassman_accuracy_2026_10_18/intrinsic.py",
        "experiments/grassman_accuracy_2026_10_18/polish.py",
        "experiments/grassman_accuracy_2026_10_18/pilot_intrinsic.py",
        "experiments/grassman_accuracy_2026_10_18/pilot_polish.py",
        "experiments/r_sdr/design.py",
        "experiments/r_sdr/runner.py",
    )
    result = {
        "seed": SEED,
        "method_seed": method_seed,
        "dtype": "float64",
        "fit_settings": {
            "max_steps": {"grassman": 5, "polish": list(BUDGETS)},
            "tol": 1e-6,
            "local_ridge": 0.0,
            "threads": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {p: digest(ROOT / p) for p in files},
        "cases": cases,
        "timer_note": "model.fit wall time; no profiling or tracemalloc",
    }
    (OUT / "pilot_polish.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    run()
