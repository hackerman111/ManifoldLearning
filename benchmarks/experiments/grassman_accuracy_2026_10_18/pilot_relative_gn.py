"""Пилот на seed90800 для выбора бюджета относительного intrinsic GN."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import relative_gn
from experiments.grassman_accuracy_2026_10_18.pilot_intrinsic import (
    ROOT,
    SEED,
    digest,
    fit,
)
from experiments.r_sdr.design import Case, generate

OUT = Path(__file__).resolve().parent
BUDGETS = (5, 10, 15)


def run() -> None:
    output = OUT / "pilot_relative_gn.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    thread_keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(k) != "1" for k in thread_keys):
        raise ValueError("set all BLAS thread environment variables to 1")
    files = (
        "ADP/solver/grassman.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/backtracking.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/relative_gn.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/pilot_intrinsic.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/pilot_relative_gn.py",
        "benchmarks/experiments/r_sdr/design.py",
        "benchmarks/experiments/r_sdr/runner.py",
    )
    snapshot = OUT / "pilot_relative_gn_sources"
    snapshot.mkdir()
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {"grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5)}
        for budget in BUDGETS:
            fits[f"relative{budget}"] = fit(
                case, x, y, truth, method_seed, relative_gn.solve, budget
            )
        cases.append(
            {
                "case": case.name,
                "shape": list(x.shape),
                "input_sha256": {
                    k: hashlib.sha256(v.tobytes()).hexdigest()
                    for k, v in (("x", x), ("y", y), ("truth", truth))
                },
                "fits": fits,
            }
        )
        print(
            json.dumps(
                {
                    "case": case.name,
                    "fits": {
                        k: {
                            field: v[field]
                            for field in ("projector_distance", "fit_seconds")
                        }
                        for k, v in fits.items()
                    },
                }
            ),
            flush=True,
        )
    result = {
        "seed": SEED,
        "method_seed": method_seed,
        "dtype": "float64",
        "fit_settings": {
            "budgets": list(BUDGETS),
            "original_budget": 5,
            "smooth_max_steps": 5,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "lambda_prox": 0.05,
            "normalized_damping": True,
            "threads": {k: os.environ[k] for k in thread_keys},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {p: digest(ROOT / p) for p in files},
        "cases": cases,
        "timer_note": "unprofiled model.fit; reused exploratory inputs, not selection",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
