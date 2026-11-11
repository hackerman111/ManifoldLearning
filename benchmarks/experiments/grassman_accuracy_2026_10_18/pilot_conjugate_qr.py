"""Single-seed exploratory comparison for the QR conjugate solver."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import conjugate_guard, conjugate_qr
from experiments.grassman_accuracy_2026_10_18.pilot_intrinsic import (
    ROOT,
    SEED,
    digest,
    fit,
)
from experiments.r_sdr.design import Case, generate

OUT = Path(__file__).resolve().parent
OUTPUT = OUT / "pilot_conjugate_qr.json"


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    method_seed = int(np.random.SeedSequence([SEED, 1]).generate_state(1)[0]) % (
        2**31 - 1
    )
    cases = []
    for d in (5, 20, 50):
        case = Case(f"d-{d}", 500, d)
        x, y, truth = generate(case, SEED)
        fits = {
            "grassman5": fit(case, x, y, truth, method_seed, grassman.solve, 5),
            "conjugate10": fit(
                case, x, y, truth, method_seed, conjugate_guard.solve, 10
            ),
            "conjugate_qr10": fit(
                case, x, y, truth, method_seed, conjugate_qr.solve, 10
            ),
        }
        base = fits["grassman5"]
        old = fits["conjugate10"]
        qr = fits["conjugate_qr10"]
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
                "comparisons": {
                    name: {
                        "fit_time_ratio_to_grassman5": value["fit_seconds"]
                        / base["fit_seconds"],
                        "edr_delta_to_grassman5": value["projector_distance"]
                        - base["projector_distance"],
                    }
                    for name, value in fits.items()
                    if name != "grassman5"
                },
                "conjugate_qr_vs_old_conjugate": {
                    "fit_time_ratio": qr["fit_seconds"] / old["fit_seconds"],
                    "edr_delta": qr["projector_distance"] - old["projector_distance"],
                },
            }
        )
    files = (
        "ADP/solver/grassman.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/backtracking.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/selective.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/conjugate_guard.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/conjugate_qr.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/pilot_conjugate_qr.py",
        "benchmarks/experiments/grassman_accuracy_2026_10_18/pilot_intrinsic.py",
        "benchmarks/experiments/r_sdr/design.py",
        "benchmarks/experiments/r_sdr/runner.py",
    )
    result = {
        "seed": SEED,
        "method_seed": method_seed,
        "dtype": "float64",
        "fit_settings": {
            "max_steps": {"grassman": 5, "conjugate_guard": 10, "conjugate_qr": 10},
            "smooth_max_steps": 5,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "threads": {
                k: os.environ[k]
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
        "timer_note": "single-thread model.fit wall time; unprofiled exploratory pilot",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
