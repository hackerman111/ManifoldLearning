"""Ограниченный exploratory90800: состав поиска отдельно от бюджета."""

from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18.pilot_intrinsic import (
    ROOT,
    SEED,
    digest,
    fit,
)
from experiments.r_sdr.design import Case, generate

OUT = Path(__file__).resolve().parent
VARIANTS = {
    "old5": (5, dict(relative_gn=False, quadratic_trials=0, constant_centers=False)),
    "relative10": (
        10,
        dict(
            relative_gn=True,
            quadratic_trials=0,
            constant_centers=False,
            smooth_max_steps=5,
        ),
    ),
    "relative_quad10": (
        10,
        dict(
            relative_gn=True,
            quadratic_trials=3,
            constant_centers=False,
            smooth_max_steps=5,
        ),
    ),
    "relative_zero10": (
        10,
        dict(
            relative_gn=True,
            quadratic_trials=0,
            constant_centers=True,
            smooth_max_steps=5,
        ),
    ),
    "combined10": (
        10,
        dict(
            relative_gn=True,
            quadratic_trials=3,
            constant_centers=True,
            smooth_max_steps=5,
        ),
    ),
    **{
        f"combined{budget}": (
            budget,
            dict(
                relative_gn=True,
                quadratic_trials=3,
                constant_centers=True,
                smooth_max_steps=None,
            ),
        )
        for budget in (12, 20, 30)
    },
}


def run() -> None:
    output = OUT / "pilot_combined.json"
    if output.exists():
        raise FileExistsError(output)
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(k) != "1" for k in keys):
        raise ValueError("set BLAS threads to 1")
    files = (
        "ADP/solver/grassman.py",
        "experiments/grassman_accuracy_2026_10_18/pilot_combined.py",
        "experiments/grassman_accuracy_2026_10_18/pilot_intrinsic.py",
        "experiments/r_sdr/design.py",
        "experiments/r_sdr/runner.py",
    )
    snapshot = OUT / "pilot_combined_sources"
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
        fits = {}
        for name, (budget, settings) in VARIANTS.items():
            fits[name] = fit(
                case,
                x,
                y,
                truth,
                method_seed,
                partial(grassman.solve, **settings),
                budget,
            )
            print(
                json.dumps(
                    dict(
                        case=case.name,
                        method=name,
                        edr=fits[name]["projector_distance"],
                        seconds=fits[name]["fit_seconds"],
                    )
                ),
                flush=True,
            )
        cases.append(
            dict(
                case=case.name,
                data_sha256=hashlib.sha256(
                    x.tobytes() + y.tobytes() + truth.tobytes()
                ).hexdigest(),
                fits=fits,
            )
        )
    output.write_text(
        json.dumps(
            dict(
                seed=SEED,
                method_seed=method_seed,
                variants=VARIANTS,
                threads={k: os.environ[k] for k in keys},
                source_sha256={p: digest(ROOT / p) for p in files},
                cases=cases,
                note="exploration only; full-fit timer; no fresh selection data",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
