"""Paired CPU single-index fits with current and reference HPAO actions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy

from ADP import ADP_Config, ADP_single_index, ADP_solver
from ADP.solver import LSMR
from benchmarks.fit_bottlenecks import TimeProfiler, _source_state
from benchmarks.optimization_candidates import hpao_operator_reference


def one(variant: str) -> dict:
    data_seed, noise_seed = np.random.SeedSequence(20260923).spawn(2)
    X = np.random.default_rng(data_seed).uniform(-1.0, 1.0, (600, 8))
    Y = np.sin(2.0 * X[:, 0])
    Y += 0.02 * np.random.default_rng(noise_seed).normal(size=len(X))
    config = ADP_Config(
        seed=17,
        N_loc=30,
        N_lin=40,
        N_J=48,
        N_phi=16,
        outer_steps=2,
        h_min=0.3,
        index_init="local",
        estimator="new",
        batch_size=32,
    )
    model = ADP_single_index(config, ADP_solver(LSMR.solve, max_steps=50, tol=1e-6))
    settings = asdict(config)
    settings["kernel"] = "max(1-q**2,0), q=distance2/h**2"
    context = (
        patch.object(LSMR, "_linear_operator", hpao_operator_reference)
        if variant == "reference"
        else nullcontext()
    )
    model_module = importlib.import_module("ADP.core.single.ADP_single_index")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    with context, patch.object(model_module, "IndexProfiler", TimeProfiler):
        start = perf_counter()
        model.fit(X, Y)
        fit_seconds = perf_counter() - start
    return {
        "variant": variant,
        "settings": settings,
        "solver": {"max_steps": 50, "tol": 1e-6},
        "fit_seconds": fit_seconds,
        "rss_pre_fit_bytes": rss_before,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "beta": model.beta_.tolist(),
        "stop_reason": model.result_.stop_reason,
        "diagnostics": model.solver_diagnostics_,
        "profile": model.profile_,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("reference", "current"), required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        print(json.dumps(one(args.variant)))
        return
    if args.runs < 1 or args.output is None or args.output.exists():
        parser.error("--runs must be positive and --output must be a new path")
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "benchmarks.optimization_single",
        "--child",
        "--variant",
        args.variant,
    ]
    sources_before = _source_state()
    rows = []
    for repeat in range(args.runs + 1):
        run = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        row = (
            json.loads(run.stdout)
            if run.returncode == 0
            else {"error": run.stderr[-4000:], "returncode": run.returncode}
        )
        row.update(repeat=repeat, warmup=repeat == 0)
        rows.append(row)
        print(
            f"{args.variant} {repeat}/{args.runs}: {row.get('fit_seconds', 'error')}",
            flush=True,
        )
    report = {
        "command": command,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "source_sha256_before": sources_before,
        "source_sha256_after": _source_state(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
