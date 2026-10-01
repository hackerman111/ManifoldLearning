"""Paired preconditioner on/off run of the saved 25-second Spokoini fit.

Run from repository root:
  rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python \
    experiments/svd_preconditioner_spokoini_2026-09-30/benchmark.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CASE = "spokoini_m2_dhigh_30s"


def _worker(seed: int, precondition_v: bool) -> dict[str, object]:
    import ADP.solver.SVD as svd
    from benchmarks.svd_vs_hybrid import _worker as fit

    original_solve = svd.solve

    def solve_with_flag(*args, **kwargs):
        kwargs["precondition_v"] = precondition_v
        return original_solve(*args, **kwargs)

    svd.solve = solve_with_flag
    result = fit(CASE, seed, "svd_direct")
    traces = result.pop("solver_trace")
    result["precondition_v"] = precondition_v
    result["updates_with_flag_enabled"] = sum(
        bool(trace.get("precondition_v", False)) for trace in traces
    )
    result["diagnostic_preconditioner_cache_bytes"] = sorted(
        {int(trace.get("preconditioner_cache_bytes", 0)) for trace in traces}
    )
    return result


def _run_worker(seed: int, flag: bool) -> dict[str, object]:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT),
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--seed",
        str(seed),
        "--precondition-v",
        str(int(flag)),
    ]
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240
        )
        if process.returncode:
            raise RuntimeError(
                f"worker exit={process.returncode}: {process.stderr[-4000:]}"
            )
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (RuntimeError, subprocess.TimeoutExpired, IndexError, json.JSONDecodeError):
        return {
            "case": CASE,
            "seed": seed,
            "precondition_v": flag,
            "status": "error",
            "error": traceback.format_exc(limit=8),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precondition-v", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.seed, bool(args.precondition_v))))
        return

    import numpy as np
    import scipy

    manifest = {
        "task": "svd-preconditioner-spokoini-2026-09-30",
        "case": CASE,
        "seeds": [0, 1, 2],
        "paired_order": {
            "0": [False, True],
            "1": [True, False],
            "2": [False, True],
        },
        "fit_configuration": (
            "Spokoiny m=2,d=50,n=800,N_loc=10,N_lin=100,N_J=800,N_phi=10, "
            "a=exp(1/100), h_min=1, select_step=last, full outer schedule, "
            "lambda_penalty=0.05, SVD rank=1, matrix objective; defaults otherwise."
        ),
        "solver_path": (
            "svd_direct with direct_max_dimension=128; precondition_v flag "
            "passed through to solve."
        ),
        "protocol": (
            "Fresh subprocess per fit, perf_counter around fit, float64, one "
            "BLAS thread, no tracemalloc; same generated data and init per seed."
        ),
        "git_commit": subprocess.run(
            ["rtk", "proxy", "git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["rtk", "proxy", "git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "production_svd_sha256": hashlib.sha256(
            (ROOT / "ADP/solver/SVD.py").read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": dict.fromkeys(
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1"
        ),
        "blas_one_thread": True,
    }
    manifest_path = OUT / "manifest.json"
    runs_path = OUT / "runs.jsonl"
    if manifest_path.exists() or runs_path.exists():
        raise FileExistsError("Refusing to overwrite existing benchmark artifacts")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with runs_path.open("w") as stream:
        for seed in (0, 1, 2):
            flags = (False, True) if seed % 2 == 0 else (True, False)
            for flag in flags:
                row = _run_worker(seed, flag)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            key: row.get(key)
                            for key in (
                                "seed",
                                "precondition_v",
                                "status",
                                "fit_time_sec",
                                "projector_distance",
                                "outer_iterations",
                                "direct_solves",
                                "linear_iterations",
                                "updates_with_flag_enabled",
                                "peak_rss_kib",
                                "error",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
