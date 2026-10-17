"""Frozen serial benchmark harness; each solver arm runs in a fresh subprocess."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from experiments.edr_structured_2026_10_15.benchmark_workers import (
    _cx_worker,
    _frozen_worker,
    _fullfit_worker,
    digest,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
TIMEOUT = 120
NEW_VARIANTS = {"schur_gn_preconditioned", "schur_newton_preconditioned", "schur_newton_unpreconditioned", "schur_newton_recycle2", "metric"}
STAGES = ("frozen", "calibration", "weak-signal", "cx", "fullfit")
CASES = ("isotropic", "weak_index", "correlated", "near_truth", "weak_signal")


def _child(args: argparse.Namespace) -> int:
    workers = {
        "frozen": lambda: _frozen_worker(args.case, args.seed, args.variant, args.tau, args.snapshot, args.max_steps),
        "calibration": lambda: _frozen_worker("near_truth", args.seed, args.variant, 0.0, None, args.max_steps),
        "weak-signal": lambda: _frozen_worker("weak_signal", args.seed, args.variant, 0.0, None, args.max_steps),
        "cx": lambda: _cx_worker(args.seed, args.support, args.variant if args.variant.startswith("cx_") else None),
        "fullfit": lambda: _fullfit_worker(args.seed, args.variant, args.snapshot),
    }
    try:
        row = workers[args.stage]()
    except BaseException as error:
        row = {
            "stage": args.stage,
            "case": args.case,
            "seed": args.seed,
            "variant": args.variant,
            "status": "worker_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-6000:],
        }
    print(json.dumps(row, allow_nan=False, default=repr))
    return 0


def _run(args: list[str], env: dict[str, str]) -> dict[str, Any]:
    command = ["uv", "run", "--no-sync", "python", "-m", "experiments.edr_structured_2026_10_15.bench", *args]
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT, check=False)
        if proc.returncode:
            return {
                "status": "process_error",
                "returncode": proc.returncode,
                "wall_including_startup_sec": time.perf_counter() - started,
                "stderr": proc.stderr[-6000:],
            }
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        row["wall_including_startup_sec"] = time.perf_counter() - started
        if proc.stderr:
            row["stderr"] = proc.stderr[-2000:]
        return row
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "timeout_sec": TIMEOUT,
            "wall_including_startup_sec": time.perf_counter() - started,
            "error": str(error)[:3000],
        }
    except (IndexError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_worker_output",
            "wall_including_startup_sec": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="frozen")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES, default="isotropic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="core_gn")
    parser.add_argument("--variants", nargs="+", default=["core_gn"])
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--support", type=int, default=8)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--allow-new", action="store_true")
    parser.add_argument("--cx-solver-parity", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.worker:
        return_code = _child(args)
        raise SystemExit(return_code)
    if not args.allow_new and NEW_VARIANTS.intersection(args.variants):
        raise SystemExit("new solver arm refused before independent correctness gate; pass --allow-new only after lead clearance")
    if args.cx_solver_parity and not args.allow_new:
        raise SystemExit("CX solver parity requires the independent correctness gate")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = args.output / "runs.jsonl"
    protocol_run = args.output / "run_metadata.json"
    if runs.exists() or protocol_run.exists():
        raise FileExistsError(f"refusing to overwrite benchmark outputs in {args.output}")
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["UV_CACHE_DIR"] = "/tmp/adp-uv-cache"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rows = []
    with runs.open("x", encoding="utf-8") as stream:
        cases = (
            ("isotropic", "weak_index", "correlated")
            if args.stage == "frozen" and args.snapshot_dir is None
            else ("actual_statistics",)
            if args.stage == "frozen"
            else ("near_truth",)
            if args.stage == "calibration"
            else ("weak_signal",)
            if args.stage == "weak-signal"
            else ("isotropic",)
        )
        supports = (8, 400) if args.stage == "cx" else (args.support,)
        for case in cases:
            for seed in (0, 1, 2):
                for support in supports:
                    variants = args.variants if args.stage != "cx" else (["cx_dense", "cx_factorized"] if args.cx_solver_parity else ["operator_actions"])
                    for variant in variants:
                        snapshot = args.output / "snapshots" / f"seed{seed}_first_outer.npz" if args.stage == "fullfit" and variant == "core_gn" else None
                        if args.stage == "frozen" and args.snapshot_dir is not None:
                            snapshot = args.snapshot_dir / f"seed{seed}_first_outer.npz"
                        command = [
                            "--worker",
                            "--stage",
                            args.stage,
                            "--seed",
                            str(seed),
                            "--variant",
                            variant,
                            "--tau",
                            str(args.tau),
                            "--support",
                            str(support),
                            "--max-steps",
                            str(args.max_steps),
                        ]
                        if args.snapshot_dir is None:
                            command += ["--case", case]
                        if snapshot is not None:
                            command += ["--snapshot", str(snapshot)]
                        row = _run(command, env)
                        row.update(
                            stage=args.stage,
                            case=case,
                            seed=seed,
                            variant=variant,
                            tau=args.tau,
                            support=support,
                        )
                        stream.write(json.dumps(row, allow_nan=False, default=repr) + "\n")
                        stream.flush()
                        rows.append(row)
    metadata = {
        "task": "edr-structured-2026-10-15",
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "harness_sha256": digest(Path(__file__).read_bytes()),
        "worker_sha256": digest(HERE.joinpath("benchmark_workers.py").read_bytes()),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "dirty": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines(),
        "stage": args.stage,
        "variants": args.variants,
        "tau": args.tau,
        "threads": dict.fromkeys(THREADS, "1"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threadpools": threadpool_info(),
        },
        "timeout_sec_per_arm": TIMEOUT,
        "command": "UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.edr_structured_2026_10_15.bench",
        "row_count": len(rows),
        "failures": [row for row in rows if row.get("status") != "ok"],
        "sha256_relevant_sources": {
            name: digest((ROOT / name).read_bytes())
            for name in (
                "ADP/solver/grassman.py",
                "ADP/solver/grassman_research.py",
                "ADP/solver/_grassman_schur.py",
                "ADP/solver/statistic_operator.py",
            )
            if (ROOT / name).exists()
        },
    }
    protocol_run.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "failures": len(metadata["failures"]),
                "results": str(runs),
                "metadata": str(protocol_run),
            }
        )
    )


if __name__ == "__main__":
    main()
