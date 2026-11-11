"""Bounded inactive-block SchurCore kernel timing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(seed: int) -> dict[str, Any]:
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info

    from ADP.solver import grassman
    from ADP.solver._grassman_schur import SchurCore

    rng = np.random.default_rng(seed)
    J, p, d, m, q, a = 100, 12, 40, 6, 2, 2
    U = rng.standard_normal((J, p, d))
    Y, _ = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((d, q)), mode="reduced")
    V, _ = np.linalg.qr(V - Y @ (Y.T @ V), mode="reduced")
    M, W = grassman._project(U, Y), grassman._project(U, V)
    mass = np.full(J, 1 / J)
    coeff = rng.standard_normal((J, m))
    target = np.einsum("jpm,jm->jp", M, coeff) + 0.1 * rng.standard_normal((J, p))
    A = np.eye(m)[:, :a]
    rss0 = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    start = time.perf_counter()
    core = SchurCore(M, W, target, mass, 0.0, A=A)
    setup = time.perf_counter() - start
    trials = [(0.02 * rng.standard_normal((q, a)), rng.standard_normal((q, a))) for _ in range(3)]
    warm_K, warm_E = 0.02 * rng.standard_normal((q, a)), rng.standard_normal((q, a))

    def schur(K: np.ndarray, E: np.ndarray) -> float:
        value, _ = core.evaluate(K)
        core.hvp(K, E)
        return value

    def reference(K: np.ndarray) -> float:
        L = K @ A.T
        _, Mn = grassman._polar(Y, M, V, W, L)
        return grassman._profile(Mn, target, mass, 0.0).value

    schur(warm_K, warm_E)
    reference(warm_K)
    schur_times, reference_times, gaps = [], [], []
    for K, E in trials:
        start = time.perf_counter()
        value = schur(K, E)
        schur_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        ref_value = reference(K)
        reference_times.append(time.perf_counter() - start)
        gaps.append(abs(value - ref_value))
    zero = core.evaluate(np.zeros((q, a)))[0]
    full_zero = grassman._profile(M, target, mass, 0.0).value
    return {
        "seed": seed,
        "status": "ok",
        "shape": {"J": J, "p": p, "d": d, "m": m, "q": q, "active_a": a},
        "dtype": str(U.dtype),
        "tau": 0.0,
        "setup_sec": setup,
        "schur_evaluate_hvp_sec": schur_times,
        "polar_profile_sec": reference_times,
        "trial_value_gaps": gaps,
        "timed_trials": len(trials),
        "schur_calls": {"evaluate": 3, "hvp": 3},
        "schur_median_sec": float(np.median(schur_times)),
        "polar_profile_median_sec": float(np.median(reference_times)),
        "base_profile_value_abs_error": abs(zero - full_zero),
        "trial_profile_value_gap_max": max(gaps),
        "peak_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "additional_peak_rss_platform_units": max(0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - rss0),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threadpools": threadpool_info(),
            "threads": {key: os.environ.get(key) for key in THREADS},
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "schur_kernel_distinct")
    args = parser.parse_args()
    if args.worker:
        try:
            print(json.dumps(worker(args.seed), allow_nan=False))
        except BaseException as exc:
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "status": "worker_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    runs, meta = args.output / "runs.jsonl", args.output / "run_metadata.json"
    if runs.exists() or meta.exists():
        raise FileExistsError(args.output)
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["UV_CACHE_DIR"] = "/tmp/adp-uv-cache"
    rows = []
    with runs.open("x", encoding="utf-8") as stream:
        for seed in (0, 1, 2):
            cmd = [
                sys.executable,
                "-m",
                "experiments.edr_structured_2026_10_15.schur_kernel",
                "--worker",
                "--seed",
                str(seed),
            ]
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                row = (
                    json.loads(proc.stdout.strip().splitlines()[-1])
                    if proc.returncode == 0
                    else {
                        "seed": seed,
                        "status": "process_error",
                        "stderr": proc.stderr[-4000:],
                    }
                )
            except subprocess.TimeoutExpired as exc:
                row = {
                    "seed": seed,
                    "status": "timeout",
                    "timeout_sec": 120,
                    "error": str(exc)[:2000],
                }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    meta.write_text(
        json.dumps(
            {
                "protocol_sha256": sha(HERE / "protocol.json"),
                "harness_sha256": sha(Path(__file__)),
                "commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "dirty": subprocess.run(
                    ["git", "status", "--short"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines(),
                "command": "UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.edr_structured_2026_10_15.schur_kernel",
                "threads": dict.fromkeys(THREADS, "1"),
                "seeds": [0, 1, 2],
                "rows": rows,
                "failures": [row for row in rows if row.get("status") != "ok"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "failures": sum(row.get("status") != "ok" for row in rows),
                "results": str(runs),
            }
        )
    )


if __name__ == "__main__":
    main()
