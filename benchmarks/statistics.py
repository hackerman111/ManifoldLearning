"""Парные замеры стадии статистик; каждый вариант в отдельном процессе."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy

from ADP.core.ADP_Manifold import ADP_Manifold
from ADP.engine.statistic import calculate_statistics
from benchmarks.statistics_reference import (
    calculate_statistics as reference_statistics,
    manifold_statistics,
)

CASES = {
    "sparse": (4000, 150, 64, 20, 0.05),
    "dense": (4000, 150, 64, 20, 1.0),
    "stress_sparse": (10000, 1000, 32, 20, 0.01),
    "manifold": (4000, 150, 64, 20, 1.0),
}


def worker(args: argparse.Namespace) -> None:
    n, d, J, P, density = CASES[args.case]
    streams = np.random.SeedSequence(args.seed).spawn(4)
    X = np.random.default_rng(streams[0]).normal(size=(n, d))
    Y = np.random.default_rng(streams[1]).normal(size=n)
    phi = np.random.default_rng(streams[2]).normal(size=(J, P, d))
    phi /= np.linalg.norm(phi, axis=2, keepdims=True)
    W = np.random.default_rng(streams[3]).random((J, n))
    W[1 - density > W] = 0
    if args.case == "manifold":
        model = SimpleNamespace(
            batch_size=32, _weight_block=lambda *a: W[a[-1] : a[-1] + 32]
        )
        method = (
            manifold_statistics
            if args.variant == "reference"
            else ADP_Manifold._calculate_statistics
        )

        def calculate() -> dict[str, np.ndarray]:
            values = method(model, X, Y, X[:J], phi, None, None, 1.0, 1.0)
            return dict(zip(("I", "U", "mass", "n_eff", "edges"), values, strict=True))

    else:
        method = (
            reference_statistics
            if args.variant == "reference"
            else calculate_statistics
        )

        def calculate() -> dict[str, np.ndarray]:
            result = method(X, Y, W, phi, normalized=True)
            return {name: result[name] for name in result}

    calculate()  # Прогрев; замер времени отдельно от tracemalloc.
    times = []
    for _ in range(5):
        start = perf_counter()
        calculate()
        times.append(perf_counter() - start)
    tracemalloc.start()
    result = calculate()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.savez(args.output.with_suffix(".npz"), **result)
    sources = [
        Path("ADP/engine/statistic.py"),
        Path("ADP/core/ADP_Manifold.py"),
        Path("benchmarks/statistics_reference.py"),
        Path(__file__),
    ]
    row = {
        "case": args.case,
        "variant": args.variant,
        "seed": args.seed,
        "shape_n_d_J_P": [n, d, J, P],
        "density": density,
        "batch_size": 32,
        "dtype": "float64",
        "seconds": times,
        "median_seconds": float(np.median(times)),
        "tracemalloc_peak_bytes": peak,
        "rss_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "blas": np.show_config(mode="dicts"),
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "protocol": (
            "Fixed statistics inputs, no fit or solver. W=uniform(0,1), "
            "zero if W<1-density; normalized=True. Four SeedSequence streams: "
            "X,Y,directions,W. Standard Gaussian directions normalized per row, "
            "no refresh. No kernel, bandwidth, anisotropy or ridge. "
            "Manifold weight construction excluded. RSS includes inputs and warmup; "
            "tracemalloc excludes untraced native allocations."
        ),
    }
    args.output.write_text(json.dumps(row, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--variant", choices=("reference", "optimized"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.variant:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    rows = []
    for case in CASES:
        for seed in (1, 2, 3):
            order = (
                ("reference", "optimized") if seed % 2 else ("optimized", "reference")
            )
            pair = {}
            for variant in order:
                output = args.output / f"{case}-{seed}-{variant}.json"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.statistics",
                    "--output",
                    str(output),
                    "--case",
                    case,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                ]
                try:
                    subprocess.run(command, env=environment, check=True, timeout=120)
                    pair[variant] = json.loads(output.read_text())
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as error:
                    pair[variant] = {"error": str(error)}
            if all("error" not in value for value in pair.values()):
                with (
                    np.load(args.output / f"{case}-{seed}-reference.npz") as before,
                    np.load(args.output / f"{case}-{seed}-optimized.npz") as after,
                ):
                    pair["max_absolute_errors"] = {
                        name: float(np.max(np.abs(before[name] - after[name])))
                        for name in before.files
                    }
            rows.append({"case": case, "seed": seed, **pair})
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
            print(case, seed, "complete", flush=True)


if __name__ == "__main__":
    main()
