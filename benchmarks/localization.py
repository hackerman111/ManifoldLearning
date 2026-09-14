from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.calculus import (
    calculate_alpha_k,
    pairwise_distance2,
    search_bandwidth,
)


def environment() -> dict[str, object]:
    from threadpoolctl import threadpool_info

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": threadpool_info(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("ADP").rglob("*.py"))
        },
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def measure(function: Callable[[], object], repeats: int) -> dict[str, object]:
    value = function()
    times = []
    for _ in range(repeats):
        started = perf_counter()
        value = function()
        times.append(perf_counter() - started)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "value": value,
        "median_seconds": float(np.median(times)),
        "peak_bytes": peak,
    }


def main() -> None:
    """Сравнить compact search против общего пути того же ядра."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--centers", type=int, default=500)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(713)
    X = rng.normal(size=(args.n, args.d))
    centers = X[: args.centers]
    basis, _ = np.linalg.qr(rng.normal(size=(args.d, args.m)))
    spectrum = np.linspace(1, 0.1, args.m)
    distance2 = pairwise_distance2(X, centers)
    rows = {}
    for label, kernel in (
        ("dense", lambda t: epanechnikov(t)),
        ("compact", epanechnikov),
    ):
        rows[label] = {
            "bandwidth": measure(
                lambda kernel=kernel: search_bandwidth(
                    distance2, 20, kernel, lower=1.0
                ),
                args.repeats,
            ),
            "alpha": measure(
                lambda kernel=kernel: calculate_alpha_k(
                    X,
                    centers,
                    basis,
                    spectrum,
                    1.5,
                    20,
                    kernel,
                    distance2=distance2,
                ),
                args.repeats,
            ),
        }
    report = {
        **environment(),
        "shape": vars(args) | {"output": str(args.output)},
        "seed": 713,
        "dtype": "float64",
        "kernel": "max(1-t**2,0), squared anisotropic distance / h**2",
        "alpha_h": 1.5,
        "target_mean_mass": 20,
        "alpha_tolerance": float(np.sqrt(np.finfo(float).eps)),
        "block_size": 32,
        "measurement": "warmup; untraced timing; separate tracemalloc peak (not RSS)",
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
