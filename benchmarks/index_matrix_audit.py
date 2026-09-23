"""Воспроизводимые CPU-пробы для аудита single/multi; алгоритм не изменяется."""

# ruff: noqa: RUF002

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ADP.engine.common.ADP_Statistic_engine import calculate_statistics as current
from ADP.engine.common.statistic import calculate_statistics as gemm
from ADP.solver._multi_operator import adjoint, forward


def measure(action: Callable[[], Any]) -> dict[str, Any]:
    """Прогрев, пять повторов и отдельный traced peak без входных массивов."""
    action()
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    tracemalloc.start()
    action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": samples,
        "median_seconds": float(np.median(samples)),
        "traced_peak_bytes": peak,
    }


def pair(
    old: Callable[[], np.ndarray], new: Callable[[], np.ndarray]
) -> dict[str, Any]:
    """Сопоставить фиксированные операторы и измерить две реализации."""
    reference, actual = old(), new()
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
    return {
        "before": measure(old),
        "candidate": measure(new),
        "relative_error": float(
            np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-300)
        ),
    }


def main() -> None:
    """Записать JSON с протоколом, окружением и проверками тождеств."""
    rows: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        for n, d, J, P, density in (
            (4000, 150, 64, 20, 1.0),
            (4000, 150, 64, 20, 0.05),
            (10000, 1000, 32, 20, 0.01),
        ):
            streams = np.random.SeedSequence(seed).spawn(4)
            X = np.random.default_rng(streams[0]).normal(size=(n, d))
            Y = np.random.default_rng(streams[1]).normal(size=n)
            phi = np.random.default_rng(streams[2]).normal(size=(J, P, d))
            phi /= np.linalg.norm(phi, axis=2, keepdims=True)
            W = np.random.default_rng(streams[3]).random((J, n))
            W[1 - density > W] = 0
            before = current(X, Y, W, phi)
            after = gemm(X, Y, W, phi)
            errors = {}
            for name in ("I", "U", "mass", "mean", "n_eff", "eta"):
                a, b = getattr(before, name), getattr(after, name)
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
                errors[name] = float(np.max(np.abs(a - b)))
            rows.append(
                {
                    "kind": "statistics",
                    "seed": seed,
                    "shape_n_d_J_P": [n, d, J, P],
                    "density": density,
                    "before": measure(
                        lambda X=X, Y=Y, W=W, phi=phi: current(X, Y, W, phi)
                    ),
                    "candidate": measure(
                        lambda X=X, Y=Y, W=W, phi=phi: gemm(X, Y, W, phi)
                    ),
                    "max_absolute_errors": errors,
                }
            )
        rng = np.random.default_rng(seed)
        J, P, d, m = 1000, 20, 1000, 10
        U = rng.normal(size=(J, P, d))
        B = rng.normal(size=(m, d))
        L = rng.normal(size=(J, m))
        R = rng.normal(size=(J, P))
        rows.append(
            {
                "kind": "multi_operator",
                "seed": seed,
                "shape_J_P_d_m": [J, P, d, m],
                "forward": pair(
                    lambda U=U, B=B, L=L: np.einsum(
                        "jpd,dm,jm->jp", U, B.T, L, optimize=True
                    ),
                    lambda U=U, B=B, L=L: forward(U, B, L),
                ),
                "adjoint": pair(
                    lambda U=U, R=R, L=L: np.einsum(
                        "jpd,jp,jm->dm", U, R, L, optimize=True
                    ),
                    lambda U=U, R=R, L=L: adjoint(U, R, L).T,
                ),
            }
        )
        np.testing.assert_allclose(
            np.sum(forward(U, B, L) * R), np.sum(B * adjoint(U, R, L)), rtol=1e-12
        )
        Q, _ = np.linalg.qr(B.T)
        V, _ = np.linalg.qr(rng.normal(size=(d, m)))
        rows.append(
            {
                "kind": "subspace_distance",
                "seed": seed,
                "shape_d_m": [d, m],
                **pair(
                    lambda Q=Q, V=V: np.linalg.norm(Q @ Q.T - V @ V.T) / np.sqrt(2),
                    lambda Q=Q, V=V: np.linalg.norm(Q - V @ (V.T @ Q)),
                ),
            }
        )
    output = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "blas": np.show_config(mode="dicts"),
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "protocol": "CPU float64; batch=32; fixed synthetic statistics, no fit. "
        "No kernel/bandwidth/anisotropy/solver/ridge. Gaussian unit directions; "
        "W uniform, zero below 1-density. Four SeedSequence streams X,Y,Phi,W. "
        "Operators use sequential local RNG draws. Five repeats after warmup. "
        "tracemalloc separate from time; excludes inputs "
        "and untraced native allocations. "
        "No GPU or EDR recovery claims; candidate statistics also returns S.",
        "results": rows,
    }
    Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
