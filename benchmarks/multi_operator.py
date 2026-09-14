from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ADP.solver._multi_operator import adjoint, forward


def measure(action: Callable[[], np.ndarray], repeats: int) -> dict[str, float]:
    """Прогрев; wall-clock без трассировки, отдельный пик traced allocations."""
    action()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    tracemalloc.start()
    action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"median_seconds": float(np.median(samples)), "peak_bytes": float(peak)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    for J, P, d, m in [(100, 20, 100, 2), (1000, 20, 1000, 2), (1000, 20, 1000, 10)]:
        rng = np.random.default_rng(187)
        U = rng.normal(size=(J, P, d))
        B = rng.normal(size=(m, d))
        L = rng.normal(size=(J, m))
        data = rng.normal(size=(J, P))
        mass = rng.uniform(0.1, 2.0, size=J)

        def old_forward(U=U, B=B, L=L) -> np.ndarray:
            return np.einsum("jpd,md,jm->jp", U, B, L, optimize=True)

        def new_forward(U=U, B=B, L=L) -> np.ndarray:
            return forward(U, B, L)

        def old_adjoint(mass=mass, L=L, U=U, data=data) -> np.ndarray:
            return np.einsum("j,jm,jpd,jp->md", mass, L, U, data, optimize=True)

        def new_adjoint(mass=mass, L=L, U=U, data=data) -> np.ndarray:
            return adjoint(U, mass[:, None] * data, L)

        row: dict[str, object] = {"shape_J_P_d_m": [J, P, d, m], "dtype": "float64"}
        for name, old, new in (
            ("forward", old_forward, new_forward),
            ("adjoint", old_adjoint, new_adjoint),
        ):
            reference = old()
            error = np.linalg.norm(new() - reference) / np.linalg.norm(reference)
            row[name] = {
                "before": measure(old, 15),
                "after": measure(new, 15),
                "relative_error": float(error),
            }
        results.append(row)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
