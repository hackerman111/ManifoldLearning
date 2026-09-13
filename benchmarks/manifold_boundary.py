"""Воспроизводимая проверка локального EDR и стоимости manifold-fit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ADP import ADP_Manifold


class ReferenceManifold(ADP_Manifold):
    """Контроль прежнего einsum-kernel при неизменных остальных шагах."""

    @staticmethod
    def _penalty_action(
        B: np.ndarray, source_projectors: np.ndarray, normalized_weights: np.ndarray
    ) -> np.ndarray:
        coordinates = np.einsum("ad,jrd->jar", B, source_projectors, optimize=True)
        return B - np.einsum(
            "j,jar,jrd->ad",
            normalized_weights,
            coordinates,
            source_projectors,
            optimize=True,
        )

    def _build_B_system(
        self,
        U: np.ndarray,
        I: np.ndarray,
        mass: np.ndarray,
        weights: np.ndarray,
        source_projectors: np.ndarray,
        slopes: np.ndarray,
    ) -> tuple[Any, Any, np.ndarray]:
        operator, preconditioner, rhs = super()._build_B_system(
            U, I, mass, weights, source_projectors, slopes
        )
        m, d = slopes.shape[1], U.shape[2]
        gamma = mass * weights
        normalized = weights / weights.sum()

        def matvec(vector: np.ndarray) -> np.ndarray:
            B = vector.reshape(m, d)
            projected = np.einsum("jpd,jd->jp", U, slopes @ B, optimize=True)
            pulled = np.einsum("jpd,jp->jd", U, projected, optimize=True)
            result = np.einsum("j,ja,jd->ad", gamma, slopes, pulled, optimize=True)
            result += self.lambda_manifold * self._penalty_action(
                B, source_projectors, normalized
            )
            return result.ravel()

        return (
            scipy.sparse.linalg.LinearOperator(
                operator.shape, matvec=matvec, rmatvec=matvec, dtype=float
            ),
            preconditioner,
            rhs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    rows = []
    cases = ((32, 0.01),) if args.large else ((4, 0.01), (8, 0.01), (4, 0.15))
    for d, noise in cases:
        for seed in (7, 19, 31):
            data_seed, noise_seed, model_seed = np.random.SeedSequence(seed).spawn(3)
            rng = np.random.default_rng(data_seed)
            X = rng.uniform(-1.0, 1.0, (2000 if args.large else 600, d))
            # Меняющееся направление градиента; |grad f| >= 1.
            Y = X[:, 0] + args.curvature * np.square(X[:, 1])
            Y += noise * np.random.default_rng(noise_seed).normal(size=len(X))
            settings = {
                "N_loc": 30,
                "N_lin": 80,
                "N_J": 40,
                "N_phi": 15,
                "N_manifold": 6,
                "sync_steps": 2,
                "h_min": 0.001 if args.boundary else 0.8,
                "seed": int(model_seed.generate_state(1)[0]),
            }
            if args.boundary:
                settings["scale_boundary"] = "stop"
            if args.large:
                settings.update(N_lin=96, N_J=100, N_phi=20, N_manifold=12)
            model_type = ReferenceManifold if args.reference else ADP_Manifold
            model = model_type(1, **settings)
            row = {
                "n": len(X),
                "d": d,
                "m": 1,
                "noise": noise,
                "seed": seed,
                "curvature": args.curvature,
                "settings": settings,
            }
            started = time.perf_counter()
            try:
                model.fit(X, Y)
                true = np.zeros((len(model.centers_), d))
                true[:, 0] = 1.0
                true[:, 1] = 2 * args.curvature * model.centers_[:, 1]
                true /= np.linalg.norm(true, axis=1, keepdims=True)
                overlap = np.square(
                    np.einsum("jd,jd->j", model.projectors_[:, 0], true)
                )
                error = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * overlap))
                row.update(
                    status="success",
                    projector_error_mean=float(error.mean()),
                    projector_error_max=float(error.max()),
                    trace=model.trace_,
                    recovered=bool(error.mean() <= 0.1 and error.max() <= 0.2),
                )
            except RuntimeError as exc:
                row.update(status="failure", error=str(exc))
            row["seconds"] = time.perf_counter() - started
            # Отдельный повтор: tracemalloc существенно искажает wall-clock.
            tracemalloc.start()
            with contextlib.suppress(RuntimeError):
                model.fit(X, Y)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            row["tracemalloc_peak_mib"] = peak / 2**20
            rows.append(row)
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    report = {
        "source_sha256": hashlib.sha256(
            Path("ADP/ADP_Manifold.py").read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_kernel": args.reference,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": config.getvalue(),
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "protocol": "float64; K(q)=(1-q^2)+; squared distance; see settings and trace; "
        "independent SeedSequence data/noise/model; model splits centers/directions; "
        "normalized penalty; CG rtol=1e-8; no ridge; batch_size=32; "
        "first timed fit includes warm-up; "
        "memory excludes untracked native allocations",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {k: v for k, v in row.items() if k not in {"trace", "settings"}}
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
