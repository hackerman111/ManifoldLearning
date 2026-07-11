from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..common.types import LocalStatistics, TrainingStep


class SolverMixin:
    """Совместимые solver-методы поверх единственного цикла ADPAlgorithm."""

    def _progress_record(
        self,
        *,
        stats: LocalStatistics,
        step: TrainingStep,
        outer_index: int,
        outer_total: int,
        inner_count: int,
        started: float,
    ) -> dict[str, Any]:
        """Формирует программный снимок прогресса."""

        record: dict[str, Any] = {
            "variant": self.variant,
            "backend": self.backend.name,
            "outer": outer_index + 1,
            "outer_total": outer_total,
            "inner": inner_count,
            "h": float(stats.h),
            "weights": float(stats.weights_mean),
            "objective": float(step.objective),
            "delta": float(step.beta_delta),
            "elapsed": float(time.perf_counter() - started),
        }
        if stats.anisotropy is not None:
            record["rho"] = float(stats.anisotropy)
        if stats.n_directions is not None:
            record["directions"] = int(stats.n_directions)
        elif stats.directions is not None:
            record["directions"] = int(stats.directions.shape[1])
        if stats.N is not None:
            local_mass = np.asarray(stats.N, dtype=float)
            finite_mass = local_mass[np.isfinite(local_mass)]
            if finite_mass.size:
                record["local_mass_mean"] = float(np.mean(finite_mass))
                record["local_mass_q05"] = float(np.quantile(finite_mass, 0.05))
                record["local_mass_min"] = float(np.min(finite_mass))
        return record

    def _alternating_solve(
        self,
        stats: LocalStatistics,
        beta_start: np.ndarray,
        lambda_penalty: float,
        outer: int,
        outer_started: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TrainingStep]]:
        """Делегирует legacy-вызов единственной реализации внутреннего цикла."""

        return self.algorithm._alternating_solve(
            stats,
            beta_start,
            lambda_penalty,
            outer,
            outer_started,
            use_protected_adapters=True,
        )
