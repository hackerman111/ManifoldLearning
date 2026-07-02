from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..common.types import LocalStatistics, TrainingStep
from ..common.utils import unit_vector


class SolverMixin:
    """Общий alternating solver для вариантов ADP."""

    def _progress_record(
        self,
        *,
        stats: LocalStatistics,  # Статистики текущего outer-шага.
        step: TrainingStep,  # Последний inner-шаг.
        outer_index: int,  # Номер outer-шага с нуля.
        outer_total: int,  # Общее число outer-шагов.
        inner_count: int,  # Число выполненных inner-шагов.
        started: float,  # time.perf_counter() начала fit.
    ) -> dict[str, Any]:
        """Формирует программный снимок прогресса.

        Вход:
            stats: локальные статистики.
            step: последняя запись истории.
            outer_index: номер outer-шага.
            outer_total: число outer-шагов.
            inner_count: число inner-шагов в этом outer.
            started: время начала fit.
        Выход:
            Словарь числовых диагностик.
        """

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
        if stats.b is not None:
            record["b"] = float(stats.b)
        if stats.directions is not None:
            record["directions"] = int(stats.directions.shape[1])
        return record

    def _alternating_solve(
        self,
        stats: LocalStatistics,  # Локальные статистики варианта.
        beta_start: np.ndarray,  # Начальное beta для outer-шага.
        lambda_penalty: float,  # Регуляризация к prior.
        outer: int,  # Номер outer-шага.
        outer_started: float,  # Время начала outer-шага.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TrainingStep]]:
        """Запускает внутреннюю попеременную оптимизацию.

        Вход:
            stats: локальные статистики.
            beta_start: стартовое направление beta.
            lambda_penalty: сила штрафа к prior.
            outer: номер внешнего шага.
            outer_started: время начала внешнего шага.
        Выход:
            Кортеж beta, intercepts, slopes и history.
        """

        beta = unit_vector(beta_start)
        prior = beta.copy()
        history: list[TrainingStep] = []
        intercepts = np.zeros(stats.centers.shape[0])
        slopes = np.ones(stats.centers.shape[0])
        last_objective = math.inf

        for inner in range(max(1, self.config.inner_steps)):
            old_beta = beta.copy()
            intercepts, slopes = self._solve_local_coefficients(stats, beta)
            beta = self._solve_beta(stats, intercepts, slopes, prior, lambda_penalty)

            norm = np.linalg.norm(beta)
            if norm > 0:
                beta = beta / norm
                slopes = slopes * norm

            objective = self._objective(stats, beta, intercepts, slopes, prior, lambda_penalty)
            beta_delta = float(np.linalg.norm(beta - old_beta))
            history.append(
                TrainingStep(
                    outer=outer,
                    inner=inner,
                    objective=float(objective),
                    beta_delta=beta_delta,
                    h=float(stats.h),
                    anisotropy=stats.anisotropy,
                    elapsed=time.perf_counter() - outer_started,
                )
            )
            if beta_delta < self.config.tol or abs(last_objective - objective) < self.config.tol:
                break
            last_objective = objective

        return unit_vector(beta), intercepts, slopes, history
