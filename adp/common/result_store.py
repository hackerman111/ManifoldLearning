from __future__ import annotations

import time
from typing import Any

import numpy as np

from .types import ADPResult, LocalStatistics


def store_fit_result(
    model: Any,  # Модель ADP, куда записывается состояние.
    beta: np.ndarray,  # Финальное направление beta.
    intercepts: np.ndarray,  # Локальные свободные члены.
    slopes: np.ndarray,  # Локальные наклоны.
    statistics: LocalStatistics,  # Последние локальные статистики.
    history: list[Any],  # История внутренних итераций.
    progress: list[dict[str, Any]],  # История внешних шагов.
    timings: dict[str, float],  # Накопленные времена этапов.
    started: float,  # Момент начала обучения.
    X: np.ndarray,  # Приведенная матрица наблюдений.
    y: np.ndarray,  # Приведенный вектор ответов.
    centers: np.ndarray,  # Использованные центры.
    directions: np.ndarray | None,  # Использованные направления или None.
    objective: float,  # Финальное значение целевой функции.
    beta_path: list[np.ndarray] | None = None,  # beta после каждого outer-шага.
) -> ADPResult:
    """Сохраняет итог fit в модель и возвращает ADPResult.

    Вход:
        Все аргументы описывают финальное состояние model.fit(...).
    Выход:
        ADPResult, записанный в model.result_.
    """

    timings["total"] = time.perf_counter() - started
    result = ADPResult(
        beta=beta,
        intercepts=intercepts,
        slopes=slopes,
        statistics=statistics,
        history=history,
        progress=progress,
        objective=float(objective),
        backend=model.backend.name,
        timings=timings,
        beta_path=list(beta_path or []),
    )
    model.result_ = result
    model.data_ = (X, y)
    model.centers_ = centers
    model.directions_ = directions
    model.diagnostic_plots_ = {}
    return result
