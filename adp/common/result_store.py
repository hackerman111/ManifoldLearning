from __future__ import annotations

import time
from typing import Any

import numpy as np

from .types import ADPResult, LocalStatistics


def store_fit_result(
    model: Any,
    beta: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    statistics: LocalStatistics,
    history: list[Any],
    progress: list[dict[str, Any]],
    timings: dict[str, float],
    started: float,
    X: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray | None,
    objective: float,
) -> ADPResult:
    """Сохраняет итог fit в объекте модели.

    Вход:
        model: модель ADP, в которую записывается результат.
        beta: финальное направление EDR.
        intercepts: локальные свободные члены.
        slopes: локальные наклоны.
        statistics: последние локальные статистики.
        history: история inner-итераций.
        progress: история outer-прогресса.
        timings: накопленные времена этапов.
        started: время начала fit.
        X: приведённая матрица наблюдений.
        y: приведённый вектор ответов.
        centers: использованные центры.
        directions: использованные направления или None.
        objective: финальное значение целевой функции.
    Выход:
        ADPResult, также сохранённый в model.result_.
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
    )
    model.result_ = result
    model.data_ = (X, y)
    model.centers_ = centers
    model.directions_ = directions
    model.diagnostic_plots_ = {}
    return result
