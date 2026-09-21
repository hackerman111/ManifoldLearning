"""Общие низкоуровневые утилиты manifold-движка."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ...core.manifold import ADP_Manifold_utils as model_utils
from ..common.calculus import pairwise_distance2


def bandwidth_floor(values: np.ndarray) -> float:
    """Вернуть численно безопасный нижний предел ширины окна."""
    scale = float(np.max(np.ptp(values, axis=0)))
    scale = max(scale, math.sqrt(np.finfo(float).tiny))
    return math.sqrt(np.finfo(float).eps) * scale


def orient_rows(basis: np.ndarray) -> np.ndarray:
    """Зафиксировать знак каждой строки базиса по ведущей координате."""
    columns = np.argmax(np.abs(basis), axis=1)
    signs = np.sign(basis[np.arange(len(basis)), columns])
    basis *= np.where(signs == 0, 1.0, signs)[:, None]
    return basis


def trace_entry(
    phase: str,
    iteration: int,
    h: float,
    h_manifold: float,
    alpha: float,
    alpha_manifold: float,
    mass: np.ndarray,
    n_eff: np.ndarray,
    function_edges: int,
    graph: csr_matrix,
    diagnostics: dict[str, float | int],
) -> dict[str, float | int | str]:
    """Собрать диагностическую запись одной manifold-итерации."""
    manifold_mass = np.asarray(graph.sum(axis=1)).ravel()
    entry: dict[str, float | int | str] = {
        "phase": phase,
        "iteration": iteration,
        "h": float(h),
        "h_manifold": float(h_manifold),
        "alpha": float(alpha),
        "alpha_manifold": float(alpha_manifold),
        "function_edges": function_edges,
        "manifold_edges": int(graph.nnz),
        "function_mass_q10": float(np.quantile(mass, 0.1)),
        "function_mass_min": float(np.min(mass)),
        "function_mass_mean": float(np.mean(mass)),
        "function_mass_median": float(np.median(mass)),
        "function_mass_q90": float(np.quantile(mass, 0.9)),
        "n_eff_q10": float(np.quantile(n_eff, 0.1)),
        "n_eff_min": float(np.min(n_eff)),
        "n_eff_median": float(np.median(n_eff)),
        "n_eff_q90": float(np.quantile(n_eff, 0.9)),
        "manifold_mass_q10": float(np.quantile(manifold_mass, 0.1)),
        "manifold_mass_median": float(np.median(manifold_mass)),
        "manifold_mass_q90": float(np.quantile(manifold_mass, 0.9)),
    }
    entry.update(diagnostics)
    return entry


def feasible_scale(
    model: Any,
    X: np.ndarray,
    centers: np.ndarray,
    projectors: np.ndarray,
    eigenvalues: np.ndarray,
    proposed: float,
    previous: float,
) -> tuple[float, bool]:
    """Уточнить допустимую границу средней массы для текущей геометрии."""

    def feasible(h: float) -> bool:
        return (
            model._mean_mass(X, centers, projectors, eigenvalues, h, 0.0) >= model.N_loc
        )

    if feasible(proposed):
        return proposed, False
    if not feasible(previous):
        model_utils.require_feasible_bracket(False)
    low, high = proposed, previous
    while high - low > math.sqrt(np.finfo(float).eps) * high:
        middle = (low + high) / 2.0
        if feasible(middle):
            high = middle
        else:
            low = middle
    return high, True


def nearest_center_indices(model: Any, X: np.ndarray) -> np.ndarray:
    """Найти ближайший chart-центр пакетной Gram-формулой."""
    indices = np.empty(len(X), dtype=np.intp)
    centered_centers = model.centers_ - model.x_offset_
    for start in range(0, len(X), model.batch_size):
        stop = min(start + model.batch_size, len(X))
        distance2 = pairwise_distance2(
            X[start:stop] - model.x_offset_, centered_centers
        )
        indices[start:stop] = np.argmin(distance2, axis=0)
    return indices


def local_coordinates(model: Any, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Проецировать смещения точек на базис выбранного локального chart."""
    coordinates = np.empty((len(X), model.index_dim))
    for start in range(0, len(X), model.batch_size):
        stop = min(start + model.batch_size, len(X))
        selected = indices[start:stop]
        coordinates[start:stop] = np.einsum(
            "bmd,bd->bm",
            model.projectors_[selected],
            (X[start:stop] - model.x_offset_)
            - (model.centers_[selected] - model.x_offset_),
            optimize=True,
        )
    return coordinates
