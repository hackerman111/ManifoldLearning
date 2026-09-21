"""Оркестрация manifold-fit и публичные преобразования результата."""

# ruff: noqa: RUF002

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import bandwidth_floor, trace_entry


@dataclass(slots=True)
class ManifoldFitState:
    """Результат численного manifold-fit до публикации состояния модели."""

    centers: np.ndarray
    projectors: np.ndarray
    eigenvalues: np.ndarray
    gradients: np.ndarray
    center_values: np.ndarray
    x_offset: np.ndarray
    trace: list[dict[str, float | int | str]]
    bandwidth: float
    linear_bandwidth: float
    manifold_bandwidth: float
    alpha: float
    manifold_alpha: float
    n_scales: int
    center_indices: np.ndarray
    effective_config: dict[str, float | int]
    stop_reason: str


def fit(model: Any, X: np.ndarray, Y: np.ndarray) -> ManifoldFitState:
    """Выполнить полный manifold-ADP, не публикуя состояние в модели.

    ``model`` остаётся тонким API-фасадом: engine вызывает его совместимые
    private-методы, поэтому monkeypatch/subclass hooks для reference-решателей
    продолжают работать. Все массивные вычисления и solver-шаги находятся здесь.
    """
    X, Y = model._prepare_xy(X, Y)
    n, d = X.shape
    config = model._effective_config(X)
    N_lin = int(config["N_lin"])
    N_J = int(config["N_J"])
    N_phi = int(config["N_phi"])
    N_manifold = int(config["N_manifold"])
    a = float(config["a"])
    h_min = float(config["h_min"])

    # NUMERICAL: единое центрирование защищает Gram-формулу от offset-cancelation.
    x_offset = X.mean(axis=0)
    y_offset = float(Y.mean())
    Xc = X - x_offset
    Yc = Y - y_offset

    center_seed, direction_seed = np.random.SeedSequence(model.seed).spawn(2)
    center_rng = np.random.default_rng(center_seed)
    direction_rng = np.random.default_rng(direction_seed)
    center_indices = center_rng.choice(n, size=N_J, replace=False)
    centers = Xc[center_indices].copy()

    h_floor = bandwidth_floor(Xc)
    h_lin = model._search_bandwidth(Xc, centers, N_lin, lower=h_floor)
    gradients, gradient_mass, center_values = model._local_gradients(
        Xc, Yc, centers, h_lin
    )

    h_manifold = model._search_bandwidth(
        centers,
        centers,
        N_manifold,
        lower=bandwidth_floor(centers),
    )
    manifold_graph = model._build_manifold_graph(centers, None, None, h_manifold, 1.0)
    projectors, eigenvalues = model._initialize_projectors(
        gradients, gradient_mass, manifold_graph, model.index_dim
    )

    h = model._search_bandwidth(Xc, centers, model.N_loc, lower=h_min)
    directions = model._random_directions(direction_rng, len(centers), N_phi, d)
    I, U, mass, n_eff, function_edges = model._calculate_statistics(
        Xc, Yc, centers, directions, None, None, h, 1.0
    )

    trace: list[dict[str, float | int | str]] = []
    for iteration in range(model.sync_steps):
        projectors, eigenvalues, diagnostics = model._one_step(
            I, U, mass, manifold_graph, projectors
        )
        trace.append(
            trace_entry(
                "sync",
                iteration,
                h,
                h_manifold,
                1.0,
                1.0,
                mass,
                n_eff,
                function_edges,
                manifold_graph,
                diagnostics,
            )
        )

    alpha = 1.0
    alpha_manifold = 1.0
    scale = 0
    stop_reason = "h_min"
    while h / a >= h_min:
        proposed_h = h / a
        if model.scale_boundary == "stop":
            if (
                model._mean_mass(
                    centers, centers, projectors, eigenvalues, h_manifold, 0.0
                )
                < N_manifold
            ):
                stop_reason = "manifold_mass_boundary"
                break
            # ESTIMATOR: отдельное правило остановки на границе массы.
            proposed_h, boundary = model._feasible_scale(
                Xc, centers, projectors, eigenvalues, proposed_h, h
            )
            if boundary:
                stop_reason = "function_mass_boundary"
            if proposed_h >= h:
                break
        h = proposed_h
        scale += 1
        alpha = model._search_anisotropy(
            Xc,
            centers,
            projectors,
            eigenvalues,
            h,
            model.N_loc,
            kind="function",
        )
        directions = model._random_directions(direction_rng, len(centers), N_phi, d)
        I, U, mass, n_eff, function_edges = model._calculate_statistics(
            Xc,
            Yc,
            centers,
            directions,
            projectors,
            eigenvalues,
            h,
            alpha,
        )

        alpha_manifold = model._search_anisotropy(
            centers,
            centers,
            projectors,
            eigenvalues,
            h_manifold,
            N_manifold,
            kind="manifold",
        )
        manifold_graph = model._build_manifold_graph(
            centers,
            projectors,
            eigenvalues,
            h_manifold,
            alpha_manifold,
        )
        projectors, eigenvalues, diagnostics = model._one_step(
            I, U, mass, manifold_graph, projectors
        )
        trace.append(
            trace_entry(
                "scale",
                scale,
                h,
                h_manifold,
                alpha,
                alpha_manifold,
                mass,
                n_eff,
                function_edges,
                manifold_graph,
                diagnostics,
            )
        )
        if stop_reason == "function_mass_boundary":
            break

    trace[-1]["stop_reason"] = stop_reason
    return ManifoldFitState(
        centers=centers + x_offset,
        projectors=projectors,
        eigenvalues=eigenvalues,
        gradients=gradients,
        center_values=center_values + y_offset,
        x_offset=x_offset,
        trace=trace,
        bandwidth=float(h),
        linear_bandwidth=float(h_lin),
        manifold_bandwidth=float(h_manifold),
        alpha=float(alpha),
        manifold_alpha=float(alpha_manifold),
        n_scales=scale,
        center_indices=center_indices,
        effective_config=config,
        stop_reason=stop_reason,
    )


def transform(model: Any, X: np.ndarray) -> np.ndarray:
    """Вернуть локальные координаты ближайших chart-центров."""
    queries = model._prepare_queries(X)
    indices = model._nearest_center_indices(queries)
    return model._local_coordinates(queries, indices)


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Предсказать отклик ближайшей локально-линейной моделью."""
    queries = model._prepare_queries(X)
    indices = model._nearest_center_indices(queries)
    coordinates = model._local_coordinates(queries, indices)
    slopes = np.einsum(
        "nmd,nd->nm",
        model.projectors_[indices],
        model.gradients_[indices],
        optimize=True,
    )
    return model.center_values_[indices] + np.einsum(
        "nm,nm->n", coordinates, slopes, optimize=True
    )
