"""Подключение экспериментального отбора к общему multi-index outer loop."""

from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, vstack

from ...core.ADP_Config import ADP_Config, epanechnikov
from ...core.ADP_Statistic import ADP_Statistics
from .experimental_centers import (
    coverage,
    greedy_select,
    information_vectors,
    projected_statistics,
    quadrature_mass,
)
from .experimental_geometry import ProjectedComponents, projected_components
from .statistic import calculate_statistics
from .weights import calculate_multi_weight


def validate_experiment(config: ADP_Config, mode: str, J: int, d: int, m: int) -> None:
    exp = config.experimental_centers
    if exp is None or not exp.active:
        return
    if mode != "multi" or config.estimator != "new" or config.gpu:
        raise NotImplementedError(
            "adaptive centers require CPU multi-index estimator=new"
        )
    if config.kernel is not epanechnikov:
        raise NotImplementedError(
            "adaptive centers require the built-in Epanechnikov kernel"
        )
    minimum = min(J, max(4 * m, 2 * d))
    if exp.budget is not None and not minimum <= exp.budget <= J:
        raise ValueError(f"center budget must lie in [{minimum}, {J}]")


def experimental_statistics(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    alpha: float,
    distance2: np.ndarray,
    directions: np.ndarray,
    config: ADP_Config,
    tensor: str,
    rng: np.random.Generator,
    geometry: ProjectedComponents | None,
) -> tuple[ADP_Statistics, np.ndarray, dict]:
    """Полный пул для alpha/ошибки шага; полные U только для выбранных центров."""
    exp = config.experimental_centers
    assert exp is not None
    started = perf_counter()
    if exp.projection_search:
        if geometry is None:
            geometry = projected_components(
                X, centers, basis, eigenvalues, h, distance2, tensor=tensor
            )
        weights = geometry.weights(alpha)
        Z = geometry.projected_X
        candidates = len(geometry.indices)
    else:
        weights = csr_matrix(
            vstack(
                [
                    csr_matrix(W)
                    for _, W in calculate_multi_weight(
                        X,
                        centers,
                        basis,
                        eigenvalues,
                        h,
                        alpha,
                        config.kernel,
                        config.batch_size,
                        distance2=distance2,
                        tensor=tensor,
                    )
                ],
                format="csr",
            )
        )
        Z = (X - X[0]) @ basis
        candidates = len(X) * len(centers)
    J, m = len(centers), basis.shape[1]
    selected = np.arange(J)
    mass = np.asarray(weights.sum(axis=1)).ravel()
    if np.any(mass <= 0):
        raise RuntimeError("experimental full pool has zero local mass")
    response_mean = np.asarray(weights @ Y).ravel() / mass
    diagnostic = {
        "selection": exp.selection,
        "candidate_edges": candidates,
        "support_edges": weights.nnz,
        "fallback": None,
        "weight_time_seconds": perf_counter() - started,
        "projected_time_seconds": 0.0,
        "selection_time_seconds": 0.0,
    }
    if exp.projected_statistics or exp.selection != "all":
        started = perf_counter()
        moments = projected_statistics(Z, Y, weights, ridge=exp.score_ridge)
        diagnostic["projected_time_seconds"] = perf_counter() - started
        started = perf_counter()
        feasible = np.flatnonzero(moments.feasible)
        diagnostic["feasible_centers"] = len(feasible)
        minimum = min(J, max(4 * m, 2 * X.shape[1]))
        required = exp.budget if exp.budget is not None else minimum
        if exp.selection != "all" and len(feasible) < required:
            diagnostic["fallback"] = "insufficient_feasible_full_pool"
        elif exp.selection != "all":
            vectors = information_vectors(moments, Y, config.N_loc)
            if exp.selection == "random":
                selected = np.asarray(
                    rng.choice(feasible, size=required, replace=False)
                )
            elif exp.selection == "score":
                scores = np.sum(vectors[feasible] ** 2, axis=1)
                selected = feasible[np.argsort(-scores, kind="stable")[: exp.budget]]
            else:
                selected, greedy = greedy_select(
                    weights,
                    vectors,
                    feasible,
                    budget=exp.budget,
                    minimum=minimum,
                    coverage_weight=(
                        exp.coverage_weight
                        if exp.selection == "logdet_coverage"
                        else 0.0
                    ),
                    ridge=exp.logdet_ridge,
                )
                diagnostic.update(greedy)
        diagnostic["selection_time_seconds"] = perf_counter() - started
    started = perf_counter()
    statistics = calculate_statistics(
        X,
        Y,
        csr_matrix(weights[selected]),
        directions[selected],
        batch_size=config.batch_size,
        normalized=True,
    )
    if exp.quadrature and exp.selection != "all":
        statistics = replace(statistics, mass=quadrature_mass(weights, selected))
    diagnostic["assembly_time_seconds"] = perf_counter() - started
    diagnostic.update(
        selected_centers=len(selected),
        selected_pool_indices=tuple(map(int, selected)),
        mass_coverage=coverage(weights, selected),
        step_error_centers=J,
        quadrature=exp.quadrature,
        localization_eigenvalues=tuple(map(float, eigenvalues)),
    )
    return statistics, response_mean, diagnostic


def experimental_metadata(config: ADP_Config) -> dict | None:
    exp = config.experimental_centers
    return asdict(exp) if exp is not None else None
