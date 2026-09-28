"""Весовые функции, bandwidth/anisotropy search и локальные градиенты."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...core.ADP_Config import epanechnikov as kernel
from ...core.manifold import ADP_Manifold_utils as model_utils
from ..common.calculus import pairwise_distance2
from ..common.statistic import _dense_moments


def weight_block(
    model: Any,
    X: np.ndarray,
    centers: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
    start: int,
) -> np.ndarray:
    """Построить блок изотропных или проекторно-анизотропных весов."""
    stop = min(start + model.batch_size, len(centers))
    center_block = centers[start:stop]
    distance2 = pairwise_distance2(X, center_block)  # (B, n)
    if projectors is None:
        distance2 /= h**2
        return kernel(distance2)
    assert eigenvalues is not None

    basis = projectors[start:stop]  # (B, m, d)
    coordinates = np.einsum("bmd,nd->bmn", basis, X, optimize=True)
    coordinates -= np.einsum("bmd,bd->bm", basis, center_block, optimize=True)[
        ..., None
    ]
    np.square(coordinates, out=coordinates)
    projected2 = coordinates.sum(axis=1)
    if model.localization_spectrum == "unit":
        principal2 = projected2
    else:
        principal2 = np.einsum(
            "bm,bmn->bn", eigenvalues[start:stop], coordinates, optimize=True
        )
    residual2 = distance2 - projected2
    np.maximum(residual2, 0.0, out=residual2)
    principal2 += alpha**2 * residual2
    principal2 /= h**2
    return kernel(principal2)


def mean_mass(
    model: Any,
    X: np.ndarray,
    centers: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
) -> float:
    """Посчитать среднюю локальную массу по всем центрам."""
    total = 0.0
    for start in range(0, len(centers), model.batch_size):
        total += float(
            model._weight_block(
                X, centers, projectors, eigenvalues, h, alpha, start
            ).sum()
        )
    return total / len(centers)


def search_bandwidth(
    model: Any,
    X: np.ndarray,
    centers: np.ndarray,
    target: int,
    *,
    lower: float,
) -> float:
    """Найти минимальный bandwidth при требуемой средней локальной массе."""
    model_utils.require_bandwidth_target(target, len(X))
    if model._mean_mass(X, centers, None, None, lower, 1.0) >= target:
        return float(lower)

    maximum_distance2 = 0.0
    for start in range(0, len(centers), model.batch_size):
        block = centers[start : start + model.batch_size]
        maximum_distance2 = max(
            maximum_distance2, float(pairwise_distance2(X, block).max())
        )
    high = max(2.0 * lower, math.sqrt(maximum_distance2), lower + 1.0e-12)
    for _ in range(100):
        if model._mean_mass(X, centers, None, None, high, 1.0) >= target:
            break
        high *= 2.0
    else:
        model_utils.raise_bandwidth_bracket_failure()

    low = lower
    for _ in range(60):
        middle = 0.5 * (low + high)
        if model._mean_mass(X, centers, None, None, middle, 1.0) >= target:
            high = middle
        else:
            low = middle
    return float(high)


def search_anisotropy(
    model: Any,
    X: np.ndarray,
    centers: np.ndarray,
    projectors: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    target: int,
    *,
    kind: str,
) -> float:
    """Найти максимальный alpha при требуемой средней массе."""
    if model._mean_mass(X, centers, projectors, eigenvalues, h, 1.0) >= target:
        return 1.0
    if model._mean_mass(X, centers, projectors, eigenvalues, h, 0.0) < target:
        model_utils.raise_infeasible_mass(kind)
    low, high = 0.0, 1.0
    tolerance = math.sqrt(np.finfo(float).eps)
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if model._mean_mass(X, centers, projectors, eigenvalues, h, middle) >= target:
            low = middle
        else:
            high = middle
    return float(low)


def local_gradients(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оценить local-linear gradients без normal matrix и ridge."""
    gradients = np.empty((len(centers), X.shape[1]))
    masses = np.empty(len(centers))
    center_values = np.empty(len(centers))
    for start in range(0, len(centers), model.batch_size):
        weights = model._weight_block(X, centers, None, None, h, 1.0, start)
        for offset, weights_j in enumerate(weights):
            j = start + offset
            mass = float(weights_j.sum())
            model_utils.require_positive_mass(
                f"empty local-linear neighborhood at center {j}", mass
            )
            normalized = weights_j / mass
            mean = normalized @ X
            y_mean = float(normalized @ Y)
            root_weight = np.sqrt(weights_j)
            design = (X - mean) * root_weight[:, None]
            response = (Y - y_mean) * root_weight
            gradient, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
            if rank != X.shape[1] or not np.all(np.isfinite(gradient)):
                model_utils.require_full_rank(
                    rank,
                    X.shape[1],
                    f"rank-deficient local-linear fit at center {j}: "
                    f"rank={rank}, required={X.shape[1]}; increase N_lin",
                )
            gradients[j] = gradient
            masses[j] = mass
            center_values[j] = y_mean + gradient @ (centers[j] - mean)
    return gradients, masses, center_values


def local_quadratic_pilot(
    X: np.ndarray, Y: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оценить градиент и упакованные квадратичные коэффициенты по 60 соседям.

    Этот отдельный ESTIMATOR-вариант предназначен для ``m=1``. Полином
    второй степени убирает смещение directional moments от квадратичной
    части отклика без тензора ``(J,n,d)`` или матриц ``(J,d,d)``.
    """
    n, d = X.shape
    n_terms = 1 + d + d * (d + 1) // 2
    neighbors_count = min(60, n)
    if neighbors_count < n_terms or n <= 6:
        raise ValueError(
            "local_quadratic requires n > 6 and at least "
            f"{n_terms} observations for its polynomial terms"
        )
    pairs = [(a, b) for a in range(d) for b in range(a, d)]
    gradients = np.empty((len(centers), d))
    values = np.empty(len(centers))
    quadratic_coefficients = np.empty((len(centers), len(pairs)))
    for j, center in enumerate(centers):
        distance2 = np.sum(np.square(X - center), axis=1)
        neighbors = np.argpartition(distance2, neighbors_count - 1)[:neighbors_count]
        delta = X[neighbors] - center
        design = np.column_stack(
            (
                np.ones(neighbors_count),
                delta,
                *(
                    (0.5 if a == b else 1.0) * delta[:, a] * delta[:, b]
                    for a, b in pairs
                ),
            )
        )
        coefficient, _, rank, _ = np.linalg.lstsq(design, Y[neighbors], rcond=None)
        model_utils.require_full_rank(
            int(rank),
            n_terms,
            f"rank-deficient local quadratic pilot at center {j}: "
            f"rank={rank}, required={n_terms}",
        )
        if not np.all(np.isfinite(coefficient)):
            raise RuntimeError(f"non-finite local quadratic pilot at center {j}")
        values[j] = coefficient[0]
        gradients[j] = coefficient[1 : d + 1]
        quadratic_coefficients[j] = coefficient[d + 1 :]
    return gradients, values, quadratic_coefficients


def quadratic_statistics(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
    quadratic_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Вычесть момент квадратичной части и защитить массу каждого центра."""
    I, U, mass, n_eff, edges = model._calculate_statistics(
        X, Y, centers, directions, projectors, eigenvalues, h, alpha
    )
    d = X.shape[1]
    pairs = [(a, b) for a in range(d) for b in range(a, d)]
    for j, center in enumerate(centers):
        basis = None if projectors is None else projectors[j : j + 1]
        spectrum = None if eigenvalues is None else eigenvalues[j : j + 1]

        def local_weight(
            bandwidth: float,
            center_block: np.ndarray = centers[j : j + 1],
            basis: np.ndarray | None = basis,
            spectrum: np.ndarray | None = spectrum,
        ) -> np.ndarray:
            return model._weight_block(
                X, center_block, basis, spectrum, bandwidth, alpha, 0
            )[0]

        weight = local_weight(h)
        if mass[j] < 6.0:
            old_support = np.count_nonzero(weight)
            low, high = h, h
            for _ in range(30):
                if local_weight(high).sum() >= 6.0:
                    break
                high *= 2
            else:
                raise RuntimeError(f"observation mass below six at center {j}")
            for _ in range(35):
                middle = (low + high) / 2
                if local_weight(middle).sum() >= 6.0:
                    high = middle
                else:
                    low = middle
            weight = local_weight(high)
            mass[j] = weight.sum()
            normalized = weight / mass[j]
            moment = _dense_moments(
                X,
                Y,
                normalized[None, :],
                directions[j : j + 1],
                mass[j : j + 1],
                True,
                include_eta=False,
            )
            I[j] = moment.I[0]
            U[j] = moment.U[0]
            n_eff[j] = 1.0 / np.square(normalized).sum()
            edges += int(np.count_nonzero(weight) - old_support)
        else:
            normalized = weight / mass[j]
        delta = X - center
        quadratic = np.zeros(len(X))
        for coefficient, (a, b) in zip(quadratic_coefficients[j], pairs, strict=True):
            quadratic += (
                (0.5 if a == b else 1.0) * coefficient * delta[:, a] * delta[:, b]
            )
        correction = _dense_moments(
            X,
            quadratic,
            normalized[None, :],
            directions[j : j + 1],
            mass[j : j + 1],
            True,
            include_eta=False,
        )
        I[j] -= correction.I[0]
    model_utils.require_statistics(mass, I, U)
    return I, U, mass, n_eff, edges


def random_directions(rng: np.random.Generator, J: int, P: int, d: int) -> np.ndarray:
    """Сгенерировать единичные направления random-direction sketch."""
    directions = rng.standard_normal((J, P, d))
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    model_utils.require_finite_directions(norms)
    return directions / norms


def calculate_statistics(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    directions: np.ndarray,
    projectors: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    h: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Вычислить normalized ``I/U`` при рабочей памяти ``O(B*P*n)``."""
    J, P, d = directions.shape
    I = np.empty((J, P))
    U = np.empty((J, P, d))
    mass = np.empty(J)
    n_eff = np.empty(J)
    edges = 0
    for start in range(0, J, model.batch_size):
        weights = model._weight_block(
            X, centers, projectors, eigenvalues, h, alpha, start
        )
        stop = start + len(weights)
        mass_block = weights.sum(axis=1)
        model_utils.require_mass(mass_block)
        normalized = weights / mass_block[:, None]
        moments = _dense_moments(
            X,
            Y,
            normalized,
            directions[start:stop],
            mass_block,
            True,
            include_eta=False,
        )
        I[start:stop] = moments.I
        U[start:stop] = moments.U
        mass[start:stop] = mass_block
        n_eff[start:stop] = 1.0 / np.square(normalized).sum(axis=1)
        edges += int(np.count_nonzero(weights))
    model_utils.require_statistics(mass, I, U)
    return I, U, mass, n_eff, edges
