"""P2/P3: проекционные моменты и экспериментальный отбор до сборки U."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csr_matrix


@dataclass(frozen=True, slots=True)
class ProjectedStatistics:
    covariance: np.ndarray
    cross: np.ndarray
    slopes: np.ndarray
    mass: np.ndarray
    n_eff: np.ndarray
    response_mean: np.ndarray
    feasible: np.ndarray
    conditioning: np.ndarray


def projected_statistics(
    Z: np.ndarray,
    Y: np.ndarray,
    weights: csr_matrix,
    *,
    ridge: float,
) -> ProjectedStatistics:
    """Точные Cov_w(Z), Cov_w(Z,Y); ridge — только surrogate-наклон.

    Память O(J*m²+n*m), scratch O(k*m); плохой rank явно исключён из feasible.
    """
    J, n = len(weights.indptr) - 1, Z.shape[0]
    m = Z.shape[1]
    if (
        weights.shape != (J, n)
        or Y.shape != (n,)
        or not np.isfinite(ridge)
        or ridge < 0
    ):
        raise ValueError("invalid projected statistics input")
    covariance = np.zeros((J, m, m))
    cross, slopes = np.zeros((J, m)), np.zeros((J, m))
    mass, neff, mean_y = np.zeros(J), np.zeros(J), np.zeros(J)
    feasible, conditioning = np.zeros(J, dtype=bool), np.zeros(J)
    for j in range(J):
        section = slice(weights.indptr[j], weights.indptr[j + 1])
        idx, w = weights.indices[section], weights.data[section]
        mass[j] = w.sum()
        if mass[j] <= 0:
            continue
        a = w / mass[j]
        centered = Z[idx] - a @ Z[idx]
        mean_y[j] = a @ Y[idx]
        neff[j] = 1.0 / (a @ a)
        covariance[j] = (centered.T * a) @ centered
        cross[j] = centered.T @ (a * (Y[idx] - mean_y[j]))
        values, vectors = np.linalg.eigh(covariance[j])
        threshold = np.finfo(float).eps * max(m, len(idx)) * max(values[-1], 0.0)
        feasible[j] = neff[j] >= m + 1 and values[0] > threshold
        if not feasible[j]:
            continue
        slopes[j] = vectors @ ((vectors.T @ cross[j]) / (values + ridge))
        conditioning[j] = values[0] / (values[0] + 0.01 * values[-1])
    return ProjectedStatistics(
        covariance, cross, slopes, mass, neff, mean_y, feasible, conditioning
    )


def information_vectors(
    moments: ProjectedStatistics,
    Y: np.ndarray,
    target: int,
) -> np.ndarray:
    """Фиксированные внутри greedy u; kappa0=.01, varsigma=std(Y)."""
    count = int(moments.feasible.sum())
    scale = float(np.std(Y))
    if scale == 0 or count == 0:
        return np.zeros_like(moments.slopes)
    confidence = np.sqrt(moments.n_eff / target * moments.conditioning)
    return (
        np.sqrt(len(moments.mass) / count)
        / scale
        * confidence[:, None]
        * moments.slopes
    )


def coverage(weights: csr_matrix, selected: np.ndarray) -> float:
    pool = np.asarray(weights.sum(axis=0)).ravel()
    covered = np.asarray(weights[selected].sum(axis=0)).ravel() > 0
    total = float(pool.sum())
    return float(pool[covered].sum() / total) if total else 0.0


def quadrature_mass(weights: csr_matrix, selected: np.ndarray) -> np.ndarray:
    """Ненормированная PoU: sum M_selected = масса покрытых наблюдений."""
    subset = weights[selected]
    pool = np.asarray(weights.sum(axis=0)).ravel()
    covered_mass = np.asarray(subset.sum(axis=0)).ravel()
    ratio = np.divide(
        pool, covered_mass, out=np.zeros_like(pool), where=covered_mass > 0
    )
    return np.asarray(subset @ ratio).ravel()


def greedy_select(
    weights: csr_matrix,
    vectors: np.ndarray,
    feasible: np.ndarray,
    *,
    budget: int | None,
    minimum: int,
    coverage_weight: float,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    """Lazy greedy для нормированного logdet(I+sum uuT/ridge)+nu*coverage.

    Фиксированные u и pool masses сохраняют убывание marginal gains.
    Cholesky solve вместо явного inverse; O(m²+J+n) кроме sparse weights.
    """
    m = vectors.shape[1]
    pool = np.asarray(weights.sum(axis=0)).ravel()
    shape = cast(tuple[int, int], weights.shape)
    active = np.zeros(shape[1], dtype=bool)
    for j in feasible:
        active[weights.indices[weights.indptr[j] : weights.indptr[j + 1]]] = True
    total = float(pool[active].sum())
    V = vectors[feasible]
    log_reference = float(np.linalg.slogdet(np.eye(m) + V.T @ V / ridge)[1])
    matrix = ridge * np.eye(m)
    chol = np.linalg.cholesky(matrix)
    covered = np.zeros(shape[1], dtype=bool)

    def gain(j: int) -> tuple[float, float]:
        solved = solve_triangular(chol, vectors[j], lower=True, check_finite=False)
        info = float(np.log1p(solved @ solved))
        idx = weights.indices[weights.indptr[j] : weights.indptr[j + 1]]
        mass = float(pool[idx[~covered[idx]]].sum()) / total if total else 0.0
        return (
            info / log_reference if log_reference > 0 else 0.0
        ) + coverage_weight * mass, info

    heap = [(-gain(int(j))[0], int(j)) for j in feasible]
    heapq.heapify(heap)
    selected = []
    information = 0.0
    limit = len(feasible) if budget is None else min(budget, len(feasible))
    while heap and len(selected) < limit:
        _, j = heapq.heappop(heap)
        marginal, info = gain(j)
        if heap and (
            marginal < -heap[0][0] - 1e-14
            or (abs(marginal + heap[0][0]) <= 1e-14 and j > heap[0][1])
        ):
            heapq.heappush(heap, (-marginal, j))
            continue
        selected.append(j)
        matrix += np.outer(vectors[j], vectors[j])
        chol = np.linalg.cholesky(matrix)
        information += info
        covered[weights.indices[weights.indptr[j] : weights.indptr[j + 1]]] = True
        info_fraction = information / log_reference if log_reference > 0 else 1.0
        mass_fraction = float(pool[covered].sum() / total) if total else 0.0
        if (
            budget is None
            and len(selected) >= minimum
            and info_fraction >= 0.965
            and mass_fraction >= 0.95
        ):
            break
    chosen = np.asarray(selected, dtype=np.intp)
    return chosen, {
        "information_fraction": information / log_reference
        if log_reference > 0
        else 1.0,
        "mass_coverage": coverage(weights, chosen),
        "active_mass_coverage": float(pool[covered].sum() / total) if total else 0.0,
        "logdet_reference": log_reference,
        "thresholds_met": bool(
            len(chosen) >= minimum
            and (log_reference == 0 or information >= 0.965 * log_reference)
            and (float(pool[covered].sum()) >= 0.95 * total)
        ),
    }
