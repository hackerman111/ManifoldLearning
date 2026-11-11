"""Независимый reference для фиксированной multi-index задачи."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Certificate:
    objective: float
    riemannian_gradient: float
    horizontal_gradient: float
    local_gradient: float
    orthogonality: float
    rank_loss: int
    min_relative_singular: float


def evaluate(
    B: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray
) -> Certificate:
    """Пересчитать local least squares и текущие нормировки stationarity.

    B=(m,d), U=(J,P,d), I=(J,P), mass=(J,). Этот медленный путь
    независим от solver и предназначен для проверки frozen задач.
    """
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if (
        B.ndim != 2
        or U.ndim != 3
        or I.shape != U.shape[:2]
        or B.shape[1] != U.shape[2]
        or mass.shape != (U.shape[0],)
        or B.shape[0] > min(U.shape[1], U.shape[2])
    ):
        raise ValueError("incompatible multi-index shapes")
    if not all(np.all(np.isfinite(x)) for x in (B, U, I, mass)):
        raise ValueError("all arrays must be finite")
    if np.any(mass <= 0):
        raise ValueError("mass must be positive")

    J, _, _ = U.shape
    m = B.shape[0]
    projected = U @ B.T  # (J,P,m)
    coefficients = np.empty((J, m))
    ranks = np.empty(J, dtype=int)
    relative_singular = np.empty(J)
    for j in range(J):
        coefficients[j], _, ranks[j], singular = np.linalg.lstsq(
            projected[j], I[j], rcond=None
        )
        relative_singular[j] = singular[-1] / singular[0] if singular[0] else 0.0

    residual = I - np.einsum("jpm,jm->jp", projected, coefficients, optimize=True)
    objective = 0.5 * float(np.einsum("j,jp,jp->", mass, residual, residual))
    transposed_residual = np.einsum("jpd,jp->jd", U, residual, optimize=True)
    gradient = -np.einsum(
        "j,jm,jd->md", mass, coefficients, transposed_residual, optimize=True
    )
    product = gradient @ B.T
    riemannian = gradient - (0.5 * (product + product.T)) @ B
    horizontal = gradient - product @ B
    local = -mass[:, None] * np.einsum("jpm,jp->jm", projected, residual, optimize=True)
    local_scale = np.maximum(
        1.0,
        np.linalg.norm(projected, axis=(1, 2)) * np.linalg.norm(I, axis=1),
    )
    local_score = float(np.max(np.linalg.norm(local, axis=1) / local_scale))
    coefficient_norm2 = np.einsum("jm,jm->j", coefficients, coefficients, optimize=True)
    U_norm2 = np.einsum("jpd,jpd->j", U, U, optimize=True)
    operator_norm2 = float(np.sum(mass * coefficient_norm2 * U_norm2))
    scale = max(1.0, math.sqrt(operator_norm2 * 2.0 * objective))
    return Certificate(
        objective=objective,
        riemannian_gradient=float(np.linalg.norm(riemannian) / scale),
        horizontal_gradient=float(np.linalg.norm(horizontal) / scale),
        local_gradient=local_score,
        orthogonality=float(np.linalg.norm(B @ B.T - np.eye(m), ord="fro")),
        rank_loss=int(np.count_nonzero(ranks < m)),
        min_relative_singular=float(np.min(relative_singular)),
    )
