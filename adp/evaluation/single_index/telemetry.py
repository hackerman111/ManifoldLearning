from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class WeightTelemetry:
    sum_w: np.ndarray
    sum_w2: np.ndarray
    nonzero: np.ndarray
    min_weight: np.ndarray
    max_weight: np.ndarray

    @property
    def ess(self) -> np.ndarray:
        return np.divide(
            self.sum_w**2,
            self.sum_w2,
            out=np.zeros_like(self.sum_w, dtype=float),
            where=self.sum_w2 > 0.0,
        )


@dataclass(frozen=True, slots=True)
class LocalSystemDiagnostic:
    determinant: float
    lambda_min: float
    lambda_max: float
    condition: float
    rank: int
    residual: float
    regularization: float
    singular: bool


def summarize_weights(weights: np.ndarray) -> WeightTelemetry:
    values = np.asarray(weights)
    if values.ndim != 2:
        raise ValueError("weights must be a two-dimensional array")
    if values.shape[1] == 0:
        raise ValueError("weights must contain at least one observation")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    return WeightTelemetry(
        sum_w=np.asarray(values.sum(axis=1), dtype=float),
        sum_w2=np.asarray(np.square(values).sum(axis=1), dtype=float),
        nonzero=np.asarray(np.count_nonzero(values, axis=1), dtype=int),
        min_weight=np.asarray(values.min(axis=1), dtype=float),
        max_weight=np.asarray(values.max(axis=1), dtype=float),
    )


def diagnose_local_systems(
    *,
    S: np.ndarray,
    U: np.ndarray,
    imav: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    regularization: float,
) -> tuple[LocalSystemDiagnostic, ...]:
    s_values = np.asarray(S)
    u_values = np.asarray(U)
    responses = np.asarray(imav)
    beta_values = np.asarray(beta).reshape(-1)
    intercept_values = np.asarray(intercepts).reshape(-1)
    slope_values = np.asarray(slopes).reshape(-1)
    _validate_local_shapes(
        s_values,
        u_values,
        responses,
        beta_values,
        intercept_values,
        slope_values,
    )
    regularization_value = float(regularization)
    if not np.isfinite(regularization_value) or regularization_value < 0.0:
        raise ValueError("regularization must be finite and nonnegative")

    projected = u_values @ beta_values
    dtype = np.result_type(s_values.dtype, u_values.dtype, responses.dtype)
    eps = float(np.finfo(dtype).eps)
    diagnostics: list[LocalSystemDiagnostic] = []
    for center_index in range(s_values.shape[0]):
        s_row = s_values[center_index]
        u_row = projected[center_index]
        response = responses[center_index]
        system = np.array(
            [
                [np.dot(s_row, s_row), np.dot(s_row, u_row)],
                [np.dot(s_row, u_row), np.dot(u_row, u_row)],
            ],
            dtype=dtype,
        )
        eigenvalues = np.linalg.eigvalsh(system)
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
        threshold = 2.0 * eps * max(lambda_max, 1.0)
        rank = int(np.count_nonzero(eigenvalues > threshold))
        singular = bool(rank < 2 or lambda_min <= threshold)
        condition = (
            float("inf") if singular else float(lambda_max / lambda_min)
        )
        fitted = (
            intercept_values[center_index] * s_row
            + slope_values[center_index] * u_row
        )
        diagnostics.append(
            LocalSystemDiagnostic(
                determinant=float(np.linalg.det(system)),
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                condition=condition,
                rank=rank,
                residual=float(np.linalg.norm(response - fitted)),
                regularization=regularization_value,
                singular=singular,
            )
        )
    return tuple(diagnostics)


def encode_beta(beta: np.ndarray) -> str:
    values = np.asarray(beta)
    if values.ndim != 1:
        raise ValueError("beta must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("beta must contain only finite values")
    precision = 9 if values.dtype == np.dtype("float32") else 17
    return "|".join(format(float(value), f".{precision}g") for value in values)


def timing_remainder(total: float, *parts: float) -> float:
    total_value = float(total)
    part_values = tuple(float(part) for part in parts)
    if not np.isfinite(total_value) or total_value < 0.0:
        raise ValueError("total timing must be finite and nonnegative")
    if any(not np.isfinite(part) or part < 0.0 for part in part_values):
        raise ValueError("timing parts must be finite and nonnegative")
    return max(0.0, total_value - sum(part_values))


def _validate_local_shapes(
    S: np.ndarray,
    U: np.ndarray,
    imav: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
) -> None:
    if S.ndim != 2 or imav.shape != S.shape:
        raise ValueError("S and imav must have the same two-dimensional shape")
    if U.ndim != 3 or U.shape[:2] != S.shape:
        raise ValueError("U must have shape (centers, directions, dimension)")
    if beta.shape != (U.shape[2],):
        raise ValueError("beta dimension must match U")
    if intercepts.shape != (S.shape[0],) or slopes.shape != (S.shape[0],):
        raise ValueError("local coefficients must match the number of centers")
    if not all(
        np.all(np.isfinite(values))
        for values in (S, U, imav, beta, intercepts, slopes)
    ):
        raise ValueError("local diagnostics require finite inputs")


__all__ = [
    "LocalSystemDiagnostic",
    "WeightTelemetry",
    "diagnose_local_systems",
    "encode_beta",
    "summarize_weights",
    "timing_remainder",
]
