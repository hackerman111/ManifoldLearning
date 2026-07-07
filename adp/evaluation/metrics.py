from __future__ import annotations

import numpy as np


def direction_metrics(
    beta_hat: np.ndarray,  # Оцененное направление.
    beta_true: np.ndarray,  # Истинное направление.
) -> dict[str, float]:
    """Считает метрики близости двух направлений.

    Вход:
        beta_hat: оцененный beta.
        beta_true: истинный beta.
    Выход:
        Словарь cosine, cosine_abs, angle_deg и signed_l2.
    """

    if not np.all(np.isfinite(beta_hat)):
        return {
            "cosine": np.nan,
            "cosine_abs": np.nan,
            "angle_deg": np.nan,
            "signed_l2": np.nan,
        }
    estimated = unit_vector(beta_hat)
    expected = unit_vector(beta_true)
    cosine = float(np.clip(expected @ estimated, -1.0, 1.0))
    cosine_abs = abs(cosine)
    return {
        "cosine": cosine,
        "cosine_abs": cosine_abs,
        "angle_deg": float(np.degrees(np.arccos(np.clip(cosine_abs, -1.0, 1.0)))),
        "signed_l2": float(
            min(
                np.linalg.norm(estimated - expected),
                np.linalg.norm(estimated + expected),
            )
        ),
    }


def unit_vector(
    value: np.ndarray,  # Исходное направление.
) -> np.ndarray:
    """Нормирует направление, возвращая NaN-вектор для нулевой нормы.

    Вход:
        value: исходный вектор.
    Выход:
        Единичный вектор или NaN-вектор.
    """

    vector = np.asarray(value, dtype=float).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm < np.finfo(float).eps:
        return np.full_like(vector, np.nan)
    return vector / norm
