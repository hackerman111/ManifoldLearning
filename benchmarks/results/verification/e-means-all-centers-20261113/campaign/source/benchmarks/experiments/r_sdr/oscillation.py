"""Exact Gaussian reference for the explicit oscillatory frequency sweep."""

from __future__ import annotations

import numpy as np


def signal_gradient(z: np.ndarray, frequency: float) -> np.ndarray:
    a, b, c = frequency * np.array([1.0, 1.25, 1.5])
    return np.column_stack(
        (
            a * np.cos(a * z[:, 0]) * np.cos(b * z[:, 1]),
            -b * np.sin(a * z[:, 0]) * np.sin(b * z[:, 1]),
            c * np.cos(c * z[:, 2]),
        )
    )


def population_moments(frequency: float) -> dict:
    """Moments for population-variance-normalized f, Z~N(0,I_3), not sample std."""
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")
    a, b, c = frequency * np.array([1.0, 1.25, 1.5])
    cos2 = (1 + np.exp(-2 * np.array([a, b, c]) ** 2)) / 2
    sin2 = -np.expm1(-2 * np.array([a, b, c]) ** 2) / 2
    variance = sin2[0] * cos2[1] + sin2[2]
    mean = np.array([a * np.exp(-(a * a + b * b) / 2), 0.0, c * np.exp(-c * c / 2)])
    gram = np.diag(
        [a * a * cos2[0] * cos2[1], b * b * sin2[0] * sin2[1], c * c * cos2[2]]
    )
    gram[0, 2] = gram[2, 0] = mean[0] * mean[2]
    return {
        "signal_variance": float(variance),
        "gradient_mean": mean / np.sqrt(variance),
        "gradient_gram": gram / variance,
    }
