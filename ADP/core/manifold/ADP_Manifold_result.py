from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ADP_Manifold_result:
    centers: np.ndarray
    projectors: np.ndarray
    eigenvalues: np.ndarray
    gradients: np.ndarray
    center_values: np.ndarray
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


__all__ = ["ADP_Manifold_result"]
