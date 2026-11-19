"""Конфигурация opt-in multi-index варианта LOO1-WLS-ADP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_real(name: str, value: object) -> float:
    """Проверить вещественный конечный скаляр; отделить TypeError от ValueError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _require_positive_int(name: str, value: object) -> int:
    """Проверить положительное целое число на границе конфигурации."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be positive")
    return integer


@dataclass(frozen=True, slots=True)
class LOO1WLSConfig:
    """Параметры LOO1-WLS estimator, сублинейного спектра и Grassmann solver."""

    spectrum_exponent: float = 0.5
    ridge: float = 1e-4
    leverage_cap: float = 0.85
    smooth_cap_eps: float = 0.0
    preconditioner_reg: float = 1e-3
    min_neighbors: int | None = None
    center_chunk: int = 512
    stream_distances: bool | None = None
    stream_distance_threshold: int = 4_000_000
    max_steps: int = 25
    tol: float = 1e-6
    armijo_c1: float = 1e-4
    max_line_search: int = 14

    def __post_init__(self) -> None:
        """Проверить типы и математические границы параметров LOO1-WLS."""
        spectrum_exponent = _require_real("spectrum_exponent", self.spectrum_exponent)
        if spectrum_exponent < 0.0:
            raise ValueError("spectrum_exponent must be nonnegative")

        ridge = _require_real("ridge", self.ridge)
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")

        leverage_cap = _require_real("leverage_cap", self.leverage_cap)
        if not 0.0 < leverage_cap < 1.0:
            raise ValueError("leverage_cap must lie strictly in (0, 1)")

        smooth_cap_eps = _require_real("smooth_cap_eps", self.smooth_cap_eps)
        if smooth_cap_eps < 0.0:
            raise ValueError("smooth_cap_eps must be nonnegative")

        preconditioner_reg = _require_real(
            "preconditioner_reg", self.preconditioner_reg
        )
        if preconditioner_reg <= 0.0:
            raise ValueError("preconditioner_reg must be positive")

        if self.min_neighbors is not None:
            _require_positive_int("min_neighbors", self.min_neighbors)
        _require_positive_int("center_chunk", self.center_chunk)

        if self.stream_distances is not None and not isinstance(
            self.stream_distances, (bool, np.bool_)
        ):
            raise TypeError("stream_distances must be boolean or None")

        _require_positive_int(
            "stream_distance_threshold", self.stream_distance_threshold
        )
        _require_positive_int("max_steps", self.max_steps)

        tol = _require_real("tol", self.tol)
        if tol <= 0.0:
            raise ValueError("tol must be positive")

        armijo_c1 = _require_real("armijo_c1", self.armijo_c1)
        if not 0.0 < armijo_c1 < 1.0:
            raise ValueError("armijo_c1 must lie strictly in (0, 1)")

        _require_positive_int("max_line_search", self.max_line_search)
