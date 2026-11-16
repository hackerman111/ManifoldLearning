"""Явные экспериментальные варианты локализации и выбора центров."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AdaptiveCentersConfig:
    """P0/P2-отбор/P3 меняют estimator; P1 сохраняет носитель Epanechnikov."""

    lambda_floor: float = 0.0
    projection_search: bool = False
    projected_statistics: bool = False
    selection: str = "all"
    budget: int | None = None
    quadrature: bool = False
    coverage_weight: float = 1.0
    score_ridge: float = 1e-6
    logdet_ridge: float = 1.0

    def __post_init__(self) -> None:
        for name in ("projection_search", "projected_statistics", "quadrature"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in ("lambda_floor", "coverage_weight", "score_ridge", "logdet_ridge"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_floor > 1:
            raise ValueError("lambda_floor must lie in [0, 1]")
        if self.logdet_ridge == 0:
            raise ValueError("logdet_ridge must be positive")
        if self.selection not in {
            "all",
            "random",
            "score",
            "logdet",
            "logdet_coverage",
        }:
            raise ValueError("unknown center selection")
        if self.budget is not None:
            if isinstance(self.budget, bool) or not isinstance(
                self.budget, (int, np.integer)
            ):
                raise TypeError("budget must be an integer")
            if self.budget < 1:
                raise ValueError("budget must be positive")
        if self.selection in {"random", "score"} and self.budget is None:
            raise ValueError("random/score selection requires a fixed budget")
        if self.selection == "all" and self.budget is not None:
            raise ValueError("budget requires a center selection method")

    @property
    def active(self) -> bool:
        return bool(
            self.lambda_floor
            or self.projection_search
            or self.projected_statistics
            or self.selection != "all"
        )
