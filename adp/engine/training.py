from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..common.types import ADPResult


class TrainingMixin:
    """Публичный интерфейс обучения поверх явного `ADPAlgorithm`."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        centers: np.ndarray | None = None,
        beta0: np.ndarray | None = None,
        directions: np.ndarray | None = None,
    ) -> ADPResult:
        """Обучает модель через настроенную композицию этапов."""

        return self.algorithm.fit(
            X,
            y,
            centers=centers,
            beta0=beta0,
            directions=directions,
        )

    def _make_progress_bar(self, outer_iter: range) -> Any:
        """Создает tqdm progress bar при включенном выводе."""

        if not self.config.show_progress:
            return None
        from .base import tqdm

        progress_factory = getattr(sys.modules.get("adp.core"), "tqdm", tqdm)
        if progress_factory is None:
            return None
        return progress_factory(
            outer_iter,
            desc=f"ADP-{self.variant} {self.backend.name}",
            leave=False,
        )

    def _prepare_outer_step(
        self,
        outer: int,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        d: int,
    ) -> tuple[float | None, float, np.ndarray | None]:
        """Совместимый адаптер подготовки внешнего шага."""

        if outer == 0:
            return None, h, directions
        h = max(h / self.config.bandwidth_decay, np.finfo(float).eps)
        anisotropy = self.algorithm.components[
            "bandwidth_selector"
        ].select_anisotropy(X, centers, h, beta)
        if self.config.renew_directions:
            directions = self.algorithm.components["direction_sampler"].prepare(
                centers,
                d,
                None,
                beta=beta,
                anisotropy=anisotropy,
            )
        return anisotropy, h, directions
