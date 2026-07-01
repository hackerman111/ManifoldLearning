from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.neighbors import NeighborIndex
from ..backends.numpy_backend import NumpyBackend
from ..common.types import ADPConfig, ADPResult, LocalStatistics, VariantName
from .bandwidth import BandwidthMixin
from .data import DataPreparationMixin
from .diagnostics import DiagnosticsMixin
from .solver import SolverMixin
from .training import TrainingMixin

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm является удобством, а не ядром.
    tqdm = None


class ADP:
    """Фабрика Average Derivative Procedure."""

    variant: VariantName

    @classmethod
    def create(
        cls,
        variant: VariantName = "new",
        config: ADPConfig | None = None,
        **config_kwargs: Any,
    ) -> "ADPBase":
        """Создаёт ADP-модель нужного варианта.

        Вход:
            variant: имя варианта, 'new' или 'old'.
            config: готовая конфигурация или None.
            config_kwargs: точечные переопределения ADPConfig.
        Выход:
            Экземпляр RandomProjectionADP или FullMomentADP.
        """

        from ..variants import FullMomentADP, RandomProjectionADP

        if config is None:
            config = ADPConfig(**config_kwargs)
        elif config_kwargs:
            config = replace(config, **config_kwargs)

        if variant == "new":
            return RandomProjectionADP(config)
        if variant == "old":
            return FullMomentADP(config)
        raise ValueError("variant должен быть 'new' или 'old'")


class ADPBase(
    DiagnosticsMixin,
    TrainingMixin,
    DataPreparationMixin,
    BandwidthMixin,
    SolverMixin,
    ADP,
):
    """Общая часть ADP без формул конкретного варианта."""

    variant: VariantName = "new"

    def __init__(self, config: ADPConfig | None = None):
        """Инициализирует модель ADP.

        Вход:
            config: конфигурация модели или None для ADPConfig().
        Выход:
            None; создаёт rng, backend и пустое состояние fit.
        """

        self.config = config or ADPConfig()
        if self.config.target_dim != 1:
            raise NotImplementedError(
                "Сейчас реализован target_dim=1; multi-index оставлен следующим слоем."
            )
        self.rng = np.random.default_rng(self.config.random_state)
        self.backend = NumpyBackend(self.config.dtype)
        self.result_: ADPResult | None = None
        self.data_: tuple[np.ndarray, np.ndarray] | None = None
        self.centers_: np.ndarray | None = None
        self.directions_: np.ndarray | None = None
        self.neighbor_index_: NeighborIndex | None = None
        self.diagnostic_plots_: dict[str, Path] = {}

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
        b_value: float | None,
    ) -> LocalStatistics:
        """Вычисляет локальные статистики конкретного варианта.

        Вход:
            X: матрица наблюдений n x d.
            y: вектор ответов длины n.
            centers: матрица центров J x d.
            h: текущая bandwidth.
            beta: текущее направление EDR.
            directions: направления для new-варианта.
            anisotropy: значение rho или None.
            b_value: значение b для old-варианта или None.
        Выход:
            LocalStatistics, определённые подклассом.
        """

        raise NotImplementedError

    def _solve_local_coefficients(
        self, stats: LocalStatistics, beta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные intercepts и slopes.

        Вход:
            stats: локальные статистики варианта.
            beta: текущее направление EDR.
        Выход:
            Кортеж intercepts и slopes.
        """

        raise NotImplementedError

    def _solve_beta(
        self,
        stats: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> np.ndarray:
        """Решает глобальный шаг по beta.

        Вход:
            stats: локальные статистики варианта.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации к prior.
        Выход:
            Новый ненормированный вектор beta.
        """

        raise NotImplementedError

    def _objective(
        self,
        stats: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
    ) -> float:
        """Считает целевую функцию варианта ADP.

        Вход:
            stats: локальные статистики варианта.
            beta: текущее направление EDR.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: beta предыдущего outer-шага.
            lambda_penalty: сила регуляризации к prior.
        Выход:
            Значение objective.
        """

        raise NotImplementedError
