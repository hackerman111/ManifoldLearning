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
except Exception:
    tqdm = None


class ADP:
    """Фабрика Average Derivative Procedure."""

    variant: VariantName

    @classmethod
    def create(
        cls,
        variant: VariantName = "new",  # Единственный рабочий вариант.
        config: ADPConfig | None = None,  # Готовая конфигурация или None.
        **config_kwargs: Any,  # Точечные переопределения ADPConfig.
    ) -> "ADPBase":
        """Создает ADP-модель нужного варианта.

        Вход:
            variant: 'new' для случайных проекций.
            config: готовая конфигурация.
            config_kwargs: поля ADPConfig, если config не задан или надо переопределить.
        Выход:
            Экземпляр RandomProjectionADP.
        """

        from ..variants import RandomProjectionADP

        if config is None:
            config = ADPConfig(**config_kwargs)
        elif config_kwargs:
            config = replace(config, **config_kwargs)

        if variant == "new":
            return RandomProjectionADP(config)
        raise ValueError("variant должен быть только 'new'")


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

    def __init__(
        self,
        config: ADPConfig | None = None,  # Конфигурация модели или None.
    ) -> None:
        """Инициализирует модель ADP.

        Вход:
            config: настройки ADP.
        Выход:
            None; создает rng, backend и пустое состояние fit.
        """

        self.config = config or ADPConfig()
        if self.config.target_dim != 1:
            raise NotImplementedError("Сейчас реализован target_dim=1; multi-index оставлен следующим слоем.")
        self.rng = np.random.default_rng(self.config.random_state)
        self.backend = NumpyBackend(self.config.dtype)
        self.result_: ADPResult | None = None
        self.data_: tuple[np.ndarray, np.ndarray] | None = None
        self.centers_: np.ndarray | None = None
        self.directions_: np.ndarray | None = None
        self.neighbor_index_: NeighborIndex | None = None
        self.diagnostic_plots_: dict[str, Path] = {}
        self._pairwise_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def _compute_statistics(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        centers: np.ndarray,  # Матрица центров J x d.
        h: float,  # Текущий масштаб h.
        beta: np.ndarray,  # Текущее направление beta.
        directions: np.ndarray | None,  # Направления для new или None.
        anisotropy: float | None,  # rho для new или None.
    ) -> LocalStatistics:
        """Вычисляет локальные статистики конкретного варианта.

        Вход:
            X, y, centers, h, beta: данные текущего outer-шага.
            directions: случайные направления для new-варианта.
            anisotropy: rho из new-варианта.
        Выход:
            LocalStatistics, определенные подклассом.
        """

        raise NotImplementedError

    def _solve_local_coefficients(
        self,
        stats: LocalStatistics,  # Локальные статистики варианта.
        beta: np.ndarray,  # Текущее направление beta.
    ) -> tuple[np.ndarray, np.ndarray]:
        """Решает локальные intercepts и slopes.

        Вход:
            stats: локальные статистики.
            beta: текущее направление.
        Выход:
            Кортеж intercepts, slopes.
        """

        raise NotImplementedError

    def _solve_beta(
        self,
        stats: LocalStatistics,  # Локальные статистики варианта.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # beta предыдущего внешнего шага.
        lambda_penalty: float,  # Сила регуляризации.
        x0: np.ndarray | None = None,  # Старт CG или None.
    ) -> np.ndarray:
        """Решает глобальный шаг по beta.

        Вход:
            stats: локальные статистики.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
            x0: warm-start для численного solve.
        Выход:
            Новый ненормированный beta.
        """

        raise NotImplementedError

    def _objective(
        self,
        stats: LocalStatistics,  # Локальные статистики варианта.
        beta: np.ndarray,  # Текущее направление beta.
        intercepts: np.ndarray,  # Локальные свободные члены.
        slopes: np.ndarray,  # Локальные наклоны.
        prior: np.ndarray,  # Направление регуляризации.
        lambda_penalty: float,  # Сила регуляризации.
    ) -> float:
        """Считает целевую функцию варианта ADP.

        Вход:
            stats: локальные статистики.
            beta: текущее направление.
            intercepts: локальные свободные члены.
            slopes: локальные наклоны.
            prior: направление регуляризации.
            lambda_penalty: сила регуляризации.
        Выход:
            Значение objective.
        """

        raise NotImplementedError
