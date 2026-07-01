# Добавить больше переменных для более тщательного тестирования в подборе весов

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..backends.neighbors import NeighborIndex
from ..common.utils import average_kernel_weight, pairwise_norm2, pairwise_projection2


class BandwidthMixin:
    """Методы выбора локальных масштабов ADP."""

    def _select_isotropic_bandwidth(
        self, X: np.ndarray, centers: np.ndarray, index: NeighborIndex | None = None
    ) -> float:
        """Подбирает начальную isotropic bandwidth.

        Вход:
            X: матрица наблюдений n x d.
            centers: матрица центров J x d.
            index: индекс соседей для начальной верхней оценки.
        Выход:
            Положительное значение h.
        """

        diff_norm2 = pairwise_norm2(X, centers)
        high_hint = None
        if index is not None:
            k = min(max(1, int(math.ceil(self.config.min_neighbors))), X.shape[0])
            kth = index.kth_distances(centers, k)
            if kth is not None and np.all(np.isfinite(kth)):
                high_hint = float(np.nanmedian(kth))

        def avg_for(h: float) -> float:
            """Считает среднюю массу для isotropic h.

            Вход:
                h: кандидат bandwidth.
            Выход:
                Средняя локальная масса ядра.
            """

            return average_kernel_weight(diff_norm2 / (h * h), self.config.kernel)

        return self._binary_search_scale(avg_for, high_hint)

    def _select_new_anisotropy(
        self, X: np.ndarray, centers: np.ndarray, h: float, beta: np.ndarray
    ) -> float:
        """Подбирает rho для anisotropic new-варианта.

        Вход:
            X: матрица наблюдений n x d.
            centers: матрица центров J x d.
            h: текущая bandwidth.
            beta: текущее направление EDR.
        Выход:
            Значение rho в диапазоне [0, 1].
        """

        norm2 = pairwise_norm2(X, centers)
        proj2 = pairwise_projection2(X, centers, beta)

        def avg_for(rho: float) -> float:
            """Считает среднюю массу для фиксированного rho.

            Вход:
                rho: кандидат anisotropy.
            Выход:
                Средняя локальная масса ядра.
            """

            q = (rho * rho * norm2 + proj2) / (h * h)
            return average_kernel_weight(q, self.config.kernel)

        if avg_for(1.0) >= self.config.min_neighbors:
            return 1.0
        if avg_for(0.0) < self.config.min_neighbors:
            return 0.0
        low, high = 0.0, 1.0
        for _ in range(50):
            mid = (low + high) / 2.0
            if avg_for(mid) >= self.config.min_neighbors:
                low = mid
            else:
                high = mid
        return float(low)

    def _select_old_bandwidth(
        self, X: np.ndarray, centers: np.ndarray, beta: np.ndarray, b_value: float
    ) -> float:
        """Подбирает h при фиксированном b для old-варианта.

        Вход:
            X: матрица наблюдений n x d.
            centers: матрица центров J x d.
            beta: текущее направление EDR.
            b_value: продольная bandwidth.
        Выход:
            Положительное значение h.
        """

        norm2 = pairwise_norm2(X, centers)
        proj2 = pairwise_projection2(X, centers, beta)

        def avg_for(h: float) -> float:
            """Считает среднюю массу для фиксированного h.

            Вход:
                h: кандидат bandwidth.
            Выход:
                Средняя локальная масса ядра.
            """

            q = norm2 / (h * h) + proj2 / (b_value * b_value)
            return average_kernel_weight(q, self.config.kernel)

        return self._binary_search_scale(avg_for, b_value)

    def _binary_search_scale(
        self, avg_fn: Callable[[float], float], high_hint: float | None = None
    ) -> float:
        """Ищет минимальный масштаб с достаточной локальной массой.

        Вход:
            avg_fn: функция средней массы от масштаба.
            high_hint: начальная верхняя оценка масштаба.
        Выход:
            Найденный положительный масштаб.
        """

        target = float(self.config.min_neighbors)
        low = np.finfo(float).eps
        high = max(float(high_hint or 1.0), low * 2.0)
        for _ in range(80):
            if avg_fn(high) >= target:
                break
            high *= 2.0
        for _ in range(70):
            mid = (low + high) / 2.0
            if avg_fn(mid) >= target:
                high = mid
            else:
                low = mid
        return float(high)
