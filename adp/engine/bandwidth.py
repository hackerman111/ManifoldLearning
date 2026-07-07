from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..backends.neighbors import NeighborIndex
from ..common.utils import average_kernel_weight, pairwise_norm2, pairwise_projection2


class BandwidthMixin:
    """Методы выбора локальных масштабов ADP."""

    def _select_isotropic_bandwidth(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
        index: NeighborIndex | None = None,  # Индекс соседей для верхней оценки.
    ) -> float:
        """Подбирает начальную isotropic bandwidth h.

        Вход:
            X: матрица наблюдений.
            centers: матрица центров.
            index: faiss/sklearn индекс или None.
        Выход:
            Положительный масштаб h.
        """

        # В TeX h выбирается как минимальный масштаб, при котором почти каждая
        # локальная окрестность содержит достаточно массы ядра.
        diff_norm2 = pairwise_norm2(X, centers)
        high_hint = None
        if index is not None:
            # Индекс соседей дает только стартовую верхнюю оценку для бинарного
            # поиска, чтобы не раздувать h от единицы слишком долго.
            k = min(max(1, int(math.ceil(self.config.min_neighbors))), X.shape[0])
            kth = index.kth_distances(centers, k)
            if kth is not None and np.all(np.isfinite(kth)):
                high_hint = float(np.nanmedian(kth))

        def avg_for(
            h: float,  # Кандидат масштаба h.
        ) -> float:
            """Считает среднюю локальную массу для isotropic h."""

            return average_kernel_weight(diff_norm2 / (h * h), self.config.kernel)

        return self._binary_search_scale(avg_for, high_hint)

    def _select_new_anisotropy(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
        h: float,  # Текущий масштаб h.
        beta: np.ndarray,  # Текущее направление beta.
    ) -> float:
        """Подбирает rho для anisotropic new-варианта.

        Вход:
            X: матрица наблюдений.
            centers: матрица центров.
            h: текущий масштаб.
            beta: текущее EDR-направление.
        Выход:
            rho в диапазоне [0, 1].
        """

        # manifold_new.tex использует q = (rho^2 ||dx||^2 + <dx,beta>^2) / h^2.
        # Чем меньше rho, тем слабее штрафуются направления, ортогональные beta.
        norm2 = pairwise_norm2(X, centers)
        proj2 = pairwise_projection2(X, centers, beta)

        def avg_for(
            rho: float,  # Кандидат rho.
        ) -> float:
            """Считает среднюю массу при фиксированном rho."""

            q = (rho * rho * norm2 + proj2) / (h * h)
            return average_kernel_weight(q, self.config.kernel)

        # Ищем наименьшее rho, которое еще сохраняет нужную массу.
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
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
        beta: np.ndarray,  # Текущее направление beta.
        b_value: float,  # Продольный масштаб b для old.
    ) -> float:
        """Подбирает h при фиксированном b для old-варианта.

        Вход:
            X: матрица наблюдений.
            centers: матрица центров.
            beta: текущее EDR-направление.
            b_value: bandwidth вдоль beta.
        Выход:
            Положительное значение h.
        """

        # manifold_old.tex задает анизотропные веса через два масштаба:
        # h для полной нормы и b для проекции на текущее beta.
        norm2 = pairwise_norm2(X, centers)
        proj2 = pairwise_projection2(X, centers, beta)

        def avg_for(
            h: float,  # Кандидат масштаба h.
        ) -> float:
            """Считает среднюю массу при фиксированном h."""

            q = norm2 / (h * h) + proj2 / (b_value * b_value)
            return average_kernel_weight(q, self.config.kernel)

        return self._binary_search_scale(avg_for, b_value)

    def _binary_search_scale(
        self,
        avg_fn: Callable[[float], float],  # Средняя масса от масштаба.
        high_hint: float | None = None,  # Начальная верхняя оценка.
    ) -> float:
        """Ищет минимальный масштаб с достаточной локальной массой.

        Вход:
            avg_fn: функция средней массы.
            high_hint: начальная верхняя оценка или None.
        Выход:
            Положительный масштаб.
        """

        # Сначала расширяем правую границу до выполнимого масштаба, затем
        # бинарным поиском возвращаемся к минимальному h/rho/b.
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
