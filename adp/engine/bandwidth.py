from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..backends.neighbors import NeighborIndex


class BandwidthMixin:
    """Методы выбора локальных масштабов ADP."""

    def _clear_pairwise_cache(
        self,
    ) -> None:
        """Очищает cache матриц расстояний для одного вызова fit."""

        self._pairwise_cache = {}

    def _cached_pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
    ) -> np.ndarray:
        """Возвращает ||X_i - c_j||^2 с cache на текущий fit."""

        cache = getattr(self, "_pairwise_cache", None)
        if cache is None:
            cache = {}
            self._pairwise_cache = cache
        key = ("norm2", id(X), X.shape, id(centers), centers.shape)
        if key not in cache:
            cache[key] = self.backend.pairwise_norm2(X, centers)
        return cache[key]

    def _cached_pairwise_projection2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
        beta: np.ndarray,  # Направление beta.
    ) -> np.ndarray:
        """Возвращает <X_i - c_j, beta>^2 с cache на текущий fit."""

        cache = getattr(self, "_pairwise_cache", None)
        if cache is None:
            cache = {}
            self._pairwise_cache = cache
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        key = ("proj2", id(X), X.shape, id(centers), centers.shape)
        beta_key = ("proj2_beta", id(X), X.shape, id(centers), centers.shape)
        cached_beta = cache.get(beta_key)
        if cached_beta is None or not np.array_equal(cached_beta, beta_arr):
            cache[key] = self.backend.pairwise_projection2(X, centers, beta_arr)
            cache[beta_key] = beta_arr.copy()
        return cache[key]

    def _local_mass_score(
        self,
        q: np.ndarray,  # Матрица квадратичной формы J x n.
    ) -> float:
        """Возвращает выбранную статистику локальной массы."""

        quantile = (
            self.config.local_mass_quantile
            if self.config.local_mass_mode == "quantile"
            else None
        )
        return self.backend.local_mass_score(q, self.config.kernel, quantile=quantile)

    def _select_isotropic_bandwidth_default(
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
        diff_norm2 = self._cached_pairwise_norm2(X, centers)
        high_hint = None
        if index is not None:
            # Индекс соседей дает только стартовую верхнюю оценку для бинарного
            # поиска, чтобы не раздувать h от единицы слишком долго.
            k = min(max(1, int(math.ceil(self.config.min_neighbors))), X.shape[0])
            kth = index.kth_distances(centers, k)
            if kth is not None and np.all(np.isfinite(kth)):
                hint_quantile = (
                    1.0 - self.config.local_mass_quantile
                    if self.config.local_mass_mode == "quantile"
                    else 0.5
                )
                hint_quantile = min(1.0, max(0.0, hint_quantile))
                high_hint = float(np.nanquantile(kth, hint_quantile))

        def score_for(
            h: float,  # Кандидат масштаба h.
        ) -> float:
            """Считает выбранную статистику локальной массы для isotropic h."""

            return self._local_mass_score(diff_norm2 / (h * h))

        return self._binary_search_scale(score_for, high_hint)

    def _select_isotropic_bandwidth(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        index: NeighborIndex | None = None,
    ) -> float:
        """Совместимый адаптер к выбранному bandwidth selector."""

        algorithm = getattr(self, "algorithm", None)
        if algorithm is None:
            return self._select_isotropic_bandwidth_default(X, centers, index)
        return algorithm.components["bandwidth_selector"].select_initial(
            X, centers, index
        )

    def _select_new_anisotropy_default(
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
        norm2 = self._cached_pairwise_norm2(X, centers)
        proj2 = self._cached_pairwise_projection2(X, centers, beta)

        def score_for(
            rho: float,  # Кандидат rho.
        ) -> float:
            """Считает выбранную статистику массы при фиксированном rho."""

            q = (rho * rho * norm2 + proj2) / (h * h)
            return self._local_mass_score(q)

        # Ищем максимальное rho, которое еще сохраняет нужную локальную массу.
        if score_for(1.0) >= self.config.min_neighbors:
            return 1.0
        if score_for(0.0) < self.config.min_neighbors:
            return 0.0
        low, high = 0.0, 1.0
        for _ in range(self.config.anisotropy_search_steps):
            mid = (low + high) / 2.0
            if score_for(mid) >= self.config.min_neighbors:
                low = mid
            else:
                high = mid
            if high - low <= 1e-3:
                break
        return float(low)

    def _select_new_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
    ) -> float:
        """Совместимый адаптер к выбранному bandwidth selector."""

        algorithm = getattr(self, "algorithm", None)
        if algorithm is None:
            return self._select_new_anisotropy_default(X, centers, h, beta)
        return algorithm.components["bandwidth_selector"].select_anisotropy(
            X, centers, h, beta
        )

    def _binary_search_scale(
        self,
        avg_fn: Callable[[float], float],  # Локальная масса от масштаба.
        high_hint: float | None = None,  # Начальная верхняя оценка.
    ) -> float:
        """Ищет минимальный масштаб с достаточной локальной массой.

        Вход:
            avg_fn: функция локальной массы.
            high_hint: начальная верхняя оценка или None.
        Выход:
            Положительный масштаб.
        """

        # Сначала расширяем правую границу до выполнимого масштаба, затем
        # бинарным поиском возвращаемся к минимальному h/rho/b.
        target = float(self.config.min_neighbors)
        low = np.finfo(float).eps
        high = max(float(high_hint or 1.0), low * 2.0)
        for _ in range(self.config.scale_expand_steps):
            if avg_fn(high) >= target:
                break
            high *= 2.0
        for _ in range(self.config.scale_search_steps):
            mid = (low + high) / 2.0
            if avg_fn(mid) >= target:
                high = mid
            else:
                low = mid
            if high > 0.0 and (high - low) / high <= 1e-3:
                break
        return float(high)
