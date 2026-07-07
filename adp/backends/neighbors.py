from __future__ import annotations

from typing import Any

import numpy as np


class NeighborIndex:
    """Опциональный faiss/sklearn индекс для оценки локального масштаба."""

    def __init__(
        self,
        enabled: bool = True,  # Включать ли попытку построить индекс.
    ) -> None:
        """Создает пустой индекс соседей.

        Вход:
            enabled: если False, индекс не строится.
        Выход:
            None; состояние индекса сохраняется в объекте.
        """

        self.backend = "none"
        self.index: Any = None
        self.enabled = enabled

    def fit(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
    ) -> "NeighborIndex":
        """Строит доступный faiss или sklearn индекс.

        Вход:
            X: матрица наблюдений n x d.
        Выход:
            self с заполненным индексом или backend='none'.
        """

        if not self.enabled:
            return self
        try:
            import faiss

            index = faiss.IndexFlatL2(X.shape[1])
            index.add(np.asarray(X, dtype=np.float32))
            self.index = index
            self.backend = "faiss"
            return self
        except Exception:
            pass
        try:
            from sklearn.neighbors import NearestNeighbors

            index = NearestNeighbors(algorithm="auto")
            index.fit(X)
            self.index = index
            self.backend = "sklearn"
        except Exception:
            self.index = None
            self.backend = "none"
        return self

    def kth_distances(
        self,
        centers: np.ndarray,  # Центры локальных окрестностей J x d.
        k: int,  # Номер соседа.
    ) -> np.ndarray | None:
        """Возвращает расстояния до k-го соседа.

        Вход:
            centers: матрица центров J x d.
            k: номер соседа для масштаба.
        Выход:
            Вектор расстояний длины J или None, если индекс недоступен.
        """

        if self.index is None or k <= 0:
            return None
        if self.backend == "faiss":
            distances, _ = self.index.search(np.asarray(centers, dtype=np.float32), k)
            return np.sqrt(np.maximum(distances[:, -1], 0.0))
        distances, _ = self.index.kneighbors(centers, n_neighbors=k)
        return distances[:, -1]
