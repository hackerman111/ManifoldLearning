from __future__ import annotations

from typing import Any

import numpy as np


class NeighborIndex:
    """Опциональный индекс соседей для оценки масштаба."""

    def __init__(self, enabled: bool = True):
        """Создаёт пустой индекс.

        Вход:
            enabled: включает попытку построить faiss/sklearn индекс.
        Выход:
            None; инициализирует состояние индекса.
        """

        self.backend = "none"
        self.index: Any = None
        self.enabled = enabled

    def fit(self, X: np.ndarray) -> "NeighborIndex":
        """Строит доступный индекс по матрице X.

        Вход:
            X: матрица наблюдений размера n x d.
        Выход:
            self с заполненным индексом или backend='none'.
        """

        if not self.enabled:
            return self
        try:
            import faiss  # type: ignore

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

    def kth_distances(self, centers: np.ndarray, k: int) -> np.ndarray | None:
        """Возвращает расстояния до k-го соседа.

        Вход:
            centers: центры локальных окрестностей.
            k: номер соседа для оценки масштаба.
        Выход:
            Вектор расстояний или None, если индекс недоступен.
        """

        if self.index is None or k <= 0:
            return None
        if self.backend == "faiss":
            distances, _ = self.index.search(np.asarray(centers, dtype=np.float32), k)
            return np.sqrt(np.maximum(distances[:, -1], 0.0))
        distances, _ = self.index.kneighbors(centers, n_neighbors=k)
        return distances[:, -1]
