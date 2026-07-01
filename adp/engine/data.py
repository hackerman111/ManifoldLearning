from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import linalg

from ..common.types import ADPData
from ..common.utils import link_function, normalize_rows, unit_vector


class DataPreparationMixin:
    """Методы генерации данных, центров, направлений и масштабов."""

    def generate_data(
        self,
        n: int,
        d: int,
        *,
        n_centers: int | None = None,
        n_directions: int | None = None,
        beta: np.ndarray | None = None,
        noise: float = 0.1,
        sigma_x: float = 1.0,
        corr: float = 0.5,
        link: str | Callable[[np.ndarray], np.ndarray] = "quadratic",
    ) -> ADPData:
        """Генерирует single-index данные из TeX-описания.

        Вход:
            n: число наблюдений.
            d: размерность признаков.
            n_centers: число локальных центров.
            n_directions: число случайных направлений.
            beta: истинное EDR-направление или None для случайного.
            noise: стандартное отклонение шума.
            sigma_x: масштаб признаков.
            corr: сила общей компоненты признаков.
            link: функция связи или её имя.
        Выход:
            ADPData с X, y, beta, центрами, направлениями и шумом.
        """

        if n <= 0 or d <= 0:
            raise ValueError("n и d должны быть положительными")
        if not 0 <= corr < 1:
            raise ValueError("corr должен лежать в [0, 1)")

        beta_vec = unit_vector(beta if beta is not None else self.rng.normal(size=d))
        shared = self.rng.normal(size=d)
        individual = self.rng.normal(size=(n, d))
        X = sigma_x * (corr * shared[None, :] + (1.0 - corr) * individual)
        eps = self.rng.normal(scale=noise, size=n)
        link_fn, link_name = link_function(link)
        y = link_fn(X @ beta_vec) + eps

        j_count = int(n_centers or self.config.n_centers or n)
        j_count = min(max(j_count, 1), n)
        selected = self.rng.choice(n, size=j_count, replace=False)
        centers = X[selected] + self.config.center_noise_scale * sigma_x * self.rng.normal(size=(j_count, d))

        directions = None
        if self.variant == "new":
            p_count = int(n_directions or self.config.n_directions)
            directions = self._sample_directions(j_count, p_count, d)
        return ADPData(X=X, y=y, beta=beta_vec, centers=centers, directions=directions, noise=eps, link_name=link_name)

    def _choose_centers(self, X: np.ndarray) -> np.ndarray:
        """Выбирает и зашумляет центры из X.

        Вход:
            X: матрица наблюдений n x d.
        Выход:
            Матрица центров J x d.
        """

        n, d = X.shape
        j_count = int(self.config.n_centers or n)
        j_count = min(max(j_count, 1), n)
        selected = self.rng.choice(n, size=j_count, replace=False)
        scale = float(np.std(X)) if np.std(X) > 0 else 1.0
        return X[selected] + self.config.center_noise_scale * scale * self.rng.normal(size=(j_count, d))

    def _prepare_directions(self, centers: np.ndarray, d: int, directions: np.ndarray | None) -> np.ndarray | None:
        """Готовит направления для new-варианта.

        Вход:
            centers: матрица центров J x d.
            d: размерность признаков.
            directions: пользовательские направления или None.
        Выход:
            Нормированные направления J x P x d или None для old-варианта.
        """

        if self.variant == "old":
            return None
        if directions is None:
            return self._sample_directions(centers.shape[0], self.config.n_directions, d)
        directions_arr = np.asarray(directions, dtype=float)
        expected = (centers.shape[0], self.config.n_directions, d)
        if directions_arr.shape != expected:
            raise ValueError(f"directions должны иметь форму {expected}, получено {directions_arr.shape}")
        return normalize_rows(directions_arr)

    def _sample_directions(
        self,
        n_centers: int,
        n_directions: int,
        d: int,
        *,
        beta: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray:
        """Сэмплирует нормированные случайные направления.

        Вход:
            n_centers: число центров.
            n_directions: число направлений на центр.
            d: размерность признаков.
            beta: текущее EDR-направление для anisotropic sampling.
            anisotropy: коэффициент rho или None.
        Выход:
            Массив направлений размера n_centers x n_directions x d.
        """

        z = self.rng.normal(size=(n_centers, n_directions, d))
        if beta is not None and anisotropy is not None:
            beta_unit = unit_vector(beta)
            noise = self.rng.normal(size=(n_centers, n_directions, 1))
            z = float(anisotropy) * z + noise * beta_unit
        return normalize_rows(z)

    def _initial_beta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Строит начальное направление beta.

        Вход:
            X: матрица наблюдений n x d.
            y: вектор ответов длины n.
        Выход:
            Единичный начальный вектор beta.
        """

        x_centered = X - X.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        try:
            beta, *_ = linalg.lstsq(x_centered, y_centered)
        except Exception:
            beta = x_centered.T @ y_centered
        if np.linalg.norm(beta) < np.finfo(float).eps:
            beta = np.zeros(X.shape[1])
            beta[0] = 1.0
        return unit_vector(beta)
