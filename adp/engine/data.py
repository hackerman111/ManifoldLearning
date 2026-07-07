from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

from ..common.types import ADPData
from ..common.utils import link_function, normalize_rows, unit_vector


class DataPreparationMixin:
    """Генерация входных данных, центров и случайных направлений."""

    def generate_data(
        self,
        n: int,  # Число наблюдений.
        d: int,  # Размерность признаков.
        *,
        n_centers: int | None = None,  # Число локальных центров.
        n_directions: int | None = None,  # Число направлений на центр для new.
        beta: np.ndarray | None = None,  # Истинное направление или None.
        noise: float = 0.1,  # Стандартное отклонение шума.
        sigma_x: float = 1.0,  # Масштаб признаков.
        corr: float = 0.5,  # Сила общей компоненты признаков.
        link: str | Callable[[np.ndarray], np.ndarray] = "quadratic",  # Связь f.
    ) -> ADPData:
        """Генерирует single-index данные из manifold_*.tex.

        Вход:
            n: число наблюдений.
            d: размерность признаков.
            n_centers: число центров или None для значения из config.
            n_directions: число случайных направлений для new-варианта.
            beta: истинный EDR-вектор или None для случайного.
            noise: стандартное отклонение шума eps.
            sigma_x: масштаб признаков X.
            corr: коррелированность признаков в диапазоне [0, 1).
            link: имя или callable функции f.
        Выход:
            ADPData с X, y, beta, центрами, направлениями и шумом.
        """

        if n <= 0 or d <= 0:
            raise ValueError("n и d должны быть положительными")
        if not 0 <= corr < 1:
            raise ValueError("corr должен лежать в [0, 1)")

        # Истинный beta задает одномерный индекс beta^T X из одноиндексной
        # модели Y = f(beta^T X) + eps в обоих TeX-файлах.
        beta_vec = unit_vector(beta if beta is not None else self.rng.normal(size=d))

        # Простая коррелированная схема X: общая компонента + индивидуальный
        # шум. Она нужна для воспроизводимых сценариев замеров, не для теории.
        shared = self.rng.normal(size=d)
        individual = self.rng.normal(size=(n, d))
        X = sigma_x * (corr * shared[None, :] + (1.0 - corr) * individual)

        eps = self.rng.normal(scale=noise, size=n)
        link_fn, link_name = link_function(link)
        y = link_fn(X @ beta_vec) + eps

        # Центры x_j выбираются из облака X и слегка зашумляются, чтобы
        # локальные окрестности не сводились только к исходным наблюдениям.
        j_count = int(n_centers or self.config.n_centers or n)
        j_count = min(max(j_count, 1), n)
        selected = self.rng.choice(n, size=j_count, replace=False)
        center_noise = self.config.center_noise_scale * sigma_x * self.rng.normal(size=(j_count, d))
        centers = X[selected] + center_noise

        directions = None
        if self.variant == "new":
            # Только manifold_new.tex требует наборы phi_j случайных направлений.
            p_count = int(n_directions or self.config.n_directions)
            directions = self._sample_directions(j_count, p_count, d)
        return ADPData(X=X, y=y, beta=beta_vec, centers=centers, directions=directions, noise=eps, link_name=link_name)

    def _choose_centers(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
    ) -> np.ndarray:
        """Выбирает центры из X и добавляет небольшой шум.

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

    def _prepare_directions(
        self,
        centers: np.ndarray,  # Матрица центров J x d.
        d: int,  # Размерность признаков.
        directions: np.ndarray | None,  # Пользовательские направления или None.
    ) -> np.ndarray | None:
        """Готовит направления для new-варианта.

        Вход:
            centers: матрица центров.
            d: размерность признаков.
            directions: готовый массив направлений или None.
        Выход:
            Нормированный массив J x P x d или None для old-варианта.
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
        n_centers: int,  # Число центров.
        n_directions: int,  # Число направлений на центр.
        d: int,  # Размерность признаков.
        *,
        beta: np.ndarray | None = None,  # Текущее beta для анизотропной выборки.
        anisotropy: float | None = None,  # rho или None.
    ) -> np.ndarray:
        """Сэмплирует направления на единичной сфере.

        Вход:
            n_centers: число центров.
            n_directions: число направлений на центр.
            d: размерность признаков.
            beta: текущее направление для anisotropic обновления.
            anisotropy: коэффициент rho из new-варианта.
        Выход:
            Массив направлений n_centers x n_directions x d.
        """

        z = self.rng.normal(size=(n_centers, n_directions, d))
        if beta is not None and anisotropy is not None:
            # При обновлении new-варианта направления смещаются к текущему beta,
            # как обновление направлений после обновления локального тензора.
            beta_unit = unit_vector(beta)
            along_beta = self.rng.normal(size=(n_centers, n_directions, 1))
            z = float(anisotropy) * z + along_beta * beta_unit
        return normalize_rows(z)

    def _initial_beta(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
    ) -> np.ndarray:
        """Строит начальное направление beta.

        Вход:
            X: матрица наблюдений.
            y: вектор ответов.
        Выход:
            Единичный начальный вектор beta.
        """

        # Стартовый beta нужен только как prior. Метод наименьших квадратов
        # хорошо работает для линейной связи, а для сложной связи дает
        # устойчивое ненулевое зерно.
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
