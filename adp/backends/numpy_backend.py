from __future__ import annotations

from typing import Any

import numpy as np

from ..common.types import KernelName


class NumpyBackend:
    """NumPy-реализация тяжелых локальных сумм."""

    def __init__(
        self,
        dtype: str = "float64",  # Числовая точность вычислителя.
    ) -> None:
        """Создает backend с нужной точностью.

        Вход:
            dtype: строковое имя точности, float64 или float32.
        Выход:
            None; параметры сохраняются в объекте.
        """

        self.name = "numpy"
        self.dtype_name = dtype
        self.dtype = np.float64 if dtype == "float64" else np.float32

    def asarray(
        self,
        value: np.ndarray,  # Исходное значение.
    ) -> Any:
        """Приводит массив к dtype backend.

        Вход:
            value: объект, совместимый с np.asarray.
        Выход:
            NumPy-массив выбранной точности.
        """

        return np.asarray(value, dtype=self.dtype)

    def to_numpy(
        self,
        value: Any,  # Значение вычислителя.
    ) -> np.ndarray:
        """Возвращает значение как NumPy-массив.

        Вход:
            value: значение backend.
        Выход:
            NumPy-представление value.
        """

        return np.asarray(value)

    def kernel(
        self,
        q: Any,  # Значения квадратичной формы.
        name: KernelName,  # Имя ядра.
    ) -> Any:
        """Вычисляет локальные веса ядра.

        Вход:
            q: массив значений квадратичной формы.
            name: epanechnikov, quartic или gaussian.
        Выход:
            Массив весов той же формы.
        """

        if name == "gaussian":
            return np.exp(-0.5 * q)
        if name == "quartic":
            return np.maximum(1.0 - q * q, 0.0)
        return np.maximum(1.0 - q, 0.0)

    def random_projection_sums(
        self,
        diff: np.ndarray,  # Разности X_i - c_j для блока центров.
        y: np.ndarray,  # Вектор ответов длины n.
        directions: np.ndarray,  # Направления блока C x P x d.
        q: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает суммы Ima, S, U для new-варианта.

        Вход:
            diff: массив C x n x d.
            y: вектор ответов.
            directions: массив C x P x d.
            q: значения локальной квадратичной формы.
            kernel: имя ядра.
        Выход:
            Кортеж imav, S, U и средней локальной массы.
        """

        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xdirs = self.asarray(directions)
        xq = self.asarray(q)
        weights = self.kernel(xq, kernel)

        # manifold_new.tex: для каждого центра j и направления phi считаем
        # <X_i - x_j, phi>, а затем суммы Ima_{j,phi}, S_{j,phi}, U_{j,phi}.
        projected = np.einsum("cnd,cpd->cnp", xdiff, xdirs)
        imav = np.einsum("n,cn,cnp->cp", xy, weights, projected)
        s_vec = np.einsum("cn,cnp->cp", weights, projected)
        u_mat = np.einsum("cnd,cn,cnp->cpd", xdiff, weights, projected)
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            float(self.to_numpy(weights.sum(axis=1)).mean()),
        )

    def full_moment_sums(
        self,
        diff: np.ndarray,  # Разности X_i - c_j для блока центров.
        y: np.ndarray,  # Вектор ответов длины n.
        q: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает Ima, N, S, VP для old-варианта.

        Вход:
            diff: массив C x n x d.
            y: вектор ответов.
            q: значения локальной квадратичной формы.
            kernel: имя ядра.
        Выход:
            Кортеж imav, N, S, VP и средней локальной массы.
        """

        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xq = self.asarray(q)
        weights = self.kernel(xq, kernel)

        # manifold_old.tex: старая версия хранит полный локальный момент
        # [N_j, S_j; S_j^T, VP_j], без сжатия через случайные направления.
        im0 = np.einsum("n,cn->c", xy, weights)
        im1 = np.einsum("n,cn,cnd->cd", xy, weights, xdiff)
        n_vec = weights.sum(axis=1)
        s_vec = np.einsum("cn,cnd->cd", weights, xdiff)
        vp = np.einsum("cn,cnd,cne->cde", weights, xdiff, xdiff)
        imav = np.concatenate([im0[:, None], im1], axis=1)
        return (
            self.to_numpy(imav),
            self.to_numpy(n_vec),
            self.to_numpy(s_vec),
            self.to_numpy(vp),
            float(self.to_numpy(n_vec).mean()),
        )
