from __future__ import annotations

from typing import Any

import numpy as np

from ..common.types import KernelName


class NumpyBackend:
    """NumPy-реализация тяжёлых локальных сумм."""

    def __init__(self, dtype: str = "float64"):
        """Создаёт backend с нужной точностью.

        Вход:
            dtype: строковое имя точности, поддерживаются float64 и float32.
        Выход:
            None; сохраняет параметры backend в объекте.
        """

        self.name = "numpy"
        self.dtype_name = dtype
        self.dtype = np.float64 if dtype == "float64" else np.float32

    def asarray(self, value: np.ndarray) -> Any:
        """Приводит массив к backend dtype.

        Вход:
            value: исходный массив.
        Выход:
            NumPy-массив с выбранной точностью.
        """

        return np.asarray(value, dtype=self.dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Возвращает значение как NumPy-массив.

        Вход:
            value: объект, совместимый с np.asarray.
        Выход:
            NumPy-представление value.
        """

        return np.asarray(value)

    def kernel(self, q: Any, name: KernelName) -> Any:
        """Вычисляет веса ядра.

        Вход:
            q: квадратичная форма расстояний.
            name: имя ядра.
        Выход:
            Массив весов той же формы, что и q.
        """

        if name == "gaussian":
            return np.exp(-0.5 * q)
        if name == "quartic":
            return np.maximum(1.0 - q * q, 0.0)
        return np.maximum(1.0 - q, 0.0)

    def random_projection_sums(
        self,
        diff: np.ndarray,
        y: np.ndarray,
        directions: np.ndarray,
        q: np.ndarray,
        kernel: KernelName,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает суммы Ima, S, U для new-варианта.

        Вход:
            diff: разности X_i - c_j для блока центров.
            y: вектор ответов.
            directions: случайные направления для блока центров.
            q: значения квадратичной формы для ядра.
            kernel: имя ядра.
        Выход:
            Кортеж imav, S, U и среднего локального веса.
        """

        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xdirs = self.asarray(directions)
        xq = self.asarray(q)
        w = self.kernel(xq, kernel)
        proj = np.einsum("cnd,cpd->cnp", xdiff, xdirs)
        imav = np.einsum("n,cn,cnp->cp", xy, w, proj)
        s_vec = np.einsum("cn,cnp->cp", w, proj)
        u_mat = np.einsum("cnd,cn,cnp->cpd", xdiff, w, proj)
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            float(self.to_numpy(w.sum(axis=1)).mean()),
        )

    def full_moment_sums(
        self,
        diff: np.ndarray,
        y: np.ndarray,
        q: np.ndarray,
        kernel: KernelName,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает Ima, N, S, VP для old-варианта.

        Вход:
            diff: разности X_i - c_j для блока центров.
            y: вектор ответов.
            q: значения квадратичной формы для ядра.
            kernel: имя ядра.
        Выход:
            Кортеж imav, N, S, VP и среднего локального веса.
        """

        xdiff = self.asarray(diff)
        xy = self.asarray(y)
        xq = self.asarray(q)
        w = self.kernel(xq, kernel)
        im0 = np.einsum("n,cn->c", xy, w)
        im1 = np.einsum("n,cn,cnd->cd", xy, w, xdiff)
        n_vec = w.sum(axis=1)
        s_vec = np.einsum("cn,cnd->cd", w, xdiff)
        vp = np.einsum("cn,cnd,cne->cde", w, xdiff, xdiff)
        imav = np.concatenate([im0[:, None], im1], axis=1)
        return (
            self.to_numpy(imav),
            self.to_numpy(n_vec),
            self.to_numpy(s_vec),
            self.to_numpy(vp),
            float(self.to_numpy(n_vec).mean()),
        )
