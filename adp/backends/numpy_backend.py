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
        if dtype not in {"float64", "float32"}:
            raise ValueError("dtype должен быть 'float64' или 'float32'")
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
            return np.square(np.maximum(1.0 - q, 0.0))
        return np.maximum(1.0 - q, 0.0)

    def pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
    ) -> Any:
        """Считает ||X_i - c_j||^2 на backend."""

        x = self.asarray(X)
        xcenters = self.asarray(centers)
        x_sq = np.einsum("ij,ij->i", x, x)
        center_sq = np.einsum("ij,ij->i", xcenters, xcenters)
        norm2 = center_sq[:, None] + x_sq[None, :] - 2.0 * (xcenters @ x.T)
        np.maximum(norm2, 0.0, out=norm2)
        return norm2

    def pairwise_projection2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
        beta: np.ndarray,  # Направление beta.
    ) -> Any:
        """Считает <X_i - c_j, beta>^2 на backend."""

        x = self.asarray(X)
        xcenters = self.asarray(centers)
        xbeta = self.asarray(beta).reshape(-1)
        x_proj = x @ xbeta
        center_proj = xcenters @ xbeta
        return np.square(x_proj[None, :] - center_proj[:, None])

    def local_mass_score(
        self,
        q: Any,  # Матрица квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
        *,
        quantile: float | None = None,  # Квантиль масс или None для среднего.
    ) -> float:
        """Считает среднюю или квантильную локальную массу на backend."""

        weights = self.kernel(q, kernel)
        masses = weights.sum(axis=1)
        if quantile is None:
            return float(masses.mean())
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile должен быть в диапазоне [0, 1]")
        return float(np.quantile(masses, quantile))

    def random_projection_sums(
        self,
        *,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        centers: np.ndarray,  # Центры блока C x d.
        directions: np.ndarray,  # Направления блока C x P x d.
        q: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает блочные суммы Ima, S, U для new-варианта.

        Вход:
            X: матрица наблюдений n x d.
            y: вектор ответов.
            centers: центры блока C x d.
            directions: массив C x P x d.
            q: значения локальной квадратичной формы.
            kernel: имя ядра.
        Выход:
            Кортеж imav, S, U, N и средней локальной массы.
        """

        x = self.asarray(X)
        xy = self.asarray(y)
        xcenters = self.asarray(centers)
        xdirs = self.asarray(directions)
        xq = self.asarray(q)
        if xcenters.shape[0] != xdirs.shape[0] or xq.shape[0] != xdirs.shape[0]:
            raise ValueError("centers, directions и q должны иметь одинаковое число центров")
        if xq.shape[1] != x.shape[0]:
            raise ValueError("q должен иметь форму C x n")
        if kernel in {"epanechnikov", "quartic"}:
            return self._compact_random_projection_sums(x, xy, xdirs, xq, kernel)
        return self._dense_random_projection_sums(x, xy, xdirs, xq, kernel)

    def _dense_random_projection_sums(
        self,
        x: np.ndarray,  # Матрица наблюдений n x d.
        xy: np.ndarray,  # Вектор ответов длины n.
        xdirs: np.ndarray,  # Направления блока C x P x d.
        xq: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает суммы плотным блочным путем для Gaussian kernel."""

        weights = self.kernel(xq, kernel)
        counts = self.to_numpy(weights.sum(axis=1))
        safe_counts = np.maximum(counts, np.finfo(float).eps)

        # centered_i = X_i - Xbar_j, где Xbar_j нормируется на N_j.
        # Алгебра ниже избегает временного массива C x n x d.
        xbar = (weights @ x) / safe_counts[:, None]
        projected = np.matmul(x[None, :, :], np.swapaxes(xdirs, 1, 2))
        xbar_projected = np.einsum("cd,cpd->cp", xbar, xdirs, optimize=True)
        projected -= xbar_projected[:, None, :]
        imav = np.einsum("n,cn,cnp->cp", xy, weights, projected, optimize=True)
        s_vec = np.einsum("cn,cnp->cp", weights, projected, optimize=True)
        projected *= weights[:, :, None]
        u_raw = np.matmul(np.swapaxes(projected, 1, 2), x)
        u_mat = u_raw - s_vec[:, :, None] * xbar[:, None, :]
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            counts,
            float(counts.mean()),
        )

    def _compact_random_projection_sums(
        self,
        x: np.ndarray,  # Матрица наблюдений n x d.
        xy: np.ndarray,  # Вектор ответов длины n.
        xdirs: np.ndarray,  # Направления блока C x P x d.
        xq: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя compact-ядра.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Считает суммы compact kernels только по точкам с ненулевым весом."""

        c_count, p_count, d = xdirs.shape
        imav = np.zeros((c_count, p_count), dtype=self.dtype)
        s_vec = np.zeros((c_count, p_count), dtype=self.dtype)
        u_mat = np.zeros((c_count, p_count, d), dtype=self.dtype)
        counts = np.zeros(c_count, dtype=self.dtype)
        tiny = np.finfo(float).eps
        for center_index in range(c_count):
            active = xq[center_index] < 1.0
            if not np.any(active):
                continue
            q_active = xq[center_index, active]
            weights = self.kernel(q_active, kernel).astype(self.dtype, copy=False)
            count = weights.sum(dtype=self.dtype)
            counts[center_index] = count
            safe_count = max(float(count), tiny)
            x_active = x[active]
            y_active = xy[active]
            xbar = (weights @ x_active) / safe_count
            centered = x_active - xbar[None, :]
            projected = centered @ xdirs[center_index].T
            weighted_projected = projected * weights[:, None]
            s_vec[center_index] = weighted_projected.sum(axis=0)
            imav[center_index] = (weights * y_active) @ projected
            u_mat[center_index] = weighted_projected.T @ centered
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            self.to_numpy(counts),
            float(counts.mean()),
        )
