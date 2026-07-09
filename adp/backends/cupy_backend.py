from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ..common.types import KernelName


class CupyBackend:
    """Опциональный CuPy-backend для тяжелых локальных сумм ADP."""

    def __init__(
        self,
        dtype: str = "float64",  # Числовая точность вычислителя.
    ) -> None:
        """Создает backend и проверяет доступность CuPy."""

        self.name = "cupy"
        self.dtype_name = dtype
        if dtype not in {"float64", "float32"}:
            raise ValueError("dtype должен быть 'float64' или 'float32'")
        try:
            self.xp = importlib.import_module("cupy")
        except Exception as exc:
            raise ImportError("CuPy backend requires installed cupy") from exc
        self.dtype = np.float64 if dtype == "float64" else np.float32

    def asarray(
        self,
        value: np.ndarray,  # Исходное значение.
    ) -> np.ndarray:
        """Оставляет внешний pipeline на NumPy, чтобы SciPy solver работал без копий."""

        return np.asarray(value, dtype=self.dtype)

    def _gpu_array(
        self,
        value: Any,  # NumPy/CuPy-совместимое значение.
    ) -> Any:
        """Переносит значение на GPU в dtype backend."""

        return self.xp.asarray(value, dtype=self.dtype)

    def to_numpy(
        self,
        value: Any,  # Значение backend.
    ) -> np.ndarray:
        """Возвращает NumPy-представление GPU/CPU значения."""

        asnumpy = getattr(self.xp, "asnumpy", None)
        if asnumpy is None:
            return np.asarray(value)
        return np.asarray(asnumpy(value))

    def kernel(
        self,
        q: Any,  # Значения квадратичной формы.
        name: KernelName,  # Имя ядра.
    ) -> Any:
        """Вычисляет веса ядра на GPU."""

        xq = self._gpu_array(q)
        if name == "gaussian":
            return self.xp.exp(-0.5 * xq)
        if name == "quartic":
            return self.xp.maximum(1.0 - xq * xq, 0.0)
        return self.xp.maximum(1.0 - xq, 0.0)

    def pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
    ) -> Any:
        """Считает ||X_i - c_j||^2 на GPU."""

        x = self._gpu_array(X)
        xcenters = self._gpu_array(centers)
        x_sq = self.xp.einsum("ij,ij->i", x, x)
        center_sq = self.xp.einsum("ij,ij->i", xcenters, xcenters)
        norm2 = center_sq[:, None] + x_sq[None, :] - 2.0 * (xcenters @ x.T)
        return self.xp.maximum(norm2, 0.0)

    def pairwise_projection2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
        beta: np.ndarray,  # Направление beta.
    ) -> Any:
        """Считает <X_i - c_j, beta>^2 на GPU."""

        x = self._gpu_array(X)
        xcenters = self._gpu_array(centers)
        xbeta = self._gpu_array(beta).reshape(-1)
        x_proj = x @ xbeta
        center_proj = xcenters @ xbeta
        return (x_proj[None, :] - center_proj[:, None]) ** 2

    def local_mass_score(
        self,
        q: Any,  # Матрица квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> float:
        """Считает среднюю локальную массу на GPU и возвращает scalar."""

        masses = self.kernel(q, kernel).sum(axis=1)
        return float(self.to_numpy(masses.mean()))

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
        """Считает блочные суммы Ima, S, U на GPU и возвращает NumPy-результаты."""

        x = self._gpu_array(X)
        xy = self._gpu_array(y)
        xcenters = self._gpu_array(centers)
        xdirs = self._gpu_array(directions)
        xq = self._gpu_array(q)
        if xcenters.shape[0] != xdirs.shape[0] or xq.shape[0] != xdirs.shape[0]:
            raise ValueError("centers, directions и q должны иметь одинаковое число центров")
        if xq.shape[1] != x.shape[0]:
            raise ValueError("q должен иметь форму C x n")

        weights = self.kernel(xq, kernel)
        counts_gpu = weights.sum(axis=1)
        counts = self.to_numpy(counts_gpu)
        safe_counts = self.xp.maximum(counts_gpu, np.finfo(float).eps)

        xbar = (weights @ x) / safe_counts[:, None]
        projected = self.xp.matmul(x[None, :, :], self.xp.swapaxes(xdirs, 1, 2))
        xbar_projected = self.xp.einsum("cd,cpd->cp", xbar, xdirs, optimize=True)
        projected = projected - xbar_projected[:, None, :]
        imav = self.xp.einsum("n,cn,cnp->cp", xy, weights, projected, optimize=True)
        s_vec = self.xp.einsum("cn,cnp->cp", weights, projected, optimize=True)
        weighted_projected = projected * weights[:, :, None]
        u_raw = self.xp.matmul(self.xp.swapaxes(weighted_projected, 1, 2), x)
        u_mat = u_raw - s_vec[:, :, None] * xbar[:, None, :]
        return (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            counts,
            float(counts.mean()),
        )
