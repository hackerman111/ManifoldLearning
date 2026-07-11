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
        self._device_cache: dict[str, tuple[int, Any]] = {}
        self._device_ids: set[int] = set()
        self._validate_cuda_device()

    def _validate_cuda_device(self) -> None:
        """Проверяет наличие CUDA, если runtime предоставляет такую проверку."""

        runtime = getattr(getattr(self.xp, "cuda", None), "runtime", None)
        get_device_count = getattr(runtime, "getDeviceCount", None)
        if get_device_count is None:
            return
        try:
            device_count = int(get_device_count())
        except Exception as exc:
            raise RuntimeError(
                "CuPy установлен, но CUDA-устройство недоступно"
            ) from exc
        if device_count < 1:
            raise RuntimeError("CuPy установлен, но CUDA-устройство недоступно")

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

        ndarray_type = getattr(self.xp, "ndarray", None)
        if (ndarray_type is not None and isinstance(value, ndarray_type)) or id(value) in self._device_ids:
            return value
        return self.xp.asarray(value, dtype=self.dtype)

    def _cached_gpu_array(
        self,
        key: str,
        value: Any,
    ) -> Any:
        """Возвращает один device-copy для стабильного fit-входа."""

        token = id(value)
        cached = self._device_cache.get(key)
        if cached is None or cached[0] != token:
            device_value = self._gpu_array(value)
            self._device_cache[key] = (token, device_value)
            self._device_ids.add(id(device_value))
        return self._device_cache[key][1]

    def prepare_statistics_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[Any, Any, Any, Any]:
        """Кеширует стабильные входы локальных статистик на GPU."""

        return (
            self._cached_gpu_array("X", X),
            self._cached_gpu_array("y", y),
            self._cached_gpu_array("centers", centers),
            self._cached_gpu_array("directions", directions),
        )

    def clear_device_cache(self) -> None:
        """Удаляет ссылки на временные device-входы, не трогая memory pool."""

        self._device_cache.clear()
        self._device_ids.clear()

    def release_device_memory(self) -> None:
        """Очищает кеш и свободные блоки CuPy после завершения fit."""

        self.clear_device_cache()
        get_pool = getattr(self.xp, "get_default_memory_pool", None)
        if get_pool is not None:
            get_pool().free_all_blocks()
        get_pinned_pool = getattr(self.xp, "get_default_pinned_memory_pool", None)
        if get_pinned_pool is not None:
            get_pinned_pool().free_all_blocks()

    def create_statistics_accumulator(
        self,
        n_centers: int,
        n_directions: int,
        dimension: int,
    ) -> dict[str, Any]:
        """Создает GPU-аккумуляторы для всех center-блоков."""

        return {
            "imav": self.xp.zeros((n_centers, n_directions), dtype=self.dtype),
            "S": self.xp.zeros((n_centers, n_directions), dtype=self.dtype),
            "U": self.xp.zeros((n_centers, n_directions, dimension), dtype=self.dtype),
            "N": self.xp.zeros(n_centers, dtype=self.dtype),
        }

    def accumulate_statistics(
        self,
        accumulator: dict[str, Any],
        start: int,
        stop: int,
        chunk: tuple[Any, Any, Any, Any, Any],
    ) -> None:
        """Записывает GPU-блок без промежуточного D2H-копирования."""

        imav, s_vec, u_mat, counts, _ = chunk
        accumulator["imav"][start:stop] = imav
        accumulator["S"][start:stop] = s_vec
        accumulator["U"][start:stop] = u_mat
        accumulator["N"][start:stop] = counts

    def finalize_statistics(
        self,
        accumulator: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Однократно переносит итоговые локальные статистики на CPU."""

        imav = self.to_numpy(accumulator["imav"])
        s_vec = self.to_numpy(accumulator["S"])
        u_mat = self.to_numpy(accumulator["U"])
        counts = self.to_numpy(accumulator["N"])
        return (
            imav,
            s_vec,
            u_mat,
            counts,
            float(counts.mean()) if counts.size else 0.0,
        )

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
            return self.xp.square(self.xp.maximum(1.0 - xq, 0.0))
        return self.xp.maximum(1.0 - xq, 0.0)

    def pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
    ) -> Any:
        """Считает ||X_i - c_j||^2 на GPU."""

        x = self._cached_gpu_array("X", X)
        xcenters = self._cached_gpu_array("centers", centers)
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

        x = self._cached_gpu_array("X", X)
        xcenters = self._cached_gpu_array("centers", centers)
        xbeta = self._gpu_array(beta).reshape(-1)
        x_proj = x @ xbeta
        center_proj = xcenters @ xbeta
        return (x_proj[None, :] - center_proj[:, None]) ** 2

    def kernel_argument(
        self,
        norm2: Any,
        *,
        h: float,
        projection2: Any | None = None,
        anisotropy: float | None = None,
    ) -> Any:
        """Builds the kernel quadratic-form argument on the GPU."""

        xnorm2 = self._gpu_array(norm2)
        inverse_h2 = self.dtype(1.0 / (float(h) * float(h)))
        if anisotropy is None:
            return xnorm2 * inverse_h2
        if projection2 is None:
            raise ValueError("projection2 is required when anisotropy is set")
        return (
            self.dtype(float(anisotropy) ** 2) * xnorm2
            + self._gpu_array(projection2)
        ) * inverse_h2

    def local_mass_score(
        self,
        q: Any,  # Матрица квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
        *,
        quantile: float | None = None,  # Квантиль масс или None для среднего.
    ) -> float:
        """Считает среднюю или квантильную локальную массу на GPU."""

        masses = self.kernel(q, kernel).sum(axis=1)
        if quantile is None:
            return float(self.to_numpy(masses.mean()))
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile должен быть в диапазоне [0, 1]")
        return float(self.to_numpy(self.xp.quantile(masses, quantile)))

    def random_projection_sums(
        self,
        *,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        centers: np.ndarray,  # Центры блока C x d.
        directions: np.ndarray,  # Направления блока C x P x d.
        q: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Считает блочные суммы Ima, S, U на GPU без D2H до finalize."""

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
        safe_counts = self.xp.maximum(counts_gpu, np.finfo(self.dtype).eps)

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
            imav,
            s_vec,
            u_mat,
            counts_gpu,
            counts_gpu.mean(),
        )
