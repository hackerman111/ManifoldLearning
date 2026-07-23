from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from ..common.types import KernelName


PAIRWISE_TEMPORARY_BYTES = 64 * 1024 * 1024


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

        if (
            not isinstance(name, str)
            or name not in {"epanechnikov", "quartic", "gaussian"}
        ):
            raise ValueError(
                "kernel должен быть 'epanechnikov', 'quartic' или 'gaussian'"
            )
        xq = self._gpu_array(q)
        if not self._device_scalar_bool(self.xp.isfinite(xq).all()):
            raise ValueError("q должен содержать только конечные значения")
        return self._evaluate_kernel(xq, name)

    def _evaluate_kernel(self, q: Any, name: KernelName) -> Any:
        """Вычисляет ядро для уже проверенного device-массива."""

        if name == "gaussian":
            return self.xp.exp(-0.5 * q)
        if name == "quartic":
            return self.xp.square(self.xp.maximum(1.0 - q, 0.0))
        return self.xp.maximum(1.0 - q, 0.0)

    @staticmethod
    def _device_scalar_bool(value: Any) -> bool:
        """Копирует на CPU только один логический device-скаляр."""

        item = getattr(value, "item", None)
        return bool(item() if item is not None else value)

    def pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
    ) -> Any:
        """Считает ||X_i - c_j||^2 на GPU."""

        x = self._cached_gpu_array("X", X)
        xcenters = self._cached_gpu_array("centers", centers)
        n_centers = int(xcenters.shape[0])
        norm2 = self.xp.zeros((n_centers, x.shape[0]), dtype=self.dtype)
        block_size = self._pairwise_center_block_size(x)
        for start in range(0, n_centers, block_size):
            stop = min(start + block_size, n_centers)
            differences = x[None, :, :] - xcenters[start:stop, None, :]
            norm2[start:stop] = self.xp.einsum(
                "cnd,cnd->cn",
                differences,
                differences,
            )
        return norm2

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
        n_centers = int(xcenters.shape[0])
        projection2 = self.xp.zeros((n_centers, x.shape[0]), dtype=self.dtype)
        block_size = self._pairwise_center_block_size(x)
        for start in range(0, n_centers, block_size):
            stop = min(start + block_size, n_centers)
            differences = x[None, :, :] - xcenters[start:stop, None, :]
            projections = self.xp.einsum("cnd,d->cn", differences, xbeta)
            projection2[start:stop] = self.xp.square(projections)
        return projection2

    def _pairwise_center_block_size(self, X: Any) -> int:
        """Ограничивает временный C x n x d массив прямых разностей."""

        elements_per_center = max(1, int(X.shape[0]) * int(X.shape[1]))
        itemsize = np.dtype(self.dtype).itemsize
        return max(1, PAIRWISE_TEMPORARY_BYTES // (itemsize * elements_per_center))

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
        record_telemetry: bool = False,
    ) -> tuple[Any, ...]:
        """Считает блочные суммы Ima, S, U на GPU без D2H до finalize."""

        x = self._gpu_array(X)
        xy = self._gpu_array(y)
        xcenters = self._gpu_array(centers)
        xdirs = self._gpu_array(directions)
        xq = self._gpu_array(q)
        self._validate_random_projection_inputs(
            x,
            xy,
            xcenters,
            xdirs,
            xq,
            kernel,
        )

        weights_started = time.perf_counter()
        weights = self._evaluate_kernel(xq, kernel)
        counts_gpu = weights.sum(axis=1)
        weight_payload = None
        if record_telemetry:
            weight_payload = {
                "sum_w2": self.to_numpy((weights * weights).sum(axis=1)),
                "nonzero": self.to_numpy((weights > 0.0).sum(axis=1)).astype(
                    int,
                    copy=False,
                ),
                "min_weight": self.to_numpy(weights.min(axis=1)),
                "max_weight": self.to_numpy(weights.max(axis=1)),
            }
        weights_time = time.perf_counter() - weights_started

        statistics_started = time.perf_counter()
        c_count, p_count, dimension = xdirs.shape
        imav = self.xp.zeros((c_count, p_count), dtype=self.dtype)
        s_vec = self.xp.zeros((c_count, p_count), dtype=self.dtype)
        u_mat = self.xp.zeros(
            (c_count, p_count, dimension),
            dtype=self.dtype,
        )
        block_size = self._pairwise_center_block_size(x)
        for start in range(0, c_count, block_size):
            stop = min(start + block_size, c_count)
            differences = x[None, :, :] - xcenters[start:stop, None, :]
            projected = self.xp.matmul(
                differences,
                self.xp.swapaxes(xdirs[start:stop], 1, 2),
            )
            projected *= weights[start:stop, :, None]
            s_vec[start:stop] = projected.sum(axis=1)
            imav[start:stop] = self.xp.einsum(
                "cnp,n->cp",
                projected,
                xy,
                optimize=True,
            )
            u_mat[start:stop] = self.xp.matmul(
                self.xp.swapaxes(projected, 1, 2),
                differences,
            )
        result = (
            imav,
            s_vec,
            u_mat,
            counts_gpu,
            counts_gpu.mean(),
        )
        if not record_telemetry:
            return result
        assert weight_payload is not None
        weight_payload["weights_time_sec"] = weights_time
        weight_payload["statistics_time_sec"] = (
            time.perf_counter() - statistics_started
        )
        return (*result, weight_payload)

    def _validate_random_projection_inputs(
        self,
        X: Any,
        y: Any,
        centers: Any,
        directions: Any,
        q: Any,
        kernel: KernelName,
    ) -> None:
        """Проверяет публичный контракт статистик без копирования массивов на CPU."""

        if (
            not isinstance(kernel, str)
            or kernel not in {"epanechnikov", "quartic", "gaussian"}
        ):
            raise ValueError(
                "kernel должен быть 'epanechnikov', 'quartic' или 'gaussian'"
            )
        if X.ndim != 2:
            raise ValueError("X должен быть двумерным массивом")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y должен быть вектором длины n")
        if centers.ndim != 2 or centers.shape[1] != X.shape[1]:
            raise ValueError("centers должен иметь форму C x d")
        if directions.ndim != 3 or directions.shape[2] != X.shape[1]:
            raise ValueError("directions должен иметь форму C x P x d")
        if q.ndim != 2:
            raise ValueError("q должен иметь форму C x n")
        if centers.shape[0] != directions.shape[0] or q.shape[0] != directions.shape[0]:
            raise ValueError(
                "centers, directions и q должны иметь одинаковое число центров"
            )
        if q.shape[1] != X.shape[0]:
            raise ValueError("q должен иметь форму C x n")

        finite_checks = tuple(
            (name, self.xp.isfinite(value).all())
            for name, value in (
                ("X", X),
                ("y", y),
                ("centers", centers),
                ("directions", directions),
                ("q", q),
            )
        )
        all_finite = finite_checks[0][1]
        for _, is_finite in finite_checks[1:]:
            all_finite = all_finite & is_finite
        if self._device_scalar_bool(all_finite):
            return
        for name, is_finite in finite_checks:
            if not self._device_scalar_bool(is_finite):
                raise ValueError(
                    f"{name} должен содержать только конечные значения"
                )
