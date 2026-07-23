from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..common.types import KernelName
from ..common.utils import (
    pairwise_norm2 as stable_pairwise_norm2,
    pairwise_projection2 as stable_pairwise_projection2,
)

PARALLEL_STATISTICS_MIN_WORK = 1_000_000


class NumpyBackend:
    """NumPy-реализация тяжелых локальных сумм."""

    def __init__(
        self,
        dtype: str = "float64",  # Числовая точность вычислителя.
        *,
        statistics_workers: int = 1,
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
        if (
            isinstance(statistics_workers, bool)
            or not isinstance(statistics_workers, int)
            or statistics_workers < 1
        ):
            raise ValueError("statistics_workers должен быть положительным")
        self.dtype = np.float64 if dtype == "float64" else np.float32
        self.statistics_workers = int(statistics_workers)

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

    def prepare_statistics_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Готовит входы блочных статистик без изменения CPU-пути."""

        return (
            self.asarray(X),
            self.asarray(y),
            self.asarray(centers),
            self.asarray(directions),
        )

    def create_statistics_accumulator(
        self,
        n_centers: int,
        n_directions: int,
        dimension: int,
    ) -> dict[str, np.ndarray]:
        """Создает host-аккумуляторы для блочных локальных сумм."""

        return {
            "imav": np.zeros((n_centers, n_directions), dtype=self.dtype),
            "S": np.zeros((n_centers, n_directions), dtype=self.dtype),
            "U": np.zeros((n_centers, n_directions, dimension), dtype=self.dtype),
            "N": np.zeros(n_centers, dtype=self.dtype),
        }

    def accumulate_statistics(
        self,
        accumulator: dict[str, np.ndarray],
        start: int,
        stop: int,
        chunk: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    ) -> None:
        """Записывает один host-блок в аккумуляторы."""

        imav, s_vec, u_mat, counts, _ = chunk
        accumulator["imav"][start:stop] = imav
        accumulator["S"][start:stop] = s_vec
        accumulator["U"][start:stop] = u_mat
        accumulator["N"][start:stop] = counts

    def finalize_statistics(
        self,
        accumulator: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Возвращает host-статистики после обработки всех блоков."""

        counts = self.to_numpy(accumulator["N"])
        return (
            self.to_numpy(accumulator["imav"]),
            self.to_numpy(accumulator["S"]),
            self.to_numpy(accumulator["U"]),
            counts,
            float(counts.mean()) if counts.size else 0.0,
        )

    def clear_device_cache(self) -> None:
        """Совместимый no-op для общего training lifecycle."""

    def release_device_memory(self) -> None:
        """Совместимый no-op для общего training lifecycle."""

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

        if (
            not isinstance(name, str)
            or name not in {"epanechnikov", "quartic", "gaussian"}
        ):
            raise ValueError(
                "kernel должен быть 'epanechnikov', 'quartic' или 'gaussian'"
            )
        xq = self.asarray(q)
        if not np.all(np.isfinite(xq)):
            raise ValueError("q должен содержать только конечные значения")
        if name == "gaussian":
            return np.exp(-0.5 * xq)
        if name == "quartic":
            return np.square(np.maximum(1.0 - xq, 0.0))
        return np.maximum(1.0 - xq, 0.0)

    def pairwise_norm2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
    ) -> Any:
        """Считает ||X_i - c_j||^2 на backend."""

        x = self.asarray(X)
        xcenters = self.asarray(centers)
        return stable_pairwise_norm2(x, xcenters)

    def pairwise_projection2(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров C x d.
        beta: np.ndarray,  # Направление beta.
    ) -> Any:
        """Считает <X_i - c_j, beta>^2 на backend."""

        x = self.asarray(X)
        xcenters = self.asarray(centers)
        xbeta = self.asarray(beta)
        return stable_pairwise_projection2(x, xcenters, xbeta)

    def kernel_argument(
        self,
        norm2: np.ndarray,
        *,
        h: float,
        projection2: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray:
        """Builds the kernel quadratic-form argument in one output buffer."""

        xnorm2 = self.asarray(norm2)
        inverse_h2 = self.dtype(1.0 / (float(h) * float(h)))
        q = np.empty_like(xnorm2)
        if anisotropy is None:
            np.multiply(xnorm2, inverse_h2, out=q)
            return q
        if projection2 is None:
            raise ValueError("projection2 is required when anisotropy is set")
        np.multiply(xnorm2, self.dtype(float(anisotropy) ** 2), out=q)
        np.add(q, self.asarray(projection2), out=q)
        np.multiply(q, inverse_h2, out=q)
        return q

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
        record_telemetry: bool = False,
    ) -> tuple[Any, ...]:
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
        self._validate_random_projection_inputs(
            x,
            xy,
            xcenters,
            xdirs,
            xq,
            kernel,
        )
        return self._centered_random_projection_sums(
            x,
            xy,
            xcenters,
            xdirs,
            xq,
            kernel,
            record_telemetry=record_telemetry,
        )

    def _validate_random_projection_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        directions: np.ndarray,
        q: np.ndarray,
        kernel: KernelName,
    ) -> None:
        """Validates the public statistics input contract."""

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
            raise ValueError("centers, directions и q должны иметь одинаковое число центров")
        if q.shape[1] != X.shape[0]:
            raise ValueError("q должен иметь форму C x n")

        for name, value in (
            ("X", X),
            ("y", y),
            ("centers", centers),
            ("directions", directions),
            ("q", q),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} должен содержать только конечные значения")

    def _centered_random_projection_sums(
        self,
        x: np.ndarray,
        xy: np.ndarray,
        xcenters: np.ndarray,
        xdirs: np.ndarray,
        xq: np.ndarray,
        kernel: KernelName,
        *,
        record_telemetry: bool = False,
    ) -> tuple[Any, ...]:
        """Computes stable per-center sums from explicit feature differences."""

        started = time.perf_counter()
        c_count, p_count, d = xdirs.shape
        imav = np.zeros((c_count, p_count), dtype=self.dtype)
        s_vec = np.zeros((c_count, p_count), dtype=self.dtype)
        u_mat = np.zeros((c_count, p_count, d), dtype=self.dtype)
        counts = np.zeros(c_count, dtype=self.dtype)
        sum_w2 = np.zeros(c_count, dtype=self.dtype)
        nonzero = np.zeros(c_count, dtype=int)
        min_weight = np.zeros(c_count, dtype=self.dtype)
        max_weight = np.zeros(c_count, dtype=self.dtype)
        weight_durations = np.zeros(c_count, dtype=float)
        is_compact_kernel = kernel in {"epanechnikov", "quartic"}

        def compute_center(center_index: int) -> None:
            active = (
                xq[center_index] < 1.0
                if is_compact_kernel
                else np.ones(x.shape[0], dtype=bool)
            )
            if not np.any(active):
                return
            weights_started = time.perf_counter()
            weights = self.kernel(xq[center_index, active], kernel).astype(
                self.dtype,
                copy=False,
            )
            count = weights.sum(dtype=self.dtype)
            if record_telemetry:
                sum_w2[center_index] = np.square(weights).sum(dtype=self.dtype)
                nonzero[center_index] = int(np.count_nonzero(weights))
                min_weight[center_index] = (
                    self.dtype(0.0)
                    if np.count_nonzero(active) < x.shape[0]
                    else weights.min()
                )
                max_weight[center_index] = weights.max()
                weight_durations[center_index] = (
                    time.perf_counter() - weights_started
                )
            counts[center_index] = count
            x_active = x[active]
            y_active = xy[active]
            differences = x_active - xcenters[center_index]
            projected = differences @ xdirs[center_index].T
            projected *= weights[:, None]
            s_vec[center_index] = projected.sum(axis=0)
            imav[center_index] = y_active @ projected
            u_mat[center_index] = projected.T @ differences

        work_proxy = c_count * x.shape[0] * p_count * d
        use_parallel = (
            self.statistics_workers > 1
            and c_count > 1
            and work_proxy >= PARALLEL_STATISTICS_MIN_WORK
        )
        if use_parallel:
            worker_count = min(self.statistics_workers, c_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(
                    executor.map(
                        compute_center,
                        range(c_count),
                        buffersize=worker_count,
                    )
                )
        else:
            for center_index in range(c_count):
                compute_center(center_index)

        result = (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            self.to_numpy(counts),
            float(counts.mean()),
        )
        if not record_telemetry:
            return result
        total_time = time.perf_counter() - started
        weights_time = min(float(weight_durations.sum()), total_time)
        return (
            *result,
            {
                "sum_w2": self.to_numpy(sum_w2),
                "nonzero": self.to_numpy(nonzero).astype(int, copy=False),
                "min_weight": self.to_numpy(min_weight),
                "max_weight": self.to_numpy(max_weight),
                "weights_time_sec": weights_time,
                "statistics_time_sec": max(0.0, total_time - weights_time),
            },
        )
