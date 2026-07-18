from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..common.types import KernelName

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
        if xcenters.shape[0] != xdirs.shape[0] or xq.shape[0] != xdirs.shape[0]:
            raise ValueError("centers, directions и q должны иметь одинаковое число центров")
        if xq.shape[1] != x.shape[0]:
            raise ValueError("q должен иметь форму C x n")
        if kernel in {"epanechnikov", "quartic"}:
            return self._compact_random_projection_sums(
                x,
                xy,
                xdirs,
                xq,
                kernel,
                record_telemetry=record_telemetry,
            )
        return self._dense_random_projection_sums(
            x,
            xy,
            xdirs,
            xq,
            kernel,
            record_telemetry=record_telemetry,
        )

    def _dense_random_projection_sums(
        self,
        x: np.ndarray,  # Матрица наблюдений n x d.
        xy: np.ndarray,  # Вектор ответов длины n.
        xdirs: np.ndarray,  # Направления блока C x P x d.
        xq: np.ndarray,  # Значения квадратичной формы C x n.
        kernel: KernelName,  # Имя ядра.
        *,
        record_telemetry: bool = False,
    ) -> tuple[Any, ...]:
        """Считает суммы плотным блочным путем для Gaussian kernel."""

        weights_started = time.perf_counter()
        weights = self.kernel(xq, kernel)
        counts = self.to_numpy(weights.sum(axis=1))
        weight_payload = None
        if record_telemetry:
            weight_payload = {
                "sum_w2": self.to_numpy(np.square(weights).sum(axis=1)),
                "nonzero": self.to_numpy((weights > 0.0).sum(axis=1)).astype(
                    int,
                    copy=False,
                ),
                "min_weight": self.to_numpy(weights.min(axis=1)),
                "max_weight": self.to_numpy(weights.max(axis=1)),
            }
        weights_time = time.perf_counter() - weights_started

        statistics_started = time.perf_counter()
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
        result = (
            self.to_numpy(imav),
            self.to_numpy(s_vec),
            self.to_numpy(u_mat),
            counts,
            float(counts.mean()),
        )
        if not record_telemetry:
            return result
        assert weight_payload is not None
        weight_payload["weights_time_sec"] = weights_time
        weight_payload["statistics_time_sec"] = (
            time.perf_counter() - statistics_started
        )
        return (*result, weight_payload)

    def _compact_random_projection_sums(
        self,
        x: np.ndarray,
        xy: np.ndarray,
        xdirs: np.ndarray,
        xq: np.ndarray,
        kernel: KernelName,
        *,
        record_telemetry: bool = False,
    ) -> tuple[Any, ...]:
        """Computes compact-kernel sums with in-place projected weights."""

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
        tiny = np.finfo(self.dtype).eps

        def compute_center(center_index: int) -> None:
            active = xq[center_index] < 1.0
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
            safe_count = max(float(count), float(tiny))
            x_active = x[active]
            y_active = xy[active]
            xbar = (weights @ x_active) / safe_count
            centered = x_active - xbar[None, :]
            projected = centered @ xdirs[center_index].T
            projected *= weights[:, None]
            imav[center_index] = y_active @ projected
            u_mat[center_index] = projected.T @ centered

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
