from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from ..backends.neighbors import NeighborIndex
from ..common.progress import format_progress_postfix
from ..common.result_store import store_fit_result
from ..common.types import ADPResult, LocalStatistics
from ..common.utils import as_1d_float, as_2d_float, unit_vector


class TrainingMixin:
    """Внешний цикл обучения ADP."""

    def fit(
        self,
        X: np.ndarray,  # Матрица наблюдений n x d.
        y: np.ndarray,  # Вектор ответов длины n.
        *,
        centers: np.ndarray | None = None,  # Пользовательские центры или None.
        beta0: np.ndarray | None = None,  # Начальное beta или None.
        directions: np.ndarray | None = None,  # Направления для new или None.
    ) -> ADPResult:
        """Обучает ADP-модель.

        Вход:
            X: матрица наблюдений.
            y: вектор ответов.
            centers: готовые центры или None.
            beta0: стартовое направление beta или None.
            directions: готовые случайные направления для new-варианта.
        Выход:
            ADPResult с beta, локальными коэффициентами и историей.
        """

        X_arr = as_2d_float(X, "X")
        y_arr = as_1d_float(y, "y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X и y имеют разные размеры по n")

        started = time.perf_counter()
        _, d = X_arr.shape
        centers_arr = as_2d_float(centers, "centers") if centers is not None else self._choose_centers(X_arr)
        if centers_arr.shape[1] != d:
            raise ValueError("centers должны иметь ту же размерность d, что и X")

        beta_prev = unit_vector(beta0 if beta0 is not None else self._initial_beta(X_arr, y_arr))
        lambda_penalty = self.config.resolved_lambda()
        neighbor_index = NeighborIndex(self.config.use_neighbor_index).fit(X_arr)
        self.neighbor_index_ = neighbor_index

        h = self._select_isotropic_bandwidth(X_arr, centers_arr, neighbor_index)
        b_old = h
        directions_arr = self._prepare_directions(centers_arr, d, directions)

        history = []
        progress = []
        timings: dict[str, float] = {}
        intercepts = np.zeros(centers_arr.shape[0])
        slopes = np.ones(centers_arr.shape[0])
        statistics: LocalStatistics | None = None

        outer_iter = range(max(1, self.config.outer_steps))
        progress_bar = self._make_progress_bar(outer_iter)
        if progress_bar is not None:
            outer_iter = progress_bar

        try:
            for outer in outer_iter:
                step_started = time.perf_counter()
                anisotropy, b_value, h, b_old, directions_arr = self._prepare_outer_step(
                    outer,
                    X_arr,
                    centers_arr,
                    h,
                    b_old,
                    beta_prev,
                    directions_arr,
                    d,
                )

                stats_started = time.perf_counter()
                statistics = self._compute_statistics(
                    X_arr,
                    y_arr,
                    centers_arr,
                    h,
                    beta_prev,
                    directions_arr,
                    anisotropy,
                    b_value,
                )
                timings["statistics"] = timings.get("statistics", 0.0) + time.perf_counter() - stats_started

                solve_started = time.perf_counter()
                beta_new, intercepts, slopes, inner_history = self._alternating_solve(
                    statistics,
                    beta_prev,
                    lambda_penalty,
                    outer,
                    step_started,
                )
                timings["solve"] = timings.get("solve", 0.0) + time.perf_counter() - solve_started
                history.extend(inner_history)
                beta_prev = beta_new

                if inner_history:
                    record = self._progress_record(
                        stats=statistics,
                        step=inner_history[-1],
                        outer_index=outer,
                        outer_total=max(1, self.config.outer_steps),
                        inner_count=len(inner_history),
                        started=started,
                    )
                    progress.append(record)
                    if progress_bar is not None and hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix(format_progress_postfix(record), refresh=True)

                if self.config.anisotropy_min is not None and self.variant == "new" and anisotropy is not None:
                    if anisotropy <= self.config.anisotropy_min:
                        break
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                progress_bar.close()

        if statistics is None:
            raise RuntimeError("fit не смог вычислить локальные статистики")
        objective = history[-1].objective if history else self._objective(statistics, beta_prev, intercepts, slopes, beta_prev, lambda_penalty)
        if progress:
            progress[-1]["objective"] = float(objective)

        return store_fit_result(
            self,
            beta_prev,
            intercepts,
            slopes,
            statistics,
            history,
            progress,
            timings,
            started,
            X_arr,
            y_arr,
            centers_arr,
            directions_arr,
            float(objective),
        )

    def _make_progress_bar(
        self,
        outer_iter: range,  # Диапазон outer-итераций.
    ) -> Any:
        """Создает tqdm progress bar при включенном выводе.

        Вход:
            outer_iter: range внешних шагов.
        Выход:
            tqdm-обертка или None.
        """

        if not self.config.show_progress:
            return None
        from .base import tqdm

        progress_factory = getattr(sys.modules.get("adp.core"), "tqdm", tqdm)
        if progress_factory is None:
            return None
        return progress_factory(outer_iter, desc=f"ADP-{self.variant} {self.backend.name}", leave=False)

    def _prepare_outer_step(
        self,
        outer: int,  # Номер внешнего шага.
        X: np.ndarray,  # Матрица наблюдений n x d.
        centers: np.ndarray,  # Матрица центров J x d.
        h: float,  # Текущий h.
        b_old: float,  # Текущий old-bandwidth b.
        beta: np.ndarray,  # Текущее beta.
        directions: np.ndarray | None,  # Текущие directions или None.
        d: int,  # Размерность признаков.
    ) -> tuple[float | None, float | None, float, float, np.ndarray | None]:
        """Готовит bandwidth, anisotropy и directions для outer-шага.

        Вход:
            outer: номер outer-шага.
            X: матрица наблюдений.
            centers: матрица центров.
            h: текущий bandwidth.
            b_old: текущий old-bandwidth.
            beta: текущее направление.
            directions: текущие направления.
            d: размерность признаков.
        Выход:
            Кортеж anisotropy, b_value, h, b_old, directions.
        """

        if outer == 0:
            return None, None, h, b_old, directions
        if self.variant == "new":
            h = max(h / self.config.bandwidth_decay, np.finfo(float).eps)
            anisotropy = self._select_new_anisotropy(X, centers, h, beta)
            if self.config.renew_directions:
                directions = self._sample_directions(
                    centers.shape[0],
                    self.config.n_directions,
                    d,
                    beta=beta,
                    anisotropy=anisotropy,
                )
            return anisotropy, None, h, b_old, directions

        b_old = max(b_old / self.config.bandwidth_decay, np.finfo(float).eps)
        h = self._select_old_bandwidth(X, centers, beta, b_old)
        return None, b_old, h, b_old, directions
