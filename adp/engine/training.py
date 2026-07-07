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

        # Приводим вход к строгой форме n x d и n, чтобы дальше формулы
        # из manifold_new.tex работали с ожидаемыми осями.
        X_arr = self.backend.asarray(as_2d_float(X, "X"))
        y_arr = self.backend.asarray(as_1d_float(y, "y"))
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X и y имеют разные размеры по n")
        self._clear_pairwise_cache()

        started = time.perf_counter()
        _, d = X_arr.shape

        # Центры x_j задают локальные окрестности из предварительного раздела.
        centers_arr = self.backend.asarray(as_2d_float(centers, "centers")) if centers is not None else self._choose_centers(X_arr)
        if centers_arr.shape[1] != d:
            raise ValueError("centers должны иметь ту же размерность d, что и X")

        # beta_prev играет роль текущего prior: beta_0 на первом внешнем шаге
        # и beta_{k-1} в адаптивных шагах TeX-алгоритма.
        beta_prev = self.backend.asarray(unit_vector(beta0 if beta0 is not None else self._initial_beta(X_arr, y_arr)))
        lambda_penalty = self.config.resolved_lambda()

        # Индекс соседей нужен только для быстрой верхней оценки h; сами веса
        # потом считаются векторизованными формулами ядра, как в TeX.
        neighbor_index = NeighborIndex(self.config.use_neighbor_index).fit(X_arr)
        self.neighbor_index_ = neighbor_index

        # Первый шаг всегда изотропный: T = h^{-2} I. Для new заранее готовим
        # случайные направления phi.
        h = self._select_isotropic_bandwidth(X_arr, centers_arr, neighbor_index)
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

                # На внешних шагах после первого локализация становится адаптивной:
                # new обновляет rho и направления phi.
                anisotropy, h, directions_arr = self._prepare_outer_step(
                    outer,
                    X_arr,
                    centers_arr,
                    h,
                    beta_prev,
                    directions_arr,
                    d,
                )

                # Здесь строятся наблюдаемые локальные суммы Ima/S/U.
                # Это единственный дорогой шаг по данным X.
                stats_started = time.perf_counter()
                statistics = self._compute_statistics(
                    X_arr,
                    y_arr,
                    centers_arr,
                    h,
                    beta_prev,
                    directions_arr,
                    anisotropy,
                )
                timings["statistics"] = timings.get("statistics", 0.0) + time.perf_counter() - stats_started

                # При фиксированных статистиках решаем попеременно локальные
                # коэффициенты (c_j, l_j) и глобальное направление beta.
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
                    # Запись остается машинно-читаемой, а индикатор получает только
                    # компактно отформатированную версию этого же состояния.
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
                    # Для new можно остановиться, когда rho уже достиг заданного
                    # уровня сильной локализации вокруг текущего beta.
                    if anisotropy <= self.config.anisotropy_min:
                        break
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                progress_bar.close()
            self._clear_pairwise_cache()

        if statistics is None:
            raise RuntimeError("fit не смог вычислить локальные статистики")

        # Финальное значение цели берем из последнего внутреннего шага.
        # Запасной путь нужен только для формально пустой истории, если
        # inner_steps был обнулен извне.
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
        outer_iter: range,  # Диапазон внешних шагов.
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
        beta: np.ndarray,  # Текущее beta.
        directions: np.ndarray | None,  # Текущие направления или None.
        d: int,  # Размерность признаков.
    ) -> tuple[float | None, float, np.ndarray | None]:
        """Готовит bandwidth, anisotropy и directions для outer-шага.

        Вход:
            outer: номер outer-шага.
            X: матрица наблюдений.
            centers: матрица центров.
            h: текущий bandwidth.
            beta: текущее направление.
            directions: текущие направления.
            d: размерность признаков.
        Выход:
            Кортеж anisotropy, h, directions.
        """

        if outer == 0:
            # Стартовый шаг соответствует изотропным весам из предварительного раздела:
            # K(h^{-2} ||X_i - x_j||^2), без анизотропных поправок.
            return None, h, directions

        # manifold_new.tex: уменьшаем h, выбираем rho из условия на
        # локальную массу и при необходимости пересэмплируем phi.
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
        return anisotropy, h, directions
