# ruff: noqa: RUF002

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..engine.common.calculus import (
    calculate_h0_from_adp,
    calculate_rho_k_from_adp,
    generate_isotropic_proj,
    generate_proj_from_adp,
)
from ..engine.common.statistic import calculate_statistics
from ..engine.common.weights import calculate_weight_from_adp
from ..solver.LSMR import HPAOResult, lsmr, solve as solve_lsmr
from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .ADP_Statistic import ADP_Statistics


@dataclass(slots=True)
class ADP_SolverResult:
    """Результат совместимого solver-адаптера single/multi моделей."""

    index: np.ndarray
    coefficients: np.ndarray | None
    diagnostics: dict[str, object]


class ADP_solver:
    """Настраиваемый адаптер текущего HPAO и явного legacy solver API."""

    def __init__(self, method: Callable, **settings: object) -> None:
        """Создать адаптер метода solver с фиксированными настройками."""
        if not callable(method):
            raise TypeError("method must be callable")
        self.method = method
        self.settings = dict(settings)

    def fit(self, statistics, initial_index, **problem_params) -> ADP_SolverResult:
        """Запустить solver и проверить его индекс и тип результата."""
        overlap = self.settings.keys() & problem_params.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"duplicate solver settings: {names}")

        result = self.method(
            statistics,
            initial_index,
            **problem_params,
            **self.settings,
        )
        if not isinstance(result, ADP_SolverResult):
            raise TypeError("solver method must return ADP_SolverResult")

        index = np.asarray(result.index, dtype=float)
        if not np.all(np.isfinite(index)):
            raise RuntimeError("solver returned a non-finite index")
        result.index = index
        return result

    def fit_current(
        self,
        statistics: ADP_Statistics,
        initial_index: np.ndarray,
        *,
        normalized: bool,
        lambda_prox: float,
    ) -> HPAOResult:
        """Вызвать HPAO solver; multi-index передаётся строками (m,d).

        Нормированные I/U требуют внешней mass. Старый ``fit`` оставлен
        для явного legacy API; модели используют только этот контракт.
        """
        if (
            hasattr(statistics.U, "__cuda_array_interface__")
            and self.method is not solve_lsmr
        ):
            raise NotImplementedError("GPU statistics require LSMR or transfer to CPU")
        overlap = self.settings.keys() & {"mass", "lambda_prox"}
        if overlap:
            raise ValueError(f"duplicate solver settings: {', '.join(sorted(overlap))}")
        result = self.method(
            initial_index,
            statistics.U,
            statistics.I,
            mass=statistics.mass if normalized else None,
            lambda_prox=lambda_prox,
            **self.settings,
        )
        if not isinstance(result, HPAOResult):
            raise TypeError(
                "current index models require a solver returning HPAOResult"
            )
        return result


class ADP_Solver:
    def __init__(
        self,
        config: ADP_Config,
        data: ADP_Data,
        solver: Callable[..., np.ndarray] | None = None,
    ):
        """Собрать совместимый single/multi solver поверх core-статистик."""
        self.config = config
        self.data = data
        self.solver = solver or lsmr

    def fit(self) -> np.ndarray:
        """Выполнить legacy-последовательность bandwidth, weights и solver."""
        X, Y = self.data.X, self.data.Y
        beta_init = self.data.beta_init
        a = self.config.a
        n_phi = self.config.N_phi
        if n_phi is None:
            raise ValueError("N_phi must be configured before fitting")
        rng = np.random.default_rng(seed=self.config.seed)
        normalized = self.config.estimator == "new"

        def run_step(beta: np.ndarray, h: float, rho: float) -> np.ndarray:
            """Построить статистики одного масштаба и решить локальную задачу."""
            if self.config.direction_mode == "isotropic":
                proj = generate_isotropic_proj(
                    rng,
                    len(self.data.x_j),
                    n_phi,
                    X.shape[1],
                )
            else:
                proj = generate_proj_from_adp(
                    self.config,
                    self.data,
                    rng,
                    beta,
                    rho,
                )
            weight = calculate_weight_from_adp(
                self.config,
                self.data,
                beta,
                h,
                rho,
            )
            stat = calculate_statistics(
                X,
                Y,
                weight,
                proj,
                batch_size=self.config.batch_size,
                normalized=normalized,
            )
            return self.solver(
                beta,
                stat.U,
                stat.I,
                mass=stat.mass if normalized else None,
                lambda_prox=self.config.lambda_penalty,
                max_steps=3,
            )

        rho_k = 1.0
        h_k = calculate_h0_from_adp(self.config, self.data)
        beta_k = run_step(beta_init, h_k, rho_k)

        outer_iteration = 1
        while h_k / a > self.config.h_min and (
            self.config.outer_steps is None or outer_iteration < self.config.outer_steps
        ):
            h_k = h_k / a
            rho_k = calculate_rho_k_from_adp(self.config, self.data, beta_k, h_k)

            if rho_k is None:
                raise RuntimeError("could not find a feasible localization factor")
            beta_k = run_step(beta_k, h_k, rho_k)
            outer_iteration += 1

        return beta_k
