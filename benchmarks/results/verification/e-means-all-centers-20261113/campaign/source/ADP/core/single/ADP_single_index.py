"""Публичная single-index модель на общем HPAO-движке."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from ...engine.common.index_fit import fit_index
from ...engine.common.logger import IndexProfiler
from ...solver.LSMR import solve as solve_lsmr
from ..ADP_Config import ADP_Config
from ..ADP_Solver import ADP_solver
from . import ADP_single_index_utils as model_utils
from .ADP_single_index_result import ADP_single_index_result


class ADP_single_index:
    def __init__(
        self,
        config: ADP_Config | None = None,
        solver: ADP_solver | None = None,
    ) -> None:
        """Выбрать актуальный HPAO solver; custom solver возвращает HPAOResult."""
        self.config = config or ADP_Config()
        self.solver = solver or ADP_solver(solve_lsmr, max_steps=3, tol=1e-6)
        self.ADP_single_index_result = ADP_single_index_result

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        progress: Callable[[dict], None] | None = None,
    ) -> ADP_single_index:
        """Оценить индекс тем же outer-циклом и seed scheme, что CLI.

        ESTIMATOR: используется HPAO, normalized statistics для estimator=new,
        выбор best/last и независимые потоки centers/init/directions.
        """
        config, solver = self.config, self.solver
        if config.gpu_solver and solver.method is not solve_lsmr:
            raise NotImplementedError(
                "gpu_solver supports built-in LSMR only; "
                "use gpu=True, gpu_solver=False for CPU solvers"
            )
        profiler = IndexProfiler()
        try:
            fitted = fit_index(
                X,
                Y,
                config=config,
                mode="single",
                index_dim=1,
                profiler=profiler,
                solve=lambda index, statistics: solver.fit_current(
                    statistics,
                    index,
                    normalized=config.estimator == "new",
                    lambda_prox=config.lambda_penalty,
                ),
                progress=progress,
                trace_indices=True,
            )
        finally:
            profiler.finish()
            self.profile_ = profiler.model_profile()

        result = self.ADP_single_index_result(
            beta_init=fitted.initial_index.copy(),
            beta_final=fitted.index.copy(),
            trace=cast(list[dict], fitted.metadata["trace"]),
            stop_reason=cast(str, fitted.metadata["stop_reason"]),
        )
        self.result_ = result
        self.beta_ = result.beta_final
        self.trace_ = result.trace
        self.coefficients_ = fitted.coefficients.copy()
        self.solver_diagnostics_ = [row["solver"] for row in self.trace_]
        self.effective_parameters_ = dict(fitted.metadata)
        self.effective_parameters_.update(
            gpu=config.gpu,
            gpu_solver=config.gpu_solver,
            smart_weights=config.smart_weights,
            h_min=config.h_min,
        )
        self.n_features_in_ = fitted.index.shape[-1]
        return self

    def _check_fitted(self) -> None:
        """Проверить наличие результата fit."""
        model_utils.require_fitted(self)
