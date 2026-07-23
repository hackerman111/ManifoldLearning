from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..backends.neighbors import NeighborIndex
from ..common.progress import format_progress_postfix
from ..common.resource_monitor import ResourceMonitor
from ..common.result_store import store_fit_result
from ..common.types import ADPResult, LocalStatistics, TrainingStep
from ..common.utils import as_1d_float, as_2d_float, stable_l2_norm, unit_vector
from ..stages.contracts import ADPState, StageContext, StageExecutionError, StageFactory
from ..stages.registry import (
    DEFAULT_STAGE_NAMES,
    STAGE_METHODS,
    StageRegistry,
)


class ADPAlgorithm:
    """Явная композиция заменяемых этапов алгоритма ADP."""

    def __init__(
        self,
        context: StageContext,
        *,
        stages: Mapping[str, str] | None = None,
        stage_factories: Mapping[str, StageFactory] | None = None,
        registry: StageRegistry | None = None,
    ) -> None:
        self.context = context
        self.registry = (registry or StageRegistry.with_defaults()).copy()
        selected = dict(stages or {})
        direct = dict(stage_factories or {})
        self._validate_categories(selected)
        self._validate_categories(direct)

        self.stage_names: dict[str, str] = {}
        self.components: dict[str, Any] = {}
        for category in STAGE_METHODS:
            if category in direct:
                factory = direct[category]
                name = "custom"
                if not callable(factory):
                    raise TypeError(f"Фабрика этапа {category!r} должна быть callable")
            else:
                name = selected.get(category, DEFAULT_STAGE_NAMES[category])
                factory = self.registry.resolve(category, name)
            component = factory(context)
            self._validate_component(category, name, component)
            self.stage_names[category] = name
            self.components[category] = component

        self.stage_timings: dict[str, float] = {
            category: 0.0 for category in self.components
        }
        self.stage_calls: dict[str, int] = {
            category: 0 for category in self.components
        }
        self.state: ADPState | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        centers: np.ndarray | None = None,
        beta0: np.ndarray | None = None,
        directions: np.ndarray | None = None,
    ) -> ADPResult:
        """Выполняет ADP как явную последовательность фабричных этапов."""

        monitor = ResourceMonitor()
        result: ADPResult | None = None
        try:
            with monitor:
                result = self._fit_impl(
                    X,
                    y,
                    centers=centers,
                    beta0=beta0,
                    directions=directions,
                )
        finally:
            usage = monitor.usage.to_dict("algorithm")
            self.context.model.last_resource_usage_ = usage
            if result is not None:
                result.resource_usage = dict(usage)
        return result

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        centers: np.ndarray | None = None,
        beta0: np.ndarray | None = None,
        directions: np.ndarray | None = None,
    ) -> ADPResult:
        """Выполняет численную часть fit внутри окна мониторинга ресурсов."""

        model = self.context.model
        config = self.context.config
        self.stage_timings = {category: 0.0 for category in self.components}
        self.stage_calls = {category: 0 for category in self.components}

        X_arr = model.backend.asarray(as_2d_float(X, "X"))
        y_arr = model.backend.asarray(as_1d_float(y, "y"))
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X и y имеют разные размеры по n")
        model.backend.clear_device_cache()
        model._clear_pairwise_cache()
        state = ADPState(X=X_arr, y=y_arr)
        self.state = state

        started = time.perf_counter()
        _, d = X_arr.shape
        if centers is None:
            centers_arr = self._invoke("center_selector", "select", X_arr)
        else:
            centers_arr = model.backend.asarray(as_2d_float(centers, "centers"))
        centers_arr = model.backend.asarray(as_2d_float(centers_arr, "centers"))
        if centers_arr.shape[1] != d:
            raise ValueError("centers должны иметь ту же размерность d, что и X")
        state.centers = centers_arr

        if beta0 is None:
            initial_beta = self._invoke("beta_initializer", "initialize", X_arr, y_arr)
        else:
            initial_beta = beta0
            self._validate_user_beta(initial_beta, d, "beta0")
        beta_prev = model.backend.asarray(unit_vector(initial_beta))
        state.beta = beta_prev
        state.prior = beta_prev.copy()
        lambda_penalty = config.resolved_lambda()

        try:
            neighbor_index = NeighborIndex(config.use_neighbor_index).fit(X_arr)
            model.neighbor_index_ = neighbor_index
            h = float(
                self._invoke(
                    "bandwidth_selector",
                    "select_initial",
                    X_arr,
                    centers_arr,
                    neighbor_index,
                )
            )
            h *= float(config.initial_bandwidth_inflation)
            state.h = h
            directions_arr = self._invoke(
                "direction_sampler",
                "prepare",
                centers_arr,
                d,
                directions,
            )
            state.directions = directions_arr
        except Exception:
            model._clear_pairwise_cache()
            model.backend.release_device_memory()
            raise

        history = state.history
        progress = state.progress
        beta_path: list[np.ndarray] = []
        timings: dict[str, float] = {}
        outer_telemetry: list[dict[str, Any]] = []
        local_telemetry: list[dict[str, Any]] = []
        fit_stop_reason = "scheduled_completion"
        intercepts = np.zeros(centers_arr.shape[0])
        slopes = np.ones(centers_arr.shape[0])
        statistics: LocalStatistics | None = None

        outer_iter = range(max(1, config.outer_steps))
        progress_bar = model._make_progress_bar(outer_iter)
        if progress_bar is not None:
            outer_iter = progress_bar

        try:
            for outer in outer_iter:
                step_started = time.perf_counter()
                beta_before_outer = np.asarray(beta_prev).copy()
                stage_timings_before = dict(self.stage_timings)
                stage_calls_before = dict(self.stage_calls)
                anisotropy: float | None = None
                if outer > 0:
                    h = max(h / config.bandwidth_decay, np.finfo(float).eps)
                    anisotropy = float(
                        self._invoke(
                            "bandwidth_selector",
                            "select_anisotropy",
                            X_arr,
                            centers_arr,
                            h,
                            beta_prev,
                            _outer=outer,
                        )
                    )
                    state.h = h
                    state.anisotropy = anisotropy
                    if config.renew_directions:
                        directions_arr = self._invoke(
                            "direction_sampler",
                            "prepare",
                            centers_arr,
                            d,
                            None,
                            beta=beta_prev,
                            anisotropy=anisotropy,
                            _outer=outer,
                        )
                        state.directions = directions_arr
                else:
                    state.anisotropy = None

                stats_started = time.perf_counter()
                bandwidth_update_time = stats_started - step_started
                statistics = self._invoke(
                    "statistics_builder",
                    "compute",
                    X_arr,
                    y_arr,
                    centers_arr,
                    h,
                    beta_prev,
                    directions_arr,
                    anisotropy,
                    _outer=outer,
                )
                state.statistics = statistics
                timings["statistics"] = (
                    timings.get("statistics", 0.0)
                    + time.perf_counter()
                    - stats_started
                )

                solve_started = time.perf_counter()
                beta_new, intercepts, slopes, inner_history = self._alternating_solve(
                    statistics,
                    beta_prev,
                    lambda_penalty,
                    outer,
                    step_started,
                    state=state,
                )
                solve_time = time.perf_counter() - solve_started
                timings["solve"] = (
                    timings.get("solve", 0.0)
                    + solve_time
                )
                beta_prev = beta_new
                state.beta = beta_prev
                state.intercepts = intercepts
                state.slopes = slopes
                beta_path.append(beta_prev.copy())

                if inner_history:
                    record = model._progress_record(
                        stats=statistics,
                        step=inner_history[-1],
                        outer_index=outer,
                        outer_total=max(1, config.outer_steps),
                        inner_count=len(inner_history),
                        started=started,
                    )
                    progress.append(record)
                    if progress_bar is not None and hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix(
                            format_progress_postfix(record), refresh=True
                        )

                should_stop = self._invoke(
                    "stop_rule",
                    "should_stop",
                    "outer",
                    state,
                    step=inner_history[-1] if inner_history else None,
                    anisotropy=anisotropy,
                    beta=beta_prev,
                    outer=outer,
                    _outer=outer,
                )
                if config.record_telemetry:
                    outer_row, local_rows = self._build_outer_telemetry(
                        statistics,
                        beta_before_outer,
                        beta_prev,
                        intercepts,
                        slopes,
                        inner_history,
                        outer=outer,
                        n_observations=int(X_arr.shape[0]),
                        iteration_started=step_started,
                        bandwidth_update_time=bandwidth_update_time,
                        optimization_time=solve_time,
                        stage_timings_before=stage_timings_before,
                        stage_calls_before=stage_calls_before,
                    )
                    outer_telemetry.append(outer_row)
                    local_telemetry.extend(local_rows)
                if should_stop:
                    fit_stop_reason = "tolerance"
                    break
        finally:
            if progress_bar is not None and hasattr(progress_bar, "close"):
                progress_bar.close()
            model._clear_pairwise_cache()
            model.backend.release_device_memory()

        if statistics is None:
            raise RuntimeError("fit не смог вычислить локальные статистики")

        objective = (
            history[-1].objective
            if history
            else model._objective(
                statistics,
                beta_prev,
                intercepts,
                slopes,
                beta_prev,
                lambda_penalty,
            )
        )
        if progress:
            progress[-1]["objective"] = float(objective)
        if not config.save_directions:
            directions_arr = None

        result = store_fit_result(
            model,
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
            beta_path,
        )
        result.stage_names = dict(self.stage_names)
        result.stage_timings = dict(self.stage_timings)
        result.stage_calls = dict(self.stage_calls)
        result.outer_telemetry = outer_telemetry
        result.local_telemetry = local_telemetry
        result.stop_reason = fit_stop_reason
        return result

    def _alternating_solve(
        self,
        stats: LocalStatistics,
        beta_start: np.ndarray,
        lambda_penalty: float,
        outer: int,
        outer_started: float,
        *,
        state: ADPState | None = None,
        use_protected_adapters: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TrainingStep]]:
        model = self.context.model
        config = self.context.config
        beta = unit_vector(beta_start)
        initial_prior = beta.copy()
        if state is None:
            state = ADPState(
                X=np.empty((0, initial_prior.size)),
                y=np.empty(0),
                centers=stats.centers,
                beta=beta,
                prior=initial_prior,
                h=float(stats.h),
                anisotropy=stats.anisotropy,
                statistics=stats,
            )
        history: list[TrainingStep] = []
        intercepts = np.zeros(stats.centers.shape[0])
        slopes = np.ones(stats.centers.shape[0])
        last_objective = math.inf
        objective_interval = max(1, int(config.objective_check_every))

        inner_steps = max(1, config.inner_steps)
        for inner in range(inner_steps):
            inner_started = time.perf_counter()
            old_beta = beta.copy()
            prior = old_beta.copy()
            old_intercepts = intercepts.copy()
            old_slopes = slopes.copy()
            should_check_objective = inner == 0 or inner % objective_interval == 0
            evaluated_objective_before = None
            if config.record_telemetry or (should_check_objective and inner > 0):
                evaluated_objective_before = float(
                    model._objective(
                        stats,
                        old_beta,
                        old_intercepts,
                        old_slopes,
                        prior,
                        lambda_penalty,
                    )
                )
            objective_before = (
                evaluated_objective_before if config.record_telemetry else None
            )
            model._last_solver_telemetry = None
            if use_protected_adapters:
                intercepts, slopes = model._solve_local_coefficients(stats, beta)
                intercepts, slopes = self._validate_local_solution(
                    stats,
                    intercepts,
                    slopes,
                    outer,
                    inner,
                    category="local_solver",
                )
                beta = model._solve_beta(
                    stats,
                    intercepts,
                    slopes,
                    prior,
                    lambda_penalty,
                    x0=beta,
                )
                beta = self.context.backend.asarray(beta)
                self._validate_beta(
                    beta, initial_prior.size, "beta_solver", outer, inner
                )
            else:
                intercepts, slopes = self._invoke(
                    "local_solver",
                    "solve",
                    stats,
                    beta,
                    _outer=outer,
                    _inner=inner,
                )
                beta = self._invoke(
                    "beta_solver",
                    "solve",
                    stats,
                    intercepts,
                    slopes,
                    prior,
                    lambda_penalty,
                    x0=beta,
                    _outer=outer,
                    _inner=inner,
                )
                beta = self.context.backend.asarray(beta)

            norm = stable_l2_norm(beta)
            beta = self.context.backend.asarray(unit_vector(beta))
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_slopes = slopes * norm
            if not np.all(np.isfinite(np.asarray(scaled_slopes))):
                raise StageExecutionError(
                    "beta_solver",
                    self.stage_names["beta_solver"],
                    "масштаб beta нельзя конечным образом перенести в slopes",
                    outer=outer,
                    inner=inner,
                )
            slopes = scaled_slopes
            evaluated_objective_after = None
            if config.record_telemetry or should_check_objective:
                evaluated_objective_after = float(
                    model._objective(
                        stats,
                        beta,
                        intercepts,
                        slopes,
                        prior,
                        lambda_penalty,
                    )
                )
            objective_after = (
                evaluated_objective_after if config.record_telemetry else None
            )

            objective_delta = math.inf
            if should_check_objective:
                assert evaluated_objective_after is not None
                objective = evaluated_objective_after
                if inner > 0:
                    assert evaluated_objective_before is not None
                    objective_delta = abs(
                        evaluated_objective_before - evaluated_objective_after
                    )
                last_objective = objective
            else:
                objective = last_objective
            beta_delta = float(
                min(
                    stable_l2_norm(beta - old_beta),
                    stable_l2_norm(beta + old_beta),
                )
            )
            solver_telemetry = model._last_solver_telemetry or {}
            history.append(
                TrainingStep(
                    outer=outer,
                    inner=inner,
                    objective=float(objective),
                    beta_delta=beta_delta,
                    h=float(stats.h),
                    anisotropy=stats.anisotropy,
                    elapsed=time.perf_counter() - outer_started,
                    objective_before=objective_before,
                    objective_after=objective_after,
                    pre_normalization_beta_norm=float(norm),
                    gradient_norm=_optional_float(
                        solver_telemetry.get("gradient_norm")
                    ),
                    linear_residual_norm=_optional_float(
                        solver_telemetry.get("linear_residual_norm")
                    ),
                    relative_linear_residual=_optional_float(
                        solver_telemetry.get("relative_linear_residual")
                    ),
                    linear_solver_iterations=_optional_int(
                        solver_telemetry.get("linear_solver_iterations")
                    ),
                    linear_solver_status=_optional_str(
                        solver_telemetry.get("linear_solver_status")
                    ),
                    intercept_change_mean=float(
                        np.mean(np.abs(intercepts - old_intercepts))
                    ),
                    slope_change_mean=float(
                        np.mean(np.abs(slopes - old_slopes))
                    ),
                    inner_iteration_time_sec=time.perf_counter() - inner_started,
                    beta=beta.copy() if config.record_telemetry else None,
                    solver_residual_trace=tuple(
                        float(value)
                        for value in solver_telemetry.get(
                            "solver_residual_trace",
                            (),
                        )
                    ),
                )
            )
            state.beta = beta
            state.prior = prior
            state.statistics = stats
            state.intercepts = intercepts
            state.slopes = slopes
            state.history.append(history[-1])
            should_stop = self._invoke(
                "stop_rule",
                "should_stop",
                "inner",
                state,
                step=history[-1],
                beta_delta=beta_delta,
                objective_delta=objective_delta,
                outer=outer,
                inner=inner,
                _outer=outer,
                _inner=inner,
            )
            if should_stop:
                history[-1].inner_stop_reason = "tolerance"
                break
            if inner == inner_steps - 1:
                history[-1].inner_stop_reason = "iteration_limit"

        if history and history[-1].inner % objective_interval != 0:
            history[-1].objective = float(
                model._objective(
                    stats, beta, intercepts, slopes, prior, lambda_penalty
                )
            )
        return unit_vector(beta), intercepts, slopes, history

    def _build_outer_telemetry(
        self,
        statistics: LocalStatistics,
        beta_before: np.ndarray,
        beta_after: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        inner_history: list[TrainingStep],
        *,
        outer: int,
        n_observations: int,
        iteration_started: float,
        bandwidth_update_time: float,
        optimization_time: float,
        stage_timings_before: dict[str, float],
        stage_calls_before: dict[str, int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        iteration_time = time.perf_counter() - iteration_started
        conditions, singular_count, local_rows = self._build_local_telemetry(
            statistics,
            beta_after,
            intercepts,
            slopes,
            outer=outer,
            n_observations=n_observations,
        )
        masses = (
            np.asarray(statistics.N, dtype=float)
            if statistics.N is not None
            else np.zeros(0, dtype=float)
        )
        sum_w2 = (
            np.asarray(statistics.weight_sum2, dtype=float)
            if statistics.weight_sum2 is not None
            else np.zeros_like(masses)
        )
        ess = np.divide(
            masses**2,
            sum_w2,
            out=np.zeros_like(masses),
            where=sum_w2 > 0.0,
        )
        nonzero = (
            np.asarray(statistics.weight_nonzero, dtype=int)
            if statistics.weight_nonzero is not None
            else np.zeros_like(masses, dtype=int)
        )
        # Prior меняется на каждом inner-шаге, поэтому сравнима только пара
        # objective до/после последнего proximal-шага с одним и тем же prior.
        last_proximal_step = inner_history[-1] if inner_history else None
        objective_before = (
            last_proximal_step.objective_before
            if last_proximal_step is not None
            else None
        )
        objective_after = (
            last_proximal_step.objective_after
            if last_proximal_step is not None
            else None
        )
        relative_objective_decrease = math.nan
        if objective_before is not None and objective_after is not None:
            relative_objective_decrease = (
                float(objective_before) - float(objective_after)
            ) / max(abs(float(objective_before)), float(np.finfo(float).eps))
        beta_before_values = np.asarray(beta_before, dtype=float).reshape(-1)
        beta_after_values = np.asarray(beta_after, dtype=float).reshape(-1)
        beta_before_unit = unit_vector(beta_before_values)
        beta_after_unit = unit_vector(beta_after_values)
        adjacent_cosine = float(
            np.clip(abs(beta_before_unit @ beta_after_unit), 0.0, 1.0)
        )
        distance_time = float(statistics.distance_time_sec)
        weights_time = float(statistics.weights_time_sec)
        statistics_time = float(statistics.statistics_time_sec)
        service_overhead = max(
            0.0,
            iteration_time
            - bandwidth_update_time
            - distance_time
            - weights_time
            - statistics_time
            - optimization_time,
        )
        return (
            {
                "outer": int(outer),
                "h": float(statistics.h),
                "anisotropy": statistics.anisotropy,
                "beta": beta_after_values.copy(),
                "beta_norm": stable_l2_norm(beta_after_values),
                "beta_delta": float(
                    min(
                        stable_l2_norm(beta_after_unit - beta_before_unit),
                        stable_l2_norm(beta_after_unit + beta_before_unit),
                    )
                ),
                "adjacent_angle_rad": float(math.acos(adjacent_cosine)),
                "objective_before": objective_before,
                "objective_after": objective_after,
                "relative_objective_decrease": relative_objective_decrease,
                "inner_iterations": len(inner_history),
                "local_mass_mean": _array_stat(masses, np.mean),
                "local_mass_min": _array_stat(masses, np.min),
                "local_mass_q05": _array_quantile(masses, 0.05),
                "local_mass_median": _array_quantile(masses, 0.5),
                "local_mass_q95": _array_quantile(masses, 0.95),
                "ess_mean": _array_stat(ess, np.mean),
                "ess_min": _array_stat(ess, np.min),
                "condition_median": _array_quantile(conditions, 0.5),
                "condition_max": _array_stat(conditions, np.max),
                "singular_centers": int(singular_count),
                "zero_weight_fraction": (
                    float(np.mean(nonzero == 0)) if nonzero.size else math.nan
                ),
                "bandwidth_update_time_sec": float(bandwidth_update_time),
                "distance_time_sec": distance_time,
                "weights_time_sec": weights_time,
                "statistics_time_sec": statistics_time,
                "optimization_time_sec": float(optimization_time),
                "iteration_time_sec": float(iteration_time),
                "service_overhead_sec": service_overhead,
                "stage_timings": {
                    category: float(
                        elapsed - stage_timings_before.get(category, 0.0)
                    )
                    for category, elapsed in self.stage_timings.items()
                },
                "stage_calls": {
                    category: int(calls - stage_calls_before.get(category, 0))
                    for category, calls in self.stage_calls.items()
                },
            },
            local_rows,
        )

    def _build_local_telemetry(
        self,
        statistics: LocalStatistics,
        beta: np.ndarray,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        *,
        outer: int,
        n_observations: int,
    ) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
        if statistics.S is None or statistics.U is None:
            return np.zeros(0, dtype=float), 0, []
        s_values = np.asarray(statistics.S)
        u_values = np.asarray(statistics.U)
        responses = np.asarray(statistics.imav)
        beta_values = np.asarray(beta).reshape(-1)
        projected = u_values @ beta_values
        dtype = np.result_type(s_values.dtype, u_values.dtype, responses.dtype)
        eps = float(np.finfo(dtype).eps)
        regularization = float(self.context.config.ridge)
        record_rows = self.context.config.record_local_trace
        conditions = np.empty(s_values.shape[0], dtype=float)
        singular_count = 0
        rows: list[dict[str, Any]] = []
        if record_rows:
            masses = (
                np.asarray(statistics.N, dtype=float)
                if statistics.N is not None
                else np.zeros(s_values.shape[0], dtype=float)
            )
            sum_w2 = (
                np.asarray(statistics.weight_sum2, dtype=float)
                if statistics.weight_sum2 is not None
                else np.zeros_like(masses)
            )
            ess = np.divide(
                masses**2,
                sum_w2,
                out=np.zeros_like(masses),
                where=sum_w2 > 0.0,
            )
            nonzero = (
                np.asarray(statistics.weight_nonzero, dtype=int)
                if statistics.weight_nonzero is not None
                else np.zeros_like(masses, dtype=int)
            )
            min_weight = _optional_vector(statistics.min_weight, masses.size)
            max_weight = _optional_vector(statistics.max_weight, masses.size)
        for center_index in range(s_values.shape[0]):
            s_row = s_values[center_index]
            u_row = projected[center_index]
            response = responses[center_index]
            cross = float(np.dot(s_row, u_row))
            system = np.array(
                [
                    [np.dot(s_row, s_row), cross],
                    [cross, np.dot(u_row, u_row)],
                ],
                dtype=dtype,
            )
            eigenvalues = np.linalg.eigvalsh(system)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            threshold = 2.0 * eps * max(lambda_max, 1.0)
            rank = int(np.count_nonzero(eigenvalues > threshold))
            singular = bool(rank < 2 or lambda_min <= threshold)
            condition = (
                math.inf if singular else float(lambda_max / lambda_min)
            )
            conditions[center_index] = condition
            singular_count += int(singular)
            if not record_rows:
                continue
            fitted = (
                float(intercepts[center_index]) * s_row
                + float(slopes[center_index]) * u_row
            )
            rows.append(
                {
                    "outer": int(outer),
                    "center": int(center_index),
                    "local_mass": float(masses[center_index]),
                    "ess": float(ess[center_index]),
                    "nonzero_weights": int(nonzero[center_index]),
                    "support_fraction": float(nonzero[center_index])
                    / max(1, n_observations),
                    "min_weight": float(min_weight[center_index]),
                    "max_weight": float(max_weight[center_index]),
                    "intercept": float(intercepts[center_index]),
                    "slope": float(slopes[center_index]),
                    "determinant": float(np.linalg.det(system)),
                    "lambda_min": lambda_min,
                    "lambda_max": lambda_max,
                    "condition": condition,
                    "rank": rank,
                    "residual": stable_l2_norm(response - fitted),
                    "regularization": regularization,
                    "singular": singular,
                }
            )
        return conditions, singular_count, rows

    def _invoke(
        self,
        category: str,
        method: str,
        *args: Any,
        _outer: int | None = None,
        _inner: int | None = None,
        **kwargs: Any,
    ) -> Any:
        component = self.components[category]
        started = time.perf_counter()
        try:
            result = getattr(component, method)(*args, **kwargs)
            return self._validate_stage_output(
                category,
                method,
                result,
                args,
                outer=_outer,
                inner=_inner,
            )
        except StageExecutionError:
            raise
        except Exception as exc:
            raise StageExecutionError(
                category,
                self.stage_names[category],
                str(exc),
                outer=_outer,
                inner=_inner,
            ) from exc
        finally:
            self.stage_timings[category] += time.perf_counter() - started
            self.stage_calls[category] += 1

    def _validate_stage_output(
        self,
        category: str,
        method: str,
        result: Any,
        args: tuple[Any, ...],
        *,
        outer: int | None,
        inner: int | None,
    ) -> Any:
        if category == "beta_initializer":
            d = int(np.asarray(args[0]).shape[1])
            self._validate_beta(result, d, category, outer, inner)
            return np.asarray(result)

        if category == "center_selector":
            centers = np.asarray(result)
            d = int(np.asarray(args[0]).shape[1])
            if (
                centers.ndim != 2
                or centers.shape[0] == 0
                or centers.shape[1] != d
                or not np.all(np.isfinite(centers))
            ):
                self._raise_invalid_stage_output(
                    category,
                    f"ожидалась конечная непустая матрица центров с d={d}, "
                    f"получено {centers.shape}",
                    outer,
                    inner,
                )
            return self.context.backend.asarray(centers)

        if category == "bandwidth_selector":
            value = float(result)
            valid = np.isfinite(value)
            if method == "select_initial":
                valid = bool(valid and value > 0.0)
            else:
                valid = bool(valid and 0.0 <= value <= 1.0)
            if not valid:
                expected = "положительный bandwidth" if method == "select_initial" else "rho в [0, 1]"
                self._raise_invalid_stage_output(
                    category,
                    f"ожидался {expected}, получено {value}",
                    outer,
                    inner,
                )
            return value

        if category == "direction_sampler":
            if result is None:
                self._raise_invalid_stage_output(
                    category,
                    "new-вариант требует массив направлений",
                    outer,
                    inner,
                )
            directions = np.asarray(result)
            centers, d = np.asarray(args[0]), int(args[1])
            expected = (centers.shape[0], self.context.config.n_directions, d)
            if directions.shape != expected or not np.all(np.isfinite(directions)):
                self._raise_invalid_stage_output(
                    category,
                    f"ожидался конечный массив формы {expected}, получено {directions.shape}",
                    outer,
                    inner,
                )
            return self.context.backend.asarray(directions)

        if category == "statistics_builder":
            if not isinstance(result, LocalStatistics):
                self._raise_invalid_stage_output(
                    category,
                    f"ожидался LocalStatistics, получено {type(result).__name__}",
                    outer,
                    inner,
                )
            centers, directions = np.asarray(args[2]), np.asarray(args[5])
            J, P, d = centers.shape[0], directions.shape[1], centers.shape[1]
            arrays = {
                "imav": (result.imav, (J, P)),
                "S": (result.S, (J, P)),
                "U": (result.U, (J, P, d)),
                "N": (result.N, (J,)),
            }
            for name, (value, expected) in arrays.items():
                value_arr = np.asarray(value) if value is not None else np.asarray([])
                if value_arr.shape != expected or not np.all(np.isfinite(value_arr)):
                    self._raise_invalid_stage_output(
                        category,
                        f"{name} должен иметь конечные значения и форму {expected}; "
                        f"получено {value_arr.shape}",
                        outer,
                        inner,
                    )
            return result

        if category == "local_solver":
            if not isinstance(result, tuple) or len(result) != 2:
                self._raise_invalid_stage_output(
                    category,
                    "ожидался кортеж (intercepts, slopes)",
                    outer,
                    inner,
                )
            return self._validate_local_solution(
                args[0],
                result[0],
                result[1],
                outer if outer is not None else -1,
                inner if inner is not None else -1,
                category=category,
            )

        if category == "beta_solver":
            d = int(np.asarray(args[3]).size)
            self._validate_beta(result, d, category, outer, inner)
            return self.context.backend.asarray(result)

        if category == "stop_rule":
            if not isinstance(result, (bool, np.bool_)):
                self._raise_invalid_stage_output(
                    category,
                    f"ожидался bool, получено {type(result).__name__}",
                    outer,
                    inner,
                )
            return bool(result)

        return result

    def _validate_beta(
        self,
        beta: np.ndarray,
        d: int,
        category: str,
        outer: int | None = None,
        inner: int | None = None,
    ) -> None:
        beta_arr = np.asarray(beta)
        if (
            beta_arr.shape != (d,)
            or not np.all(np.isfinite(beta_arr))
            or stable_l2_norm(beta_arr) == 0.0
        ):
            raise StageExecutionError(
                category,
                self.stage_names[category],
                f"ожидался конечный ненулевой вектор формы {(d,)}, получено {beta_arr.shape}",
                outer=outer,
                inner=inner,
            )

    def _validate_user_beta(self, beta: np.ndarray, d: int, name: str) -> None:
        beta_arr = np.asarray(beta)
        if (
            beta_arr.shape != (d,)
            or not np.all(np.isfinite(beta_arr))
            or stable_l2_norm(beta_arr) == 0.0
        ):
            raise ValueError(
                f"{name} должен быть конечным ненулевым вектором формы {(d,)}, "
                f"получено {beta_arr.shape}"
            )

    def _raise_invalid_stage_output(
        self,
        category: str,
        message: str,
        outer: int | None,
        inner: int | None,
    ) -> None:
        raise StageExecutionError(
            category,
            self.stage_names[category],
            message,
            outer=outer,
            inner=inner,
        )

    def _validate_local_solution(
        self,
        statistics: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        outer: int,
        inner: int,
        *,
        category: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        intercepts_arr = np.asarray(intercepts)
        slopes_arr = np.asarray(slopes)
        expected = (statistics.centers.shape[0],)
        if (
            intercepts_arr.shape != expected
            or slopes_arr.shape != expected
            or not np.all(np.isfinite(intercepts_arr))
            or not np.all(np.isfinite(slopes_arr))
        ):
            raise StageExecutionError(
                category,
                self.stage_names[category],
                "intercepts и slopes должны быть конечными векторами "
                f"формы {expected}; получено {intercepts_arr.shape} и {slopes_arr.shape}",
                outer=outer,
                inner=inner,
            )
        return intercepts_arr, slopes_arr

    def _validate_categories(self, values: Mapping[str, Any]) -> None:
        unknown = sorted(set(values) - set(STAGE_METHODS))
        if unknown:
            available = ", ".join(STAGE_METHODS)
            raise ValueError(
                f"Неизвестный этап {unknown[0]!r}; доступны: {available}"
            )

    def _validate_component(self, category: str, name: str, component: Any) -> None:
        missing = [
            method
            for method in STAGE_METHODS[category]
            if not callable(getattr(component, method, None))
        ]
        if missing:
            methods = ", ".join(missing)
            raise TypeError(
                f"Фабрика {category!r} ({name!r}) вернула объект без методов: {methods}"
            )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _array_stat(values: np.ndarray, function: Any) -> float:
    return float(function(values)) if values.size else math.nan


def _array_quantile(values: np.ndarray, quantile: float) -> float:
    if not values.size:
        return math.nan
    ordered = np.sort(np.asarray(values, dtype=float))
    position = (ordered.size - 1) * float(quantile)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    if lower_index == upper_index or lower == upper:
        return lower
    if math.isinf(lower):
        return lower
    if math.isinf(upper):
        return upper
    fraction = position - lower_index
    return lower + fraction * (upper - lower)


def _optional_vector(values: np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, math.nan, dtype=float)
    return np.asarray(values, dtype=float)
