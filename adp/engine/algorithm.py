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
from ..common.utils import as_1d_float, as_2d_float, unit_vector
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
                timings["solve"] = (
                    timings.get("solve", 0.0)
                    + time.perf_counter()
                    - solve_started
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

                if self._invoke(
                    "stop_rule",
                    "should_stop",
                    "outer",
                    state,
                    step=inner_history[-1] if inner_history else None,
                    anisotropy=anisotropy,
                    beta=beta_prev,
                    outer=outer,
                    _outer=outer,
                ):
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
        prior = beta.copy()
        if state is None:
            state = ADPState(
                X=np.empty((0, prior.size)),
                y=np.empty(0),
                centers=stats.centers,
                beta=beta,
                prior=prior,
                h=float(stats.h),
                anisotropy=stats.anisotropy,
                statistics=stats,
            )
        history: list[TrainingStep] = []
        intercepts = np.zeros(stats.centers.shape[0])
        slopes = np.ones(stats.centers.shape[0])
        last_objective = math.inf
        objective_interval = max(1, int(config.objective_check_every))

        for inner in range(max(1, config.inner_steps)):
            old_beta = beta.copy()
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
                self._validate_beta(beta, prior.size, "beta_solver", outer, inner)
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

            norm = np.linalg.norm(beta)
            beta = beta / norm
            slopes = slopes * norm

            should_check_objective = inner == 0 or inner % objective_interval == 0
            objective_delta = math.inf
            if should_check_objective:
                objective = model._objective(
                    stats, beta, intercepts, slopes, prior, lambda_penalty
                )
                objective_delta = abs(last_objective - objective)
                last_objective = objective
            else:
                objective = last_objective
            beta_delta = float(
                min(
                    np.linalg.norm(beta - old_beta),
                    np.linalg.norm(beta + old_beta),
                )
            )
            history.append(
                TrainingStep(
                    outer=outer,
                    inner=inner,
                    objective=float(objective),
                    beta_delta=beta_delta,
                    h=float(stats.h),
                    anisotropy=stats.anisotropy,
                    elapsed=time.perf_counter() - outer_started,
                )
            )
            state.beta = beta
            state.prior = prior
            state.statistics = stats
            state.intercepts = intercepts
            state.slopes = slopes
            state.history.append(history[-1])
            if self._invoke(
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
            ):
                break

        if history and history[-1].inner % objective_interval != 0:
            history[-1].objective = float(
                model._objective(
                    stats, beta, intercepts, slopes, prior, lambda_penalty
                )
            )
        return unit_vector(beta), intercepts, slopes, history

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
            or np.linalg.norm(beta_arr) == 0.0
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
            or np.linalg.norm(beta_arr) == 0.0
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
