from __future__ import annotations

import math
import time
import traceback
from dataclasses import asdict, fields
from typing import Any, Mapping

import numpy as np
from threadpoolctl import threadpool_limits

from ...common.experiment_log import Scalar
from ...common.types import ADPConfig, ADPResult, TrainingStep
from ...engine.base import ADP
from .datasets import GeneratedSingleIndexData, generate_synthetic_data
from .schema import ALGORITHM_RESOURCE_COLUMNS
from .telemetry import encode_beta
from .types import RunOutcome, SingleIndexJob, SingleIndexSeriesConfig


_NUMERICAL_EXCEPTIONS = (
    ArithmeticError,
    RuntimeError,
    ValueError,
    np.linalg.LinAlgError,
)


def execute_job(
    job: SingleIndexJob,
    config: SingleIndexSeriesConfig,
) -> RunOutcome:
    """Execute one deterministic ADP fit and normalize its diagnostics."""

    job_started = time.perf_counter()
    generated: GeneratedSingleIndexData | None = None
    result: ADPResult | None = None
    caught: Exception | None = None
    caught_traceback = ""
    resource_usage: dict[str, Scalar] = {}
    data_generation_time_sec = math.nan
    fit_wall_time_sec = math.nan
    adp_config = _benchmark_adp_config(job)
    try:
        generation_started = time.perf_counter()
        try:
            generated = generate_synthetic_data(job)
        finally:
            data_generation_time_sec = time.perf_counter() - generation_started

        model = ADP.create(
            "new",
            adp_config,
            stages={
                "statistics_builder": job.parameters.statistics_builder,
            },
        )
        fit_started = time.perf_counter()
        try:
            result = _fit_adp(model, generated)
        finally:
            fit_wall_time_sec = time.perf_counter() - fit_started
        resource_usage = dict(result.resource_usage)
    except Exception as exc:
        if not isinstance(exc, _NUMERICAL_EXCEPTIONS):
            raise
        caught = exc
        caught_traceback = traceback.format_exc()
        partial = getattr(exc, "partial_result", None)
        if isinstance(partial, ADPResult):
            result = partial
        partial_usage = getattr(exc, "partial_resource_usage", None)
        if isinstance(partial_usage, Mapping):
            resource_usage = dict(partial_usage)
        elif result is not None:
            resource_usage = dict(result.resource_usage)

    serialization_started = time.perf_counter()
    outer_rows = _outer_rows(job, result, generated)
    inner_rows = _inner_rows(job, result, generated)
    all_local_rows = _local_rows(job, result)
    solver_rows = _solver_rows(job, result)
    invalid_value_count = _invalid_value_count(result)
    status, stop_reason = _classify_status(
        result,
        caught,
        invalid_value_count=invalid_value_count,
    )
    local_rows = (
        all_local_rows
        if job.diagnostic or status != "success"
        else ()
    )
    run_row = _run_row(
        job,
        generated,
        result,
        adp_config=adp_config,
        resource_usage=resource_usage,
        status=status,
        stop_reason=stop_reason,
        invalid_value_count=invalid_value_count,
        error=caught,
        error_traceback=caught_traceback,
        data_generation_time_sec=data_generation_time_sec,
        fit_wall_time_sec=fit_wall_time_sec,
        outer_rows=outer_rows,
        inner_rows=inner_rows,
        local_rows=all_local_rows,
        solver_rows=solver_rows,
    )
    run_row["telemetry_serialization_time_sec"] = (
        time.perf_counter() - serialization_started
    )
    run_row["job_wall_time_sec"] = time.perf_counter() - job_started
    return RunOutcome(
        run_row=run_row,
        outer_rows=outer_rows,
        inner_rows=inner_rows,
        local_rows=local_rows,
        solver_rows=solver_rows,
    )


def _fit_adp(
    model: Any,
    generated: GeneratedSingleIndexData,
) -> ADPResult:
    try:
        with threadpool_limits(limits=1):
            return model.fit(
                generated.data.X,
                generated.data.y,
                centers=generated.data.centers,
                directions=generated.data.directions,
            )
    except _NUMERICAL_EXCEPTIONS as exc:
        if model.result_ is not None:
            setattr(exc, "partial_result", model.result_)
        usage = getattr(model, "last_resource_usage_", None)
        if isinstance(usage, Mapping):
            setattr(exc, "partial_resource_usage", dict(usage))
        raise


def _benchmark_adp_config(job: SingleIndexJob) -> ADPConfig:
    return ADPConfig(
        n_centers=job.parameters.n_centers,
        n_directions=max(4, min(job.parameters.d, 32)),
        statistics_workers=1,
        show_progress=False,
        record_telemetry=True,
        record_local_trace=job.diagnostic,
        record_solver_trace=job.diagnostic,
        random_state=job.seeds.init,
    )


def _run_row(
    job: SingleIndexJob,
    generated: GeneratedSingleIndexData | None,
    result: ADPResult | None,
    *,
    adp_config: ADPConfig,
    resource_usage: Mapping[str, Scalar],
    status: str,
    stop_reason: str,
    invalid_value_count: int,
    error: Exception | None,
    error_traceback: str,
    data_generation_time_sec: float,
    fit_wall_time_sec: float,
    outer_rows: tuple[dict[str, Scalar], ...],
    inner_rows: tuple[dict[str, Scalar], ...],
    local_rows: tuple[dict[str, Scalar], ...],
    solver_rows: tuple[dict[str, Scalar], ...],
) -> dict[str, Scalar]:
    parameters = job.parameters
    values: dict[str, Scalar] = {
        "run_id": job.run_id,
        "experiment": job.experiment,
        "seed": job.seed,
        "diagnostic": job.diagnostic,
        **asdict(parameters),
        "n": parameters.n,
        "n_centers": parameters.n_centers,
        "n_directions": max(4, min(parameters.d, 32)),
        "statistics_workers": 1,
        **{
            f"adp_{name}": value
            for name, value in asdict(adp_config).items()
        },
    }
    for seed_field in fields(job.seeds):
        values[f"seed_{seed_field.name}"] = int(
            getattr(job.seeds, seed_field.name)
        )
    if generated is not None:
        values.update(generated.metadata)

    beta = None if result is None else np.asarray(result.beta)
    truth = None if generated is None else generated.data.beta
    cosine_abs, projector_error = _truth_metrics(beta, truth)
    first_outer = outer_rows[0] if outer_rows else {}
    last_outer = outer_rows[-1] if outer_rows else {}
    values.update(
        {
            "h_initial": first_outer.get("h_k", math.nan),
            "h_final": last_outer.get("h_k", math.nan),
            "rho_final": last_outer.get("rho_k", math.nan),
            "outer_iterations": len(outer_rows),
            "inner_iterations_total": len(inner_rows),
            "cosine_abs": cosine_abs,
            "projector_error": projector_error,
            "objective": (
                math.nan if result is None else _safe_float(result.objective)
            ),
            "data_generation_time_sec": float(data_generation_time_sec),
            "fit_wall_time_sec": float(fit_wall_time_sec),
            "statistics_builder_time_sec": (
                math.nan
                if result is None
                else _safe_float(
                    result.stage_timings.get("statistics_builder")
                )
            ),
            "statistics_builder_calls": (
                0
                if result is None
                else int(result.stage_calls.get("statistics_builder", 0))
            ),
            "singular_local_count": sum(
                bool(row.get("is_singular", False)) for row in local_rows
            ),
            "invalid_value_count": int(invalid_value_count),
            "stop_reason": stop_reason,
            "status": status,
            "error_type": "" if error is None else type(error).__name__,
            "error_message": "" if error is None else str(error),
            "error_traceback": error_traceback,
            "outer_row_count": len(outer_rows),
            "inner_row_count": len(inner_rows),
            "local_row_count": len(local_rows),
            "solver_row_count": len(solver_rows),
        }
    )
    for column in ALGORITHM_RESOURCE_COLUMNS:
        values[column] = resource_usage.get(
            column,
            "" if column == "algorithm_memory_source" else math.nan,
        )
    return values


def _outer_rows(
    job: SingleIndexJob,
    result: ADPResult | None,
    generated: GeneratedSingleIndexData | None,
) -> tuple[dict[str, Scalar], ...]:
    if result is None:
        return ()
    truth = None if generated is None else generated.data.beta
    rows: list[dict[str, Scalar]] = []
    for telemetry in result.outer_telemetry:
        beta = np.asarray(telemetry.get("beta", np.array([])))
        cosine_abs, projector_error = _truth_metrics(beta, truth)
        row: dict[str, Scalar] = {
            "run_id": job.run_id,
            "experiment": job.experiment,
            "seed": job.seed,
            "outer_k": int(telemetry.get("outer", len(rows))),
            "h_k": _safe_float(telemetry.get("h")),
            "rho_k": _safe_float(telemetry.get("anisotropy")),
            "beta_k": _encode_optional_beta(beta),
            "beta_norm": _safe_float(telemetry.get("beta_norm")),
            "cosine_abs": cosine_abs,
            "projector_error": projector_error,
            "beta_delta": _safe_float(telemetry.get("beta_delta")),
            "adjacent_angle_rad": _safe_float(
                telemetry.get("adjacent_angle_rad")
            ),
            "objective_before": _safe_float(
                telemetry.get("objective_before")
            ),
            "objective_after": _safe_float(telemetry.get("objective_after")),
            "relative_objective_decrease": _safe_float(
                telemetry.get("relative_objective_decrease")
            ),
            "inner_iterations": int(telemetry.get("inner_iterations", 0)),
            "local_mass_mean": _safe_float(telemetry.get("local_mass_mean")),
            "local_mass_min": _safe_float(telemetry.get("local_mass_min")),
            "local_mass_q05": _safe_float(telemetry.get("local_mass_q05")),
            "local_mass_median": _safe_float(
                telemetry.get("local_mass_median")
            ),
            "local_mass_q95": _safe_float(telemetry.get("local_mass_q95")),
            "ess_mean": _safe_float(telemetry.get("ess_mean")),
            "ess_min": _safe_float(telemetry.get("ess_min")),
            "condition_median": _safe_float(
                telemetry.get("condition_median")
            ),
            "condition_max": _safe_float(telemetry.get("condition_max")),
            "singular_centers": int(telemetry.get("singular_centers", 0)),
            "zero_weight_fraction": _safe_float(
                telemetry.get("zero_weight_fraction")
            ),
            "bandwidth_update_time_sec": _safe_float(
                telemetry.get("bandwidth_update_time_sec")
            ),
            "distance_time_sec": _safe_float(
                telemetry.get("distance_time_sec")
            ),
            "weights_time_sec": _safe_float(telemetry.get("weights_time_sec")),
            "statistics_time_sec": _safe_float(
                telemetry.get("statistics_time_sec")
            ),
            "optimization_time_sec": _safe_float(
                telemetry.get("optimization_time_sec")
            ),
            "iteration_time_sec": _safe_float(
                telemetry.get("iteration_time_sec")
            ),
            "service_overhead_sec": _safe_float(
                telemetry.get("service_overhead_sec")
            ),
        }
        _append_stage_columns(row, telemetry)
        rows.append(row)
    return tuple(rows)


def _inner_rows(
    job: SingleIndexJob,
    result: ADPResult | None,
    generated: GeneratedSingleIndexData | None,
) -> tuple[dict[str, Scalar], ...]:
    if result is None:
        return ()
    truth = None if generated is None else generated.data.beta
    rows: list[dict[str, Scalar]] = []
    for step in result.history:
        beta = None if step.beta is None else np.asarray(step.beta)
        cosine_abs, projector_error = _truth_metrics(beta, truth)
        objective_before = _safe_float(step.objective_before)
        objective_after = _safe_float(step.objective_after)
        relative_change = math.nan
        if math.isfinite(objective_before) and math.isfinite(objective_after):
            relative_change = abs(objective_after - objective_before) / max(
                abs(objective_before),
                float(np.finfo(float).eps),
            )
        rows.append(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": int(step.outer),
                "inner_k": int(step.inner),
                "objective": (
                    objective_after
                    if math.isfinite(objective_after)
                    else _safe_float(step.objective)
                ),
                "objective_before": objective_before,
                "objective_after": objective_after,
                "relative_objective_change": relative_change,
                "beta_delta": _safe_float(step.beta_delta),
                "pre_normalization_beta_norm": _safe_float(
                    step.pre_normalization_beta_norm
                ),
                "cosine_abs": cosine_abs,
                "projector_error": projector_error,
                "gradient_norm": _safe_float(step.gradient_norm),
                "linear_residual_norm": _safe_float(
                    step.linear_residual_norm
                ),
                "relative_linear_residual": _safe_float(
                    step.relative_linear_residual
                ),
                "linear_solver_iterations": (
                    None
                    if step.linear_solver_iterations is None
                    else int(step.linear_solver_iterations)
                ),
                "linear_solver_status": step.linear_solver_status,
                "intercept_change_mean": _safe_float(
                    step.intercept_change_mean
                ),
                "slope_change_mean": _safe_float(step.slope_change_mean),
                "inner_iteration_time_sec": _safe_float(
                    step.inner_iteration_time_sec
                ),
            }
        )
    return tuple(rows)


def _local_rows(
    job: SingleIndexJob,
    result: ADPResult | None,
) -> tuple[dict[str, Scalar], ...]:
    if result is None:
        return ()
    rows: list[dict[str, Scalar]] = []
    for telemetry in result.local_telemetry:
        rows.append(
            {
                "run_id": job.run_id,
                "experiment": job.experiment,
                "seed": job.seed,
                "outer_k": int(telemetry.get("outer", 0)),
                "center_j": int(telemetry.get("center", 0)),
                "local_mass": _safe_float(telemetry.get("local_mass")),
                "ess": _safe_float(telemetry.get("ess")),
                "nonzero_weights": int(telemetry.get("nonzero_weights", 0)),
                "support_fraction": _safe_float(
                    telemetry.get("support_fraction")
                ),
                "min_weight": _safe_float(telemetry.get("min_weight")),
                "max_weight": _safe_float(telemetry.get("max_weight")),
                "intercept": _safe_float(telemetry.get("intercept")),
                "slope": _safe_float(telemetry.get("slope")),
                "determinant": _safe_float(telemetry.get("determinant")),
                "lambda_min": _safe_float(telemetry.get("lambda_min")),
                "lambda_max": _safe_float(telemetry.get("lambda_max")),
                "condition": _safe_float(telemetry.get("condition")),
                "rank": int(telemetry.get("rank", 0)),
                "residual": _safe_float(telemetry.get("residual")),
                "regularization": _safe_float(
                    telemetry.get("regularization")
                ),
                "is_singular": bool(telemetry.get("singular", False)),
            }
        )
    return tuple(rows)


def _solver_rows(
    job: SingleIndexJob,
    result: ADPResult | None,
) -> tuple[dict[str, Scalar], ...]:
    if result is None:
        return ()
    rows: list[dict[str, Scalar]] = []
    for step in result.history:
        for solver_k, residual in enumerate(step.solver_residual_trace, start=1):
            rows.append(
                {
                    "run_id": job.run_id,
                    "experiment": job.experiment,
                    "seed": job.seed,
                    "outer_k": int(step.outer),
                    "inner_k": int(step.inner),
                    "solver_k": solver_k,
                    "relative_residual": float(residual),
                }
            )
    return tuple(rows)


def _classify_status(
    result: ADPResult | None,
    error: Exception | None,
    *,
    invalid_value_count: int,
) -> tuple[str, str]:
    if error is not None:
        return "numerical_failure", "numerical_exception"
    if result is None or not _valid_direction(result.beta) or invalid_value_count:
        return "numerical_failure", "invalid_numerical_result"

    solver_statuses = {
        step.linear_solver_status
        for step in result.history
        if step.linear_solver_status is not None
    }
    if solver_statuses & {"breakdown", "invalid_solution"}:
        return "numerical_failure", "linear_solver_failure"
    if "max_iterations" in solver_statuses:
        return "nonconverged", "linear_iteration_limit"

    if any(
        step.inner_stop_reason == "iteration_limit"
        for step in result.history
    ):
        return "nonconverged", "alternating_iteration_limit"
    return "success", result.stop_reason


def _invalid_value_count(result: ADPResult | None) -> int:
    if result is None:
        return 0
    values: list[Any] = [result.beta, result.objective]
    for telemetry in result.outer_telemetry:
        values.extend(
            telemetry.get(name)
            for name in (
                "beta",
                "h",
                "beta_norm",
                "beta_delta",
                "objective_before",
                "objective_after",
                "local_mass_mean",
                "ess_mean",
                "iteration_time_sec",
            )
        )
    for step in result.history:
        values.extend(
            (
                step.objective,
                step.objective_before,
                step.objective_after,
                step.beta,
                step.pre_normalization_beta_norm,
                step.linear_residual_norm,
                step.relative_linear_residual,
            )
        )
    return sum(_count_nonfinite(value) for value in values if value is not None)


def _truth_metrics(
    beta: np.ndarray | None,
    beta_true: np.ndarray | None,
) -> tuple[float, float]:
    if beta is None or beta_true is None:
        return math.nan, math.nan
    estimate = np.asarray(beta, dtype=float).reshape(-1)
    truth = np.asarray(beta_true, dtype=float).reshape(-1)
    if estimate.shape != truth.shape or not _valid_direction(estimate):
        return math.nan, math.nan
    if not _valid_direction(truth):
        return math.nan, math.nan
    estimate = estimate / np.linalg.norm(estimate)
    truth = truth / np.linalg.norm(truth)
    cosine_abs = float(np.clip(abs(estimate @ truth), 0.0, 1.0))
    projector_error = float(
        np.linalg.norm(
            np.outer(estimate, estimate) - np.outer(truth, truth),
            ord="fro",
        )
    )
    return cosine_abs, projector_error


def _append_stage_columns(
    row: dict[str, Scalar],
    telemetry: Mapping[str, Any],
) -> None:
    for stage, elapsed in dict(telemetry.get("stage_timings", {})).items():
        row[f"stage_{stage}_time_sec"] = _safe_float(elapsed)
    for stage, calls in dict(telemetry.get("stage_calls", {})).items():
        row[f"stage_{stage}_calls"] = int(calls)


def _encode_optional_beta(beta: np.ndarray) -> str:
    values = np.asarray(beta)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return ""
    return encode_beta(values)


def _valid_direction(beta: Any) -> bool:
    values = np.asarray(beta, dtype=float).reshape(-1)
    return bool(
        values.size
        and np.all(np.isfinite(values))
        and np.linalg.norm(values) > np.finfo(float).eps
    )


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _count_nonfinite(value: Any) -> int:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return 1
    return int(np.count_nonzero(~np.isfinite(array)))


__all__ = ["RunOutcome", "execute_job"]
