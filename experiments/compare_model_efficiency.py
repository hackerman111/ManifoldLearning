from __future__ import annotations

# One process per fit is the unit of isolation. Keep BLAS single-threaded so the
# comparison measures implementations rather than different thread scheduling.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import multiprocessing as mp
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "adp-model-comparison-matplotlib"),
)

import matplotlib.pyplot as plt
import cloudpickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adp import ADP, ADPConfig
from adp.common.resource_monitor import ResourceMonitor
from adp.evaluation.single_index.datasets import generate_synthetic_data
from adp.evaluation.single_index.runner import _build_single_index_job
from adp.evaluation.single_index.scenarios import (
    full_parameter_grid,
    parse_seed_selection,
    smoke_parameter_grid,
)
from adp.evaluation.single_index.types import ExperimentParameters


_PAIR_KEYS = ("case_id", "seed", "d", "n", "n_over_d")
_PAIR_CONTEXT_KEYS = (*_PAIR_KEYS, "requested_n_over_d", "actual_n_over_d")
_MODEL_METRICS = (
    "fit_time_sec",
    "rss_start_mib",
    "rss_min_mib",
    "rss_mean_mib",
    "rss_max_mib",
    "rss_peak_delta_mib",
    "memory_samples",
    "memory_source",
    "objective",
    "cosine_abs",
    "beta_encoded",
    "beta_dimension",
    "beta_norm",
    "beta_finite",
    "objective_finite",
    "result_finite",
    "statistics_builder_time_sec",
    "statistics_builder_calls",
    "worker_pid",
    "fit_started_ns",
    "fit_finished_ns",
    "actual_fit_order",
    "assigned_cpu",
    "worker_cpu_affinity",
    "worker_cpu_count",
    "cpu_affinity_supported",
    "cpu_affinity_pinned",
    "parallel_pairs",
)

DEFAULT_MODEL_NAMES = (
    "random_projection_baseline",
    "random_projection_candidate",
)
DEFAULT_BETA_ATOL = 1e-5
DEFAULT_PROJECTOR_ATOL = 1e-5
DEFAULT_OBJECTIVE_RTOL = 1e-5
DEFAULT_OBJECTIVE_ATOL = 1e-8


@dataclass(frozen=True, slots=True)
class _FitTask:
    index: int
    row: dict[str, object]
    model_payload: bytes
    X: np.ndarray
    y: np.ndarray
    centers: np.ndarray
    directions: np.ndarray
    beta0: np.ndarray
    beta_true: np.ndarray
    sample_interval_sec: float


@dataclass(frozen=True, slots=True)
class _FitPairTask:
    index: int
    fits: tuple[_FitTask, _FitTask]


@dataclass(frozen=True, slots=True)
class _AffinityState:
    supported: bool
    effective_cpus: tuple[int, ...]
    pinned: bool


def compare_models(
    first_model: Any,
    second_model: Any,
    *,
    model_names: tuple[str, str] = ("first", "second"),
    parameter_grid: Iterable[ExperimentParameters] | None = None,
    seeds: Sequence[int] = tuple(range(100)),
    sample_interval_sec: float = 0.01,
    jobs: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compare two ADP-compatible model objects on paired experiment-2 data.

    Every ``fit`` runs in a fresh spawned child. Up to ``jobs`` pairs run in
    parallel, but the two fits of each pair run strictly sequentially in the
    recorded AB/BA order. Each active pair owns one CPU from the caller's
    affinity mask where process affinity is supported. The model is serialized
    with ``cloudpickle`` so factory-based implementations are supported, while
    mutations, caches, and allocated memory disappear with the process. Both
    models receive identical ``X``, ``y``, centers, initial beta, and directions.
    """

    names = _validate_model_names(model_names)
    grid = tuple(full_parameter_grid("2") if parameter_grid is None else parameter_grid)
    _validate_experiment_two_grid(grid)
    selected_seeds = _validate_seeds(seeds)
    if sample_interval_sec <= 0.0 or not math.isfinite(sample_interval_sec):
        raise ValueError("sample_interval_sec must be finite and positive")
    worker_count = _validate_jobs(jobs)
    if not isinstance(show_progress, bool):
        raise ValueError("show_progress must be boolean")
    comparison_started = time.perf_counter()
    model_payloads = (
        _serialize_model(first_model),
        _serialize_model(second_model),
    )
    pair_tasks = iter(
        _iter_fit_pair_tasks(
            _iter_fit_tasks(
                names,
                model_payloads,
                grid,
                selected_seeds,
                sample_interval_sec,
                worker_count,
            )
        )
    )
    total_pairs = len(grid) * len(selected_seeds)
    total = total_pairs * 2
    available_cpus = _available_cpu_ids()
    parallel_pairs = min(worker_count, total_pairs, len(available_cpus))
    rows: dict[int, dict[str, object]] = {}
    pool = ProcessPoolExecutor(
        max_workers=parallel_pairs,
        mp_context=_spawn_context(),
        max_tasks_per_child=1,
    )
    pending: dict[
        Future[tuple[int, dict[str, object]]],
        tuple[_FitPairTask, int, int],
    ] = {}

    def submit_fit(pair: _FitPairTask, fit_order: int, cpu_id: int) -> None:
        fit = pair.fits[fit_order]
        future = pool.submit(
            _execute_fit_task,
            fit,
            cpu_id,
            fit_order,
            parallel_pairs,
        )
        pending[future] = (pair, fit_order, cpu_id)

    try:
        for cpu_id in available_cpus[:parallel_pairs]:
            pair = next(pair_tasks)
            submit_fit(pair, 0, cpu_id)
        with tqdm(
            total=total,
            desc="model comparison",
            unit="fit",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            while pending:
                future = next(as_completed(tuple(pending)))
                pair, fit_order, cpu_id = pending.pop(future)
                fit = pair.fits[fit_order]
                try:
                    index, row = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"model fit failed for {fit.row['model']} "
                        f"({fit.row['case_id']}, seed={fit.row['seed']})"
                    ) from exc
                rows[index] = row
                progress.update(1)
                if fit_order == 0:
                    submit_fit(pair, 1, cpu_id)
                    continue
                try:
                    next_pair = next(pair_tasks)
                except StopIteration:
                    continue
                submit_fit(next_pair, 0, cpu_id)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    comparison_wall_time_sec = time.perf_counter() - comparison_started
    frame = pd.DataFrame(rows[index] for index in sorted(rows))
    frame["comparison_total_fits"] = len(frame)
    frame["comparison_wall_time_sec"] = comparison_wall_time_sec
    frame["comparison_fits_per_sec"] = (
        len(frame) / comparison_wall_time_sec
    )
    return frame


def pair_model_runs(
    runs: pd.DataFrame,
    *,
    model_names: tuple[str, str] = ("first", "second"),
    beta_atol: float = DEFAULT_BETA_ATOL,
    projector_atol: float = DEFAULT_PROJECTOR_ATOL,
    objective_rtol: float = DEFAULT_OBJECTIVE_RTOL,
    objective_atol: float = DEFAULT_OBJECTIVE_ATOL,
) -> pd.DataFrame:
    """Pair runs and check efficiency plus sign-invariant equivalence."""

    baseline_name, candidate_name = _validate_model_names(model_names)
    beta_atol = _validate_tolerance("beta_atol", beta_atol)
    projector_atol = _validate_tolerance("projector_atol", projector_atol)
    objective_rtol = _validate_tolerance("objective_rtol", objective_rtol)
    objective_atol = _validate_tolerance("objective_atol", objective_atol)
    required = (
        *_PAIR_KEYS,
        "model",
        "fit_time_sec",
        "rss_max_mib",
        "rss_peak_delta_mib",
        "cosine_abs",
        "objective",
        "beta_encoded",
        "result_finite",
    )
    missing = [column for column in required if column not in runs]
    if missing:
        raise ValueError(f"runs is missing columns: {', '.join(missing)}")

    source = runs.copy()
    if "requested_n_over_d" not in source:
        source["requested_n_over_d"] = source["n_over_d"]
    if "actual_n_over_d" not in source:
        source["actual_n_over_d"] = source["n"] / source["d"]
    baseline = source.loc[source["model"].eq(baseline_name)].copy()
    candidate = source.loc[source["model"].eq(candidate_name)].copy()
    if baseline.empty or candidate.empty:
        raise ValueError("runs must contain both named models")

    rename_metrics = tuple(
        metric for metric in _MODEL_METRICS if metric in source.columns
    )
    baseline = baseline[[*_PAIR_CONTEXT_KEYS, *rename_metrics]].rename(
        columns={metric: f"baseline_{metric}" for metric in rename_metrics}
    )
    candidate = candidate[[*_PAIR_CONTEXT_KEYS, *rename_metrics]].rename(
        columns={metric: f"candidate_{metric}" for metric in rename_metrics}
    )
    paired = baseline.merge(
        candidate,
        on=list(_PAIR_CONTEXT_KEYS),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("model runs are not paired one-to-one")
    paired["baseline_model"] = baseline_name
    paired["candidate_model"] = candidate_name
    paired["time_speedup"] = _safe_ratio(
        paired["baseline_fit_time_sec"],
        paired["candidate_fit_time_sec"],
    )
    paired["peak_delta_memory_ratio"] = _safe_ratio(
        paired["baseline_rss_peak_delta_mib"],
        paired["candidate_rss_peak_delta_mib"],
    )
    paired["peak_rss_ratio"] = _safe_ratio(
        paired["baseline_rss_max_mib"],
        paired["candidate_rss_max_mib"],
    )
    paired["cosine_abs_gap"] = (
        paired["baseline_cosine_abs"] - paired["candidate_cosine_abs"]
    ).abs()
    beta_metrics = [
        _compare_encoded_betas(first, second, int(d))
        for first, second, d in zip(
            paired["baseline_beta_encoded"],
            paired["candidate_beta_encoded"],
            paired["d"],
            strict=True,
        )
    ]
    paired["beta_cosine_abs"] = [metric[0] for metric in beta_metrics]
    paired["beta_sign_invariant_error"] = [
        metric[1] for metric in beta_metrics
    ]
    paired["projector_frobenius_error"] = [
        metric[2] for metric in beta_metrics
    ]

    baseline_objective = pd.to_numeric(
        paired["baseline_objective"], errors="coerce"
    )
    candidate_objective = pd.to_numeric(
        paired["candidate_objective"], errors="coerce"
    )
    paired["objective_abs_gap"] = (
        baseline_objective - candidate_objective
    ).abs()
    objective_scale = np.maximum(
        baseline_objective.abs(), candidate_objective.abs()
    )
    paired["objective_relative_gap"] = paired["objective_abs_gap"] / np.maximum(
        objective_scale,
        np.finfo(float).eps,
    )

    baseline_finite = _boolean_series(paired["baseline_result_finite"])
    candidate_finite = _boolean_series(paired["candidate_result_finite"])
    pair_metrics_finite = np.isfinite(
        paired[
            [
                "beta_cosine_abs",
                "beta_sign_invariant_error",
                "projector_frobenius_error",
                "objective_abs_gap",
                "objective_relative_gap",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    paired["result_pair_finite"] = (
        baseline_finite.to_numpy()
        & candidate_finite.to_numpy()
        & pair_metrics_finite
    )
    objective_tolerance = objective_atol + objective_rtol * objective_scale
    paired["beta_atol"] = beta_atol
    paired["projector_atol"] = projector_atol
    paired["objective_rtol"] = objective_rtol
    paired["objective_atol"] = objective_atol
    paired["numerically_equivalent"] = (
        paired["result_pair_finite"]
        & paired["beta_sign_invariant_error"].le(beta_atol)
        & paired["projector_frobenius_error"].le(projector_atol)
        & paired["objective_abs_gap"].le(objective_tolerance)
    )
    return paired.sort_values(list(_PAIR_CONTEXT_KEYS)).reset_index(drop=True)


def summarize_paired_runs(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired comparisons by experiment-2 dimension and sample ratio."""

    metrics = (
        "baseline_fit_time_sec",
        "candidate_fit_time_sec",
        "time_speedup",
        "baseline_rss_peak_delta_mib",
        "candidate_rss_peak_delta_mib",
        "peak_delta_memory_ratio",
        "baseline_rss_max_mib",
        "candidate_rss_max_mib",
        "peak_rss_ratio",
        "cosine_abs_gap",
        "beta_cosine_abs",
        "beta_sign_invariant_error",
        "projector_frobenius_error",
        "objective_abs_gap",
        "objective_relative_gap",
    )
    group_columns = (
        "d",
        "n",
        "n_over_d",
        "requested_n_over_d",
        "actual_n_over_d",
    )
    required = (*group_columns, "result_pair_finite", "numerically_equivalent", *metrics)
    missing = [column for column in required if column not in paired]
    if missing:
        raise ValueError(f"paired runs is missing columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    grouped = paired.groupby(list(group_columns), sort=True, dropna=False)
    for keys, group in grouped:
        row = dict(zip(group_columns, keys, strict=True))
        row["pair_count"] = int(len(group))
        row["valid_pair_count"] = int(
            _boolean_series(group["result_pair_finite"]).sum()
        )
        row["equivalent_pair_count"] = int(
            _boolean_series(group["numerically_equivalent"]).sum()
        )
        row["equivalence_rate"] = (
            row["equivalent_pair_count"] / row["pair_count"]
        )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid = values.notna()
            finite = np.isfinite(values)
            valid_values = values.loc[valid]
            finite_values = values.loc[finite]
            row[f"valid_{metric}_count"] = int(valid.sum())
            row[f"finite_{metric}_count"] = int(finite.sum())
            row[f"median_{metric}"] = (
                math.nan if valid_values.empty else float(valid_values.median())
            )
            row[f"q25_{metric}"] = (
                math.nan
                if finite_values.empty
                else float(finite_values.quantile(0.25))
            )
            row[f"q75_{metric}"] = (
                math.nan
                if finite_values.empty
                else float(finite_values.quantile(0.75))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_comparison_artifacts(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_names: tuple[str, str] = ("first", "second"),
    dpi: int = 160,
) -> dict[str, Path]:
    """Write flat CSV results and time/memory comparison plots."""

    if dpi < 1:
        raise ValueError("dpi must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired = pair_model_runs(runs, model_names=model_names)
    summary = summarize_paired_runs(paired)

    artifacts = {
        "runs.csv": output / "runs.csv",
        "paired.csv": output / "paired.csv",
        "summary.csv": output / "summary.csv",
        "runtime_vs_dimension.png": output / "runtime_vs_dimension.png",
        "memory_vs_dimension.png": output / "memory_vs_dimension.png",
        "time_speedup_heatmap.png": output / "time_speedup_heatmap.png",
        "memory_ratio_heatmap.png": output / "memory_ratio_heatmap.png",
    }
    runs.to_csv(artifacts["runs.csv"], index=False)
    paired.to_csv(artifacts["paired.csv"], index=False)
    summary.to_csv(artifacts["summary.csv"], index=False)
    _line_plot(
        runs,
        value="fit_time_sec",
        ylabel="Median fit time, seconds",
        path=artifacts["runtime_vs_dimension.png"],
        dpi=dpi,
    )
    _line_plot(
        runs,
        value="rss_peak_delta_mib",
        ylabel="Median peak RSS increase, MiB",
        path=artifacts["memory_vs_dimension.png"],
        dpi=dpi,
    )
    _heatmap(
        paired,
        value="time_speedup",
        title=f"Time speedup: {model_names[0]} / {model_names[1]}",
        path=artifacts["time_speedup_heatmap.png"],
        dpi=dpi,
    )
    _heatmap(
        paired,
        value="peak_delta_memory_ratio",
        title=f"Peak-memory ratio: {model_names[0]} / {model_names[1]}",
        path=artifacts["memory_ratio_heatmap.png"],
        dpi=dpi,
    )
    return artifacts


def _iter_fit_tasks(
    names: tuple[str, str],
    model_payloads: tuple[bytes, bytes],
    grid: tuple[ExperimentParameters, ...],
    seeds: tuple[int, ...],
    sample_interval_sec: float,
    jobs: int,
) -> Iterable[_FitTask]:
    index = 0
    pair_index = 0
    for parameters in grid:
        case_id = _case_id(parameters)
        for seed in seeds:
            job = _build_single_index_job(
                "2",
                parameters,
                seed,
                diagnostic=False,
            )
            data = generate_synthetic_data(job).data
            beta0 = _initial_beta(parameters.d, job.seeds.init)
            model_specs = tuple(zip(names, model_payloads, strict=True))
            if pair_index % 2:
                model_specs = tuple(reversed(model_specs))
                pair_order = "BA"
            else:
                pair_order = "AB"
            for fit_order, (name, model_payload) in enumerate(
                model_specs
            ):
                yield _FitTask(
                    index=index,
                    row={
                        "case_id": case_id,
                        "seed": seed,
                        "d": parameters.d,
                        "n": parameters.n,
                        "n_over_d": parameters.n_over_d,
                        "requested_n_over_d": parameters.n_over_d,
                        "actual_n_over_d": parameters.n / parameters.d,
                        "n_centers": parameters.n_centers,
                        "n_directions": int(data.directions.shape[1]),
                        "pair_index": pair_index,
                        "model": name,
                        "fit_order": fit_order,
                        "pair_order": pair_order,
                        "jobs": jobs,
                    },
                    model_payload=model_payload,
                    X=data.X,
                    y=data.y,
                    centers=data.centers,
                    directions=data.directions,
                    beta0=beta0,
                    beta_true=data.beta,
                    sample_interval_sec=sample_interval_sec,
                )
                index += 1
            pair_index += 1


def _iter_fit_pair_tasks(tasks: Iterable[_FitTask]) -> Iterable[_FitPairTask]:
    iterator = iter(tasks)
    pair_index = 0
    while True:
        try:
            first = next(iterator)
        except StopIteration:
            return
        try:
            second = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("fit task stream ended inside a model pair") from exc
        fits = (first, second)
        if [fit.row.get("fit_order") for fit in fits] != [0, 1]:
            raise RuntimeError("fit task pair has an invalid planned order")
        if any(fit.row.get("pair_index") != pair_index for fit in fits):
            raise RuntimeError("fit task pair indices are inconsistent")
        yield _FitPairTask(index=pair_index, fits=fits)
        pair_index += 1


def _execute_fit_task(
    task: _FitTask,
    assigned_cpu: int,
    actual_fit_order: int,
    parallel_pairs: int,
) -> tuple[int, dict[str, object]]:
    if task.row.get("fit_order") != actual_fit_order:
        raise RuntimeError("scheduled fit order differs from the planned pair order")
    affinity = _pin_current_process(assigned_cpu)
    model = cloudpickle.loads(task.model_payload)
    _configure_child_model(
        model,
        task.centers.shape[0],
        task.directions.shape[1],
    )
    monitor = ResourceMonitor(sample_interval_sec=task.sample_interval_sec)
    with threadpool_limits(limits=1):
        with monitor:
            fit_started_ns = time.perf_counter_ns()
            result = model.fit(
                task.X,
                task.y,
                centers=task.centers,
                beta0=task.beta0,
                directions=task.directions,
            )
            fit_finished_ns = time.perf_counter_ns()
    beta = np.asarray(getattr(result, "beta"), dtype=float)
    objective = float(getattr(result, "objective", math.nan))
    stage_timings = getattr(result, "stage_timings", {})
    stage_calls = getattr(result, "stage_calls", {})
    usage = monitor.usage
    beta_flat = beta.reshape(-1)
    beta_dimension = int(beta_flat.size)
    beta_finite = bool(beta.ndim == 1 and np.all(np.isfinite(beta_flat)))
    beta_norm = float(np.linalg.norm(beta_flat)) if beta_finite else math.nan
    objective_finite = math.isfinite(objective)
    result_finite = bool(
        beta.ndim == 1
        and beta_dimension == task.beta_true.size
        and beta_finite
        and math.isfinite(beta_norm)
        and beta_norm > np.finfo(float).eps
        and objective_finite
    )
    metrics: dict[str, object] = {
        "fit_time_sec": usage.elapsed_sec,
        "rss_start_mib": usage.rss_start_mib,
        "rss_min_mib": usage.rss_min_mib,
        "rss_mean_mib": usage.rss_mean_mib,
        "rss_max_mib": usage.rss_max_mib,
        "rss_peak_delta_mib": usage.rss_peak_delta_mib,
        "memory_samples": usage.samples,
        "memory_source": usage.source,
        "objective": objective,
        "cosine_abs": _absolute_cosine(beta, task.beta_true),
        "beta_encoded": _encode_beta(beta_flat),
        "beta_dimension": beta_dimension,
        "beta_norm": beta_norm,
        "beta_finite": beta_finite,
        "objective_finite": objective_finite,
        "result_finite": result_finite,
        "statistics_builder_time_sec": float(
            stage_timings.get("statistics_builder", math.nan)
        ),
        "statistics_builder_calls": int(
            stage_calls.get("statistics_builder", 0)
        ),
        "worker_pid": os.getpid(),
        "fit_started_ns": fit_started_ns,
        "fit_finished_ns": fit_finished_ns,
        "actual_fit_order": actual_fit_order,
        "assigned_cpu": assigned_cpu,
        "worker_cpu_affinity": ",".join(
            str(cpu_id) for cpu_id in affinity.effective_cpus
        ),
        "worker_cpu_count": len(affinity.effective_cpus),
        "cpu_affinity_supported": affinity.supported,
        "cpu_affinity_pinned": affinity.pinned,
        "parallel_pairs": parallel_pairs,
    }
    return task.index, {**task.row, **metrics}


def _serialize_model(model: Any) -> bytes:
    try:
        return cloudpickle.dumps(model)
    except Exception as exc:
        raise TypeError(
            f"model {type(model).__name__} cannot be serialized for isolation"
        ) from exc


def _configure_child_model(model: Any, n_centers: int, n_directions: int) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    for name, value in (
        ("n_centers", n_centers),
        ("n_directions", n_directions),
        ("renew_directions", False),
        ("show_progress", False),
        ("statistics_workers", 1),
    ):
        if hasattr(config, name):
            setattr(config, name, value)
    backend = getattr(model, "backend", None)
    if backend is not None and hasattr(backend, "statistics_workers"):
        setattr(backend, "statistics_workers", 1)


def _spawn_context() -> Any:
    return mp.get_context("spawn")


def _available_cpu_ids() -> tuple[int, ...]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            cpu_ids = tuple(sorted(int(cpu_id) for cpu_id in get_affinity(0)))
        except OSError:
            cpu_ids = ()
        if cpu_ids:
            return cpu_ids
    return tuple(range(max(1, os.cpu_count() or 1)))


def _pin_current_process(cpu_id: int) -> _AffinityState:
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    supported = callable(get_affinity) and callable(set_affinity)
    if supported:
        try:
            set_affinity(0, {cpu_id})
        except OSError as exc:
            raise RuntimeError(
                f"failed to pin fit process to CPU {cpu_id}"
            ) from exc
    effective_cpus: tuple[int, ...] = ()
    if callable(get_affinity):
        try:
            effective_cpus = tuple(
                sorted(int(value) for value in get_affinity(0))
            )
        except OSError:
            pass
    pinned = supported and effective_cpus == (cpu_id,)
    if supported and not pinned:
        raise RuntimeError(
            f"fit process affinity is {effective_cpus}, expected {(cpu_id,)}"
        )
    return _AffinityState(
        supported=supported,
        effective_cpus=effective_cpus,
        pinned=pinned,
    )


def _validate_jobs(jobs: int) -> int:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    return jobs


def _validate_model_names(model_names: tuple[str, str]) -> tuple[str, str]:
    if not isinstance(model_names, tuple) or len(model_names) != 2:
        raise ValueError("model_names must contain exactly two names")
    names = tuple(str(name).strip() for name in model_names)
    if not all(names):
        raise ValueError("model names must not be empty")
    if names[0] == names[1]:
        raise ValueError("model names must be distinct")
    return names


def _validate_experiment_two_grid(
    grid: tuple[ExperimentParameters, ...],
) -> None:
    if not grid:
        raise ValueError("experiment 2 parameter grid must not be empty")
    for parameters in grid:
        expected = ExperimentParameters(d=parameters.d, n_over_d=parameters.n_over_d)
        if parameters != expected:
            raise ValueError(
                "comparison accepts only experiment 2 parameters: d and n_over_d"
            )


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    result = tuple(seeds)
    if not result:
        raise ValueError("seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in result):
        raise ValueError("seeds must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return result


def _case_id(parameters: ExperimentParameters) -> str:
    return f"d={parameters.d};n_over_d={parameters.n_over_d:g}"


def _initial_beta(d: int, seed: int) -> np.ndarray:
    beta = np.random.default_rng(seed).normal(size=d)
    norm = float(np.linalg.norm(beta))
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise RuntimeError("failed to generate a finite initial beta")
    return np.asarray(beta / norm, dtype=float)


def _absolute_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    if first.shape != second.shape:
        return math.nan
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps or not math.isfinite(denominator):
        return math.nan
    return float(abs(np.dot(first, second)) / denominator)


def _encode_beta(beta: np.ndarray) -> str:
    return "|".join(format(float(value), ".17g") for value in beta)


def _decode_beta(encoded: object) -> np.ndarray:
    if not isinstance(encoded, str) or not encoded:
        return np.array([], dtype=float)
    try:
        return np.asarray([float(value) for value in encoded.split("|")])
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _compare_encoded_betas(
    first_encoded: object,
    second_encoded: object,
    dimension: int,
) -> tuple[float, float, float]:
    first = _decode_beta(first_encoded)
    second = _decode_beta(second_encoded)
    if (
        first.shape != (dimension,)
        or second.shape != (dimension,)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        return math.nan, math.nan, math.nan
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    denominator = first_norm * second_norm
    if not math.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return math.nan, math.nan, math.nan
    cosine_abs = float(np.clip(abs(np.dot(first, second)) / denominator, 0.0, 1.0))
    beta_error = min(
        float(np.linalg.norm(first - second)),
        float(np.linalg.norm(first + second)),
    ) / max(first_norm, second_norm, np.finfo(float).eps)
    projector_error = math.sqrt(max(0.0, 2.0 - 2.0 * cosine_abs**2))
    return cosine_abs, beta_error, projector_error


def _boolean_series(values: pd.Series) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        if isinstance(value, (int, np.integer)):
            return int(value) == 1
        return False

    return values.map(parse).astype(bool)


def _validate_tolerance(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & numerator.ge(0.0)
        & denominator.ge(0.0)
    )
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    positive_denominator = valid & denominator.gt(0.0)
    result.loc[positive_denominator] = (
        numerator.loc[positive_denominator]
        / denominator.loc[positive_denominator]
    )
    both_zero = valid & numerator.eq(0.0) & denominator.eq(0.0)
    result.loc[both_zero] = 1.0
    positive_over_zero = valid & numerator.gt(0.0) & denominator.eq(0.0)
    result.loc[positive_over_zero] = math.inf
    return result


def _line_plot(
    runs: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    path: Path,
    dpi: int,
) -> None:
    prepared = (
        runs.groupby(["model", "n_over_d", "d"], sort=True, as_index=False)[value]
        .median()
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (model, ratio), group in prepared.groupby(["model", "n_over_d"], sort=True):
        group = group.sort_values("d")
        ax.plot(
            group["d"],
            group[value],
            marker="o",
            label=f"{model}, n/d={ratio:g}",
        )
    ax.set_xlabel("Dimension d")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _heatmap(
    paired: pd.DataFrame,
    *,
    value: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    table = paired.pivot_table(
        index="d",
        columns="n_over_d",
        values=value,
        aggfunc="median",
        sort=True,
    )
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)), labels=[f"{value:g}" for value in table.columns])
    ax.set_yticks(np.arange(len(table.index)), labels=[str(value) for value in table.index])
    ax.set_xlabel("n/d")
    ax.set_ylabel("Dimension d")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if math.isfinite(values[row, column]):
                ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two ADP model implementations by time and memory."
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--seeds",
        help="Inclusive START:STOP range or comma-separated seeds; defaults to 0 for smoke and 0:99 for full.",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help=(
            "Parallel AB/BA pairs; fits inside a pair are sequential and "
            "isolated. Use 1 for uncontended latency measurements."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _default_models() -> tuple[Any, Any]:
    common = {
        "statistics_workers": 1,
        "show_progress": False,
        "record_telemetry": True,
        "renew_directions": False,
        "random_state": 0,
    }
    baseline = ADP.create(
        "new",
        ADPConfig(**common),
        stages={"statistics_builder": "random_projection"},
    )
    candidate = ADP.create(
        "new",
        ADPConfig(**common),
        stages={"statistics_builder": "random_projection"},
    )
    return baseline, candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid = (
        full_parameter_grid("2")
        if args.profile == "full"
        else smoke_parameter_grid("2")
    )
    seeds = (
        parse_seed_selection(args.seeds)
        if args.seeds is not None
        else (tuple(range(100)) if args.profile == "full" else (0,))
    )
    names = DEFAULT_MODEL_NAMES
    runs = compare_models(
        *_default_models(),
        model_names=names,
        parameter_grid=grid,
        seeds=seeds,
        sample_interval_sec=args.sample_interval,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    artifacts = write_comparison_artifacts(
        runs,
        args.output,
        model_names=names,
        dpi=args.dpi,
    )
    paired = pair_model_runs(runs, model_names=names)
    print(f"runs: {len(runs)}")
    print(f"median_time_speedup: {paired['time_speedup'].median():.6f}")
    print(
        "median_peak_delta_memory_ratio: "
        f"{paired['peak_delta_memory_ratio'].median():.6f}"
    )
    valid_pairs = int(paired["result_pair_finite"].sum())
    equivalent_pairs = int(paired["numerically_equivalent"].sum())
    print(f"valid_result_pairs: {valid_pairs}/{len(paired)}")
    print(f"numerically_equivalent_pairs: {equivalent_pairs}/{len(paired)}")
    print(
        "comparison_fits_per_sec: "
        f"{float(runs['comparison_fits_per_sec'].iloc[0]):.6f}"
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    if equivalent_pairs != len(paired):
        print(
            f"candidate is not numerically equivalent in "
            f"{len(paired) - equivalent_pairs}/{len(paired)} paired runs",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
