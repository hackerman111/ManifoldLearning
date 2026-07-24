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
import hashlib
import json
import math
import multiprocessing as mp
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
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
    parse_experiment_selectors,
    parse_seed_selection,
    smoke_parameter_grid,
)
from adp.evaluation.single_index.types import ExperimentParameters
from experiments.model_comparison_specs import load_model_specs


_COMPARISON_EXPERIMENTS = ("2", "3", "4", "5", "6")
_PARAMETER_COLUMNS = tuple(
    field.name for field in fields(ExperimentParameters)
)
_PAIR_KEYS = ("case_id", "seed", "d", "n", "n_over_d")
_PAIR_OPTIONAL_CONTEXT = (
    "experiment",
    *(
        name
        for name in _PARAMETER_COLUMNS
        if name not in {"d", "n_over_d"}
    ),
    "requested_n_over_d",
    "actual_n_over_d",
)
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
    "beta_initializer",
    "bandwidth_selector",
    "beta_ref_encoded",
    "beta_hat0_encoded",
    "beta_ref_cosine_abs",
    "beta_hat0_cosine_abs",
    "initial_bandwidth",
    "initial_local_mass_min",
    "initial_local_mass_q05",
    "initial_local_mass_q10",
    "initial_local_mass_q25",
    "statistics_builder_time_sec",
    "statistics_builder_calls",
    "local_solver_time_sec",
    "local_solver_calls",
    "beta_solver_time_sec",
    "beta_solver_calls",
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
    beta0: np.ndarray | None
    beta_true: np.ndarray
    sample_interval_sec: float


@dataclass(frozen=True, slots=True)
class _FitPairTask:
    index: int
    fits: tuple[_FitTask, _FitTask]


@dataclass(frozen=True, slots=True)
class _FitGroupTask:
    index: int
    fits: tuple[_FitTask, ...]


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
    return compare_model_set(
        (first_model, second_model),
        model_names=names,
        parameter_grid=parameter_grid,
        seeds=seeds,
        sample_interval_sec=sample_interval_sec,
        jobs=jobs,
        show_progress=show_progress,
        continue_on_error=False,
    )


def compare_model_set(
    models: Sequence[Any],
    *,
    model_names: Sequence[str],
    parameter_grid: Iterable[ExperimentParameters] | None = None,
    experiment_parameter_grids: Mapping[
        str,
        Iterable[ExperimentParameters],
    ]
    | None = None,
    seeds: Sequence[int] = tuple(range(100)),
    sample_interval_sec: float = 0.01,
    jobs: int = 1,
    show_progress: bool = True,
    continue_on_error: bool = True,
    use_model_initializers: bool = False,
) -> pd.DataFrame:
    """Compare two or more models on identical isolated fit inputs."""

    selected_models = tuple(models)
    if len(selected_models) < 2:
        raise ValueError("models must contain at least two implementations")
    names = _validate_model_set_names(model_names, len(selected_models))
    grid, experiment_selectors = _prepare_experiment_parameter_grids(
        parameter_grid,
        experiment_parameter_grids,
    )
    selected_seeds = _validate_seeds(seeds)
    if sample_interval_sec <= 0.0 or not math.isfinite(sample_interval_sec):
        raise ValueError("sample_interval_sec must be finite and positive")
    worker_count = _validate_jobs(jobs)
    if not isinstance(show_progress, bool):
        raise ValueError("show_progress must be boolean")
    if not isinstance(continue_on_error, bool):
        raise ValueError("continue_on_error must be boolean")
    if not isinstance(use_model_initializers, bool):
        raise ValueError("use_model_initializers must be boolean")
    comparison_started = time.perf_counter()
    model_payloads = tuple(
        _serialize_model(model) for model in selected_models
    )
    group_tasks = iter(
        _iter_fit_group_tasks(
            _iter_model_fit_tasks(
                names,
                model_payloads,
                grid,
                selected_seeds,
                sample_interval_sec,
                worker_count,
                use_model_initializers,
                experiment_selectors=experiment_selectors,
            ),
            len(names),
        )
    )
    total_groups = len(grid) * len(selected_seeds)
    total = total_groups * len(names)
    available_cpus = _available_cpu_ids()
    parallel_groups = min(worker_count, total_groups, len(available_cpus))
    rows: dict[int, dict[str, object]] = {}
    pool = ProcessPoolExecutor(
        max_workers=parallel_groups,
        mp_context=_spawn_context(),
        max_tasks_per_child=1,
    )
    pending: dict[
        Future[tuple[int, dict[str, object]]],
        tuple[_FitGroupTask, int, int],
    ] = {}

    def submit_fit(group: _FitGroupTask, fit_order: int, cpu_id: int) -> None:
        fit = group.fits[fit_order]
        future = pool.submit(
            _execute_fit_task,
            fit,
            cpu_id,
            fit_order,
            parallel_groups,
        )
        pending[future] = (group, fit_order, cpu_id)

    try:
        for cpu_id in available_cpus[:parallel_groups]:
            group = next(group_tasks)
            submit_fit(group, 0, cpu_id)
        with tqdm(
            total=total,
            desc="model comparison",
            unit="fit",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            while pending:
                future = next(as_completed(tuple(pending)))
                group, fit_order, cpu_id = pending.pop(future)
                fit = group.fits[fit_order]
                try:
                    index, row = future.result()
                except Exception as exc:
                    if not continue_on_error:
                        raise RuntimeError(
                            f"model fit failed for {fit.row['model']} "
                            f"({fit.row['case_id']}, seed={fit.row['seed']})"
                        ) from exc
                    index = fit.index
                    row = _failed_fit_row(
                        fit,
                        exc,
                        assigned_cpu=cpu_id,
                        actual_fit_order=fit_order,
                        parallel_groups=parallel_groups,
                    )
                rows[index] = row
                progress.update(1)
                next_fit_order = fit_order + 1
                if next_fit_order < len(group.fits):
                    submit_fit(group, next_fit_order, cpu_id)
                    continue
                try:
                    next_group = next(group_tasks)
                except StopIteration:
                    continue
                submit_fit(next_group, 0, cpu_id)
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
    pair_context_keys = (
        *_PAIR_KEYS,
        *(
            column
            for column in _PAIR_OPTIONAL_CONTEXT
            if column in source.columns
        ),
    )
    baseline = source.loc[source["model"].eq(baseline_name)].copy()
    candidate = source.loc[source["model"].eq(candidate_name)].copy()
    if baseline.empty or candidate.empty:
        raise ValueError("runs must contain both named models")

    rename_metrics = tuple(
        metric for metric in _MODEL_METRICS if metric in source.columns
    )
    baseline = baseline[[*pair_context_keys, *rename_metrics]].rename(
        columns={metric: f"baseline_{metric}" for metric in rename_metrics}
    )
    candidate = candidate[[*pair_context_keys, *rename_metrics]].rename(
        columns={metric: f"candidate_{metric}" for metric in rename_metrics}
    )
    paired = baseline.merge(
        candidate,
        on=list(pair_context_keys),
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
    return paired.sort_values(list(pair_context_keys)).reset_index(drop=True)


def summarize_paired_runs(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired comparisons without mixing experiment cases."""

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
    group_columns = _summary_context_columns(paired)
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


def pair_model_set_runs(
    runs: pd.DataFrame,
    *,
    model_names: Sequence[str],
) -> pd.DataFrame:
    """Pair every candidate with the first, baseline model."""

    provided_names = tuple(model_names)
    names = _validate_model_set_names(
        provided_names,
        len(provided_names),
    )
    if len(names) < 2:
        raise ValueError("model_names must contain at least two names")
    frames = [
        pair_model_runs(runs, model_names=(names[0], candidate))
        for candidate in names[1:]
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_model_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time, memory, quality, and status per model and scenario."""

    group_columns = ("model", *_summary_context_columns(runs))
    metrics = (
        "fit_time_sec",
        "rss_peak_delta_mib",
        "rss_max_mib",
        "cosine_abs",
        "objective",
    )
    required = (*group_columns, "status", *metrics)
    missing = [column for column in required if column not in runs]
    if missing:
        raise ValueError(f"runs is missing columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    grouped = runs.groupby(list(group_columns), sort=True, dropna=False)
    for keys, group in grouped:
        row = dict(zip(group_columns, keys, strict=True))
        success = group["status"].astype(str).eq("ok")
        row["run_count"] = int(len(group))
        row["success_count"] = int(success.sum())
        row["failure_count"] = int((~success).sum())
        for metric in metrics:
            values = pd.to_numeric(
                group.loc[success, metric],
                errors="coerce",
            )
            finite = values.loc[np.isfinite(values)]
            row[f"valid_{metric}_count"] = int(values.notna().sum())
            row[f"finite_{metric}_count"] = int(len(finite))
            row[f"median_{metric}"] = (
                math.nan if finite.empty else float(finite.median())
            )
            row[f"q25_{metric}"] = (
                math.nan if finite.empty else float(finite.quantile(0.25))
            )
            row[f"q75_{metric}"] = (
                math.nan if finite.empty else float(finite.quantile(0.75))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline-relative metrics while preserving candidate names."""

    required = ("baseline_model", "candidate_model")
    missing = [column for column in required if column not in comparisons]
    if missing:
        raise ValueError(
            f"comparisons is missing columns: {', '.join(missing)}"
        )
    frames: list[pd.DataFrame] = []
    for candidate, group in comparisons.groupby(
        "candidate_model",
        sort=False,
        dropna=False,
    ):
        summary = summarize_paired_runs(group)
        summary.insert(0, "candidate_model", candidate)
        summary.insert(0, "baseline_model", group["baseline_model"].iloc[0])
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_model_set_artifacts(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_names: Sequence[str],
    configuration: Mapping[str, object],
    dpi: int = 160,
) -> dict[str, Path]:
    """Write multi-model CSV, manifest, and baseline-relative plots."""

    if dpi < 1:
        raise ValueError("dpi must be positive")
    provided_names = tuple(model_names)
    names = _validate_model_set_names(
        provided_names,
        len(provided_names),
    )
    if len(names) < 2:
        raise ValueError("model_names must contain at least two names")
    output = Path(output_dir)
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    _clear_model_set_line_plots(plots)

    comparisons = pair_model_set_runs(runs, model_names=names)
    model_summary = summarize_model_runs(runs)
    comparison_summary = summarize_model_comparisons(comparisons)
    artifacts: dict[str, Path] = {
        "runs.csv": output / "runs.csv",
        "model_summary.csv": output / "model_summary.csv",
        "comparisons.csv": output / "comparisons.csv",
        "comparison_summary.csv": output / "comparison_summary.csv",
        "manifest.json": output / "manifest.json",
    }
    runs.to_csv(artifacts["runs.csv"], index=False)
    model_summary.to_csv(artifacts["model_summary.csv"], index=False)
    comparisons.to_csv(artifacts["comparisons.csv"], index=False)
    comparison_summary.to_csv(
        artifacts["comparison_summary.csv"],
        index=False,
    )

    saved_runs = pd.read_csv(artifacts["runs.csv"])
    saved_comparisons = pd.read_csv(artifacts["comparisons.csv"])
    experiment_values: tuple[str | None, ...]
    if "experiment" in saved_runs.columns:
        experiment_values = tuple(
            dict.fromkeys(saved_runs["experiment"].astype(str))
        )
    else:
        experiment_values = (None,)
    separate_experiments = len(experiment_values) > 1
    metric_plots = (
        (
            "runtime_by_dimension",
            "fit_time_sec",
            "Median fit time, seconds",
        ),
        (
            "memory_by_dimension",
            "rss_peak_delta_mib",
            "Median peak RSS increase, MiB",
        ),
        (
            "cosine_abs_by_dimension",
            "cosine_abs",
            "Median absolute cosine",
        ),
    )
    for experiment in experiment_values:
        experiment_directory = plots
        experiment_key = "plots"
        experiment_label = ""
        if separate_experiments:
            assert experiment is not None
            safe_experiment = _safe_model_directory(experiment)
            experiment_directory = plots / f"experiment_{safe_experiment}"
            experiment_key = f"plots/experiment_{safe_experiment}"
            experiment_label = f", experiment {experiment}"
        experiment_runs = saved_runs
        if experiment is not None and "experiment" in experiment_runs:
            experiment_runs = experiment_runs.loc[
                experiment_runs["experiment"].astype(str).eq(experiment)
            ]
        for dimension in sorted(experiment_runs["d"].unique()):
            dimension_label = _dimension_label(dimension)
            for directory_name, value, ylabel in metric_plots:
                directory = experiment_directory / directory_name
                directory.mkdir(parents=True, exist_ok=True)
                key = f"{experiment_key}/{directory_name}/d_{dimension_label}.png"
                artifacts[key] = directory / f"d_{dimension_label}.png"
                _line_plot_by_dimension(
                    experiment_runs,
                    dimension=dimension,
                    value=value,
                    ylabel=ylabel,
                    experiment_label=experiment_label,
                    path=artifacts[key],
                    dpi=dpi,
                )
        experiment_rows = saved_comparisons
        if experiment is not None and "experiment" in experiment_rows:
            experiment_rows = experiment_rows.loc[
                experiment_rows["experiment"].astype(str).eq(experiment)
            ]
        for candidate in names[1:]:
            directory = (
                experiment_directory / _safe_model_directory(candidate)
            )
            directory.mkdir(parents=True, exist_ok=True)
            candidate_rows = experiment_rows.loc[
                experiment_rows["candidate_model"].eq(candidate)
            ]
            time_key = (
                f"{experiment_key}/{directory.name}/"
                "time_speedup_heatmap.png"
            )
            memory_key = (
                f"{experiment_key}/{directory.name}/"
                "memory_ratio_heatmap.png"
            )
            artifacts[time_key] = directory / "time_speedup_heatmap.png"
            artifacts[memory_key] = directory / "memory_ratio_heatmap.png"
            _heatmap(
                candidate_rows,
                value="time_speedup",
                title=(
                    f"Time speedup: {names[0]} / {candidate}"
                    f"{experiment_label}"
                ),
                path=artifacts[time_key],
                dpi=dpi,
            )
            _heatmap(
                candidate_rows,
                value="peak_delta_memory_ratio",
                title=(
                    f"Peak-memory ratio: {names[0]} / {candidate}"
                    f"{experiment_label}"
                ),
                path=artifacts[memory_key],
                dpi=dpi,
            )

    manifest = {
        "schema_version": 1,
        **dict(configuration),
        "model_names": list(names),
        "baseline_model": names[0],
        "artifacts": [
            key for key in artifacts if key != "manifest.json"
        ],
    }
    artifacts["manifest.json"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return artifacts


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
    yield from _iter_model_fit_tasks(
        names,
        model_payloads,
        grid,
        seeds,
        sample_interval_sec,
        jobs,
    )


def _iter_model_fit_tasks(
    names: tuple[str, ...],
    model_payloads: tuple[bytes, ...],
    grid: tuple[ExperimentParameters, ...],
    seeds: tuple[int, ...],
    sample_interval_sec: float,
    jobs: int,
    use_model_initializers: bool = False,
    *,
    experiment_selectors: tuple[str, ...] | None = None,
) -> Iterable[_FitTask]:
    if experiment_selectors is None:
        experiment_selectors = ("2",) * len(grid)
    if len(experiment_selectors) != len(grid):
        raise ValueError(
            "experiment_selectors must match the parameter grid"
        )
    index = 0
    group_index = 0
    for experiment, parameters in zip(
        experiment_selectors,
        grid,
        strict=True,
    ):
        _validate_experiment_grid(experiment, (parameters,))
        case_id = _case_id(experiment, parameters)
        for seed in seeds:
            job = _build_single_index_job(
                experiment,
                parameters,
                seed,
                diagnostic=False,
            )
            data = generate_synthetic_data(job).data
            beta0 = _initial_beta(parameters.d, job.seeds.init)
            model_specs = tuple(zip(names, model_payloads, strict=True))
            offset = group_index % len(model_specs)
            model_specs = model_specs[offset:] + model_specs[:offset]
            planned_order = "|".join(name for name, _ in model_specs)
            if len(model_specs) == 2:
                pair_order = "AB" if offset == 0 else "BA"
            else:
                pair_order = planned_order
            fingerprint_arrays = [
                data.X,
                data.y,
                data.centers,
                data.directions,
            ]
            if not use_model_initializers:
                fingerprint_arrays.append(beta0)
            fingerprint_arrays.append(data.beta)
            input_fingerprint = _input_fingerprint(*fingerprint_arrays)
            for fit_order, (name, model_payload) in enumerate(
                model_specs
            ):
                yield _FitTask(
                    index=index,
                    row={
                        "case_id": case_id,
                        "experiment": experiment,
                        "seed": seed,
                        **asdict(parameters),
                        "n": parameters.n,
                        "requested_n_over_d": parameters.n_over_d,
                        "actual_n_over_d": parameters.n / parameters.d,
                        "n_centers": parameters.n_centers,
                        "n_directions": int(data.directions.shape[1]),
                        "pair_index": group_index,
                        "comparison_group_index": group_index,
                        "model": name,
                        "fit_order": fit_order,
                        "pair_order": pair_order,
                        "planned_model_order": planned_order,
                        "input_fingerprint": input_fingerprint,
                        "beta0_source": (
                            "model_initializer"
                            if use_model_initializers
                            else "shared_generated"
                        ),
                        "jobs": jobs,
                    },
                    model_payload=model_payload,
                    X=data.X,
                    y=data.y,
                    centers=data.centers,
                    directions=data.directions,
                    beta0=None if use_model_initializers else beta0,
                    beta_true=data.beta,
                    sample_interval_sec=sample_interval_sec,
                )
                index += 1
            group_index += 1


def _iter_fit_group_tasks(
    tasks: Iterable[_FitTask],
    model_count: int,
) -> Iterable[_FitGroupTask]:
    iterator = iter(tasks)
    group_index = 0
    while True:
        fits: list[_FitTask] = []
        for _ in range(model_count):
            try:
                fits.append(next(iterator))
            except StopIteration:
                if fits:
                    raise RuntimeError(
                        "fit task stream ended inside a comparison group"
                    )
                return
        if [fit.row.get("fit_order") for fit in fits] != list(
            range(model_count)
        ):
            raise RuntimeError("fit task group has an invalid planned order")
        if any(
            fit.row.get("comparison_group_index") != group_index
            for fit in fits
        ):
            raise RuntimeError("fit task group indices are inconsistent")
        yield _FitGroupTask(index=group_index, fits=tuple(fits))
        group_index += 1


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
    parallel_groups: int,
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
    stage_names = getattr(result, "stage_names", {})
    beta_ref = _result_vector(result, "beta_ref", task.beta_true.size)
    beta_hat0 = _result_vector(result, "beta_hat0", task.beta_true.size)
    outer_telemetry = getattr(result, "outer_telemetry", ())
    initial_telemetry = (
        outer_telemetry[0]
        if outer_telemetry
        and isinstance(outer_telemetry[0], Mapping)
        else {}
    )
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
        "status": "ok",
        "error_type": "",
        "error_message": "",
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
        "beta_initializer": str(
            stage_names.get("beta_initializer", "")
        ),
        "bandwidth_selector": str(
            stage_names.get("bandwidth_selector", "")
        ),
        "beta_ref_encoded": _encode_beta(beta_ref),
        "beta_hat0_encoded": _encode_beta(beta_hat0),
        "beta_ref_cosine_abs": _absolute_cosine(
            beta_ref,
            task.beta_true,
        ),
        "beta_hat0_cosine_abs": _absolute_cosine(
            beta_hat0,
            task.beta_true,
        ),
        "initial_bandwidth": _mapping_float(initial_telemetry, "h"),
        "initial_local_mass_min": _mapping_float(
            initial_telemetry,
            "local_mass_min",
        ),
        "initial_local_mass_q05": _mapping_float(
            initial_telemetry,
            "local_mass_q05",
        ),
        "initial_local_mass_q10": _mapping_float(
            initial_telemetry,
            "local_mass_q10",
        ),
        "initial_local_mass_q25": _mapping_float(
            initial_telemetry,
            "local_mass_q25",
        ),
        "statistics_builder_time_sec": float(
            stage_timings.get("statistics_builder", math.nan)
        ),
        "statistics_builder_calls": int(
            stage_calls.get("statistics_builder", 0)
        ),
        "local_solver_time_sec": float(
            stage_timings.get("local_solver", math.nan)
        ),
        "local_solver_calls": int(stage_calls.get("local_solver", 0)),
        "beta_solver_time_sec": float(
            stage_timings.get("beta_solver", math.nan)
        ),
        "beta_solver_calls": int(stage_calls.get("beta_solver", 0)),
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
        "parallel_pairs": parallel_groups,
        "parallel_groups": parallel_groups,
    }
    return task.index, {**task.row, **metrics}


def _failed_fit_row(
    task: _FitTask,
    exc: Exception,
    *,
    assigned_cpu: int,
    actual_fit_order: int,
    parallel_groups: int,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "fit_time_sec": math.nan,
        "rss_start_mib": math.nan,
        "rss_min_mib": math.nan,
        "rss_mean_mib": math.nan,
        "rss_max_mib": math.nan,
        "rss_peak_delta_mib": math.nan,
        "memory_samples": 0,
        "memory_source": "",
        "objective": math.nan,
        "cosine_abs": math.nan,
        "beta_encoded": "",
        "beta_dimension": 0,
        "beta_norm": math.nan,
        "beta_finite": False,
        "objective_finite": False,
        "result_finite": False,
        "beta_initializer": "",
        "bandwidth_selector": "",
        "beta_ref_encoded": "",
        "beta_hat0_encoded": "",
        "beta_ref_cosine_abs": math.nan,
        "beta_hat0_cosine_abs": math.nan,
        "initial_bandwidth": math.nan,
        "initial_local_mass_min": math.nan,
        "initial_local_mass_q05": math.nan,
        "initial_local_mass_q10": math.nan,
        "initial_local_mass_q25": math.nan,
        "statistics_builder_time_sec": math.nan,
        "statistics_builder_calls": 0,
        "local_solver_time_sec": math.nan,
        "local_solver_calls": 0,
        "beta_solver_time_sec": math.nan,
        "beta_solver_calls": 0,
        "worker_pid": -1,
        "fit_started_ns": -1,
        "fit_finished_ns": -1,
        "actual_fit_order": actual_fit_order,
        "assigned_cpu": assigned_cpu,
        "worker_cpu_affinity": "",
        "worker_cpu_count": 0,
        "cpu_affinity_supported": callable(
            getattr(os, "sched_getaffinity", None)
        ) and callable(getattr(os, "sched_setaffinity", None)),
        "cpu_affinity_pinned": False,
        "parallel_pairs": parallel_groups,
        "parallel_groups": parallel_groups,
    }
    return {**task.row, **metrics}


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


def _validate_model_set_names(
    model_names: Sequence[str],
    model_count: int,
) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in model_names)
    if len(names) != model_count:
        raise ValueError("model_names must match the number of models")
    if not all(names):
        raise ValueError("model names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("model names must be distinct")
    return names


def _prepare_experiment_parameter_grids(
    parameter_grid: Iterable[ExperimentParameters] | None,
    experiment_parameter_grids: Mapping[
        str,
        Iterable[ExperimentParameters],
    ]
    | None,
) -> tuple[tuple[ExperimentParameters, ...], tuple[str, ...]]:
    if parameter_grid is not None and experiment_parameter_grids is not None:
        raise ValueError(
            "parameter_grid and experiment_parameter_grids are mutually exclusive"
        )
    if experiment_parameter_grids is None:
        grid = tuple(
            full_parameter_grid("2")
            if parameter_grid is None
            else parameter_grid
        )
        _validate_experiment_grid("2", grid)
        return grid, ("2",) * len(grid)
    if not isinstance(experiment_parameter_grids, Mapping):
        raise ValueError("experiment_parameter_grids must be a mapping")
    unknown = sorted(
        set(experiment_parameter_grids) - set(_COMPARISON_EXPERIMENTS)
    )
    if unknown:
        raise ValueError(
            "comparison supports only experiments 2-6; unknown: "
            + ", ".join(unknown)
        )
    if not experiment_parameter_grids:
        raise ValueError("experiment_parameter_grids must not be empty")

    parameters: list[ExperimentParameters] = []
    selectors: list[str] = []
    for experiment in _COMPARISON_EXPERIMENTS:
        if experiment not in experiment_parameter_grids:
            continue
        grid = tuple(experiment_parameter_grids[experiment])
        _validate_experiment_grid(experiment, grid)
        parameters.extend(grid)
        selectors.extend([experiment] * len(grid))
    return tuple(parameters), tuple(selectors)


def _validate_experiment_grid(
    experiment: str,
    grid: tuple[ExperimentParameters, ...],
) -> None:
    if experiment not in _COMPARISON_EXPERIMENTS:
        raise ValueError(
            f"comparison supports only experiments 2-6, got {experiment}"
        )
    if not grid:
        raise ValueError(
            f"experiment {experiment} parameter grid must not be empty"
        )
    varying_fields = {
        "2": set(),
        "3": {"sigma_eps"},
        "4": {"rho_corr"},
        "5": {"sigma_x"},
        "6": {"link"},
    }[experiment]
    for parameters in grid:
        if not isinstance(parameters, ExperimentParameters):
            raise ValueError(
                f"experiment {experiment} grid must contain "
                "ExperimentParameters"
            )
        baseline = ExperimentParameters(
            d=parameters.d,
            n_over_d=parameters.n_over_d,
        )
        changed = {
            name
            for name in _PARAMETER_COLUMNS
            if getattr(parameters, name) != getattr(baseline, name)
        }
        if not changed <= varying_fields:
            unexpected = ", ".join(sorted(changed - varying_fields))
            raise ValueError(
                f"comparison experiment {experiment} grid changes "
                f"unsupported parameters: {unexpected}"
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


def _case_id(experiment: str, parameters: ExperimentParameters) -> str:
    encoded = ";".join(
        f"{name}={getattr(parameters, name)}"
        for name in _PARAMETER_COLUMNS
    )
    return f"experiment={experiment};{encoded}"


def _initial_beta(d: int, seed: int) -> np.ndarray:
    beta = np.random.default_rng(seed).normal(size=d)
    norm = float(np.linalg.norm(beta))
    if not math.isfinite(norm) or norm <= np.finfo(float).eps:
        raise RuntimeError("failed to generate a finite initial beta")
    return np.asarray(beta / norm, dtype=float)


def _input_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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


def _result_vector(
    result: Any,
    attribute: str,
    dimension: int,
) -> np.ndarray:
    value = getattr(result, attribute, None)
    if value is None:
        return np.array([], dtype=float)
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        return np.array([], dtype=float)
    return vector


def _mapping_float(values: Mapping[str, object], key: str) -> float:
    try:
        result = float(values.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


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


def _summary_context_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    preferred = (
        "experiment",
        "case_id",
        "d",
        "n",
        "n_over_d",
        *(
            name
            for name in _PARAMETER_COLUMNS
            if name not in {"d", "n_over_d"}
        ),
        "requested_n_over_d",
        "actual_n_over_d",
    )
    return tuple(
        column for column in preferred if column in frame.columns
    )


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


def _safe_model_directory(name: str) -> str:
    safe = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "-"
        for character in name
    ).strip(".-")
    if not safe:
        safe = "model"
    if safe != name:
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}-{suffix}"
    return safe


def _clear_model_set_line_plots(plots: Path) -> None:
    """Remove only obsolete line charts before rewriting a model-set run."""

    for filename in (
        "runtime_vs_dimension.png",
        "memory_vs_dimension.png",
    ):
        (plots / filename).unlink(missing_ok=True)
    directories = (plots, *plots.glob("experiment_*"))
    for parent in directories:
        if not parent.is_dir():
            continue
        for directory_name in (
            "runtime_by_dimension",
            "memory_by_dimension",
            "cosine_abs_by_dimension",
        ):
            directory = parent / directory_name
            if not directory.is_dir():
                continue
            for plot in directory.glob("*.png"):
                plot.unlink()


def _dimension_label(dimension: object) -> str:
    """Return a stable, filename-safe label for an integer dimension."""

    value = float(dimension)
    if value.is_integer():
        return str(int(value))
    return format(value, ".12g")


def _line_plot_by_dimension(
    runs: pd.DataFrame,
    *,
    dimension: object,
    value: str,
    ylabel: str,
    experiment_label: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot one model line per implementation against n/d at fixed d."""

    dimension_value = float(dimension)
    dimension_runs = runs.loc[
        pd.to_numeric(runs["d"], errors="coerce").eq(dimension_value)
    ]
    prepared = (
        dimension_runs.groupby(["model", "n_over_d"], sort=True)[value]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ratios = sorted(prepared["n_over_d"].dropna().unique())
    ratio_positions = {ratio: position for position, ratio in enumerate(ratios)}
    for model, group in prepared.groupby("model", sort=True):
        group = group.sort_values("n_over_d")
        ax.plot(
            group["n_over_d"].map(ratio_positions),
            group[value],
            marker="o",
            label=str(model),
        )
    dimension_label = _dimension_label(dimension)
    if len(ratios):
        ax.set_xticks(
            range(len(ratios)),
            labels=[f"{ratio:g}" for ratio in ratios],
        )
    ax.set_xlabel("n/d")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by n/d (d={dimension_label}{experiment_label})")
    ax.grid(True, alpha=0.25)
    if not prepared.empty:
        ax.legend(title="Model", loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _line_plot(
    runs: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    path: Path,
    dpi: int,
) -> None:
    group_columns = ["model", "n_over_d", "d"]
    series_columns = ["model", "n_over_d"]
    if "experiment" in runs.columns:
        group_columns.insert(0, "experiment")
        series_columns.insert(0, "experiment")
    prepared = (
        runs.groupby(group_columns, sort=True, as_index=False)[value]
        .median()
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for keys, group in prepared.groupby(series_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        if "experiment" in runs.columns:
            experiment, model, ratio = key_values
            label = f"exp {experiment}, {model}, n/d={ratio:g}"
        else:
            model, ratio = key_values
            label = f"{model}, n/d={ratio:g}"
        group = group.sort_values("d")
        ax.plot(
            group["d"],
            group[value],
            marker="o",
            label=label,
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


def build_model_set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more ADP-compatible model implementations "
            "defined by a Python MODELS mapping."
        )
    )
    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="Python file containing an ordered MODELS factory mapping.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--experiments",
        default="2",
        help=(
            "Experiments to compare: comma list such as 2,3,5 or "
            "inclusive integer range such as 2:6. Supported: 2-6."
        ),
    )
    parser.add_argument(
        "--d",
        default=None,
        help=(
            "Comma-separated dimensions for a manual experiment-2 grid. "
            "Requires --n-over-d."
        ),
    )
    parser.add_argument(
        "--n-over-d",
        default=None,
        help=(
            "Comma-separated sample ratios for a manual experiment-2 grid. "
            "Requires --d."
        ),
    )
    parser.add_argument(
        "--seeds",
        help=(
            "Inclusive START:STOP range or comma-separated seeds; defaults "
            "to 0 for smoke and 0:99 for full."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help=(
            "Parallel comparison groups; models inside a group are "
            "sequential and isolated. Use 1 for latency measurements."
        ),
    )
    parser.add_argument(
        "--sample-interval",
        type=_positive_float,
        default=0.01,
    )
    parser.add_argument("--dpi", type=_positive_int, default=160)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--require-equivalent",
        action="store_true",
        help=(
            "Return a nonzero status when a complete candidate result is "
            "not numerically equivalent to the baseline."
        ),
    )
    parser.add_argument(
        "--use-model-initializers",
        action="store_true",
        help=(
            "Do not pass the shared generated beta0 to fit(); let each "
            "model's beta_initializer stage choose beta_ref."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run_model_set_command(argv: list[str] | None = None) -> int:
    parser = build_model_set_parser()
    args = parser.parse_args(argv)
    if (args.d is None) != (args.n_over_d is None):
        parser.error("--d and --n-over-d must be provided together")
    try:
        specs = load_model_specs(args.models)
        experiments = _parse_comparison_experiments(args.experiments)
        manual_d = (
            None
            if args.d is None
            else _parse_positive_int_selection(args.d, name="d")
        )
        manual_n_over_d = (
            None
            if args.n_over_d is None
            else _parse_positive_float_selection(
                args.n_over_d,
                name="n-over-d",
            )
        )
        if manual_d is not None and experiments != ("2",):
            raise ValueError(
                "--d and --n-over-d can only be used with --experiments 2"
            )
        seeds = (
            parse_seed_selection(args.seeds)
            if args.seeds is not None
            else (
                tuple(range(100))
                if args.profile == "full"
                else (0,)
            )
        )
    except (ImportError, OSError, SyntaxError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if manual_d is not None and manual_n_over_d is not None:
        experiment_parameter_grids = {
            "2": tuple(
                ExperimentParameters(d=d, n_over_d=n_over_d)
                for d in manual_d
                for n_over_d in manual_n_over_d
            )
        }
    else:
        grid_builder = (
            full_parameter_grid
            if args.profile == "full"
            else smoke_parameter_grid
        )
        experiment_parameter_grids = {
            experiment: grid_builder(experiment)
            for experiment in experiments
        }
    names = tuple(spec.name for spec in specs)
    runs = compare_model_set(
        tuple(spec.model for spec in specs),
        model_names=names,
        experiment_parameter_grids=experiment_parameter_grids,
        seeds=seeds,
        sample_interval_sec=args.sample_interval,
        jobs=args.jobs,
        show_progress=not args.no_progress,
        continue_on_error=True,
        use_model_initializers=args.use_model_initializers,
    )
    configuration: dict[str, object] = {
        "command": [
            sys.executable,
            "run_benchmarks.py",
            "compare",
            *(sys.argv[2:] if argv is None else argv),
        ],
        "models_file": str(args.models.expanduser().resolve()),
        "profile": args.profile,
        "experiments": list(experiments),
        "manual_d": None if manual_d is None else list(manual_d),
        "manual_n_over_d": (
            None
            if manual_n_over_d is None
            else list(manual_n_over_d)
        ),
        "seeds": list(seeds),
        "jobs": args.jobs,
        "sample_interval_sec": args.sample_interval,
        "dpi": args.dpi,
        "require_equivalent": args.require_equivalent,
        "use_model_initializers": args.use_model_initializers,
    }
    artifacts = write_model_set_artifacts(
        runs,
        args.output,
        model_names=names,
        configuration=configuration,
        dpi=args.dpi,
    )
    _print_model_set_summary(runs)
    for name, path in artifacts.items():
        print(f"{name}: {path}")

    failures = int(runs["status"].ne("ok").sum())
    if failures:
        print(
            f"model fits failed: {failures}/{len(runs)}",
            file=sys.stderr,
        )
        return 1
    if args.require_equivalent:
        comparisons = pair_model_set_runs(runs, model_names=names)
        inequivalent = int(
            (
                comparisons["result_pair_finite"]
                & ~comparisons["numerically_equivalent"]
            ).sum()
        )
        if inequivalent:
            print(
                f"candidate results are not numerically equivalent in "
                f"{inequivalent}/{len(comparisons)} paired runs",
                file=sys.stderr,
            )
            return 2
    return 0


def _print_model_set_summary(runs: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for model, group in runs.groupby("model", sort=False):
        successful = group.loc[group["status"].eq("ok")]
        rows.append(
            {
                "model": model,
                "success": f"{len(successful)}/{len(group)}",
                "median_fit_sec": _finite_median(
                    successful["fit_time_sec"]
                ),
                "median_peak_mib": _finite_median(
                    successful["rss_peak_delta_mib"]
                ),
                "median_cosine_abs": _finite_median(
                    successful["cosine_abs"]
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.loc[np.isfinite(numeric)]
    return math.nan if finite.empty else float(finite.median())


def _parse_comparison_experiments(value: str) -> tuple[str, ...]:
    selection = str(value).strip()
    if ":" in selection:
        if selection.count(":") != 1 or "," in selection:
            raise ValueError(
                "experiment range must have the form START:STOP"
            )
        start_text, stop_text = (
            part.strip() for part in selection.split(":")
        )
        try:
            start = int(start_text)
            stop = int(stop_text)
        except ValueError as exc:
            raise ValueError(
                "experiment range must contain integers"
            ) from exc
        if start > stop:
            raise ValueError(
                "experiment range start must not exceed stop"
            )
        experiments = tuple(str(number) for number in range(start, stop + 1))
    else:
        experiments = parse_experiment_selectors(selection)
    unknown = sorted(set(experiments) - set(_COMPARISON_EXPERIMENTS))
    if unknown:
        raise ValueError(
            "comparison supports only experiments 2-6; unknown: "
            + ", ".join(unknown)
        )
    if not experiments:
        raise ValueError("experiments must not be empty")
    return tuple(
        experiment
        for experiment in _COMPARISON_EXPERIMENTS
        if experiment in set(experiments)
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        )
    return parsed


def _parse_positive_int_selection(
    value: str,
    *,
    name: str,
) -> tuple[int, ...]:
    parts = _parse_selection_parts(value, name=name)
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must contain positive integers") from exc
    if any(number < 1 for number in values):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(dict.fromkeys(values))


def _parse_positive_float_selection(
    value: str,
    *,
    name: str,
) -> tuple[float, ...]:
    parts = _parse_selection_parts(value, name=name)
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain finite positive numbers"
        ) from exc
    if any(
        not math.isfinite(number) or number <= 0.0
        for number in values
    ):
        raise ValueError(f"{name} must contain finite positive numbers")
    return tuple(dict.fromkeys(values))


def _parse_selection_parts(value: str, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} selection must not be empty")
    parts = tuple(part.strip() for part in value.split(","))
    if any(not part for part in parts):
        raise ValueError(f"{name} selection contains an empty value")
    return parts


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
