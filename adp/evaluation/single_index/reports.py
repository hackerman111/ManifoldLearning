from __future__ import annotations

import csv
import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...common.experiment_log import SCHEMA_VERSION
from ...common.plotting import apply_adp_axis_style, save_figure, set_adp_figure_size
from .schema import ARTIFACT_COLUMNS


SUMMARY_METRICS = (
    "cosine_abs",
    "angle_deg",
    "signed_l2",
    "objective",
    "result_persist_time_sec",
    "algorithm_time_sec",
    "algorithm_rss_min_mib",
    "algorithm_rss_mean_mib",
    "algorithm_rss_max_mib",
    "algorithm_rss_peak_delta_mib",
    "full_run_time_sec",
    "full_run_rss_min_mib",
    "full_run_rss_mean_mib",
    "full_run_rss_max_mib",
    "full_run_rss_peak_delta_mib",
)

RUN_REPORT_COLUMNS = (
    "schema_version",
    "series_id",
    "run_id",
    "scenario_id",
    "family",
    "method",
    "repeat",
    "data_seed",
    "status",
    "failed",
    "dataset_source",
    "dataset_path",
    "dataset_size_bytes",
    "dataset_sha256",
    "dataset_rows",
    "dataset_features",
    *SUMMARY_METRICS,
)

PARAMETER_REPORT_COLUMNS = (
    "run_id",
    "data_n",
    "data_d",
    "data_link",
    "data_noise",
    "data_corr",
    "data_sigma_x",
    "algorithm_n_centers",
    "algorithm_n_directions",
    "algorithm_min_neighbors",
    "solver_outer_steps",
    "solver_inner_steps",
)

ITERATION_REPORT_COLUMNS = (
    "run_id",
    "scenario_id",
    "method",
    "outer_k",
    "h_k",
    "rho_k",
    "local_mass_mean",
    "local_mass_q05",
    "local_mass_min",
    "objective",
    "cosine_abs",
    "beta_delta",
    "runtime_sec",
)

SOLVER_REPORT_COLUMNS = (
    "run_id",
    "scenario_id",
    "method",
    "outer_k",
    "inner_k",
    "cg_k",
    "relative_objective",
    "relative_residual",
    "projective_delta",
    "cg_info",
)

FAILURE_REPORT_COLUMNS = (
    "run_id",
    "scenario_id",
    "method",
    "status",
    "category",
    "exception_type",
    "stage",
)

_TEXT_COLUMNS = {
    "series_id",
    "run_id",
    "scenario_id",
    "family",
    "method",
    "status",
    "data_link",
    "category",
    "exception_type",
    "stage",
    "dataset_source",
    "dataset_path",
    "dataset_sha256",
}

SCALING_COLUMNS = (
    "scenario_id",
    "method",
    "x_column",
    "y_column",
    "point_count",
    "x_unique",
    "exponent",
    "intercept",
    "r_squared",
)

PAIRED_COLUMNS = (
    "scenario_id",
    "reference_method",
    "comparison_method",
    "pair_count",
    "cosine_abs_delta_mean",
    "cosine_abs_delta_median",
    "cosine_abs_delta_bootstrap_ci95_low",
    "cosine_abs_delta_bootstrap_ci95_high",
)

_SCALING_SPECS = {
    "M01": ("data_n", "algorithm_time_sec"),
    "M02": ("data_d", "algorithm_time_sec"),
    "M03": ("algorithm_n_centers", "algorithm_time_sec"),
    "M04": ("algorithm_n_directions", "algorithm_time_sec"),
    "M05": ("iteration_budget", "algorithm_time_sec"),
    "M06": ("data_n", "full_run_rss_peak_delta_mib"),
}

_PLOT_SPECS = {
    "G01": ("runs", "data_n", "direction_loss", "Размер выборки", "Ошибка направления", "Восстановление с ростом выборки"),
    "G02": ("runs", "data_noise", "success_value", "Уровень шума", "Доля успешных запусков", "Граница устойчивости к шуму"),
    "G03": ("runs", "data_d", "success_value", "Размерность", "Доля успешных запусков", "Граница по размерности"),
    "G04": ("runs", "algorithm_n_directions", "direction_loss", "Число направлений", "Ошибка направления", "Бюджет направлений"),
    "G05": ("runs", "algorithm_n_directions", "algorithm_time_sec", "Число направлений", "Время алгоритма, с", "Стоимость направленного sketch"),
    "G06": ("runs", "algorithm_n_centers", "direction_loss", "Число центров", "Ошибка направления", "Чувствительность к числу центров"),
    "G07": ("runs", "algorithm_min_neighbors", "direction_loss", "Минимальная локальная масса", "Ошибка направления", "Локальная масса и качество"),
    "G08": ("runs", "algorithm_lambda_penalty", "direction_loss", "Относительная регуляризация", "Ошибка направления", "Регуляризация и устойчивость"),
    "G09": ("runs", "initial_cosine", "success_value", "Начальный косинус", "Доля успешных запусков", "Область притяжения"),
    "G10": ("runs", "data_corr", "cosine_abs", "Уровень возмущения", "Медианный модуль косинуса", "Устойчивость к возмущениям"),
    "G11": ("runs", "algorithm_time_sec", "direction_loss", "Время алгоритма, с", "Ошибка направления", "Фронт качество-время"),
    "G12": ("scaling", "fitted", "observed", "Предсказанное время", "Наблюдаемое время", "Качество модели масштабирования"),
    "G13": ("iterations", "outer_k", "cosine_abs", "Внешний шаг", "Модуль косинуса", "Качество по внешним шагам"),
    "G14": ("iterations", "outer_k", "h_k", "Внешний шаг", "Ширина окна", "Траектория ширины окна"),
    "G15": ("iterations", "outer_k", "rho_k", "Внешний шаг", "Анизотропия", "Траектория анизотропии"),
    "G16": ("iterations", "outer_k", "local_mass_q05", "Внешний шаг", "Квантиль локальной массы", "Локальная масса по шагам"),
    "G17": ("solver", "inner_k", "relative_objective", "Внутренний шаг", "Относительный функционал", "Сходимость ALS"),
    "G18": ("solver", "inner_k", "projective_delta", "Внутренний шаг", "Проективное изменение", "Стабилизация направления"),
    "G19": ("solver", "cg_k", "relative_residual", "Итерация CG", "Относительная невязка", "Сходимость CG"),
    "G20": ("iterations", "center_rank", "local_slope_norm", "Ранг центра", "Квадрат локального наклона", "Распределение локальных наклонов"),
}


def build_single_index_summary(
    runs: pd.DataFrame,
    *,
    bootstrap_resamples: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Aggregate run-level status, quality, timing, and memory statistics."""

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    _require_columns(runs, ("scenario_id", "method", "status"), "runs")
    source = runs.copy()
    if "family" not in source:
        source["family"] = source["scenario_id"].astype(str).str[:1]
    rows: list[dict[str, Any]] = []
    for (scenario_id, family, method), group in source.groupby(
        ["scenario_id", "family", "method"], sort=True, dropna=False
    ):
        statuses = group["status"].astype(str)
        total = len(group)
        successes = int((statuses == "success").sum())
        failed = int((statuses == "failed").sum())
        unavailable = int((statuses == "unavailable").sum())
        success_low, success_high = wilson_interval(successes, total)
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "family": family,
            "method": method,
            "total_count": total,
            "success_count": successes,
            "failed_count": failed,
            "unavailable_count": unavailable,
            "success_rate": successes / total if total else math.nan,
            "failure_rate": failed / total if total else math.nan,
            "unavailable_rate": unavailable / total if total else math.nan,
            "success_ci95_low": success_low,
            "success_ci95_high": success_high,
        }
        successful_group = group.loc[statuses == "success"]
        for metric in SUMMARY_METRICS:
            values = (
                _finite_values(successful_group[metric])
                if metric in successful_group
                else np.array([])
            )
            row.update(
                _metric_summary(
                    metric,
                    values,
                    bootstrap_resamples=bootstrap_resamples,
                    random_state=_derived_seed(random_state, scenario_id, method, metric),
                )
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=_summary_columns())


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must define a valid binomial count")
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int = 1000,
    random_state: int = 0,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> tuple[float, float]:
    array = _finite_values(pd.Series(values))
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1:
        value = float(statistic(array))
        return value, value
    rng = np.random.default_rng(random_state)
    samples = rng.choice(array, size=(resamples, array.size), replace=True)
    if statistic is np.median:
        estimates = np.median(samples, axis=1)
    else:
        estimates = np.asarray([statistic(sample) for sample in samples], dtype=float)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def select_worst_five(
    runs: pd.DataFrame,
    *,
    metric: str = "cosine_abs",
) -> pd.DataFrame:
    _require_columns(runs, ("status", metric), "runs")
    values = pd.to_numeric(runs[metric], errors="coerce")
    selected = runs.loc[(runs["status"] == "success") & np.isfinite(values)].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    return selected.sort_values(metric, ascending=True, kind="stable").head(5).reset_index(drop=True)


def fit_scaling_exponents(runs: pd.DataFrame) -> pd.DataFrame:
    _require_columns(runs, ("scenario_id", "method", "status"), "runs")
    source = runs.copy()
    if {"solver_outer_steps", "solver_inner_steps"}.issubset(source.columns):
        source["iteration_budget"] = (
            pd.to_numeric(source["solver_outer_steps"], errors="coerce")
            * pd.to_numeric(source["solver_inner_steps"], errors="coerce")
        )
    rows = []
    for (scenario_id, method), group in source.groupby(
        ["scenario_id", "method"], sort=True
    ):
        if scenario_id not in _SCALING_SPECS:
            continue
        x_column, y_column = _SCALING_SPECS[str(scenario_id)]
        fit = _log_log_fit(group, x_column, y_column)
        rows.append(
            {
                "scenario_id": scenario_id,
                "method": method,
                "x_column": x_column,
                "y_column": y_column,
                **fit,
            }
        )
    return pd.DataFrame(rows, columns=SCALING_COLUMNS)


def paired_method_differences(
    runs: pd.DataFrame,
    *,
    reference_method: str = "full_adp",
    bootstrap_resamples: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    required = ("scenario_id", "method", "repeat", "data_seed", "status", "cosine_abs")
    _require_columns(runs, required, "runs")
    success = runs.loc[runs["status"] == "success", list(required)].copy()
    reference = success.loc[success["method"] == reference_method]
    rows = []
    keys = ["scenario_id", "repeat", "data_seed"]
    for comparison_method in sorted(set(success["method"]) - {reference_method}):
        comparison = success.loc[success["method"] == comparison_method]
        paired = reference.merge(comparison, on=keys, suffixes=("_reference", "_comparison"))
        for scenario_id, group in paired.groupby("scenario_id", sort=True):
            delta = _finite_values(
                pd.to_numeric(group["cosine_abs_reference"], errors="coerce")
                - pd.to_numeric(group["cosine_abs_comparison"], errors="coerce")
            )
            low, high = bootstrap_interval(
                delta,
                resamples=bootstrap_resamples,
                random_state=_derived_seed(
                    random_state, scenario_id, reference_method, comparison_method
                ),
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "pair_count": int(delta.size),
                    "cosine_abs_delta_mean": float(np.mean(delta)) if delta.size else math.nan,
                    "cosine_abs_delta_median": float(np.median(delta)) if delta.size else math.nan,
                    "cosine_abs_delta_bootstrap_ci95_low": low,
                    "cosine_abs_delta_bootstrap_ci95_high": high,
                }
            )
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def write_single_index_reports(
    series_dir: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    random_state: int = 0,
    dpi: int = 150,
) -> Mapping[str, Path]:
    """Rebuild numeric reports and G01-G21 artifacts from primary CSV files."""

    root = Path(series_dir)
    frames = _read_primary_frames(root)
    runs = frames["runs"]
    summary = build_single_index_summary(
        runs,
        bootstrap_resamples=bootstrap_resamples,
        random_state=random_state,
    )
    scaling = fit_scaling_exponents(runs)
    paired = paired_method_differences(
        runs,
        bootstrap_resamples=bootstrap_resamples,
        random_state=random_state,
    )
    worst = select_worst_five(runs)

    saved: dict[str, Path] = {}
    numeric = {
        "summary": (root / "single_index_summary.csv", summary),
        "scaling": (root / "single_index_scaling.csv", scaling),
        "paired": (root / "single_index_paired_differences.csv", paired),
        "worst_five": (root / "single_index_worst_five.csv", worst),
    }
    artifact_rows = []
    series_id = _series_id(runs, root)
    for name, (path, frame) in numeric.items():
        _atomic_frame_csv(frame, path)
        saved[name] = path
        artifact_rows.append(_artifact_row(series_id, root, name, path))

    frames = {**frames, "scaling": scaling}
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    for index in range(1, 22):
        plot_id = f"G{index:02d}"
        path = plots_dir / f"{plot_id}.png"
        try:
            _render_plot(plot_id, frames, path, dpi=dpi)
        except Exception as exc:
            path.unlink(missing_ok=True)
            artifact_rows.append(
                _artifact_row(series_id, root, plot_id, path, error=exc)
            )
        else:
            saved[plot_id] = path
            artifact_rows.append(_artifact_row(series_id, root, plot_id, path))

    artifacts_path = root / "single_index_artifacts.csv"
    _merge_artifacts(artifacts_path, artifact_rows)
    saved["artifacts"] = artifacts_path
    return saved


def _read_primary_frames(root: Path) -> dict[str, pd.DataFrame]:
    runs = _read_selected_csv(
        root / "single_index_runs.csv",
        RUN_REPORT_COLUMNS,
        required=("run_id", "scenario_id", "method", "status"),
    )
    parameters = _read_selected_csv(
        root / "single_index_initial_parameters.csv",
        PARAMETER_REPORT_COLUMNS,
        required=("run_id",),
    )
    parameter_values = parameters.drop(
        columns=[column for column in parameters if column != "run_id" and column in runs],
        errors="ignore",
    )
    runs = runs.merge(parameter_values, on="run_id", how="left", validate="one_to_one")
    runs["success_value"] = (runs["status"] == "success").astype(float)
    runs["direction_loss"] = 1.0 - pd.to_numeric(
        runs.get("cosine_abs"), errors="coerce"
    )
    iterations = _read_selected_csv(
        root / "single_index_iterations.csv",
        ITERATION_REPORT_COLUMNS,
        required=("run_id", "scenario_id", "method", "outer_k"),
    )
    solver = _read_selected_csv(
        root / "single_index_solver_iterations.csv",
        SOLVER_REPORT_COLUMNS,
        required=("run_id", "scenario_id", "method", "outer_k", "inner_k", "cg_k"),
    )
    failures = _read_selected_csv(
        root / "single_index_failures.csv",
        FAILURE_REPORT_COLUMNS,
        required=("run_id", "scenario_id", "method"),
    )
    return {
        "runs": runs,
        "iterations": iterations,
        "solver": solver,
        "failures": failures,
    }


def _read_selected_csv(
    path: Path,
    columns: Sequence[str],
    *,
    required: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = tuple(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"missing columns in {path.name}: {', '.join(missing)}")
    selected = [column for column in columns if column in header]
    dtype = {
        column: "string" if column in _TEXT_COLUMNS else "float64"
        for column in selected
    }
    return pd.read_csv(path, usecols=selected, dtype=dtype)


def _metric_summary(
    metric: str,
    values: np.ndarray,
    *,
    bootstrap_resamples: int,
    random_state: int,
) -> dict[str, Any]:
    prefix = f"{metric}_"
    if values.size == 0:
        return {
            prefix + suffix: 0 if suffix == "count" else math.nan
            for suffix in (
                "count",
                "mean",
                "median",
                "iqr",
                "q05",
                "q95",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            )
        }
    q05, q25, q75, q95 = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
    low, high = bootstrap_interval(
        values,
        resamples=bootstrap_resamples,
        random_state=random_state,
    )
    return {
        prefix + "count": int(values.size),
        prefix + "mean": float(np.mean(values)),
        prefix + "median": float(np.median(values)),
        prefix + "iqr": float(q75 - q25),
        prefix + "q05": float(q05),
        prefix + "q95": float(q95),
        prefix + "bootstrap_ci95_low": low,
        prefix + "bootstrap_ci95_high": high,
    }


def _summary_columns() -> list[str]:
    columns = [
        "scenario_id",
        "family",
        "method",
        "total_count",
        "success_count",
        "failed_count",
        "unavailable_count",
        "success_rate",
        "failure_rate",
        "unavailable_rate",
        "success_ci95_low",
        "success_ci95_high",
    ]
    for metric in SUMMARY_METRICS:
        columns.extend(
            f"{metric}_{suffix}"
            for suffix in (
                "count",
                "mean",
                "median",
                "iqr",
                "q05",
                "q95",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            )
        )
    return columns


def _log_log_fit(group: pd.DataFrame, x_column: str, y_column: str) -> dict[str, Any]:
    if x_column not in group or y_column not in group:
        return _empty_scaling_fit()
    x = pd.to_numeric(group[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group[y_column], errors="coerce").to_numpy(dtype=float)
    success = group["status"].to_numpy() == "success"
    valid = success & np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.unique(x).size < 2:
        return _empty_scaling_fit(point_count=x.size, x_unique=np.unique(x).size)
    log_x = np.log(x)
    log_y = np.log(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + exponent * log_x
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return {
        "point_count": int(x.size),
        "x_unique": int(np.unique(x).size),
        "exponent": float(exponent),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def _empty_scaling_fit(*, point_count: int = 0, x_unique: int = 0) -> dict[str, Any]:
    return {
        "point_count": int(point_count),
        "x_unique": int(x_unique),
        "exponent": math.nan,
        "intercept": math.nan,
        "r_squared": math.nan,
    }


def _render_plot(
    plot_id: str,
    frames: Mapping[str, pd.DataFrame],
    path: Path,
    *,
    dpi: int,
) -> None:
    if plot_id == "G21":
        _render_failure_plot(frames["failures"], path, dpi=dpi)
        return
    if plot_id not in _PLOT_SPECS:
        raise ValueError(f"unknown report plot: {plot_id}")
    source_name, x_column, y_column, xlabel, ylabel, title = _PLOT_SPECS[plot_id]
    source = frames[source_name]
    _require_columns(source, (x_column, y_column), source_name)
    plot_data = source[[x_column, y_column]].copy()
    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_data.empty:
        raise ValueError(f"no finite data for {plot_id}")

    import matplotlib.pyplot as plt

    aggregate = plot_data.groupby(x_column, as_index=False)[y_column].median()
    fig, ax = plt.subplots()
    set_adp_figure_size(fig)
    ax.plot(aggregate[x_column], aggregate[y_column], marker="o", linewidth=1.8)
    apply_adp_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    save_figure(fig, path, dpi=dpi, close=True)


def _render_failure_plot(failures: pd.DataFrame, path: Path, *, dpi: int) -> None:
    _require_columns(failures, ("scenario_id", "category"), "failures")
    source = failures.dropna(subset=["scenario_id", "category"])
    if source.empty:
        raise ValueError("no failure data for G21")
    shares = pd.crosstab(source["scenario_id"], source["category"], normalize="index")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    set_adp_figure_size(fig, width=max(8.0, 0.8 * len(shares)))
    shares.plot(kind="bar", stacked=True, ax=ax)
    apply_adp_axis_style(
        ax,
        xlabel="Сценарий",
        ylabel="Доля категории",
        title="Причины неуспешных запусков",
        legend_title="Категория",
        x_rotation=30,
    )
    save_figure(fig, path, dpi=dpi, close=True)


def _artifact_row(
    series_id: str,
    root: Path,
    name: str,
    path: Path,
    *,
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "series_id": series_id,
        "artifact_type": path.suffix.lstrip(".") or "file",
        "name": name,
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "status": "error" if error is not None else "created",
        "error": f"{type(error).__name__}: {error}" if error is not None else "",
    }


def _merge_artifacts(path: Path, new_rows: Sequence[Mapping[str, Any]]) -> None:
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ARTIFACT_COLUMNS)
    names = {str(row["name"]) for row in new_rows}
    if "name" in existing:
        existing = existing.loc[~existing["name"].astype(str).isin(names)]
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.reindex(columns=ARTIFACT_COLUMNS)
    _atomic_frame_csv(combined, path)


def _atomic_frame_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _series_id(runs: pd.DataFrame, root: Path) -> str:
    if "series_id" in runs and not runs["series_id"].dropna().empty:
        return str(runs["series_id"].dropna().iloc[0])
    series_path = root / "single_index_series.csv"
    if series_path.exists():
        with series_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), {})
        return str(row.get("series_id", ""))
    return ""


def _finite_values(values: pd.Series | Sequence[float]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    array = np.asarray(numeric, dtype=float)
    return array[np.isfinite(array)]


def _derived_seed(base: int, *parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns in {name}: {', '.join(missing)}")


__all__ = [
    "bootstrap_interval",
    "build_single_index_summary",
    "fit_scaling_exponents",
    "paired_method_differences",
    "select_worst_five",
    "wilson_interval",
    "write_single_index_reports",
]
