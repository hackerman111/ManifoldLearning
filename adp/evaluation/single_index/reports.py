from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...common.experiment_log import SCHEMA_VERSION
from ...common.plotting import ensure_matplotlib_config_dir
from .plots import (
    boxplot,
    grouped_line,
    heatmap,
    line_with_quantile_band,
    scatter,
    stacked_runtime,
)
from .schema import ARTIFACT_COLUMNS, PUBLIC_TABLE_COLUMNS
from .types import EXPERIMENT_SELECTORS


PlotKind = Literal[
    "quantile",
    "median_line",
    "mean_line",
    "box",
    "scatter",
    "heatmap",
    "stacked",
]


@dataclass(frozen=True, slots=True)
class PlotSpec:
    filename: str
    selectors: tuple[str, ...]
    table: str
    kind: PlotKind
    x: str
    y: str
    title: str
    xlabel: str
    ylabel: str
    groups: tuple[str, ...] = ()
    value: str | None = None
    components: tuple[str, ...] = ()
    diagnostic: bool = False
    log_x: bool = False
    log_y: bool = False


_DIAGNOSTIC_SELECTOR = ("1",)
_RUNTIME_COMPONENTS = (
    "distance_time_sec",
    "weights_time_sec",
    "statistics_time_sec",
    "optimization_time_sec",
    "bandwidth_update_time_sec",
    "service_overhead_sec",
)


PLOT_MANIFEST = (
    PlotSpec(
        "quality_vs_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "outer",
        "quantile",
        "outer_k",
        "cosine_abs",
        "Direction quality by outer iteration",
        "Outer iteration",
        "Absolute cosine",
        diagnostic=True,
    ),
    PlotSpec(
        "bandwidth_vs_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "outer",
        "quantile",
        "outer_k",
        "h_k",
        "Bandwidth by outer iteration",
        "Outer iteration",
        "Bandwidth",
        groups=("d", "n_over_d"),
        diagnostic=True,
    ),
    PlotSpec(
        "rho_vs_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "outer",
        "quantile",
        "outer_k",
        "rho_k",
        "Anisotropy by outer iteration",
        "Outer iteration",
        "Rho",
        diagnostic=True,
    ),
    PlotSpec(
        "beta_step_vs_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "outer",
        "quantile",
        "outer_k",
        "beta_delta",
        "Direction step by outer iteration",
        "Outer iteration",
        "Direction step",
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "objective_vs_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "outer",
        "quantile",
        "outer_k",
        "objective_after",
        "Objective by outer iteration",
        "Outer iteration",
        "Objective",
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "objective_vs_inner_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "inner",
        "quantile",
        "inner_k",
        "objective",
        "Objective by inner iteration",
        "Inner iteration",
        "Objective",
        groups=("outer_k",),
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "beta_step_vs_inner_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "inner",
        "quantile",
        "inner_k",
        "beta_delta",
        "Direction step by inner iteration",
        "Inner iteration",
        "Direction step",
        groups=("outer_k",),
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "solver_residual_vs_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "solver",
        "quantile",
        "solver_k",
        "relative_residual",
        "Linear-solver residual",
        "Solver iteration",
        "Relative residual",
        groups=("outer_k", "inner_k"),
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "local_mass_by_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "local",
        "box",
        "outer_k",
        "local_mass",
        "Local mass by outer iteration",
        "Outer iteration",
        "Local mass",
        diagnostic=True,
    ),
    PlotSpec(
        "effective_neighbors_by_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "local",
        "box",
        "outer_k",
        "ess",
        "Effective neighbors by outer iteration",
        "Outer iteration",
        "Effective neighbors",
        diagnostic=True,
    ),
    PlotSpec(
        "local_condition_by_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "local",
        "box",
        "outer_k",
        "condition",
        "Local condition by outer iteration",
        "Outer iteration",
        "Condition number",
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "mass_vs_condition.png",
        _DIAGNOSTIC_SELECTOR,
        "local",
        "scatter",
        "local_mass",
        "condition",
        "Local mass versus condition",
        "Local mass",
        "Condition number",
        groups=("outer_k",),
        diagnostic=True,
        log_y=True,
    ),
    PlotSpec(
        "local_slopes_by_outer_iteration.png",
        _DIAGNOSTIC_SELECTOR,
        "local",
        "box",
        "outer_k",
        "slope",
        "Local slopes by outer iteration",
        "Outer iteration",
        "Local slope",
        diagnostic=True,
    ),
    PlotSpec(
        "quality_heatmap_d_nd_ratio.png",
        ("2",),
        "runs",
        "heatmap",
        "n_over_d",
        "d",
        "Direction quality over dimension and sample ratio",
        "n / d",
        "Dimension",
        value="cosine_abs",
    ),
    PlotSpec(
        "success_rate_heatmap.png",
        ("2",),
        "runs",
        "heatmap",
        "n_over_d",
        "d",
        "Success rate over dimension and sample ratio",
        "n / d",
        "Dimension",
        value="success_value",
    ),
    PlotSpec(
        "runtime_vs_dimension.png",
        ("2",),
        "runs",
        "median_line",
        "d",
        "runtime_sec",
        "Runtime versus dimension",
        "Dimension",
        "Runtime, seconds",
        groups=("n_over_d",),
        log_x=True,
        log_y=True,
    ),
    PlotSpec(
        "memory_vs_dimension.png",
        ("2",),
        "runs",
        "median_line",
        "d",
        "peak_memory_mb",
        "Peak memory versus dimension",
        "Dimension",
        "Peak memory, MB",
        groups=("n_over_d",),
        log_x=True,
    ),
    PlotSpec(
        "iterations_heatmap_d_nd_ratio.png",
        ("2",),
        "runs",
        "heatmap",
        "n_over_d",
        "d",
        "Outer iterations over dimension and sample ratio",
        "n / d",
        "Dimension",
        value="outer_iterations",
    ),
    PlotSpec(
        "quality_vs_sigma_eps.png",
        ("3",),
        "runs",
        "quantile",
        "sigma_eps",
        "cosine_abs",
        "Direction quality versus noise",
        "Noise standard deviation",
        "Absolute cosine",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "success_rate_vs_sigma_eps.png",
        ("3",),
        "runs",
        "mean_line",
        "sigma_eps",
        "success_value",
        "Success rate versus noise",
        "Noise standard deviation",
        "Success rate",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "runtime_vs_sigma_eps.png",
        ("3",),
        "runs",
        "quantile",
        "sigma_eps",
        "runtime_sec",
        "Runtime versus noise",
        "Noise standard deviation",
        "Runtime, seconds",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "outer_iterations_vs_sigma_eps.png",
        ("3",),
        "runs",
        "quantile",
        "sigma_eps",
        "outer_iterations",
        "Outer iterations versus noise",
        "Noise standard deviation",
        "Outer iterations",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "final_objective_vs_sigma_eps.png",
        ("3",),
        "runs",
        "quantile",
        "sigma_eps",
        "objective",
        "Final objective versus noise",
        "Noise standard deviation",
        "Final objective",
        groups=("d", "n_over_d"),
        log_y=True,
    ),
    PlotSpec(
        "quality_vs_correlation.png",
        ("4",),
        "runs",
        "quantile",
        "rho_corr",
        "cosine_abs",
        "Direction quality versus correlation",
        "AR(1) correlation",
        "Absolute cosine",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "success_rate_vs_correlation.png",
        ("4",),
        "runs",
        "mean_line",
        "rho_corr",
        "success_value",
        "Success rate versus correlation",
        "AR(1) correlation",
        "Success rate",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "local_condition_vs_correlation.png",
        ("4",),
        "outer",
        "quantile",
        "rho_corr",
        "condition_median",
        "Local condition versus correlation",
        "AR(1) correlation",
        "Median condition number",
        groups=("d", "n_over_d"),
        log_y=True,
    ),
    PlotSpec(
        "solver_iterations_vs_correlation.png",
        ("4",),
        "inner",
        "quantile",
        "rho_corr",
        "linear_solver_iterations",
        "Solver iterations versus correlation",
        "AR(1) correlation",
        "Linear-solver iterations",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "runtime_vs_correlation.png",
        ("4",),
        "runs",
        "quantile",
        "rho_corr",
        "runtime_sec",
        "Runtime versus correlation",
        "AR(1) correlation",
        "Runtime, seconds",
        groups=("d", "n_over_d"),
    ),
    PlotSpec(
        "quality_vs_sigma_x.png",
        ("5",),
        "runs",
        "quantile",
        "sigma_x",
        "cosine_abs",
        "Direction quality versus feature scale",
        "Feature scale",
        "Absolute cosine",
        groups=("d", "n_over_d"),
        log_x=True,
    ),
    PlotSpec(
        "h0_vs_sigma_x.png",
        ("5",),
        "runs",
        "quantile",
        "sigma_x",
        "h_initial",
        "Initial bandwidth versus feature scale",
        "Feature scale",
        "Initial bandwidth",
        groups=("d", "n_over_d"),
        log_x=True,
    ),
    PlotSpec(
        "final_bandwidth_vs_sigma_x.png",
        ("5",),
        "runs",
        "quantile",
        "sigma_x",
        "h_final",
        "Final bandwidth versus feature scale",
        "Feature scale",
        "Final bandwidth",
        groups=("d", "n_over_d"),
        log_x=True,
    ),
    PlotSpec(
        "local_mass_vs_sigma_x.png",
        ("5",),
        "outer",
        "quantile",
        "sigma_x",
        "local_mass_mean",
        "Local mass versus feature scale",
        "Feature scale",
        "Mean local mass",
        groups=("d", "n_over_d"),
        log_x=True,
    ),
    PlotSpec(
        "runtime_vs_sigma_x.png",
        ("5",),
        "runs",
        "quantile",
        "sigma_x",
        "runtime_sec",
        "Runtime versus feature scale",
        "Feature scale",
        "Runtime, seconds",
        groups=("d", "n_over_d"),
        log_x=True,
    ),
    PlotSpec(
        "quality_by_link_function.png",
        ("6",),
        "runs",
        "box",
        "link",
        "cosine_abs",
        "Direction quality by link function",
        "Link function",
        "Absolute cosine",
    ),
    PlotSpec(
        "success_rate_by_link_function.png",
        ("6",),
        "runs",
        "mean_line",
        "link",
        "success_value",
        "Success rate by link function",
        "Link function",
        "Success rate",
    ),
    PlotSpec(
        "outer_iterations_by_link_function.png",
        ("6",),
        "runs",
        "box",
        "link",
        "outer_iterations",
        "Outer iterations by link function",
        "Link function",
        "Outer iterations",
    ),
    PlotSpec(
        "objective_by_link_function.png",
        ("6",),
        "runs",
        "box",
        "link",
        "objective",
        "Final objective by link function",
        "Link function",
        "Final objective",
        log_y=True,
    ),
    PlotSpec(
        "local_slopes_by_link_function.png",
        ("6",),
        "local",
        "box",
        "link",
        "slope",
        "Local slopes by link function",
        "Link function",
        "Local slope",
    ),
    PlotSpec(
        "quality_by_x_distribution.png",
        ("7.1",),
        "runs",
        "box",
        "x_distribution",
        "cosine_abs",
        "Direction quality by feature distribution",
        "Feature distribution",
        "Absolute cosine",
    ),
    PlotSpec(
        "quality_by_noise_distribution.png",
        ("7.2",),
        "runs",
        "box",
        "noise_distribution",
        "cosine_abs",
        "Direction quality by noise distribution",
        "Noise distribution",
        "Absolute cosine",
    ),
    PlotSpec(
        "failure_rate_by_distribution.png",
        ("7.1", "7.2"),
        "runs",
        "mean_line",
        "distribution",
        "failure_value",
        "Numerical-failure rate by distribution",
        "Distribution",
        "Numerical-failure rate",
    ),
    PlotSpec(
        "runtime_by_distribution.png",
        ("7.1", "7.2"),
        "runs",
        "median_line",
        "distribution",
        "runtime_sec",
        "Runtime by distribution",
        "Distribution",
        "Runtime, seconds",
    ),
    PlotSpec(
        "quality_by_heteroscedasticity.png",
        ("8.1",),
        "runs",
        "box",
        "heteroscedastic",
        "cosine_abs",
        "Direction quality by heteroscedasticity",
        "Heteroscedastic",
        "Absolute cosine",
    ),
    PlotSpec(
        "quality_vs_outlier_fraction.png",
        ("8.2",),
        "runs",
        "quantile",
        "outlier_fraction",
        "cosine_abs",
        "Direction quality versus outlier fraction",
        "Outlier fraction",
        "Absolute cosine",
        groups=("outlier_scale",),
    ),
    PlotSpec(
        "failure_rate_vs_outliers.png",
        ("8.2",),
        "runs",
        "mean_line",
        "outlier_fraction",
        "failure_value",
        "Numerical-failure rate versus outliers",
        "Outlier fraction",
        "Numerical-failure rate",
        groups=("outlier_scale",),
    ),
    PlotSpec(
        "quality_vs_model_misspecification.png",
        ("8.3",),
        "runs",
        "quantile",
        "delta",
        "cosine_abs",
        "Direction quality versus model misspecification",
        "Misspecification strength",
        "Absolute cosine",
    ),
    PlotSpec(
        "objective_vs_model_misspecification.png",
        ("8.3",),
        "runs",
        "quantile",
        "delta",
        "objective",
        "Objective versus model misspecification",
        "Misspecification strength",
        "Final objective",
        log_y=True,
    ),
    PlotSpec(
        "runtime_breakdown.png",
        EXPERIMENT_SELECTORS,
        "outer",
        "stacked",
        "experiment",
        "iteration_time_sec",
        "Runtime breakdown",
        "Experiment",
        "Median time, seconds",
        components=_RUNTIME_COMPONENTS,
    ),
)


_PRIMARY_TABLES = {
    "runs": "run_summary",
    "outer": "outer_iterations",
    "inner": "inner_iterations",
    "local": "local_diagnostics",
    "solver": "solver_iterations",
}


def add_report_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Add explicit success/failure indicators without dropping failed runs."""

    _require_columns(runs, ("experiment", "status", "cosine_abs"), "run_summary")
    enriched = runs.copy()
    experiment = _normalise_experiment(enriched["experiment"])
    status = enriched["status"].astype("string")
    quality = pd.to_numeric(enriched["cosine_abs"], errors="coerce")
    threshold = np.where(experiment.eq("1"), 0.99, 0.9)
    numerical_failure = status.eq("numerical_failure").fillna(False)
    enriched["experiment"] = experiment
    enriched["success_value"] = (
        ~numerical_failure & np.isfinite(quality) & quality.ge(threshold)
    ).astype(float)
    enriched["failure_value"] = numerical_failure.astype(float)
    return enriched


def prepare_quantile_band(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    groups: Sequence[str] = (),
) -> pd.DataFrame:
    """Compute 5/50/95 percentiles inside every explicit experiment group."""

    keys = _unique_columns((*groups, x))
    _require_columns(frame, (*keys, y), "plot frame")
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & np.isfinite(source[y])]
    if source.empty:
        return pd.DataFrame(columns=(*keys, "q05", "median", "q95"))
    return (
        source.groupby(list(keys), sort=True, dropna=False, as_index=False)
        .agg(
            q05=(y, lambda values: values.quantile(0.05)),
            median=(y, "median"),
            q95=(y, lambda values: values.quantile(0.95)),
        )
        .reset_index(drop=True)
    )


def write_single_index_reports(
    series_dir: str | Path,
    *,
    dpi: int = 150,
) -> pd.DataFrame:
    """Rebuild every report artifact exclusively from committed public CSV files."""

    root = Path(series_dir)
    frames = _read_primary_frames(root)
    available = set(frames["runs"]["experiment"].dropna().astype(str))
    series_id = _series_id(frames["runs"])
    artifact_rows = _csv_artifact_rows(root, series_id)
    ensure_matplotlib_config_dir()

    for spec in PLOT_MANIFEST:
        path = _plot_path(root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        applicable = bool(set(spec.selectors) & available)
        if not applicable:
            artifact_rows.append(
                _artifact_row(
                    series_id,
                    root,
                    spec.filename,
                    path,
                    status="skipped",
                )
            )
            continue
        try:
            _render_plot(spec, frames, path, dpi=dpi)
        except Exception as exc:
            path.unlink(missing_ok=True)
            artifact_rows.append(
                _artifact_row(
                    series_id,
                    root,
                    spec.filename,
                    path,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            artifact_rows.append(
                _artifact_row(
                    series_id,
                    root,
                    spec.filename,
                    path,
                    status="created",
                )
            )

    return _write_artifact_manifest(root / "artifacts.csv", artifact_rows)


def _read_primary_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for alias, table in _PRIMARY_TABLES.items():
        path = root / f"{table}.csv"
        frames[alias] = _read_public_csv(
            path,
            expected=PUBLIC_TABLE_COLUMNS[table],
        )

    runs = add_report_metrics(frames["runs"])
    runs["distribution"] = pd.NA
    x_distribution = runs["experiment"].eq("7.1")
    noise_distribution = runs["experiment"].eq("7.2")
    runs.loc[x_distribution, "distribution"] = runs.loc[
        x_distribution, "x_distribution"
    ]
    runs.loc[noise_distribution, "distribution"] = runs.loc[
        noise_distribution, "noise_distribution"
    ]
    frames["runs"] = runs
    for alias in ("outer", "inner", "local", "solver"):
        frames[alias] = _enrich_detail(frames[alias], runs)
    return frames


def _read_public_csv(path: Path, *, expected: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = tuple(pd.read_csv(path, nrows=0).columns)
    missing = tuple(column for column in expected if column not in header)
    if missing:
        raise ValueError(f"missing columns in {path.name}: {', '.join(missing)}")
    frame = pd.read_csv(
        path,
        usecols=list(expected),
        dtype={
            "series_id": "string",
            "run_id": "string",
            "experiment": "string",
            "status": "string",
        },
        low_memory=False,
    )
    if "experiment" in frame:
        frame["experiment"] = _normalise_experiment(frame["experiment"])
    return frame


def _enrich_detail(detail: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in runs.columns
        if column == "run_id" or column not in detail.columns
    ]
    metadata = runs[metadata_columns].drop_duplicates("run_id", keep="last")
    return detail.merge(metadata, on="run_id", how="left", validate="many_to_one")


def _render_plot(
    spec: PlotSpec,
    frames: Mapping[str, pd.DataFrame],
    path: Path,
    *,
    dpi: int,
) -> None:
    source = _select_rows(spec, frames)
    common: dict[str, Any] = {
        "path": path,
        "xlabel": spec.xlabel,
        "ylabel": spec.ylabel,
        "title": spec.title,
        "dpi": dpi,
    }
    if spec.kind == "quantile":
        groups = _unique_columns(("experiment", *spec.groups))
        prepared = prepare_quantile_band(
            source,
            x=spec.x,
            y=spec.y,
            groups=groups,
        )
        prepared = _attach_group_labels(prepared, groups)
        line_with_quantile_band(
            prepared,
            x=spec.x,
            median="median",
            q05="q05",
            q95="q95",
            group="_group",
            log_x=spec.log_x,
            log_y=spec.log_y,
            **common,
        )
        return
    if spec.kind in {"median_line", "mean_line"}:
        aggregate = "median" if spec.kind == "median_line" else "mean"
        groups = _unique_columns(("experiment", *spec.groups))
        prepared = _prepare_aggregate_line(
            source,
            x=spec.x,
            y=spec.y,
            groups=groups,
            aggregate=aggregate,
        )
        prepared = _attach_group_labels(prepared, groups)
        grouped_line(
            prepared,
            x=spec.x,
            y=spec.y,
            group="_group",
            log_x=spec.log_x,
            log_y=spec.log_y,
            **common,
        )
        return
    if spec.kind == "box":
        boxplot(
            source,
            x=spec.x,
            y=spec.y,
            log_x=spec.log_x,
            log_y=spec.log_y,
            **common,
        )
        return
    if spec.kind == "scatter":
        groups = _unique_columns(("experiment", *spec.groups))
        prepared = _attach_group_labels(source, groups)
        scatter(
            prepared,
            x=spec.x,
            y=spec.y,
            group="_group",
            log_x=spec.log_x,
            log_y=spec.log_y,
            **common,
        )
        return
    if spec.kind == "heatmap":
        if spec.value is None:
            raise ValueError(f"heatmap {spec.filename} has no value column")
        heatmap(
            source,
            x=spec.x,
            y=spec.y,
            value=spec.value,
            **common,
        )
        return
    if spec.kind == "stacked":
        stacked_runtime(
            source,
            category=spec.x,
            components=spec.components,
            **common,
        )
        return
    raise ValueError(f"unknown plot kind: {spec.kind}")


def _select_rows(
    spec: PlotSpec,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    source = frames[spec.table]
    selected = source.loc[source["experiment"].isin(spec.selectors)].copy()
    if spec.diagnostic and "diagnostic" in selected:
        selected = selected.loc[_truthy(selected["diagnostic"])]
    return selected


def _prepare_aggregate_line(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    groups: Sequence[str],
    aggregate: Literal["mean", "median"],
) -> pd.DataFrame:
    keys = _unique_columns((*groups, x))
    _require_columns(frame, (*keys, y), "plot frame")
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & np.isfinite(source[y])]
    if source.empty:
        return pd.DataFrame(columns=(*keys, y))
    grouped = source.groupby(list(keys), sort=True, dropna=False, as_index=False)[y]
    if aggregate == "mean":
        return grouped.mean()
    return grouped.median()


def _attach_group_labels(
    frame: pd.DataFrame,
    groups: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    present = tuple(column for column in groups if column in result)
    if not present:
        result["_group"] = "all"
        return result
    if result.empty:
        result["_group"] = pd.Series(dtype="string")
        return result
    labels = result[list(present)].astype("string")
    result["_group"] = labels.apply(
        lambda row: ", ".join(
            f"{column}={row[column]}" for column in present
        ),
        axis=1,
    )
    return result


def _plot_path(root: Path, spec: PlotSpec) -> Path:
    if spec.diagnostic:
        selector = spec.selectors[0]
        return root / "plots" / f"experiment_{selector}" / spec.filename
    return root / "plots" / "summary" / spec.filename


def _csv_artifact_rows(root: Path, series_id: str) -> list[dict[str, Any]]:
    rows = []
    for table in PUBLIC_TABLE_COLUMNS:
        path = root / f"{table}.csv"
        rows.append(
            _artifact_row(
                series_id,
                root,
                table,
                path,
                status="created" if path.exists() or table == "artifacts" else "missing",
            )
        )
    return rows


def _artifact_row(
    series_id: str,
    root: Path,
    name: str,
    path: Path,
    *,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "series_id": series_id,
        "artifact_type": path.suffix.lstrip(".") or "file",
        "name": name,
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "status": status,
        "error": error,
    }


def _write_artifact_manifest(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).reindex(columns=ARTIFACT_COLUMNS)
    artifact_mask = frame["path"].eq(path.name)
    for _ in range(6):
        _atomic_frame_csv(frame, path)
        size = path.stat().st_size
        recorded = pd.to_numeric(
            frame.loc[artifact_mask, "size_bytes"], errors="coerce"
        )
        if not recorded.empty and int(recorded.iloc[0]) == size:
            break
        frame.loc[artifact_mask, "size_bytes"] = size
    return frame.reset_index(drop=True)


def _atomic_frame_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"pending-{path.name}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _series_id(runs: pd.DataFrame) -> str:
    if "series_id" not in runs:
        return ""
    values = runs["series_id"].dropna()
    return str(values.iloc[0]) if not values.empty else ""


def _normalise_experiment(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    return result.str.replace(r"\.0$", "", regex=True)


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype("string").str.lower().isin({"1", "1.0", "true", "yes"})


def _unique_columns(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = tuple(column for column in columns if column not in frame)
    if missing:
        raise ValueError(f"missing columns in {name}: {', '.join(missing)}")


__all__ = [
    "PLOT_MANIFEST",
    "PlotSpec",
    "add_report_metrics",
    "prepare_quantile_band",
    "write_single_index_reports",
]
