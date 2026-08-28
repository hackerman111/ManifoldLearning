from __future__ import annotations

import csv
import importlib
import json
import os
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
_STYLES = ("-", "--", ":")
_AXIS_FACE = "#f8fafc"
_FIGURE_FACE = "#ffffff"
_GRID_COLOR = "#cbd5e1"
_TEXT_COLOR = "#111827"
_SPINE_COLOR = "#94a3b8"
_SUMMARY_COLUMNS = (
    "experiment",
    "title",
    "factor",
    "factor_label",
    "level",
    "build",
    "quality_metric",
    "quality_direction",
    "n_total",
    "n_success",
    "n_nonconverged",
    "n_failure",
    "failure_rate",
    "initial_quality_median",
    "last_quality_median",
    "selected_quality_median",
    "selected_quality_q05",
    "selected_quality_q95",
    "fit_time_median_sec",
    "peak_memory_median_mib",
)
_TRACE_COLUMNS = (
    "experiment",
    "factor",
    "factor_label",
    "level",
    "build",
    "iteration",
    "n",
    "quality_median",
    "quality_q05",
    "quality_q95",
    "test_error_median",
    "h_median",
    "anisotropy_median",
)
_FACTOR_LABELS = {
    "n_samples": "n",
    "d": "d",
    "sigma_eps": "sigma_eps",
    "sigma_x": "sigma_X",
    "tau": "τ",
    "link": "функция связи",
    "link_scale": "s",
    "N_loc": "N_loc",
    "N_lin": "N_lin",
    "N_J": "N_J",
    "N_phi": "N_phi",
    "lambda_penalty": "λ",
    "a": "a",
    "h_min_factor": "множитель h_min",
    "solver_max_steps": "k_max",
    "index_init": "инициализация",
    "center_displacement": "nu",
    "training_set": "обучающая выборка",
    "direction_mode": "закон направлений",
    "redraw_directions": "обновление направлений",
    "multi_tensor": "тензор",
    "point_label": "точка сетки",
}


def build_report(series_dir: str | Path, *, plots: bool = True) -> Path:
    """Записать компактные таблицы и, при запросе, читаемые PNG-графики."""
    path = Path(series_dir)
    rows, fieldnames = _read_rows(path / "runs.csv")
    if not rows:
        raise ValueError("runs.csv contains no experiment rows")
    manifest = json.loads((path / "series.json").read_text(encoding="utf-8"))
    factors = tuple(manifest.get("report_fields") or ())
    if not factors:
        factors = _infer_factors(rows, fieldnames)

    summary = _summaries(rows, factors)
    trace = _trace_summaries(rows, factors)
    _write_csv(path / "summary.csv", _SUMMARY_COLUMNS, summary)
    _write_csv(path / "trace_summary.csv", _TRACE_COLUMNS, trace)
    _write_failures(path / "failures.csv", rows)
    _write_markdown(path / "summary.md", manifest, summary, factors)

    if plots:
        output = path / "plots" / str(manifest["experiment"]).replace(".", "_")
        output.mkdir(parents=True, exist_ok=True)
        quality_label = _quality_label(rows)
        _metric_plot(
            summary,
            factors,
            "selected_quality",
            output / "quality.png",
            "Качество выбранной оценки",
            quality_label,
        )
        _stages_plot(summary, factors, output / "stages.png", quality_label)
        _metric_plot(
            summary,
            factors,
            "fit_time",
            output / "runtime.png",
            "Время fit",
            "секунды",
        )
        _metric_plot(
            summary,
            factors,
            "peak_memory",
            output / "memory.png",
            "Пиковая traced-память этапа",
            "MiB",
        )
        _metric_plot(
            summary,
            factors,
            "failure_rate",
            output / "failures.png",
            "Доля numerical failure",
            "доля",
        )
        _trajectory_plot(rows, output / "trajectory.png", quality_label)
        if len({row["build"] for row in rows}) == 2:
            _delta_plot(rows, output / "paired_delta.png")
    return path


def plot_experiment(series_dir: str | Path) -> Path:
    """Совместимый wrapper: построить весь отчёт и вернуть каталог PNG."""
    return build_report(series_dir, plots=True) / "plots"


def _read_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), tuple(reader.fieldnames or ())


def _infer_factors(
    rows: list[dict[str, str]],
    fieldnames: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        start = fieldnames.index("noise_dimension_condition") + 1
        stop = fieldnames.index("requested_config")
    except ValueError:
        return ("point_label",)
    point_rows = {row["point"]: row for row in rows}.values()
    varying = tuple(
        name
        for name in fieldnames[start:stop]
        if len({row.get(name, "") for row in point_rows}) > 1
    )
    return varying or ("point_label",)


def _summaries(
    rows: list[dict[str, str]],
    factors: tuple[str, ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    builds = tuple(dict.fromkeys(row["build"] for row in rows))
    for factor in factors:
        levels = tuple(dict.fromkeys(row.get(factor, "") for row in rows))
        for level in levels:
            for build in builds:
                selected = [
                    row
                    for row in rows
                    if row.get(factor, "") == level and row["build"] == build
                ]
                if not selected:
                    continue
                quality = _quantiles(_values(selected, "quality"))
                failures = sum(row["status"] == "numerical_failure" for row in selected)
                result.append(
                    {
                        "experiment": selected[0]["experiment"],
                        "title": selected[0]["title"],
                        "factor": factor,
                        "factor_label": _factor_label(factor),
                        "level": level,
                        "build": build,
                        "quality_metric": _first(selected, "quality_metric"),
                        "quality_direction": _first(selected, "quality_direction"),
                        "n_total": len(selected),
                        "n_success": sum(
                            row["status"] == "success" for row in selected
                        ),
                        "n_nonconverged": sum(
                            row["status"] == "nonconverged" for row in selected
                        ),
                        "n_failure": failures,
                        "failure_rate": failures / len(selected),
                        "initial_quality_median": _median(
                            _values(selected, "initial_quality")
                        ),
                        "last_quality_median": _median(
                            _values(selected, "last_quality")
                        ),
                        "selected_quality_median": quality[0],
                        "selected_quality_q05": quality[1],
                        "selected_quality_q95": quality[2],
                        "fit_time_median_sec": _median(
                            _values(selected, "fit_time_sec")
                        ),
                        "peak_memory_median_mib": _median(
                            _values(selected, "max_stage_traced_peak_mib")
                        ),
                    }
                )
    return result


def _trace_summaries(
    rows: list[dict[str, str]],
    factors: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not row.get("trace"):
            continue
        try:
            steps = json.loads(row["trace"])
        except (TypeError, json.JSONDecodeError):
            continue
        for factor in factors:
            for step in steps:
                key = (
                    factor,
                    row.get(factor, ""),
                    row["build"],
                    int(step["iteration"]),
                )
                groups[key].append(step)

    result: list[dict[str, object]] = []
    for (factor, level, build, iteration), steps in groups.items():
        quality = _quantiles(_object_values(steps, "quality"))
        result.append(
            {
                "experiment": rows[0]["experiment"],
                "factor": factor,
                "factor_label": _factor_label(factor),
                "level": level,
                "build": build,
                "iteration": iteration,
                "n": len(steps),
                "quality_median": quality[0],
                "quality_q05": quality[1],
                "quality_q95": quality[2],
                "test_error_median": _median(_object_values(steps, "err")),
                "h_median": _median(_object_values(steps, "h")),
                "anisotropy_median": _median(_object_values(steps, "factor")),
            }
        )
    return result


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_failures(path: Path, rows: list[dict[str, str]]) -> None:
    columns = (
        "experiment",
        "point",
        "point_label",
        "run",
        "seed",
        "build",
        "status",
        "error",
    )
    _write_csv(
        path,
        columns,
        [row for row in rows if row["status"] != "success"],
    )


def _write_markdown(
    path: Path,
    manifest: dict[str, object],
    summary: list[dict[str, object]],
    factors: tuple[str, ...],
) -> None:
    lines = [
        f"# {manifest['title']}",
        "",
        f"Метрика: `{_first(summary, 'quality_metric')}`; направление: "
        f"`{_first(summary, 'quality_direction')}`.",
        "",
        "Полные наблюдения — в `runs.csv`; ниже агрегаты по факторам.",
    ]
    for factor in factors:
        lines.extend(
            (
                "",
                f"## {_factor_label(factor)}",
                "",
                "| Уровень | Build | Success | Nonconv | Fail | Init | Last | "
                "Selected | Время, sec | Память, MiB |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in (item for item in summary if item["factor"] == factor):
            values = (
                row["level"],
                row["build"],
                f"{row['n_success']}/{row['n_total']}",
                row["n_nonconverged"],
                row["n_failure"],
                _format_number(row["initial_quality_median"]),
                _format_number(row["last_quality_median"]),
                _format_number(row["selected_quality_median"]),
                _format_number(row["fit_time_median_sec"]),
                _format_number(row["peak_memory_median_mib"]),
            )
            lines.append("| " + " | ".join(map(str, values)) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_plot(
    summary: list[dict[str, object]],
    factors: tuple[str, ...],
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt = _pyplot()
    fig, axes = _factor_axes(plt, len(factors))
    try:
        builds = tuple(dict.fromkeys(str(row["build"]) for row in summary))
        for ax, factor in zip(axes, factors, strict=True):
            factor_rows = [row for row in summary if row["factor"] == factor]
            levels = tuple(dict.fromkeys(str(row["level"]) for row in factor_rows))
            x = np.arange(len(levels))
            for color, build in zip(_COLORS, builds, strict=False):
                by_level = {
                    str(row["level"]): row
                    for row in factor_rows
                    if row["build"] == build
                }
                median = np.asarray(
                    [
                        _summary_value(by_level.get(level), metric, "median")
                        for level in levels
                    ]
                )
                low = np.asarray(
                    [
                        _summary_value(by_level.get(level), metric, "q05")
                        for level in levels
                    ]
                )
                high = np.asarray(
                    [
                        _summary_value(by_level.get(level), metric, "q95")
                        for level in levels
                    ]
                )
                valid = np.isfinite(median)
                if not np.any(valid):
                    continue
                ax.plot(
                    x[valid],
                    median[valid],
                    marker="o",
                    linewidth=1.8,
                    color=color,
                    label=build,
                )
                if metric == "selected_quality":
                    ax.fill_between(
                        x[valid],
                        low[valid],
                        high[valid],
                        color=color,
                        alpha=0.16,
                    )
            _style(ax, _factor_label(factor), ylabel, levels)
            if len(builds) > 1:
                ax.legend(fontsize=9)
        fig.suptitle(title, color=_TEXT_COLOR, fontweight="bold", fontsize=14)
        _save(fig, path, plt)
    except Exception:
        plt.close(fig)
        raise


def _summary_value(
    row: dict[str, object] | None,
    metric: str,
    statistic: str,
) -> float:
    if row is None:
        return float("nan")
    if metric == "selected_quality":
        suffix = {"median": "median", "q05": "q05", "q95": "q95"}[statistic]
        return _float(row[f"selected_quality_{suffix}"])
    column = {
        "fit_time": "fit_time_median_sec",
        "peak_memory": "peak_memory_median_mib",
        "failure_rate": "failure_rate",
    }[metric]
    return _float(row[column])


def _stages_plot(
    summary: list[dict[str, object]],
    factors: tuple[str, ...],
    path: Path,
    ylabel: str,
) -> None:
    plt = _pyplot()
    fig, axes = _factor_axes(plt, len(factors))
    stages = (
        ("initial_quality_median", "initialization"),
        ("last_quality_median", "last"),
        ("selected_quality_median", "selected"),
    )
    try:
        builds = tuple(dict.fromkeys(str(row["build"]) for row in summary))
        for ax, factor in zip(axes, factors, strict=True):
            factor_rows = [row for row in summary if row["factor"] == factor]
            levels = tuple(dict.fromkeys(str(row["level"]) for row in factor_rows))
            x = np.arange(len(levels))
            for color, build in zip(_COLORS, builds, strict=False):
                by_level = {
                    str(row["level"]): row
                    for row in factor_rows
                    if row["build"] == build
                }
                for style, (column, stage) in zip(_STYLES, stages, strict=True):
                    values = np.asarray(
                        [
                            _float(by_level[level][column])
                            if level in by_level
                            else float("nan")
                            for level in levels
                        ]
                    )
                    valid = np.isfinite(values)
                    if np.any(valid):
                        label = stage if len(builds) == 1 else f"{build} · {stage}"
                        ax.plot(
                            x[valid],
                            values[valid],
                            marker="o",
                            linewidth=1.6,
                            linestyle=style,
                            color=color,
                            label=label,
                        )
            _style(ax, _factor_label(factor), ylabel, levels)
            ax.legend(fontsize=8, ncols=2)
        fig.suptitle(
            "Initialization / last / stopping-selected",
            color=_TEXT_COLOR,
            fontweight="bold",
            fontsize=14,
        )
        _save(fig, path, plt)
    except Exception:
        plt.close(fig)
        raise


def _trajectory_plot(rows: list[dict[str, str]], path: Path, ylabel: str) -> None:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not row.get("trace"):
            continue
        try:
            steps = json.loads(row["trace"])
        except (TypeError, json.JSONDecodeError):
            continue
        for step in steps:
            groups[(row["build"], int(step["iteration"]))].append(step)

    plt = _pyplot()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        facecolor=_FIGURE_FACE,
        sharex=True,
    )
    try:
        builds = tuple(dict.fromkeys(row["build"] for row in rows))
        for color, build in zip(_COLORS, builds, strict=False):
            iterations = sorted(
                iteration for candidate, iteration in groups if candidate == build
            )
            quality = [
                _quantiles(_object_values(groups[(build, i)], "quality"))
                for i in iterations
            ]
            errors = [
                _quantiles(_object_values(groups[(build, i)], "err"))
                for i in iterations
            ]
            _trace_line(axes[0], iterations, quality, color, build)
            _trace_line(axes[1], iterations, errors, color, build)
        _style(axes[0], "Качество по outer-шагам", ylabel, ())
        _style(axes[1], "Test fit по outer-шагам", "сумма квадратов ошибок", ())
        axes[1].set_xlabel("outer iteration")
        iterations = sorted({iteration for _, iteration in groups})
        axes[1].set_xticks(iterations)
        if len(iterations) == 1:
            axes[1].set_xlim(iterations[0] - 0.5, iterations[0] + 0.5)
        for ax in axes:
            if groups:
                ax.legend(fontsize=9)
        _save(fig, path, plt)
    except Exception:
        plt.close(fig)
        raise


def _trace_line(ax, x: list[int], values, color: str, label: str) -> None:
    if not x:
        return
    median = np.asarray([value[0] for value in values])
    low = np.asarray([value[1] for value in values])
    high = np.asarray([value[2] for value in values])
    ax.plot(x, median, marker="o", linewidth=1.8, color=color, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.16)


def _delta_plot(rows: list[dict[str, str]], path: Path) -> None:
    plt = _pyplot()
    builds = tuple(dict.fromkeys(row["build"] for row in rows))
    labels = tuple(
        label
        for _, label in sorted(
            {int(row["point"]): row["point_label"] for row in rows}.items()
        )
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        facecolor=_FIGURE_FACE,
        sharex=True,
    )
    try:
        for ax, metric, ylabel in (
            (axes[0], "quality", "Качество A - B"),
            (axes[1], "fit_time_sec", "Время A - B, секунд"),
        ):
            summaries = [
                _quantiles(_paired_deltas(rows, point, metric, builds))
                for point in range(len(labels))
            ]
            x = np.arange(len(labels))
            median = np.asarray([value[0] for value in summaries])
            low = np.asarray([value[1] for value in summaries])
            high = np.asarray([value[2] for value in summaries])
            valid = np.isfinite(median)
            if np.any(valid):
                ax.plot(x[valid], median[valid], marker="o", color=_COLORS[0])
                ax.fill_between(
                    x[valid], low[valid], high[valid], color=_COLORS[0], alpha=0.16
                )
            ax.axhline(0, color="#4b5563", linestyle="--", linewidth=1.1)
            _style(ax, "", ylabel, labels)
        fig.suptitle(
            f"Парная разница: {builds[0]} - {builds[1]}",
            fontweight="bold",
        )
        _save(fig, path, plt)
    except Exception:
        plt.close(fig)
        raise


def _paired_deltas(
    rows: list[dict[str, str]],
    point: int,
    metric: str,
    builds: tuple[str, ...],
) -> list[float]:
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["point"]) != point:
            continue
        value = _number(row.get(metric, ""))
        if value is not None:
            pairs[row["seed"]][row["build"]] = value
    return [
        values[builds[0]] - values[builds[1]]
        for values in pairs.values()
        if all(build in values for build in builds)
    ]


def _factor_axes(plt, count: int):
    columns = min(2, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 4.6 * rows),
        facecolor=_FIGURE_FACE,
        squeeze=False,
    )
    flattened = list(axes.flat)
    for ax in flattened[count:]:
        ax.remove()
    return fig, flattened[:count]


def _style(
    ax,
    title: str,
    ylabel: str,
    levels: tuple[str, ...],
) -> None:
    ax.set_facecolor(_AXIS_FACE)
    ax.set_title(title, color=_TEXT_COLOR, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=_TEXT_COLOR, labelpad=7)
    if levels:
        ax.set_xticks(
            range(len(levels)),
            tuple(textwrap.fill(_short_level(level), width=14) for level in levels),
        )
        ax.tick_params(axis="x", rotation=30 if len(levels) > 6 else 0)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.grid(axis="y", color=_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)


def _short_level(value: str) -> str:
    return {
        "True": "redraw",
        "False": "fixed",
        "exclude_centers": "без test indices",
        "multi_additive": "additive",
        "multi_multiplicative": "multiplicative",
        "sin_scaled": "sin(sx)",
        "x_sin": "x sin(sx)",
    }.get(value, value)


def _quality_label(rows: list[dict[str, str]]) -> str:
    if _first(rows, "quality_metric") == "cosine_abs":
        return "|cos(β, β*)| (выше лучше)"
    return "ошибка проектора (ниже лучше)"


def _factor_label(name: str) -> str:
    return _FACTOR_LABELS.get(name, name)


def _first(rows, key: str) -> str:
    return next(
        (str(row[key]) for row in rows if row.get(key) not in (None, "")),
        "",
    )


def _values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if (value := _number(row.get(key, ""))) is not None]


def _object_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [value for row in rows if np.isfinite(value := _float(row.get(key)))]


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        missing = float("nan")
        return missing, missing, missing
    result = np.quantile(np.asarray(values), (0.5, 0.05, 0.95))
    return float(result[0]), float(result[1]), float(result[2])


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _number(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _float(value: object) -> float:
    number = _number(value)
    return number if number is not None else float("nan")


def _format_number(value: object) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.4g}"


def _pyplot() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/adp_matplotlib")
    return importlib.import_module("matplotlib.pyplot")


def _save(fig, path: Path, plt) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=_FIGURE_FACE)
    plt.close(fig)


__all__ = ["build_report", "plot_experiment"]
