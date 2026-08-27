from __future__ import annotations

import csv
import importlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_COLORS = ("#2563eb", "#dc2626")
_AXIS_FACE = "#f8fafc"
_FIGURE_FACE = "#ffffff"
_GRID_COLOR = "#cbd5e1"
_TEXT_COLOR = "#111827"
_SPINE_COLOR = "#94a3b8"


def plot_experiment(series_dir: str | Path) -> Path:
    """Построить три универсальных A/B-графика из сохранённого ``runs.csv``."""
    path = Path(series_dir)
    with (path / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("runs.csv contains no experiment rows")
    output = path / "plots"
    for selector in dict.fromkeys(row["experiment"] for row in rows):
        selected = [row for row in rows if row["experiment"] == selector]
        directory = output / selector.replace(".", "_")
        directory.mkdir(parents=True, exist_ok=True)
        _metric_plot(
            selected,
            "cosine_abs",
            directory / "quality.png",
            "Качество A/B",
            "Абсолютный косинус",
        )
        _metric_plot(
            selected,
            "fit_time_sec",
            directory / "runtime.png",
            "Время A/B",
            "Время fit, секунд",
        )
        _delta_plot(selected, directory / "paired_delta.png")
    return output


def _metric_plot(
    rows: list[dict[str, str]],
    metric: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt = _pyplot()
    point_labels = _point_labels(rows)
    builds = tuple(dict.fromkeys(row["build"] for row in rows))
    fig, ax = plt.subplots(
        figsize=_figure_size(len(point_labels)),
        facecolor=_FIGURE_FACE,
    )
    try:
        plotted = False
        x = np.arange(len(point_labels))
        for color, build in zip(_COLORS, builds, strict=False):
            medians, lows, highs = [], [], []
            for point in range(len(point_labels)):
                values = _values(rows, build, point, metric)
                median, low, high = _quantiles(values)
                medians.append(median)
                lows.append(low)
                highs.append(high)
            valid = np.isfinite(medians)
            if not np.any(valid):
                continue
            plotted = True
            ax.plot(
                x[valid],
                np.asarray(medians)[valid],
                marker="o",
                linewidth=1.8,
                color=color,
                label=build,
            )
            ax.fill_between(
                x[valid],
                np.asarray(lows)[valid],
                np.asarray(highs)[valid],
                color=color,
                alpha=0.18,
            )
        if not plotted:
            ax.text(0.5, 0.5, "нет успешных наблюдений", ha="center", va="center")
        _style(ax, title, "Точка сетки", ylabel, point_labels)
        if plotted:
            ax.legend()
        _save(fig, path, plt)
    except Exception:
        plt.close(fig)
        raise


def _delta_plot(rows: list[dict[str, str]], path: Path) -> None:
    plt = _pyplot()
    point_labels = _point_labels(rows)
    builds = tuple(dict.fromkeys(row["build"] for row in rows))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(_figure_size(len(point_labels))[0], 9.0),
        facecolor=_FIGURE_FACE,
        sharex=True,
    )
    try:
        for ax, metric, ylabel in (
            (axes[0], "cosine_abs", "Косинус A - B"),
            (axes[1], "fit_time_sec", "Время A - B, секунд"),
        ):
            summaries = [
                _quantiles(_paired_deltas(rows, point, metric, builds))
                for point in range(len(point_labels))
            ]
            median = np.asarray([summary[0] for summary in summaries])
            low = np.asarray([summary[1] for summary in summaries])
            high = np.asarray([summary[2] for summary in summaries])
            valid = np.isfinite(median)
            x = np.arange(len(point_labels))
            if np.any(valid):
                ax.plot(x[valid], median[valid], marker="o", color=_COLORS[0])
                ax.fill_between(
                    x[valid], low[valid], high[valid], color=_COLORS[0], alpha=0.18
                )
            else:
                ax.text(0.5, 0.5, "нет полных A/B-пар", ha="center", va="center")
            ax.axhline(0, color="#4b5563", linestyle="--", linewidth=1.2)
            _style(ax, "", "Точка сетки", ylabel, point_labels)
        label = "A - B" if len(builds) < 2 else f"{builds[0]} - {builds[1]}"
        fig.suptitle(
            f"Парная разница: {label}",
            color=_TEXT_COLOR,
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
    if len(builds) != 2:
        return []
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


def _values(
    rows: list[dict[str, str]],
    build: str,
    point: int,
    metric: str,
) -> list[float]:
    values = []
    for row in rows:
        if row["build"] != build or int(row["point"]) != point:
            continue
        value = _number(row.get(metric, ""))
        if value is not None:
            values.append(value)
    return values


def _point_labels(rows: list[dict[str, str]]) -> tuple[str, ...]:
    labels = {int(row["point"]): row["point_label"] for row in rows}
    return tuple(labels[index] for index in sorted(labels))


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        missing = float("nan")
        return missing, missing, missing
    array = np.asarray(values)
    quantiles = np.quantile(array, (0.5, 0.05, 0.95))
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _style(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    point_labels: tuple[str, ...],
) -> None:
    ax.set_facecolor(_AXIS_FACE)
    ax.set_title(title, color=_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=_TEXT_COLOR, labelpad=8)
    ax.set_ylabel(ylabel, color=_TEXT_COLOR, labelpad=8)
    ax.set_xticks(range(len(point_labels)), point_labels)
    ax.tick_params(
        axis="x",
        rotation=45 if len(point_labels) < 12 else 90,
        colors=_TEXT_COLOR,
    )
    ax.tick_params(axis="y", colors=_TEXT_COLOR)
    ax.grid(axis="y", color=_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)


def _figure_size(points: int) -> tuple[float, float]:
    return max(8.0, min(18.0, 0.55 * points)), 5.2


def _pyplot() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/adp_matplotlib")
    return importlib.import_module("matplotlib.pyplot")


def _save(fig, path: Path, plt) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_FIGURE_FACE)
    plt.close(fig)


__all__ = ["plot_experiment"]
