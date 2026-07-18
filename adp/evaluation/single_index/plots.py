from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...common.plotting import apply_adp_axis_style, save_figure, set_adp_figure_size


def line_with_quantile_band(
    frame: pd.DataFrame,
    *,
    x: str,
    median: str,
    q05: str,
    q95: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    group: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        plotted = False
        for label, subset in _groups(frame, group):
            values = _numeric_frame(subset, (x, median, q05, q95)).sort_values(x)
            if values.empty:
                continue
            x_values = values[x].to_numpy(dtype=float)
            center = values[median].to_numpy(dtype=float)
            low = values[q05].to_numpy(dtype=float)
            high = values[q95].to_numpy(dtype=float)
            line = ax.plot(
                x_values,
                center,
                marker="o",
                linewidth=1.7,
                label=label,
            )[0]
            ax.fill_between(x_values, low, high, color=line.get_color(), alpha=0.18)
            plotted = True
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            log_x=log_x,
            log_y=log_y,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def grouped_line(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    group: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        plotted = False
        for label, subset in _groups(frame, group):
            values = subset[[x, y]].copy() if {x, y} <= set(subset) else pd.DataFrame()
            if values.empty:
                continue
            values[y] = pd.to_numeric(values[y], errors="coerce")
            values = values.loc[values[x].notna() & np.isfinite(values[y])]
            if values.empty:
                continue
            numeric_x = pd.to_numeric(values[x], errors="coerce")
            if numeric_x.notna().all():
                values = values.assign(**{x: numeric_x}).sort_values(x)
            ax.plot(
                values[x],
                values[y],
                marker="o",
                linewidth=1.7,
                label=label,
            )
            plotted = True
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            log_x=log_x,
            log_y=log_y,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def boxplot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        groups: list[np.ndarray] = []
        labels: list[str] = []
        if {x, y} <= set(frame):
            for label, subset in frame.groupby(x, sort=True, dropna=False):
                values = pd.to_numeric(subset[y], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    groups.append(values)
                    labels.append(str(label))
        plotted = bool(groups)
        if plotted:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
            ax.tick_params(axis="x", rotation=30)
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            log_x=log_x,
            log_y=log_y,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    group: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        plotted = False
        for label, subset in _groups(frame, group):
            values = _numeric_frame(subset, (x, y))
            if values.empty:
                continue
            ax.scatter(values[x], values[y], s=22, alpha=0.68, label=label)
            plotted = True
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            log_x=log_x,
            log_y=log_y,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def heatmap(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        plotted = False
        if {x, y, value} <= set(frame):
            values = frame[[x, y, value]].copy()
            values[value] = pd.to_numeric(values[value], errors="coerce")
            values = values.loc[values[x].notna() & values[y].notna()]
            if not values.empty:
                pivot = values.pivot_table(
                    index=y,
                    columns=x,
                    values=value,
                    aggfunc="median",
                )
                matrix = pivot.to_numpy(dtype=float)
                if matrix.size and np.isfinite(matrix).any():
                    image = ax.imshow(matrix, aspect="auto", origin="lower")
                    ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns])
                    ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
                    fig.colorbar(image, ax=ax, shrink=0.82)
                    plotted = True
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def stacked_runtime(
    frame: pd.DataFrame,
    *,
    category: str,
    components: Sequence[str],
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int = 100,
) -> Path:
    fig, ax, plt = _new_figure()
    try:
        plotted = False
        required = {category, *components}
        if required <= set(frame) and not frame.empty:
            source = frame[[category, *components]].copy()
            for component in components:
                source[component] = pd.to_numeric(source[component], errors="coerce")
            source = source.groupby(category, sort=True, dropna=False)[list(components)].median()
            if not source.empty and np.isfinite(source.to_numpy(dtype=float)).any():
                bottom = np.zeros(len(source), dtype=float)
                positions = np.arange(len(source))
                for component in components:
                    values = source[component].fillna(0.0).to_numpy(dtype=float)
                    ax.bar(positions, values, bottom=bottom, label=component)
                    bottom += values
                ax.set_xticks(positions, [str(value) for value in source.index])
                ax.tick_params(axis="x", rotation=30)
                plotted = True
        _finish_axis(
            ax,
            plotted=plotted,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def _new_figure() -> tuple[Any, Any, Any]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    set_adp_figure_size(fig)
    return fig, ax, plt


def _groups(frame: pd.DataFrame, group: str | None):
    if group is None or group not in frame:
        return ((None, frame),)
    return tuple(
        (str(label), subset)
        for label, subset in frame.groupby(group, sort=True, dropna=False)
    )


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not set(columns) <= set(frame):
        return pd.DataFrame(columns=columns)
    values = frame[list(columns)].copy()
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    return values.loc[finite]


def _finish_axis(
    ax: Any,
    *,
    plotted: bool,
    xlabel: str,
    ylabel: str,
    title: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "no finite data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if log_x and plotted:
        ax.set_xscale("log")
    if log_y and plotted:
        ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    if handles and any(not str(label).startswith("_") for label in labels):
        ax.legend()
    apply_adp_axis_style(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend_title="group" if handles else None,
    )


__all__ = [
    "boxplot",
    "grouped_line",
    "heatmap",
    "line_with_quantile_band",
    "scatter",
    "stacked_runtime",
]
