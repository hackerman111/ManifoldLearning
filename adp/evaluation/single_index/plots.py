from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...common.plotting import (
    ADP_COLORS,
    apply_adp_axis_style,
    ensure_matplotlib_config_dir,
    save_figure,
    set_adp_figure_size,
)


Scale = Literal["linear", "log", "log2", "symlog"]
_FIGURE_SIZE = (10.5, 6.2)


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
    subtitle: str = "медиана и интервал 5–95%",
    group: str | None = None,
    facet: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    data_notes: Mapping[str | None, str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 300,
) -> Path:
    frame = _with_note_facets(frame, facet, data_notes)
    fig, panels, plt = _new_figure(frame, facet)
    try:
        all_handles: dict[str, Any] = {}
        styles = _series_styles(frame, group)
        for panel_label, subset, ax in panels:
            plotted = False
            for label, grouped in _groups(subset, group):
                values = _numeric_y_frame(grouped, x, (median, q05, q95), category_order)
                if values.empty:
                    continue
                line = ax.plot(
                    values["_x"], values[median], marker="o", linewidth=1.8,
                    label=label, color=styles[label][0], linestyle=styles[label][1],
                )[0]
                ax.fill_between(
                    values["_x"], values[q05], values[q95],
                    color=line.get_color(), alpha=0.18,
                )
                if label:
                    all_handles.setdefault(label, line)
                plotted = True
            _finish_axis(
                ax, frame=subset, value_columns=(median,), plotted=plotted,
                xlabel=xlabel, ylabel=ylabel, panel_title=panel_label,
                xscale="log" if log_x else xscale,
                yscale="log" if log_y else yscale,
                xlim=xlim, ylim=ylim, reference_y=reference_y,
                reference_label=reference_label, category_order=category_order,
                data_note=(data_notes or {}).get(panel_label),
            )
        _finish_figure(
            fig, title, subtitle, all_handles,
            statistic="quantile",
        )
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def line_with_wilson_interval(
    frame: pd.DataFrame,
    *,
    x: str,
    estimate: str,
    low: str,
    high: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str = "оценка доли и 95% доверительный интервал Уилсона",
    group: str | None = None,
    facet: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    data_notes: Mapping[str | None, str] | None = None,
    dpi: int = 300,
) -> Path:
    frame = _with_note_facets(frame, facet, data_notes)
    fig, panels, plt = _new_figure(frame, facet)
    try:
        all_handles: dict[str, Any] = {}
        styles = _series_styles(frame, group)
        for panel_label, subset, ax in panels:
            plotted = False
            for label, grouped in _groups(subset, group):
                values = _numeric_y_frame(grouped, x, (estimate, low, high), category_order)
                if values.empty:
                    continue
                errors = np.vstack(
                    (values[estimate] - values[low], values[high] - values[estimate])
                )
                container = ax.errorbar(
                    values["_x"], values[estimate], yerr=errors,
                    marker="o", linewidth=1.8, capsize=4, label=label,
                    color=styles[label][0], linestyle=styles[label][1],
                )
                if label:
                    all_handles.setdefault(label, container.lines[0])
                plotted = True
            _finish_axis(
                ax, frame=subset, value_columns=(estimate,), plotted=plotted,
                xlabel=xlabel, ylabel=ylabel, panel_title=panel_label,
                xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim,
                reference_y=reference_y, reference_label=reference_label,
                category_order=category_order,
                data_note=(data_notes or {}).get(panel_label),
            )
        _finish_figure(fig, title, subtitle, all_handles, statistic="wilson")
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
    subtitle: str = "медиана по запускам",
    group: str | None = None,
    facet: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    data_notes: Mapping[str | None, str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 300,
) -> Path:
    frame = _with_note_facets(frame, facet, data_notes)
    fig, panels, plt = _new_figure(frame, facet)
    try:
        all_handles: dict[str, Any] = {}
        styles = _series_styles(frame, group)
        for panel_label, subset, ax in panels:
            plotted = False
            for label, grouped in _groups(subset, group):
                values = _numeric_y_frame(grouped, x, (y,), category_order)
                if values.empty:
                    continue
                line = ax.plot(
                    values["_x"], values[y], marker="o", linewidth=1.8,
                    label=label, color=styles[label][0], linestyle=styles[label][1],
                )[0]
                if label:
                    all_handles.setdefault(label, line)
                plotted = True
            _finish_axis(
                ax, frame=subset, value_columns=(y,), plotted=plotted,
                xlabel=xlabel, ylabel=ylabel, panel_title=panel_label,
                xscale="log" if log_x else xscale,
                yscale="log" if log_y else yscale,
                xlim=xlim, ylim=ylim, reference_y=reference_y,
                reference_label=reference_label, category_order=category_order,
                data_note=(data_notes or {}).get(panel_label),
            )
        _finish_figure(fig, title, subtitle, all_handles)
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
    subtitle: str = "медиана, коробка 25–75%, усики 5–95%",
    facet: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 300,
) -> Path:
    fig, panels, plt = _new_figure(frame, facet)
    try:
        for panel_label, subset, ax in panels:
            groups: list[np.ndarray] = []
            labels: list[str] = []
            if {x, y} <= set(subset):
                order = _category_values(subset[x], category_order)
                for label in order:
                    values = pd.to_numeric(
                        subset.loc[subset[x].astype("string") == str(label), y],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        groups.append(values)
                        labels.append(_category_label(label))
            plotted = bool(groups)
            if plotted:
                ax.boxplot(
                    groups, tick_labels=labels, whis=(5, 95), showfliers=True,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.28},
                    medianprops={"color": "#111827", "linewidth": 1.7},
                )
                ax.tick_params(axis="x", rotation=25)
            _finish_axis(
                ax, frame=subset, value_columns=(y,), plotted=plotted,
                xlabel=xlabel, ylabel=ylabel, panel_title=panel_label,
                xscale="log" if log_x else xscale,
                yscale="log" if log_y else yscale,
                xlim=xlim, ylim=ylim, reference_y=reference_y,
                reference_label=reference_label,
            )
        _finish_figure(fig, title, subtitle, {}, statistic="box")
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
    subtitle: str = "отдельные локальные оценки",
    group: str | None = None,
    facet: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    log_x: bool = False,
    log_y: bool = False,
    dpi: int = 300,
) -> Path:
    fig, panels, plt = _new_figure(frame, facet)
    try:
        all_handles: dict[str, Any] = {}
        styles = _series_styles(frame, group)
        for panel_label, subset, ax in panels:
            plotted = False
            for label, grouped in _groups(subset, group):
                values = _numeric_xy(grouped, x, y)
                if values.empty:
                    continue
                artist = ax.scatter(
                    values[x], values[y], s=24, alpha=0.62, label=label,
                    color=styles[label][0], marker=styles[label][2],
                )
                if label:
                    all_handles.setdefault(label, artist)
                plotted = True
            _finish_axis(
                ax, frame=subset, value_columns=(y,), plotted=plotted,
                xlabel=xlabel, ylabel=ylabel, panel_title=panel_label,
                xscale="log" if log_x else xscale,
                yscale="log" if log_y else yscale,
                xlim=xlim, ylim=ylim, reference_y=reference_y,
                reference_label=reference_label, category_order=category_order,
            )
        _finish_figure(fig, title, subtitle, all_handles)
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
    subtitle: str = "медиана по запускам в каждой ячейке",
    colorbar_label: str = "Значение",
    value_limits: tuple[float, float] | None = None,
    integer_values: bool = False,
    dpi: int = 300,
) -> Path:
    ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    fig, ax = plt.subplots()
    set_adp_figure_size(fig, width=_FIGURE_SIZE[0], height=_FIGURE_SIZE[1])
    try:
        plotted = False
        x_tick_labels: list[str] = []
        y_tick_labels: list[str] = []
        if {x, y, value} <= set(frame):
            source = frame[[x, y, value]].copy()
            source[value] = pd.to_numeric(source[value], errors="coerce")
            source = source.loc[source[x].notna() & source[y].notna()]
            if not source.empty:
                pivot = source.pivot_table(index=y, columns=x, values=value, aggfunc="median", dropna=False)
                matrix = pivot.to_numpy(dtype=float)
                if matrix.size:
                    cmap = plt.get_cmap("viridis").with_extremes(bad="#d1d5db")
                    image_kwargs: dict[str, Any] = {"cmap": cmap}
                    if value_limits is not None:
                        image_kwargs.update(vmin=value_limits[0], vmax=value_limits[1])
                    if integer_values and np.isfinite(matrix).any():
                        finite = matrix[np.isfinite(matrix)]
                        lower, upper = int(np.floor(finite.min())), int(np.ceil(finite.max()))
                        bounds = np.arange(lower - 0.5, upper + 1.5, 1.0)
                        image_kwargs["norm"] = BoundaryNorm(bounds, cmap.N)
                        image_kwargs.pop("vmin", None)
                        image_kwargs.pop("vmax", None)
                    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", origin="lower", **image_kwargs)
                    x_tick_labels = [_format_number(v) for v in pivot.columns]
                    y_tick_labels = [_format_number(v) for v in pivot.index]
                    ax.set_xticks(range(len(x_tick_labels)), x_tick_labels)
                    ax.set_yticks(range(len(y_tick_labels)), y_tick_labels)
                    for row in range(matrix.shape[0]):
                        for column in range(matrix.shape[1]):
                            number = matrix[row, column]
                            label = "нет данных" if not np.isfinite(number) else _heatmap_value(number, integer_values)
                            ax.text(column, row, label, ha="center", va="center", fontsize=8,
                                    color=_contrast_color(number, value_limits))
                    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
                    colorbar.set_label(colorbar_label)
                    if integer_values and np.isfinite(matrix).any():
                        colorbar.set_ticks(np.arange(lower, upper + 1))
                    plotted = np.isfinite(matrix).any()
        _finish_axis(
            ax, frame=frame, value_columns=(value,), plotted=plotted,
            xlabel=xlabel, ylabel=ylabel,
        )
        if x_tick_labels:
            ax.set_xticks(range(len(x_tick_labels)), x_tick_labels)
        if y_tick_labels:
            ax.set_yticks(range(len(y_tick_labels)), y_tick_labels)
        _finish_figure(fig, title, subtitle, {})
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
    subtitle: str,
    component_labels: Mapping[str, str] | None = None,
    aggregate: Literal["mean", "median"] = "median",
    normalize: bool = False,
    dpi: int = 300,
) -> Path:
    fig, panels, plt = _new_figure(frame, None)
    ax = panels[0][2]
    try:
        plotted = False
        labels = component_labels or {}
        if {category, *components} <= set(frame) and not frame.empty:
            source = frame[[category, *components]].copy()
            for component in components:
                source[component] = pd.to_numeric(source[component], errors="coerce")
            grouped = source.groupby(category, sort=True, dropna=False)[list(components)]
            source = grouped.mean() if aggregate == "mean" else grouped.median()
            source = source.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if normalize:
                totals = source.sum(axis=1).replace(0.0, np.nan)
                source = source.div(totals, axis=0).fillna(0.0)
            if not source.empty:
                bottom = np.zeros(len(source), dtype=float)
                positions = np.arange(len(source))
                for component in components:
                    values = source[component].to_numpy(dtype=float)
                    ax.bar(positions, values, bottom=bottom, label=labels.get(component, component))
                    bottom += values
                ax.set_xticks(positions, [str(value) for value in source.index])
                ax.tick_params(axis="x", rotation=25)
                plotted = bool(np.any(bottom > 0))
        _finish_axis(
            ax, frame=frame, value_columns=tuple(components), plotted=plotted,
            xlabel=xlabel, ylabel=ylabel, ylim=(0.0, 1.0) if normalize else None,
        )
        handles, legend_labels = ax.get_legend_handles_labels()
        _finish_figure(fig, title, subtitle, dict(zip(legend_labels, handles, strict=False)))
        return save_figure(fig, path, dpi=dpi, close=False)
    finally:
        plt.close(fig)


def _new_figure(frame: pd.DataFrame, facet: str | None):
    ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    labels = [None]
    if facet and facet in frame and not frame.empty:
        labels = list(dict.fromkeys(frame[facet].dropna().astype(str))) or [None]
    count = len(labels)
    columns = min(2, count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False)
    width = 13.0 if count > 1 else _FIGURE_SIZE[0]
    height = 6.2 if rows == 1 else 5.0 * rows
    set_adp_figure_size(fig, width=width, height=height)
    flat_axes = list(axes.flat)
    panels = []
    for index, label in enumerate(labels):
        subset = frame if label is None else frame.loc[frame[facet].astype(str) == label]
        panels.append((label, subset, flat_axes[index]))
    for unused in flat_axes[count:]:
        unused.set_visible(False)
    return fig, panels, plt


def _groups(frame: pd.DataFrame, group: str | None):
    if group is None or group not in frame:
        return ((None, frame),)
    return tuple((str(label), subset) for label, subset in frame.groupby(group, sort=False, dropna=False))


def _with_note_facets(
    frame: pd.DataFrame,
    facet: str | None,
    data_notes: Mapping[str | None, str] | None,
) -> pd.DataFrame:
    if not facet or not data_notes:
        return frame
    present = set(frame[facet].dropna().astype(str)) if facet in frame else set()
    missing = [label for label in data_notes if label is not None and label not in present]
    if not missing:
        return frame
    placeholders = pd.DataFrame({facet: missing})
    return pd.concat([frame, placeholders], ignore_index=True, sort=False)


def _series_styles(
    frame: pd.DataFrame,
    group: str | None,
) -> dict[str | None, tuple[str, str, str]]:
    if group is None or group not in frame:
        return {None: (ADP_COLORS[0], "-", "o")}
    labels = list(dict.fromkeys(frame[group].dropna().astype(str)))
    color_keys = list(dict.fromkeys(label.split(", ", 1)[0] for label in labels))
    style_keys = list(
        dict.fromkeys(
            label.split(", ", 1)[1] if ", " in label else ""
            for label in labels
        )
    )
    linestyles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    result = {}
    for label in labels:
        parts = label.split(", ", 1)
        color_index = color_keys.index(parts[0])
        style_index = style_keys.index(parts[1] if len(parts) == 2 else "")
        result[label] = (
            ADP_COLORS[color_index % len(ADP_COLORS)],
            linestyles[style_index % len(linestyles)],
            markers[style_index % len(markers)],
        )
    return result


def _numeric_y_frame(
    frame: pd.DataFrame,
    x: str,
    y_columns: Sequence[str],
    category_order: Sequence[Any],
) -> pd.DataFrame:
    if not {x, *y_columns} <= set(frame):
        return pd.DataFrame()
    values = frame[[x, *y_columns]].copy()
    for column in y_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.loc[np.isfinite(values[list(y_columns)].to_numpy(dtype=float)).all(axis=1)]
    numeric_x = pd.to_numeric(values[x], errors="coerce")
    if numeric_x.notna().all():
        values["_x"] = numeric_x
        return values.sort_values("_x")
    order = _category_values(values[x], category_order)
    positions = {str(label): index for index, label in enumerate(order)}
    values["_x"] = values[x].astype("string").map(positions)
    return values.sort_values("_x")


def _numeric_xy(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    if not {x, y} <= set(frame):
        return pd.DataFrame()
    values = frame[[x, y]].apply(pd.to_numeric, errors="coerce")
    return values.loc[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]


def _finish_axis(
    ax: Any,
    *,
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    plotted: bool,
    xlabel: str,
    ylabel: str,
    panel_title: str | None = None,
    xscale: Scale = "linear",
    yscale: Scale = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    category_order: Sequence[Any] = (),
    data_note: str | None = None,
) -> None:
    computed_annotation = _data_annotation(frame, value_columns)
    if data_note:
        annotation = data_note
        if computed_annotation == "значение постоянно":
            annotation += "; " + computed_annotation
    else:
        annotation = computed_annotation
    if not plotted:
        ax.text(0.5, 0.5, annotation, ha="center", va="center", transform=ax.transAxes)
    elif annotation:
        ax.text(0.01, 0.99, annotation, ha="left", va="top", transform=ax.transAxes,
                fontsize=8, color="#4b5563")
    x_linthresh = _apply_scale(ax, "x", xscale)
    y_linthresh = _apply_scale(ax, "y", yscale)
    _set_experimental_x_ticks(ax, xscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if reference_y is not None:
        ax.axhline(reference_y, color="#4b5563", linestyle="--", linewidth=1.2)
        if reference_label:
            ax.text(0.99, reference_y, reference_label, ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), fontsize=8, color="#4b5563")
    symlog_notes = []
    if x_linthresh is not None:
        symlog_notes.append(f"x: linthresh = {x_linthresh:g}")
    if y_linthresh is not None:
        symlog_notes.append(f"linthresh = {y_linthresh:g}")
    if symlog_notes:
        ax.text(
            0.99, 0.01, "симлог: " + ", ".join(symlog_notes),
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=8, color="#4b5563",
        )
    if category_order:
        ax.set_xticks(range(len(category_order)), [_category_label(v) for v in category_order])
    from matplotlib.ticker import ScalarFormatter

    if xscale == "linear" and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    if yscale == "linear" and isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    apply_adp_axis_style(ax, xlabel=xlabel, ylabel=ylabel, title=panel_title or "")


def _finish_figure(
    fig: Any,
    title: str,
    subtitle: str,
    parameter_handles: Mapping[str, Any],
    *,
    statistic: str | None = None,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10, color="#4b5563")
    anchor_x = 1.01
    if parameter_handles:
        legend = fig.legend(
            list(parameter_handles.values()), list(parameter_handles.keys()),
            title="Параметры", loc="upper left", bbox_to_anchor=(anchor_x, 0.9),
        )
        legend._legend_box.align = "left"
    if statistic:
        if statistic == "quantile":
            handles = [
                Line2D([], [], color="#2563eb", marker="o", linewidth=1.8),
                Patch(facecolor="#2563eb", alpha=0.18),
            ]
            labels = ["линия с маркером — медиана", "полоса — интервал 5–95%"]
        elif statistic == "wilson":
            handles = [
                Line2D([], [], color="#2563eb", marker="o", linewidth=1.8),
                Line2D([], [], color="#2563eb", marker="|", markersize=10, linewidth=1.2),
            ]
            labels = [
                "линия с маркером — оценка доли",
                "усики — 95% доверительный интервал Уилсона",
            ]
        else:
            handles = [
                Line2D([], [], color="#111827", linewidth=1.8),
                Patch(facecolor="#dbeafe", edgecolor="#2563eb"),
                Line2D([], [], color="#2563eb", marker="_", markersize=10, linewidth=0),
            ]
            labels = ["линия — медиана", "коробка — 25–75%", "усики — 5–95%"]
        y = 0.62 if parameter_handles else 0.9
        legend = fig.legend(handles, labels, title="Статистика", loc="upper left",
                            bbox_to_anchor=(anchor_x, y))
        legend._legend_box.align = "left"
    fig.subplots_adjust(top=0.87, right=0.78 if (parameter_handles or statistic) else 0.96,
                        bottom=0.14, hspace=0.36, wspace=0.28)


def _apply_scale(ax: Any, axis: str, scale: Scale) -> float | None:
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if scale == "log2":
        setter("log", base=2)
        return None
    elif scale == "symlog":
        values = []
        for line in ax.lines:
            data = line.get_xdata() if axis == "x" else line.get_ydata()
            values.extend(np.asarray(data, dtype=float).ravel())
        finite = np.abs(np.asarray(values, dtype=float))
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size:
            linthresh = float(finite.min() / 10.0)
            setter("symlog", linthresh=linthresh)
            return linthresh
        else:
            setter("linear")
            return None
    else:
        setter(scale)
        return None


def _set_experimental_x_ticks(ax: Any, scale: Scale) -> None:
    if scale not in {"log", "log2"}:
        return
    values = []
    for line in ax.lines:
        values.extend(np.asarray(line.get_xdata(), dtype=float).ravel())
    finite = np.asarray(values, dtype=float)
    finite = np.unique(finite[np.isfinite(finite) & (finite > 0)])
    if not 0 < finite.size <= 16:
        return
    ax.set_xticks(finite)
    ax.set_xticklabels([_format_number(value) for value in finite])


def _data_annotation(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    arrays = []
    for column in columns:
        if column in frame:
            arrays.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    if not arrays:
        return "нет наблюдений"
    values = np.concatenate(arrays) if arrays else np.array([], dtype=float)
    if values.size == 0:
        return "нет наблюдений"
    nan_count = int(np.isnan(values).sum())
    posinf = int(np.isposinf(values).sum())
    neginf = int(np.isneginf(values).sum())
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        if posinf == values.size:
            return f"все {values.size} {_value_word(values.size)} равны +∞"
        if neginf == values.size:
            return f"все {values.size} {_value_word(values.size)} равны −∞"
        return f"нет конечных значений: NaN = {nan_count}, +∞ = {posinf}, −∞ = {neginf}"
    notes = []
    if nan_count or posinf or neginf:
        notes.append(f"невалидные: NaN = {nan_count}, +∞ = {posinf}, −∞ = {neginf}")
    if finite.size > 1 and np.allclose(finite, finite[0], rtol=0.0, atol=0.0):
        notes.append("значение постоянно")
    return "; ".join(notes)


def _value_word(count: int) -> str:
    remainder_100 = count % 100
    remainder_10 = count % 10
    if 11 <= remainder_100 <= 14:
        return "значений"
    if remainder_10 == 1:
        return "значение"
    if 2 <= remainder_10 <= 4:
        return "значения"
    return "значений"


def _category_values(values: pd.Series, order: Sequence[Any]) -> list[Any]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    if not order:
        return present
    ordered = [value for value in order if str(value) in present]
    return [*ordered, *(value for value in present if str(value) not in {str(v) for v in ordered})]


def _category_label(value: Any) -> str:
    labels = {
        "linear": "линейная", "quadratic": "квадратичная", "square": "квадрат",
        "sin": "синус", "tanh": "гиперболический тангенс", "oscillating": "осциллирующая",
        "gaussian": "гауссовское", "uniform": "равномерное",
        "student_t5": "Стьюдент, 5 степеней свободы",
        "student_t3": "Стьюдент, 3 степени свободы",
        "False": "нет", "True": "да", "success": "успех",
        "nonconverged": "нет сходимости", "numerical_failure": "численный сбой",
    }
    return labels.get(str(value), _format_number(value))


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _heatmap_value(value: float, integer: bool) -> str:
    return str(int(round(value))) if integer else f"{value:.3f}"


def _contrast_color(value: float, limits: tuple[float, float] | None) -> str:
    if not np.isfinite(value):
        return "#374151"
    if limits is None:
        return "white"
    fraction = (value - limits[0]) / max(limits[1] - limits[0], np.finfo(float).eps)
    return "white" if fraction < 0.45 or fraction > 0.82 else "#111827"


__all__ = [
    "boxplot", "grouped_line", "heatmap", "line_with_quantile_band",
    "line_with_wilson_interval", "scatter", "stacked_runtime",
]
