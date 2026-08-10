from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adp_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADP_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
)
ADP_AXIS_FACE = "#f8fafc"
ADP_FIGURE_FACE = "#ffffff"
ADP_GRID_COLOR = "#cbd5e1"
ADP_TEXT_COLOR = "#111827"
ADP_SPINE_COLOR = "#94a3b8"


def _keys(x: str, groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*groups, x)))


def _quantiles(
    frame: pd.DataFrame,
    x: str,
    y: str,
    groups: tuple[str, ...] = (),
) -> pd.DataFrame:
    keys = _keys(x, groups)
    columns = (*keys, "q05", "median", "q95")
    if not {*keys, y} <= set(frame):
        return pd.DataFrame(columns=columns)
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & np.isfinite(source[y])]
    if source.empty:
        return pd.DataFrame(columns=columns)
    return (
        source.groupby(
            list(keys),
            sort=True,
            dropna=False,
            observed=True,
            as_index=False,
        )
        .agg(
            q05=(y, lambda values: values.quantile(0.05)),
            median=(y, "median"),
            q95=(y, lambda values: values.quantile(0.95)),
        )
        .reset_index(drop=True)
    )


def _wilson(
    successes: float,
    total: float,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    if total <= 0 or successes < 0 or successes > total:
        return np.nan, np.nan, np.nan
    estimate = successes / total
    denominator = 1.0 + z**2 / total
    center = (estimate + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            estimate * (1.0 - estimate) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - margin, center, center + margin


def _style_axis(ax) -> None:
    ax.set_facecolor(ADP_AXIS_FACE)
    ax.set_prop_cycle(color=ADP_COLORS)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=ADP_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ADP_SPINE_COLOR)
    ax.spines["bottom"].set_color(ADP_SPINE_COLOR)
    ax.tick_params(axis="both", colors=ADP_TEXT_COLOR)


def _new_axis():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=ADP_FIGURE_FACE)
    _style_axis(ax)
    return fig, ax


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor=ADP_FIGURE_FACE,
        )
        return path
    finally:
        plt.close(fig)


def _iter_groups(frame: pd.DataFrame, groups: tuple[str, ...]):
    if not groups:
        yield None, frame
        return
    grouper = groups[0] if len(groups) == 1 else list(groups)
    for values, subset in frame.groupby(
        grouper,
        sort=False,
        dropna=False,
        observed=True,
    ):
        values = (values,) if len(groups) == 1 else values
        yield ", ".join(map(str, values)), subset


def _x_mapping(frame: pd.DataFrame, x: str):
    numeric = pd.to_numeric(frame[x], errors="coerce")
    if numeric.notna().all():
        return None
    labels = list(dict.fromkeys(frame[x].astype(str)))
    return {label: index for index, label in enumerate(labels)}


def _x_values(frame: pd.DataFrame, x: str, mapping):
    if mapping is None:
        return pd.to_numeric(frame[x], errors="coerce").to_numpy(dtype=float)
    return frame[x].astype(str).map(mapping).to_numpy(dtype=float)


def _validate_scales(xscale: str, yscale: str) -> None:
    allowed = {"linear", "log", "log2", "symlog"}
    if xscale not in allowed or yscale not in allowed:
        raise ValueError(f"scale must be one of {sorted(allowed)}")


def _log_x(frame: pd.DataFrame, x: str, scale: str) -> pd.DataFrame:
    if scale not in {"log", "log2"}:
        return frame
    numeric = pd.to_numeric(frame[x], errors="coerce")
    if (numeric.isna() & frame[x].notna()).any():
        raise ValueError("logarithmic x scale requires numeric values")
    source = frame.copy()
    source[x] = numeric
    return source.loc[np.isfinite(numeric) & numeric.gt(0)]


def _positive(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    scale: str,
) -> pd.DataFrame:
    if scale not in {"log", "log2"}:
        return frame
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    valid &= values.gt(0).all(axis=1).to_numpy()
    return frame.loc[valid]


def _reject_ordinal_log(scale: str, axis: str = "x") -> None:
    if scale in {"log", "log2"}:
        raise ValueError(
            f"logarithmic {axis} scale requires numeric, non-ordinal coordinates"
        )


def _validate_aggregate(aggregate: str) -> None:
    if aggregate not in {"mean", "median"}:
        raise ValueError("aggregate must be 'mean' or 'median'")


def _finish_axis(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    mapping=None,
) -> None:
    ax.set_title(title, color=ADP_TEXT_COLOR, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=ADP_TEXT_COLOR, labelpad=8)
    ax.set_ylabel(ylabel, color=ADP_TEXT_COLOR, labelpad=8)
    if mapping:
        ax.set_xticks(list(mapping.values()), list(mapping))
    if xscale == "log2":
        ax.set_xscale("log", base=2)
    else:
        ax.set_xscale(xscale)
    if yscale == "log2":
        ax.set_yscale("log", base=2)
    else:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _render_quantile(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    **_options,
):
    _validate_scales(xscale, yscale)
    summary = _quantiles(frame, x, y, groups)
    summary = _log_x(summary, x, xscale)
    summary = _positive(summary, ("q05", "median", "q95"), yscale)
    if summary.empty:
        return False
    fig, ax = _new_axis()
    try:
        mapping = _x_mapping(summary, x)
        for label, subset in _iter_groups(summary, groups):
            x_values = _x_values(subset, x, mapping)
            order = np.argsort(x_values)
            line = ax.plot(
                x_values[order],
                subset["median"].to_numpy(dtype=float)[order],
                marker="o",
                linewidth=1.8,
                label=label,
            )[0]
            ax.fill_between(
                x_values[order],
                subset["q05"].to_numpy(dtype=float)[order],
                subset["q95"].to_numpy(dtype=float)[order],
                color=line.get_color(),
                alpha=0.18,
            )
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=ylim,
            mapping=mapping,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)


def _render_median_line(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    **_options,
):
    _validate_scales(xscale, yscale)
    summary = _quantiles(frame, x, y, groups)
    summary = _log_x(summary, x, xscale)
    summary = _positive(summary, ("median",), yscale)
    if summary.empty:
        return False
    fig, ax = _new_axis()
    try:
        mapping = _x_mapping(summary, x)
        for label, subset in _iter_groups(summary, groups):
            x_values = _x_values(subset, x, mapping)
            order = np.argsort(x_values)
            ax.plot(
                x_values[order],
                subset["median"].to_numpy(dtype=float)[order],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=ylim,
            mapping=mapping,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)


def _render_proportion(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    **_options,
):
    _validate_scales(xscale, yscale)
    keys = _keys(x, groups)
    if not {*keys, y} <= set(frame):
        return False
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & source[y].isin((0.0, 1.0))]
    if source.empty:
        return False
    counts = (
        source.groupby(
            list(keys),
            sort=True,
            dropna=False,
            observed=True,
            as_index=False,
        )[y]
        .agg(successes="sum", total="count")
    )
    intervals = np.asarray(
        [_wilson(row.successes, row.total) for row in counts.itertuples()],
        dtype=float,
    )
    counts["low"] = intervals[:, 0]
    counts["estimate"] = counts["successes"] / counts["total"]
    counts["high"] = intervals[:, 2]
    counts = _log_x(counts, x, xscale)
    counts = _positive(counts, ("low", "estimate", "high"), yscale)
    if counts.empty:
        return False
    fig, ax = _new_axis()
    try:
        mapping = _x_mapping(counts, x)
        for label, subset in _iter_groups(counts, groups):
            x_values = _x_values(subset, x, mapping)
            order = np.argsort(x_values)
            estimate = subset["estimate"].to_numpy(dtype=float)[order]
            low = subset["low"].to_numpy(dtype=float)[order]
            high = subset["high"].to_numpy(dtype=float)[order]
            ax.errorbar(
                x_values[order],
                estimate,
                yerr=np.vstack((estimate - low, high - estimate)),
                marker="o",
                linewidth=1.8,
                capsize=4,
                label=label,
            )
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=ylim,
            mapping=mapping,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)


def _render_box(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    **_options,
):
    _validate_scales(xscale, yscale)
    _reject_ordinal_log(xscale)
    keys = _keys(x, groups)
    if not {*keys, y} <= set(frame):
        return False
    source = frame[[*keys, y]].copy()
    source[y] = pd.to_numeric(source[y], errors="coerce")
    source = source.loc[source[x].notna() & np.isfinite(source[y])]
    source = _positive(source, (y,), yscale)
    if source.empty:
        return False
    values = []
    labels = []
    grouper = keys[0] if len(keys) == 1 else list(keys)
    for key, subset in source.groupby(
        grouper,
        sort=True,
        dropna=False,
        observed=True,
    ):
        key = (key,) if len(keys) == 1 else key
        values.append(subset[y].to_numpy(dtype=float))
        labels.append(", ".join(f"{name}={value}" for name, value in zip(keys, key)))
    fig, ax = _new_axis()
    try:
        artists = ax.boxplot(
            values,
            tick_labels=labels,
            whis=(5, 95),
            patch_artist=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.28},
            medianprops={"color": ADP_TEXT_COLOR, "linewidth": 1.7},
        )
        for index, box in enumerate(artists["boxes"]):
            box.set_facecolor(ADP_COLORS[index % len(ADP_COLORS)])
            box.set_alpha(0.28)
        ax.tick_params(axis="x", rotation=25)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=ylim,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)


def _render_scatter(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    **_options,
):
    _validate_scales(xscale, yscale)
    if not {x, y, *groups} <= set(frame):
        return False
    source = frame[[*groups, x, y]].copy()
    source[[x, y]] = source[[x, y]].apply(pd.to_numeric, errors="coerce")
    source = source.loc[np.isfinite(source[[x, y]].to_numpy(dtype=float)).all(axis=1)]
    source = _log_x(source, x, xscale)
    source = _positive(source, (y,), yscale)
    if source.empty:
        return False
    fig, ax = _new_axis()
    try:
        for label, subset in _iter_groups(source, groups):
            ax.scatter(subset[x], subset[y], s=24, alpha=0.62, label=label)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=ylim,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)


def _render_heatmap(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    value: str | None = None,
    value_limits: tuple[float, float] | None = None,
    aggregate: str = "median",
    xscale: str = "linear",
    yscale: str = "linear",
    **_options,
):
    _validate_scales(xscale, yscale)
    _validate_aggregate(aggregate)
    _reject_ordinal_log(xscale)
    _reject_ordinal_log(yscale, "y")
    if value is None or not {x, y, value, *groups} <= set(frame):
        return False
    source = frame[[*groups, x, y, value]].copy()
    source[value] = pd.to_numeric(source[value], errors="coerce")
    source = source.loc[
        source[x].notna() & source[y].notna() & np.isfinite(source[value])
    ]
    if source.empty:
        return False
    prepared = []
    for label, subset in _iter_groups(source, groups):
        pivot = subset.pivot_table(
            index=y,
            columns=x,
            values=value,
            aggfunc=aggregate,
            dropna=False,
            observed=True,
        )
        matrix = pivot.to_numpy(dtype=float)
        if np.isfinite(matrix).any():
            prepared.append((label, pivot, matrix))
    if not prepared:
        return False
    count = len(prepared)
    columns = min(2, count)
    rows = int(np.ceil(count / columns))
    displayed = np.concatenate(
        [matrix[np.isfinite(matrix)] for _, _, matrix in prepared]
    )
    limits = value_limits or (float(displayed.min()), float(displayed.max()))
    fig, axes = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(10.5 if count == 1 else 13.0, 6.2 * rows),
        facecolor=ADP_FIGURE_FACE,
    )
    try:
        axes = list(axes.flat)
        images = []
        for index, (label, pivot, matrix) in enumerate(prepared):
            ax = axes[index]
            _style_axis(ax)
            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
            )
            images.append(image)
            ax.set_xticks(
                range(len(pivot.columns)),
                [str(item) for item in pivot.columns],
            )
            ax.set_yticks(
                range(len(pivot.index)),
                [str(item) for item in pivot.index],
            )
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    number = matrix[row, column]
                    if np.isfinite(number):
                        fraction = (number - limits[0]) / max(
                            limits[1] - limits[0], np.finfo(float).eps
                        )
                        color = (
                            "white"
                            if fraction < 0.45 or fraction > 0.82
                            else ADP_TEXT_COLOR
                        )
                        ax.text(
                            column,
                            row,
                            f"{number:.3g}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                        )
            ax.set_title(label or title, color=ADP_TEXT_COLOR, fontweight="bold")
            ax.set_xlabel(xlabel, color=ADP_TEXT_COLOR)
            ax.set_ylabel(ylabel, color=ADP_TEXT_COLOR)
        for ax in axes[count:]:
            ax.set_visible(False)
        fig.colorbar(images[0], ax=axes[:count], shrink=0.82, label=value)
        if count > 1:
            fig.suptitle(title, color=ADP_TEXT_COLOR, fontweight="bold")
        return _save(fig, path)
    finally:
        plt.close(fig)


def _status_component(name: str) -> bool:
    return name in {
        "success",
        "success_value",
        "failure_value",
        "nonconverged",
        "numerical_failure",
    }


def _render_stacked(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: tuple[str, ...] = (),
    components: tuple[str, ...] = (),
    normalize: bool = False,
    aggregate: str = "median",
    xscale: str = "linear",
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
    **_options,
):
    _validate_scales(xscale, yscale)
    _validate_aggregate(aggregate)
    _reject_ordinal_log(xscale)
    keys = _keys(x, groups)
    if not components or not {*keys, *components} <= set(frame):
        return False
    source = frame[[*keys, *components]].copy()
    source[list(components)] = source[list(components)].apply(
        pd.to_numeric, errors="coerce"
    )
    source = source.loc[source[x].notna()]
    finite = np.isfinite(source[list(components)].to_numpy(dtype=float))
    if source.empty or not finite.any():
        return False
    methods = {
        component: "mean"
        if aggregate == "mean" or _status_component(component)
        else "median"
        for component in components
    }
    summary = (
        source.replace([np.inf, -np.inf], np.nan)
        .groupby(
            list(keys),
            sort=True,
            dropna=False,
            observed=True,
            as_index=False,
        )
        .agg(methods)
    )
    if normalize:
        totals = summary[list(components)].sum(axis=1).replace(0.0, np.nan)
        summary[list(components)] = summary[list(components)].div(totals, axis=0)
    summary[list(components)] = summary[list(components)].fillna(0.0)
    summary["_stack_total"] = summary[list(components)].sum(axis=1)
    summary = _positive(summary, ("_stack_total",), yscale)
    if summary.empty:
        return False
    x_labels = list(dict.fromkeys(summary[x].astype(str)))
    x_positions = {label: index for index, label in enumerate(x_labels)}
    group_panels = list(_iter_groups(summary, groups))
    width = 0.8 / len(group_panels)
    fig, ax = _new_axis()
    try:
        for group_index, (label, subset) in enumerate(group_panels):
            positions = np.asarray(
                [x_positions[item] for item in subset[x].astype(str)], dtype=float
            )
            positions += (group_index - (len(group_panels) - 1) / 2.0) * width
            bottom = np.zeros(len(subset), dtype=float)
            for component_index, component in enumerate(components):
                values = subset[component].to_numpy(dtype=float)
                ax.bar(
                    positions,
                    values,
                    width=width,
                    bottom=bottom,
                    color=ADP_COLORS[component_index % len(ADP_COLORS)],
                    label=component if group_index == 0 else None,
                )
                bottom += values
            if label:
                for position in positions:
                    ax.text(
                        position,
                        0,
                        label,
                        ha="center",
                        va="top",
                        rotation=90,
                        fontsize=7,
                        transform=ax.get_xaxis_transform(),
                    )
        ax.set_xticks(range(len(x_labels)), x_labels)
        _finish_axis(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xscale=xscale,
            yscale=yscale,
            ylim=(0.0, 1.0) if normalize and ylim is None else ylim,
        )
        return _save(fig, path)
    finally:
        plt.close(fig)
