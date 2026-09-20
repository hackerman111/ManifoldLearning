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


@dataclass(frozen=True, slots=True)
class PlotSpec:
    filename: str
    table: str
    kind: str
    x: str
    y: str
    title: str
    xlabel: str
    ylabel: str
    scope: str = "summary"
    groups: tuple[str, ...] = ("variant",)
    required_metadata: tuple[str, ...] = ()
    required_any_metadata: tuple[str, ...] = ()
    value: str | None = None
    components: tuple[str, ...] = ()
    xscale: str = "linear"
    yscale: str = "linear"
    ylim: tuple[float, float] | None = None
    value_limits: tuple[float, float] | None = None
    normalize: bool = False
    aggregate: str = "median"


_RUNTIME_COMPONENTS = (
    "stage_initialization_time_sec",
    "stage_directions_time_sec",
    "stage_statistics_time_sec",
    "stage_solver_time_sec",
    "stage_update_time_sec",
)


PLOT_MANIFEST = (
    PlotSpec(
        "projector_error_vs_outer_iteration.png", "outer", "quantile",
        "outer_k", "projector_error", "Ошибка проектора по внешним итерациям",
        "Внешняя итерация", "Ошибка проектора", scope="point", ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "quality_vs_outer_iteration.png", "outer", "quantile", "outer_k",
        "cosine_abs", "Качество направления по внешним итерациям",
        "Внешняя итерация", "Абсолютный косинус направления", scope="point",
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "bandwidth_vs_outer_iteration.png", "outer", "quantile", "outer_k",
        "h_k", "Ширина окна по внешним итерациям", "Внешняя итерация",
        "Ширина окна h", scope="point", groups=("variant", "d", "n_over_d"),
        yscale="log",
    ),
    PlotSpec(
        "rho_vs_outer_iteration.png", "outer", "quantile", "outer_k", "rho_k",
        "Анизотропия по внешним итерациям", "Внешняя итерация",
        "Параметр анизотропии rho", scope="point", ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "beta_step_vs_outer_iteration.png", "outer", "quantile", "outer_k",
        "beta_delta", "Изменение направления по внешним итерациям",
        "Внешняя итерация", "Шаг направления", scope="point", yscale="symlog",
    ),
    PlotSpec(
        "objective_vs_outer_iteration.png", "outer", "quantile", "outer_k",
        "objective_after", "Целевая функция по внешним итерациям",
        "Внешняя итерация", "Значение целевой функции", scope="point",
        yscale="symlog",
    ),
    PlotSpec(
        "objective_vs_inner_iteration.png", "inner", "quantile", "inner_k",
        "objective", "Целевая функция по внутренним итерациям",
        "Внутренняя итерация", "Значение целевой функции", scope="point",
        groups=("variant", "outer_k"), yscale="symlog",
    ),
    PlotSpec(
        "beta_step_vs_inner_iteration.png", "inner", "quantile", "inner_k",
        "beta_delta", "Изменение направления по внутренним итерациям",
        "Внутренняя итерация", "Шаг направления", scope="point",
        groups=("variant", "outer_k"), yscale="symlog",
    ),
    PlotSpec(
        "solver_residual_vs_iteration.png", "solver", "quantile", "solver_k",
        "relative_residual", "Невязка линейного решателя",
        "Итерация линейного решателя", "Относительная невязка", scope="point",
        groups=("variant", "outer_k", "inner_k"), yscale="symlog",
    ),
    PlotSpec(
        "local_mass_by_outer_iteration.png", "local", "box", "outer_k",
        "local_mass", "Локальная масса по внешним итерациям",
        "Внешняя итерация", "Локальная масса", scope="point",
    ),
    PlotSpec(
        "effective_neighbors_by_outer_iteration.png", "local", "box", "outer_k",
        "ess", "Эффективное число соседей по внешним итерациям",
        "Внешняя итерация", "Эффективное число соседей", scope="point",
    ),
    PlotSpec(
        "local_condition_by_outer_iteration.png", "local", "box", "outer_k",
        "condition", "Обусловленность локальных систем",
        "Внешняя итерация", "Число обусловленности", scope="point", yscale="log",
    ),
    PlotSpec(
        "mass_vs_condition.png", "local", "scatter", "local_mass", "condition",
        "Связь локальной массы и обусловленности", "Локальная масса",
        "Число обусловленности", scope="point", groups=("variant", "outer_k"),
        yscale="log",
    ),
    PlotSpec(
        "local_slopes_by_outer_iteration.png", "local", "box", "outer_k", "slope",
        "Локальные наклоны по внешним итерациям", "Внешняя итерация",
        "Локальный наклон", scope="point", yscale="symlog",
    ),
    PlotSpec(
        "quality_heatmap_d_nd_ratio.png", "runs", "heatmap", "n_over_d", "d",
        "Качество по размерности и объёму выборки", "Отношение n/d", "Размерность d",
        required_metadata=("d", "n_over_d"), value="cosine_abs",
        value_limits=(0.0, 1.0),
    ),
    PlotSpec(
        "success_rate_heatmap.png", "runs", "heatmap", "n_over_d", "d",
        "Доля успешных запусков", "Отношение n/d", "Размерность d",
        required_metadata=("d", "n_over_d"), value="success_value",
        value_limits=(0.0, 1.0), aggregate="mean",
    ),
    PlotSpec(
        "runtime_vs_dimension.png", "runs", "median_line", "d",
        "algorithm_time_sec", "Время работы в зависимости от размерности",
        "Размерность d", "Время алгоритма, с", groups=("variant", "n_over_d"),
        required_metadata=("d", "n_over_d"), xscale="log", yscale="log",
    ),
    PlotSpec(
        "memory_vs_dimension.png", "runs", "median_line", "d",
        "algorithm_rss_max_mib", "Пиковая память в зависимости от размерности",
        "Размерность d", "Максимальный RSS процесса, МиБ",
        groups=("variant", "n_over_d"), required_metadata=("d", "n_over_d"),
        xscale="log",
    ),
    PlotSpec(
        "iterations_heatmap_d_nd_ratio.png", "runs", "heatmap", "n_over_d", "d",
        "Число внешних итераций", "Отношение n/d", "Размерность d",
        required_metadata=("d", "n_over_d"), value="outer_iterations",
    ),
    PlotSpec(
        "quality_vs_sigma_eps.png", "runs", "quantile", "sigma_eps", "cosine_abs",
        "Качество направления в зависимости от шума", "Стандартное отклонение шума",
        "Абсолютный косинус направления", groups=("variant", "d", "n_over_d"),
        required_metadata=("sigma_eps",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "success_rate_vs_sigma_eps.png", "runs", "proportion", "sigma_eps",
        "success_value", "Доля успешных запусков в зависимости от шума",
        "Стандартное отклонение шума", "Доля успешных запусков",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_eps",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "runtime_vs_sigma_eps.png", "runs", "quantile", "sigma_eps",
        "algorithm_time_sec", "Время работы в зависимости от шума",
        "Стандартное отклонение шума", "Время алгоритма, с",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_eps",),
    ),
    PlotSpec(
        "outer_iterations_vs_sigma_eps.png", "runs", "quantile", "sigma_eps",
        "outer_iterations", "Число внешних итераций в зависимости от шума",
        "Стандартное отклонение шума", "Число внешних итераций",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_eps",),
    ),
    PlotSpec(
        "final_objective_vs_sigma_eps.png", "runs", "quantile", "sigma_eps",
        "objective", "Финальная целевая функция в зависимости от шума",
        "Стандартное отклонение шума", "Финальная целевая функция",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_eps",),
        yscale="symlog",
    ),
    PlotSpec(
        "quality_vs_correlation.png", "runs", "quantile", "rho_corr", "cosine_abs",
        "Качество направления в зависимости от корреляции", "Корреляция AR(1)",
        "Абсолютный косинус направления", groups=("variant", "d", "n_over_d"),
        required_metadata=("rho_corr",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "success_rate_vs_correlation.png", "runs", "proportion", "rho_corr",
        "success_value", "Доля успешных запусков в зависимости от корреляции",
        "Корреляция AR(1)", "Доля успешных запусков",
        groups=("variant", "d", "n_over_d"), required_metadata=("rho_corr",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "local_condition_vs_correlation.png", "outer", "quantile", "rho_corr",
        "condition_median", "Локальная обусловленность в зависимости от корреляции",
        "Корреляция AR(1)", "Медианное число обусловленности",
        groups=("variant", "d", "n_over_d"), required_metadata=("rho_corr",),
        yscale="log",
    ),
    PlotSpec(
        "solver_iterations_vs_correlation.png", "outer", "quantile", "rho_corr",
        "linear_solver_iterations", "Итерации решателя в зависимости от корреляции",
        "Корреляция AR(1)", "Число итераций линейного решателя",
        groups=("variant", "d", "n_over_d"), required_metadata=("rho_corr",),
    ),
    PlotSpec(
        "runtime_vs_correlation.png", "runs", "quantile", "rho_corr",
        "algorithm_time_sec", "Время работы в зависимости от корреляции",
        "Корреляция AR(1)", "Время алгоритма, с",
        groups=("variant", "d", "n_over_d"), required_metadata=("rho_corr",),
    ),
    PlotSpec(
        "singular_fraction_vs_correlation.png", "outer", "quantile", "rho_corr",
        "singular_fraction", "Доля вырожденных локальных систем",
        "Корреляция AR(1)", "Доля вырожденных систем",
        groups=("variant", "d", "n_over_d"), required_metadata=("rho_corr",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "quality_vs_sigma_x.png", "runs", "quantile", "sigma_x", "cosine_abs",
        "Качество при изменении масштаба признаков", "Масштаб признаков",
        "Абсолютный косинус направления", groups=("variant", "d", "n_over_d"),
        required_metadata=("sigma_x",), xscale="log2", ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "h0_vs_sigma_x.png", "runs", "quantile", "sigma_x", "h_initial",
        "Начальная ширина окна", "Масштаб признаков", "Начальная ширина окна",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_x",),
        xscale="log2", yscale="log",
    ),
    PlotSpec(
        "final_bandwidth_vs_sigma_x.png", "runs", "quantile", "sigma_x", "h_final",
        "Финальная ширина окна", "Масштаб признаков", "Финальная ширина окна",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_x",),
        xscale="log2", yscale="log",
    ),
    PlotSpec(
        "local_mass_vs_sigma_x.png", "outer", "quantile", "sigma_x",
        "local_mass_mean", "Локальная масса при изменении масштаба признаков",
        "Масштаб признаков", "Средняя локальная масса",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_x",),
        xscale="log2",
    ),
    PlotSpec(
        "runtime_vs_sigma_x.png", "runs", "quantile", "sigma_x",
        "algorithm_time_sec", "Время работы при изменении масштаба признаков",
        "Масштаб признаков", "Время алгоритма, с",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_x",),
        xscale="log2", yscale="log",
    ),
    PlotSpec(
        "bandwidth_ratio_vs_sigma_x.png", "runs", "quantile", "sigma_x",
        "bandwidth_ratio", "Масштабная эквивариантность ширины окна",
        "Масштаб признаков", "Отношение финальной ширины окна к масштабу",
        groups=("variant", "d", "n_over_d"), required_metadata=("sigma_x",),
        xscale="log2",
    ),
    PlotSpec(
        "quality_by_link_function.png", "runs", "box", "link", "cosine_abs",
        "Качество для разных функций связи", "Функция связи",
        "Абсолютный косинус направления", required_metadata=("link",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "success_rate_by_link_function.png", "runs", "proportion", "link",
        "success_value", "Доля успешных запусков для функций связи",
        "Функция связи", "Доля успешных запусков", required_metadata=("link",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "outer_iterations_by_link_function.png", "runs", "box", "link",
        "outer_iterations", "Число внешних итераций для функций связи",
        "Функция связи", "Число внешних итераций", required_metadata=("link",),
    ),
    PlotSpec(
        "objective_by_link_function.png", "runs", "box", "link", "objective",
        "Целевая функция для функций связи", "Функция связи",
        "Финальная целевая функция", required_metadata=("link",), yscale="symlog",
    ),
    PlotSpec(
        "local_slopes_by_link_function.png", "local", "box", "link", "slope",
        "Локальные наклоны для функций связи", "Функция связи", "Локальный наклон",
        required_metadata=("link",), yscale="symlog",
    ),
    PlotSpec(
        "quality_by_x_distribution.png", "runs", "box", "x_distribution",
        "cosine_abs", "Качество для распределений признаков",
        "Распределение признаков", "Абсолютный косинус направления",
        required_metadata=("x_distribution",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "quality_by_noise_distribution.png", "runs", "box", "noise_distribution",
        "cosine_abs", "Качество для распределений шума", "Распределение шума",
        "Абсолютный косинус направления", required_metadata=("noise_distribution",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "failure_rate_by_distribution.png", "runs", "proportion", "distribution",
        "failure_value", "Доля численных сбоев для распределений", "Распределение",
        "Доля численных сбоев", groups=("variant", "experiment"),
        required_any_metadata=("x_distribution", "noise_distribution"),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "runtime_by_distribution.png", "runs", "median_line", "distribution",
        "algorithm_time_sec", "Время работы для распределений", "Распределение",
        "Время алгоритма, с", groups=("variant", "experiment"),
        required_any_metadata=("x_distribution", "noise_distribution"),
    ),
    PlotSpec(
        "quality_by_heteroscedasticity.png", "runs", "box", "heteroscedastic",
        "cosine_abs", "Влияние гетероскедастичности на качество",
        "Гетероскедастичность", "Абсолютный косинус направления",
        required_metadata=("heteroscedastic",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "quality_vs_outlier_fraction.png", "runs", "quantile",
        "effective_outlier_fraction", "cosine_abs",
        "Качество в зависимости от доли выбросов", "Фактическая доля выбросов",
        "Абсолютный косинус направления", groups=("variant", "outlier_scale"),
        required_metadata=("effective_outlier_fraction",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "failure_rate_vs_outliers.png", "runs", "proportion",
        "effective_outlier_fraction", "failure_value",
        "Доля численных сбоев в зависимости от выбросов",
        "Фактическая доля выбросов", "Доля численных сбоев",
        groups=("variant", "outlier_scale"),
        required_metadata=("effective_outlier_fraction",), ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "quality_vs_model_misspecification.png", "runs", "quantile", "delta",
        "cosine_abs", "Качество при нарушении модели", "Сила нарушения модели",
        "Абсолютный косинус направления", required_metadata=("delta",),
        ylim=(0.0, 1.0),
    ),
    PlotSpec(
        "objective_vs_model_misspecification.png", "runs", "quantile", "delta",
        "objective", "Целевая функция при нарушении модели", "Сила нарушения модели",
        "Финальная целевая функция", required_metadata=("delta",), yscale="symlog",
    ),
    PlotSpec(
        "runtime_breakdown.png", "runs", "stacked", "experiment",
        "algorithm_time_sec", "Абсолютное время по этапам алгоритма", "Эксперимент",
        "Медианное время, с", components=_RUNTIME_COMPONENTS,
    ),
    PlotSpec(
        "runtime_share_breakdown.png", "runs", "stacked", "experiment",
        "algorithm_time_sec", "Доли времени по этапам алгоритма", "Эксперимент",
        "Доля времени", components=_RUNTIME_COMPONENTS, normalize=True,
    ),
    PlotSpec(
        "status_breakdown.png", "runs", "stacked", "experiment", "status",
        "Технические статусы запусков", "Эксперимент", "Доля запусков",
        components=("success", "nonconverged", "numerical_failure"),
        normalize=True, aggregate="mean",
    ),
)


def _for_mode(spec: PlotSpec, mode: str) -> PlotSpec | None:
    if mode == "single":
        return spec
    if spec.filename == "rho_vs_outer_iteration.png":
        return replace(
            spec,
            filename="alpha_vs_outer_iteration.png",
            y="alpha_k",
            ylabel="Параметр локализации alpha",
        )
    if "quality" in spec.filename:
        heatmap = spec.kind == "heatmap"
        return replace(
            spec,
            filename=spec.filename.replace("quality", "projector_error"),
            title=spec.title.replace("Качество", "Ошибка проектора").replace(
                "качество", "ошибку проектора"
            ),
            y=spec.y if heatmap else "projector_error",
            value="projector_error" if heatmap else spec.value,
            ylabel="Ошибка проектора",
            ylim=spec.ylim if heatmap else (0.0, 1.0),
            value_limits=(0.0, 1.0) if heatmap else spec.value_limits,
        )
    if "beta_step" in spec.filename:
        return replace(spec, ylabel="Шаг базиса")
    if spec.y in {"rho_k", "slope"}:
        return None
    return spec


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
    source = source.loc[finite.all(axis=1)]
    if source.empty:
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


_TABLE_FILES = {
    "runs": "run_summary.csv",
    "outer": "outer_iterations.csv",
    "inner": "inner_iterations.csv",
    "local": "local_diagnostics.csv",
    "solver": "solver_iterations.csv",
}
_TABLE_SOURCES = {
    "runs": ("run_summary.csv",),
    "outer": ("outer_iterations.csv", "run_summary.csv"),
    "inner": ("inner_iterations.csv", "run_summary.csv"),
    "local": ("local_diagnostics.csv", "run_summary.csv"),
    "solver": ("solver_iterations.csv", "run_summary.csv"),
}
_DERIVED_SOURCE_TABLES = {
    "objective": ("outer_iterations.csv",),
    "h_initial": ("outer_iterations.csv",),
    "h_final": ("outer_iterations.csv",),
    "bandwidth_ratio": ("outer_iterations.csv",),
    "local_mass_mean": ("local_diagnostics.csv",),
    "condition_median": ("local_diagnostics.csv",),
    "singular_fraction": ("local_diagnostics.csv",),
}
_RENDERERS = {
    "quantile": "_render_quantile",
    "median_line": "_render_median_line",
    "proportion": "_render_proportion",
    "box": "_render_box",
    "scatter": "_render_scatter",
    "heatmap": "_render_heatmap",
    "stacked": "_render_stacked",
}


def _read_report_table(path: Path, *, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fill_column(
    frame: pd.DataFrame,
    name: str,
    values: pd.Series,
) -> None:
    mapped = frame["run_id"].map(values)
    frame[name] = frame[name].combine_first(mapped) if name in frame else mapped


def _ordered_run_values(
    outer: pd.DataFrame,
    column: str,
    *,
    first: bool = False,
) -> pd.Series | None:
    if not {"run_id", "outer_k", column} <= set(outer):
        return None
    ordered = outer.assign(
        _outer_order=pd.to_numeric(outer["outer_k"], errors="coerce")
    ).sort_values(["run_id", "_outer_order"], kind="stable")
    grouped = ordered.groupby("run_id", sort=False, dropna=False)[column]
    return grouped.first() if first else grouped.last()


def _prepare_runs(runs: pd.DataFrame, outer: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    status = runs.get("status", pd.Series(index=runs.index, dtype="object"))
    for name in ("success", "nonconverged", "numerical_failure"):
        runs[name] = status.eq(name).fillna(False).astype(float)
    runs["success_value"] = runs["success"]
    runs["failure_value"] = runs["numerical_failure"]

    if "run_id" in runs:
        for name, source, first in (
            ("objective", "objective_after", False),
            ("h_initial", "h_k", True),
            ("h_final", "h_k", False),
        ):
            values = _ordered_run_values(outer, source, first=first)
            if values is not None:
                _fill_column(runs, name, values)

    if {"h_final", "sigma_x"} <= set(runs):
        h_final = pd.to_numeric(runs["h_final"], errors="coerce")
        sigma_x = pd.to_numeric(runs["sigma_x"], errors="coerce")
        ratio = h_final.div(sigma_x).replace([np.inf, -np.inf], np.nan)
        runs["bandwidth_ratio"] = (
            runs["bandwidth_ratio"].combine_first(ratio)
            if "bandwidth_ratio" in runs
            else ratio
        )

    distributions = [
        runs[name]
        for name in ("x_distribution", "noise_distribution")
        if name in runs
    ]
    if distributions:
        distribution = distributions[0]
        for values in distributions[1:]:
            distribution = distribution.combine_first(values)
        runs["distribution"] = distribution
    return runs


def _prepare_outer(outer: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    if outer.empty or local.empty or not {"run_id", "outer_k"} <= set(local):
        return outer.copy()
    prepared = outer.copy()
    keys = ["run_id", "outer_k"]
    aggregations = {}
    if "local_mass" in local:
        aggregations["local_mass_mean"] = ("local_mass", "mean")
    if "condition" in local:
        aggregations["condition_median"] = ("condition", "median")
    singular = next(
        (name for name in ("singular", "is_singular") if name in local),
        None,
    )
    if singular is not None:
        aggregations["singular_fraction"] = (singular, "mean")
    if not aggregations:
        return prepared
    sources = tuple(dict.fromkeys(source for source, _ in aggregations.values()))
    numeric = local[[*keys, *sources]].copy()
    for source, _ in aggregations.values():
        numeric[source] = pd.to_numeric(numeric[source], errors="coerce")
    derived = numeric.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
    prepared = prepared.merge(derived, on=keys, how="left", suffixes=("", "_derived"))
    for name in aggregations:
        derived_name = f"{name}_derived"
        if derived_name in prepared:
            prepared[name] = prepared[name].combine_first(prepared.pop(derived_name))
    return prepared


def _join_run_metadata(detail: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if "run_id" not in detail or "run_id" not in runs:
        return detail.copy()
    additions = [name for name in runs if name == "run_id" or name not in detail]
    metadata = runs[additions].drop_duplicates("run_id", keep="last")
    return detail.merge(metadata, on="run_id", how="left")


def _report_mode(series_dir: Path, runs: pd.DataFrame) -> str:
    values = (
        runs["mode"].dropna().astype(str).unique().tolist()
        if "mode" in runs
        else []
    )
    if not values:
        series = _read_report_table(series_dir / "series.csv")
        if "mode" in series:
            values = series["mode"].dropna().astype(str).unique().tolist()
    if len(values) > 1:
        raise ValueError("run_summary.csv contains multiple modes")
    return values[0] if values else "single"


def _not_applicable(frame: pd.DataFrame, spec: PlotSpec) -> str | None:
    if frame.empty:
        return "source table is empty"
    missing_metadata = [name for name in spec.required_metadata if name not in frame]
    if missing_metadata:
        return f"missing metadata: {', '.join(missing_metadata)}"
    if spec.required_any_metadata and not any(
        name in frame for name in spec.required_any_metadata
    ):
        return "missing any metadata: " + ", ".join(spec.required_any_metadata)
    required = {spec.x, spec.y, *spec.groups, *spec.components}
    if spec.value is not None:
        required.add(spec.value)
    missing = sorted(name for name in required if name and name not in frame)
    if missing:
        return f"missing columns: {', '.join(missing)}"
    if not frame[spec.x].notna().any():
        return f"no observations for {spec.x}"
    if spec.kind == "heatmap" and not frame[spec.y].notna().any():
        return f"no observations for {spec.y}"
    metrics = spec.components or ((spec.value or spec.y),)
    values = frame[list(metrics)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        return "no finite metric observations"
    return None


def _source_tables(spec: PlotSpec) -> str:
    tables = list(_TABLE_SOURCES[spec.table])
    fields = (spec.x, spec.y, *spec.groups, *spec.components)
    if spec.value is not None:
        fields = (*fields, spec.value)
    for field in fields:
        tables.extend(_DERIVED_SOURCE_TABLES.get(field, ()))
    return ",".join(dict.fromkeys(tables))


def _plot_target(
    series_dir: Path,
    point: str | None,
    filename: str,
) -> tuple[Path, Path]:
    components = (
        ("points", str(point), filename)
        if point is not None
        else ("summary", filename)
    )
    for component in components:
        if (
            not component
            or component in {".", ".."}
            or Path(component).parts != (component,)
            or Path(component).name != component
        ):
            raise ValueError(f"unsafe plot path component: {component!r}")
    root = series_dir.resolve()
    plots = (root / "plots").resolve(strict=False)
    target = plots.joinpath(*components).resolve(strict=False)
    try:
        plots.relative_to(root)
        target.relative_to(root)
        relative = target.relative_to(plots)
    except ValueError as error:
        raise ValueError("plot path escapes the series directory") from error
    return target, Path("plots") / relative


def _atomic_artifacts(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ("filename", "path", "status", "source_tables", "error")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _remove_stale_plot(path: Path) -> str:
    try:
        path.unlink(missing_ok=True)
    except Exception as error:
        return f"cleanup failed: {type(error).__name__}: {error}"
    return ""


def write_reports(series_dir: str | Path) -> Path:
    series_dir = Path(series_dir)
    raw = {
        name: _read_report_table(
            series_dir / filename,
            required=name == "runs",
        )
        for name, filename in _TABLE_FILES.items()
    }
    mode = _report_mode(series_dir, raw["runs"])
    runs = _prepare_runs(raw["runs"], raw["outer"])
    outer = _prepare_outer(raw["outer"], raw["local"])
    tables = {
        "runs": runs,
        "outer": _join_run_metadata(outer, runs),
        "inner": _join_run_metadata(raw["inner"], runs),
        "local": _join_run_metadata(raw["local"], runs),
        "solver": _join_run_metadata(raw["solver"], runs),
    }
    points = (
        tuple(dict.fromkeys(runs["point"].dropna().astype(str)))
        if "point" in runs
        else ()
    )
    artifacts: list[dict[str, str]] = []
    seen: set[tuple[Path, str]] = set()

    for base in PLOT_MANIFEST:
        scopes = points if base.scope == "point" else (None,)
        for point in scopes:
            spec = _for_mode(base, mode)
            filename = spec.filename if spec is not None else base.filename
            source = _source_tables(spec or base)
            try:
                target, relative = _plot_target(series_dir, point, filename)
            except (OSError, ValueError) as error:
                artifacts.append(
                    {
                        "filename": filename,
                        "path": "",
                        "status": "error",
                        "source_tables": source,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            def record(status: str, error: str = "") -> None:
                artifacts.append(
                    {
                        "filename": filename,
                        "path": str(relative),
                        "status": status,
                        "source_tables": source,
                        "error": error,
                    }
                )

            def record_noncreated(status: str, error: str) -> None:
                cleanup = _remove_stale_plot(target)
                record(
                    "error" if cleanup else status,
                    f"{error}; {cleanup}" if cleanup else error,
                )

            if spec is None:
                record_noncreated("skipped", f"not applicable to mode {mode}")
                continue
            key = (target, spec.table)
            if key in seen:
                record("skipped", "duplicate adapted plot")
                continue
            seen.add(key)
            frame = tables[spec.table]
            if point is not None:
                if "point" not in frame:
                    frame = frame.iloc[:0]
                else:
                    frame = frame.loc[frame["point"].astype(str).eq(str(point))]
            reason = _not_applicable(frame, spec)
            if reason is not None:
                record_noncreated("skipped", reason)
                continue
            try:
                rendered = globals()[_RENDERERS[spec.kind]](
                    frame,
                    target,
                    x=spec.x,
                    y=spec.y,
                    title=spec.title,
                    xlabel=spec.xlabel,
                    ylabel=spec.ylabel,
                    groups=spec.groups,
                    value=spec.value,
                    components=spec.components,
                    xscale=spec.xscale,
                    yscale=spec.yscale,
                    ylim=spec.ylim,
                    value_limits=spec.value_limits,
                    normalize=spec.normalize,
                    aggregate=spec.aggregate,
                )
                if rendered is False:
                    record_noncreated(
                        "skipped", "renderer found no plottable observations"
                    )
                else:
                    record("created")
            except Exception as error:
                record_noncreated(
                    "error", f"{type(error).__name__}: {error}"
                )

    artifacts_path = series_dir / "artifacts.csv"
    _atomic_artifacts(artifacts_path, artifacts)
    return artifacts_path
