from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.plotting import (
    ADP_COLORS,
    apply_adp_axis_style,
    configure_adp_matplotlib,
    prepare_adp_axis,
    save_figure as save_common_figure,
    set_adp_figure_size,
)


def save_benchmark_report(
    frame: pd.DataFrame,  # Таблица результатов замеров.
    output_dir: str | Path,  # Каталог отчета.
    *,
    prefix: str = "adp_benchmark",  # Префикс файлов.
    dpi: int = 150,  # Разрешение изображения.
) -> dict[str, Path]:
    """Сохраняет CSV и обзорные графики benchmark.

    Вход:
        frame: таблица результатов.
        output_dir: каталог для файлов.
        prefix: префикс имен.
        dpi: разрешение PNG.
    Выход:
        Словарь имя_артефакта -> Path.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # Таблица сохраняется до графиков, чтобы численный результат был доступен
    # даже при сбое отрисовки в окружении без экрана.
    csv_path = output_path / f"{prefix}.csv"
    frame.to_csv(csv_path, index=False)
    saved["csv"] = csv_path

    configure_adp_matplotlib()
    import matplotlib.pyplot as plt

    # Графики строятся по средним значениям: детальная статистика с
    # доверительными интервалами живет отдельно в benchmark_summary(...).
    quality = frame.groupby(["scenario", "method"], as_index=False)["cosine_abs"].mean()
    fig, ax = plt.subplots()
    set_adp_figure_size(fig, width=max(7.5, 1.35 * quality["scenario"].nunique()), height=4.8)
    plot_grouped_bars(ax, quality, value="cosine_abs", ylabel="среднее |cos(beta, beta_hat)|", title="Качество восстановления EDR")
    saved["quality_plot"] = save_figure(fig, output_path / f"{prefix}_quality.png", dpi=dpi)

    timings = frame.groupby(["scenario", "method"], as_index=False)["fit_time_sec"].mean()
    fig, ax = plt.subplots()
    set_adp_figure_size(fig, width=max(7.5, 1.35 * timings["scenario"].nunique()), height=4.8)
    plot_grouped_bars(ax, timings, value="fit_time_sec", ylabel="среднее время обучения, сек", title="Время обучения EDR")
    saved["time_plot"] = save_figure(fig, output_path / f"{prefix}_time.png", dpi=dpi)

    if "peak_memory_kib" in frame.columns:
        # Память есть не у старых таблиц, поэтому график строится условно.
        memory = frame.groupby(["scenario", "method"], as_index=False)["peak_memory_kib"].mean()
        fig, ax = plt.subplots()
        set_adp_figure_size(fig, width=max(7.5, 1.35 * memory["scenario"].nunique()), height=4.8)
        plot_grouped_bars(ax, memory, value="peak_memory_kib", ylabel="пиковая память, КиБ", title="Пиковая память EDR")
        saved["memory_plot"] = save_figure(fig, output_path / f"{prefix}_memory.png", dpi=dpi)

    return saved


def benchmark_summary(
    frame: pd.DataFrame,  # Таблица результатов замеров.
) -> pd.DataFrame:
    """Строит сводную таблицу по качеству, времени и памяти.

    Вход:
        frame: таблица результатов run_benchmark_suite.
    Выход:
        DataFrame со средними, std и 95% CI.
    """

    # Группировка по scenario/method сохраняет возможность сравнивать рабочий
    # ADP new с готовыми EDR-методами на одной сетке параметров.
    source = frame.copy()
    resource_metrics = (
        "algorithm_time_sec",
        "algorithm_rss_min_mib",
        "algorithm_rss_mean_mib",
        "algorithm_rss_max_mib",
        "full_run_time_sec",
        "full_run_rss_min_mib",
        "full_run_rss_mean_mib",
        "full_run_rss_max_mib",
    )
    for metric in resource_metrics:
        if metric not in source:
            source[metric] = np.nan
    summary = (
        source.groupby(["scenario", "method"])
        .agg(
            count=("cosine_abs", "count"),
            cosine_abs_mean=("cosine_abs", "mean"),
            cosine_abs_std=("cosine_abs", "std"),
            angle_deg_mean=("angle_deg", "mean"),
            angle_deg_std=("angle_deg", "std"),
            fit_time_sec_mean=("fit_time_sec", "mean"),
            fit_time_sec_std=("fit_time_sec", "std"),
            peak_memory_kib_mean=("peak_memory_kib", "mean"),
            peak_memory_kib_std=("peak_memory_kib", "std"),
            algorithm_time_sec_mean=("algorithm_time_sec", "mean"),
            algorithm_rss_min_mib_mean=("algorithm_rss_min_mib", "mean"),
            algorithm_rss_mean_mib_mean=("algorithm_rss_mean_mib", "mean"),
            algorithm_rss_max_mib_mean=("algorithm_rss_max_mib", "mean"),
            full_run_time_sec_mean=("full_run_time_sec", "mean"),
            full_run_rss_min_mib_mean=("full_run_rss_min_mib", "mean"),
            full_run_rss_mean_mib_mean=("full_run_rss_mean_mib", "mean"),
            full_run_rss_max_mib_mean=("full_run_rss_max_mib", "mean"),
            failures=("failed", "sum"),
        )
        .reset_index()
    )
    return add_confidence_intervals(summary)


def add_confidence_intervals(
    summary: pd.DataFrame,  # Сводная таблица без доверительных интервалов.
) -> pd.DataFrame:
    """Добавляет 95% доверительные интервалы для средних значений.

    Вход:
        summary: таблица со средними, std и count.
    Выход:
        Копия таблицы с *_ci95_low/high колонками.
    """

    result = summary.copy()
    for value in ("cosine_abs", "angle_deg", "fit_time_sec", "peak_memory_kib"):
        # Для одного повтора стандартное отклонение становится NaN;
        # тогда интервал вырождается в среднее.
        mean_col = f"{value}_mean"
        std_col = f"{value}_std"
        low_col = f"{value}_ci95_low"
        high_col = f"{value}_ci95_high"
        stderr = result[std_col].fillna(0.0) / result["count"].pow(0.5)
        radius = 1.96 * stderr
        result[low_col] = result[mean_col] - radius
        result[high_col] = result[mean_col] + radius
    return result


def plot_grouped_bars(
    ax: Any,  # Ось графика.
    frame: pd.DataFrame,  # Таблица scenario/method/value.
    *,
    value: str,  # Имя числовой колонки.
    ylabel: str,  # Подпись оси y.
    title: str,  # Заголовок графика.
) -> None:
    """Рисует grouped bar chart для benchmark-таблицы.

    Вход:
        ax: axis для рисования.
        frame: таблица со столбцами scenario, method и value.
        value: имя значения.
        ylabel: подпись оси y.
        title: заголовок.
    Выход:
        None; график рисуется на ax.
    """

    pivot = frame.pivot(index="scenario", columns="method", values=value)
    if ax.figure.get_size_inches()[0] < 7.0:
        set_adp_figure_size(ax.figure)
    prepare_adp_axis(ax)
    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.78,
        color=list(ADP_COLORS[: len(pivot.columns)]),
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    ax.margins(y=0.12)
    if value == "cosine_abs":
        ax.set_ylim(0.0, max(1.0, min(1.05, float(pivot.max().max()) * 1.12)))
    apply_adp_axis_style(
        ax,
        xlabel="сценарий",
        ylabel=ylabel,
        title=title,
        legend_title="метод",
        x_rotation=30,
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def save_figure(
    fig: Any,  # Объект рисунка.
    path: Path,  # Путь сохранения.
    *,
    dpi: int = 150,  # Разрешение изображения.
) -> Path:
    """Сохраняет figure и закрывает ее.

    Вход:
        fig: matplotlib figure.
        path: путь к файлу.
        dpi: разрешение.
    Выход:
        Path сохраненного файла.
    """

    return save_common_figure(fig, path, dpi=dpi, close=True)


def ensure_matplotlib_config_dir() -> None:
    """Готовит MPLCONFIGDIR для headless-окружений.

    Вход:
        Нет явных аргументов.
    Выход:
        None; при необходимости обновляет os.environ.
    """

    if "MPLCONFIGDIR" not in os.environ:
        config_dir = Path("/tmp") / "adp_matplotlib"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(config_dir)
