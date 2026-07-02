from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def save_benchmark_report(
    frame: pd.DataFrame,  # Таблица benchmark-результатов.
    output_dir: str | Path,  # Каталог отчета.
    *,
    prefix: str = "adp_benchmark",  # Префикс файлов.
    dpi: int = 150,  # Разрешение PNG.
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

    csv_path = output_path / f"{prefix}.csv"
    frame.to_csv(csv_path, index=False)
    saved["csv"] = csv_path

    ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    quality = frame.groupby(["scenario", "method"], as_index=False)["cosine_abs"].mean()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * quality["scenario"].nunique()), 4.5))
    plot_grouped_bars(ax, quality, value="cosine_abs", ylabel="среднее |cos(beta, beta_hat)|", title="Качество восстановления EDR")
    saved["quality_plot"] = save_figure(fig, output_path / f"{prefix}_quality.png", dpi=dpi)

    timings = frame.groupby(["scenario", "method"], as_index=False)["fit_time_sec"].mean()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * timings["scenario"].nunique()), 4.5))
    plot_grouped_bars(ax, timings, value="fit_time_sec", ylabel="среднее время обучения, сек", title="Время обучения EDR")
    saved["time_plot"] = save_figure(fig, output_path / f"{prefix}_time.png", dpi=dpi)

    if "peak_memory_kib" in frame.columns:
        memory = frame.groupby(["scenario", "method"], as_index=False)["peak_memory_kib"].mean()
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * memory["scenario"].nunique()), 4.5))
        plot_grouped_bars(ax, memory, value="peak_memory_kib", ylabel="пиковая память, КиБ", title="Пиковая память EDR")
        saved["memory_plot"] = save_figure(fig, output_path / f"{prefix}_memory.png", dpi=dpi)

    return saved


def benchmark_summary(
    frame: pd.DataFrame,  # Таблица benchmark-результатов.
) -> pd.DataFrame:
    """Строит сводную таблицу по качеству, времени и памяти.

    Вход:
        frame: таблица результатов run_benchmark_suite.
    Выход:
        DataFrame со средними, std и 95% CI.
    """

    summary = (
        frame.groupby(["scenario", "method"])
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
            failures=("failed", "sum"),
        )
        .reset_index()
    )
    return add_confidence_intervals(summary)


def add_confidence_intervals(
    summary: pd.DataFrame,  # Сводная таблица без CI.
) -> pd.DataFrame:
    """Добавляет 95% доверительные интервалы для средних значений.

    Вход:
        summary: таблица со средними, std и count.
    Выход:
        Копия таблицы с *_ci95_low/high колонками.
    """

    result = summary.copy()
    for value in ("cosine_abs", "angle_deg", "fit_time_sec", "peak_memory_kib"):
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
    ax: Any,  # Matplotlib axis.
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
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("сценарий")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="method", fontsize="small")


def save_figure(
    fig: Any,  # Matplotlib figure.
    path: Path,  # Путь сохранения.
    *,
    dpi: int = 150,  # Разрешение PNG.
) -> Path:
    """Сохраняет figure и закрывает ее.

    Вход:
        fig: matplotlib figure.
        path: путь к файлу.
        dpi: разрешение.
    Выход:
        Path сохраненного файла.
    """

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def ensure_matplotlib_config_dir() -> None:
    """Готовит MPLCONFIGDIR для headless-окружений.

    Вход:
        Нет явных аргументов.
    Выход:
        None; при необходимости обновляет os.environ.
    """

    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path("/tmp") / "adp_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
