from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ADP_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4b5563",
)
ADP_AXIS_FACE = "#f8fafc"
ADP_FIGURE_FACE = "#ffffff"
ADP_GRID_COLOR = "#cbd5e1"
ADP_TEXT_COLOR = "#111827"
ADP_SPINE_COLOR = "#94a3b8"


def configure_adp_matplotlib() -> None:
    """Настраивает базовые параметры matplotlib для графиков ADP."""

    ensure_matplotlib_config_dir()
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": ADP_TEXT_COLOR,
            "xtick.color": ADP_TEXT_COLOR,
            "ytick.color": ADP_TEXT_COLOR,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": ADP_FIGURE_FACE,
            "legend.edgecolor": ADP_SPINE_COLOR,
            "savefig.facecolor": ADP_FIGURE_FACE,
        }
    )


def set_adp_figure_size(fig: Any, *, width: float = 8.0, height: float = 4.8) -> Any:
    """Задает размер и фон figure."""

    fig.set_size_inches(width, height)
    fig.patch.set_facecolor(ADP_FIGURE_FACE)
    return fig


def prepare_adp_axis(ax: Any) -> Any:
    """Готовит ось перед построением линий или столбцов."""

    configure_adp_matplotlib()
    ax.set_prop_cycle(color=ADP_COLORS)
    ax.set_facecolor(ADP_AXIS_FACE)
    ax.set_axisbelow(True)
    return ax


def apply_adp_axis_style(
    ax: Any,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_title: str | None = None,
    x_rotation: float | None = None,
) -> Any:
    """Применяет единый стиль к оси ADP-графика."""

    prepare_adp_axis(ax)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", color=ADP_GRID_COLOR, alpha=0.75, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ADP_SPINE_COLOR)
    ax.spines["bottom"].set_color(ADP_SPINE_COLOR)
    ax.tick_params(axis="both", length=0, pad=6)
    if x_rotation is not None:
        ax.tick_params(axis="x", rotation=x_rotation)
    format_adp_legend(ax, title=legend_title)
    return ax


def set_integer_x_ticks(ax: Any, *, count: int, max_ticks: int = 16) -> Any:
    """Ставит целочисленные tick-метки для индексов компонент."""

    if count <= 0:
        return ax
    if count <= max_ticks:
        ticks = list(range(count))
    else:
        step = max(1, (count - 1) // (max_ticks - 1))
        ticks = list(range(0, count, step))
        if ticks[-1] != count - 1:
            ticks.append(count - 1)
    ax.set_xticks(ticks)
    return ax


def format_adp_legend(ax: Any, *, title: str | None = None) -> Any:
    """Оформляет легенду, если она уже создана."""

    legend = ax.get_legend()
    if legend is None:
        return None
    if title is not None:
        legend.set_title(title)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor(ADP_SPINE_COLOR)
    for text in legend.get_texts():
        text.set_color(ADP_TEXT_COLOR)
    if legend.get_title() is not None:
        legend.get_title().set_color(ADP_TEXT_COLOR)
    return legend


def save_figure(
    fig: Any,  # Объект рисунка.
    path: str | Path,  # Путь сохранения.
    *,
    dpi: int = 150,  # Разрешение изображения.
    close: bool = False,  # Закрыть рисунок после сохранения.
) -> Path:
    """Сохраняет matplotlib figure на диск.

    Вход:
        fig: объект matplotlib.figure.Figure.
        path: путь к файлу.
        dpi: разрешение.
        close: закрывать ли figure.
    Выход:
        Path к сохраненному файлу.
    """

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if close:
        ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        plt.close(fig)
    return save_path


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
