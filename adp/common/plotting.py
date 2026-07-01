from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def save_figure(fig: Any, path: str | Path, *, dpi: int = 150, close: bool = False) -> Path:
    """Сохраняет matplotlib figure.

    Вход:
        fig: объект figure.
        path: путь сохранения.
        dpi: разрешение изображения.
        close: закрыть figure после сохранения.
    Выход:
        Path к сохранённому файлу.
    """

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    if close:
        ensure_matplotlib_config_dir()
        import matplotlib.pyplot as plt

        plt.close(fig)
    return save_path


def ensure_matplotlib_config_dir() -> None:
    """Готовит временный MPLCONFIGDIR, если он не задан.

    Вход:
        Нет явных аргументов.
    Выход:
        None; при необходимости меняет os.environ.
    """

    if "MPLCONFIGDIR" in os.environ:
        return
    config_dir = Path("/tmp") / "adp_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)
