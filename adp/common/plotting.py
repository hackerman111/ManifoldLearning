from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def save_figure(
    fig: Any,  # Matplotlib figure.
    path: str | Path,  # Путь сохранения.
    *,
    dpi: int = 150,  # Разрешение PNG.
    close: bool = False,  # Закрыть figure после сохранения.
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
    fig.savefig(save_path, dpi=dpi)
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
