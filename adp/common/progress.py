from __future__ import annotations

from typing import Any

import numpy as np


def format_float(
    value: float,  # Число для вывода.
) -> str:
    """Форматирует число для компактного tqdm postfix.

    Вход:
        value: исходное число.
    Выход:
        Строка для отображения в progress bar.
    """

    number = float(value)
    if number == 0:
        return "0"
    if not np.isfinite(number):
        return str(number)
    if abs(number) >= 1e4 or abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.4g}"


def format_progress_postfix(
    record: dict[str, Any],  # Сырой словарь прогресса.
) -> dict[str, Any]:
    """Готовит словарь postfix для tqdm.

    Вход:
        record: диагностический словарь одного outer-шага.
    Выход:
        Словарь с короткими строками для tqdm.set_postfix(...).
    """

    postfix: dict[str, Any] = {
        "variant": record["variant"],
        "backend": record["backend"],
        "outer": f"{record['outer']}/{record['outer_total']}",
        "inner": record["inner"],
        "h": format_float(record["h"]),
        "weights": format_float(record["weights"]),
        "objective": format_float(record["objective"]),
        "delta": format_float(record["delta"]),
        "elapsed": f"{record['elapsed']:.1f}s",
    }
    if "rho" in record:
        postfix["rho"] = format_float(record["rho"])
    if "directions" in record:
        postfix["dirs"] = record["directions"]
    return postfix
