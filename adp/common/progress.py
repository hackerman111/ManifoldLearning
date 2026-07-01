from __future__ import annotations

from typing import Any

import numpy as np


def format_float(value: float) -> str:
    """Форматирует число для компактного вывода прогресса.

    Вход:
        value: исходное число.
    Выход:
        Строка с обычной или scientific notation записью.
    """

    number = float(value)
    if number == 0:
        return "0"
    if not np.isfinite(number):
        return str(number)
    if abs(number) >= 1e4 or abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.4g}"


def format_progress_postfix(record: dict[str, Any]) -> dict[str, Any]:
    """Готовит значения postfix для tqdm.

    Вход:
        record: сырой словарь прогресса после outer-шага.
    Выход:
        Словарь компактных строк и чисел для отображения.
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
    if "b" in record:
        postfix["b"] = format_float(record["b"])
    if "directions" in record:
        postfix["dirs"] = record["directions"]
    return postfix
