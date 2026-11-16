from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


def require_cupy() -> Any:
    """Return CuPy when a CUDA device is usable; never fall back to NumPy."""
    try:
        cp = import_module("cupy")

        count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(
            f"GPU execution requires working CuPy/CUDA: {error}"
        ) from error
    if count < 1:
        raise RuntimeError("GPU execution requires an available CUDA device")
    return cp


def array_module(value: Any) -> Any:
    """Выбрать backend без импорта CUDA в CPU-пути."""
    if hasattr(value, "__cuda_array_interface__"):
        cp = import_module("cupy")

        return cp
    return np


def to_numpy(value: Any) -> np.ndarray:
    """Вернуть host-массив; NumPy-вход не копируется."""
    xp = array_module(value)
    return value if xp is np else xp.asnumpy(value)
