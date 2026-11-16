from __future__ import annotations

from typing import Any

import numpy as np


def solve(index_init: np.ndarray, U: np.ndarray, I: np.ndarray, **settings: Any):
    """Single-index HPAO route; it reuses the current LSMR implementation.

    The recycling workspace in HYBRID is specific to a multi-index basis.
    """
    shape = getattr(index_init, "shape", None)
    if shape is None:
        shape = np.shape(index_init)
    if len(shape) != 1:
        raise ValueError("single-index solver expects an index with shape (d,)")
    from ..LSMR import solve as hpao

    settings["linear_solver"] = "lsmr"
    return hpao(index_init, U, I, **settings)
