"""Относительное демпфирование intrinsic GN; исходный полный профиль."""

from __future__ import annotations

from typing import Any

import numpy as np

from ADP.solver.LSMR import HPAOResult

from .backtracking import solve as _solve


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    max_steps: int = 10,
    **kwargs: Any,
) -> HPAOResult:
    return _solve(
        index_init,
        U,
        I,
        max_steps=max_steps,
        intrinsic_gn=True,
        normalized_damping=True,
        **kwargs,
    )
