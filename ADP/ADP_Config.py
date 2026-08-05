from collections.abc import Callable

import numpy as np
from pydantic.dataclasses import dataclass


def epanechnikov(X):
    return np.max(0, 1 - X**2)


@dataclass(frozen=True, slots=True)
class ADP_Config:
    seed: int = 42
    N_loc: int = 10
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
    outer_steps: int = 8
    lamda_penalty: float = 1.0
    local_ridge: float = 1e-8
    kernel: Callable = epanechnikov
    a: float = np.sqrt(2)
    h_min: int | None = None
    batch_size: int = 32
    index_init: str = "local"
