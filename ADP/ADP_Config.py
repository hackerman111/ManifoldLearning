from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import numpy as np


def epanechnikov(value: np.ndarray) -> np.ndarray:
    result = np.square(value, dtype=float)
    np.subtract(1.0, result, out=result)
    np.maximum(result, 0.0, out=result)
    return result


@dataclass(frozen=True, slots=True)
class ADP_Config:
    seed: int = 42
    N_loc: int = 10
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
    outer_steps: int | None = None
    lambda_penalty: float = 1.0
    local_ridge: float = 1e-8
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov
    a: float = sqrt(2)
    h_min: float | None = None
    batch_size: int = 32
    index_init: str = "local"
    smart_weights: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "N_loc",
            "N_lin",
            "N_J",
            "N_phi",
            "outer_steps",
            "batch_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1):
                requirement = "nonnegative" if name == "seed" else "positive"
                raise ValueError(f"{name} must be {requirement}")

        for name in ("lambda_penalty", "local_ridge"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        if not np.isfinite(self.a) or self.a <= 1:
            raise ValueError("a must be finite and exceed one")
        if self.h_min is not None and (
            not np.isfinite(self.h_min) or self.h_min <= 0
        ):
            raise ValueError("h_min must be finite and positive")
        if not callable(self.kernel):
            raise TypeError("kernel must be callable")
        if self.index_init not in {"local", "pilot", "random"}:
            raise ValueError("index_init must be 'local', 'pilot', or 'random'")
        if not isinstance(self.smart_weights, (bool, np.bool_)):
            raise TypeError("smart_weights must be boolean")
        if self.smart_weights and self.kernel is not epanechnikov:
            raise ValueError("smart_weights requires the epanechnikov kernel")
