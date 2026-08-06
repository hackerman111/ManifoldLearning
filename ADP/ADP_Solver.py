from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ADP_SolverResult:
    index: np.ndarray
    coefficients: np.ndarray | None
    diagnostics: dict[str, object]


class ADP_solver:
    def __init__(self, method: Callable, **settings):
        if not callable(method):
            raise TypeError("method must be callable")
        self.method = method
        self.settings = dict(settings)

    def fit(self, statistics, initial_index, **problem_params) -> ADP_SolverResult:
        overlap = self.settings.keys() & problem_params.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"duplicate solver settings: {names}")

        result = self.method(
            statistics,
            initial_index,
            **problem_params,
            **self.settings,
        )
        if not isinstance(result, ADP_SolverResult):
            raise TypeError("solver method must return ADP_SolverResult")

        index = np.asarray(result.index, dtype=float)
        if not np.all(np.isfinite(index)):
            raise RuntimeError("solver returned a non-finite index")
        result.index = index
        return result
