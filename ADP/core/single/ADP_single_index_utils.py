from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...engine.common import utils as engine_utils
from ..ADP_Solver import ADP_solver

if TYPE_CHECKING:
    from .ADP_single_index_result import ADP_single_index_result


def require_gpu_solver(solver: ADP_solver, builtin_solver: object) -> None:
    if solver.method is not builtin_solver:
        raise ValueError("GPU mode requires the built-in LSMR solver")


def require_supported_initialization(index_init: str) -> None:
    if index_init == "pilot":
        raise ValueError("pilot initialization is multi-index only")


def validate_model_sizes(
    n: int,
    d: int,
    n_loc: int,
    n_lin: int,
    n_centers: int,
    index_init: str,
) -> None:
    engine_utils._check_model_sizes(n, d, n_loc, n_lin, n_centers, index_init)


def require_outer_step(outer_steps: int | None, iteration: int) -> None:
    if outer_steps is not None and iteration >= outer_steps:
        raise RuntimeError("outer_steps exhausted before reaching h_min")


def validate_solver_index(index: np.ndarray, dimension: int) -> np.ndarray:
    value = np.asarray(index, dtype=float)
    if value.shape != (dimension,):
        raise RuntimeError("single-index solver must return shape (d,)")
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("single-index solver returned an invalid index")
    return value / norm


def require_fitted(model: object) -> None:
    if not hasattr(model, "beta_"):
        raise RuntimeError("model is not fitted")


def require_result_vectors(
    result: ADP_single_index_result,
) -> tuple[np.ndarray, np.ndarray]:
    if result.beta_true is None or result.beta_final is None:
        raise RuntimeError("beta_true and beta_final are required")
    return result.beta_true, result.beta_final
