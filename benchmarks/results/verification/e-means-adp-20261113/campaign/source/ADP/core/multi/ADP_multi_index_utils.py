from __future__ import annotations

import numpy as np

from ...engine.common import utils as engine_utils
from ..ADP_Solver import ADP_solver


def validate_index_dim(index_dim: int) -> int:
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if index_dim < 1:
        raise ValueError("index_dim must be positive")
    return int(index_dim)


def require_gpu_solver(solver: ADP_solver, builtin_solver: object) -> None:
    if solver.method is not builtin_solver:
        raise ValueError("GPU mode requires the built-in LSMR solver")


def validate_dimension(index_dim: int, dimension: int) -> None:
    if index_dim >= dimension:
        raise ValueError("index_dim must be smaller than d")


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


def validate_basis(
    value: np.ndarray,
    shape: tuple[int, int] | None = None,
    *,
    name: str = "solver index",
    require_rank: bool = False,
) -> np.ndarray:
    basis = np.asarray(value, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError(f"{name} must have non-empty shape (d, m)")
    if shape is not None and basis.shape != shape:
        raise RuntimeError(f"multi-index solver must return shape {shape}")
    if not np.all(np.isfinite(basis)):
        raise RuntimeError(f"{name} must contain only finite values")
    if require_rank:
        _, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
        threshold = np.finfo(float).eps * max(basis.shape) * singular_values[0]
        if singular_values[-1] <= threshold:
            raise ValueError(f"{name} must have full column rank")
    return basis


def require_full_rank(triangular: np.ndarray) -> None:
    if np.min(np.abs(np.diag(triangular))) <= np.finfo(float).eps:
        raise RuntimeError("multi-index solver returned a rank-deficient basis")


def validate_eigenvalues(value: np.ndarray, dimension: int) -> np.ndarray:
    eigenvalues = np.asarray(value, dtype=float)
    if (
        eigenvalues.shape != (dimension,)
        or not np.all(np.isfinite(eigenvalues))
        or np.any(eigenvalues < 0)
    ):
        raise RuntimeError("multi-index solver returned invalid eigenvalues")
    return eigenvalues


def require_fitted(model: object) -> None:
    if not hasattr(model, "basis_"):
        raise RuntimeError("model is not fitted")


def projectors_for_result(
    value: np.ndarray | None,
    name: str,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if value is None:
        raise RuntimeError("beta_true and beta_final are required")
    basis = validate_basis(value, expected_shape, name=name, require_rank=True)
    left, _, _ = np.linalg.svd(basis, full_matrices=False)
    return left @ left.T, basis.shape
