from __future__ import annotations

import math
from typing import Literal

import numpy as np


def validate_solver(value: str) -> Literal["cg", "hybrid"]:
    if value not in {"cg", "hybrid"}:
        raise ValueError("solver must be 'cg' or 'hybrid'")
    return value  # type: ignore[return-value]


def validate_scale_boundary(value: str) -> Literal["raise", "stop"]:
    if value not in {"raise", "stop"}:
        raise ValueError("scale_boundary must be 'raise' or 'stop'")
    return value  # type: ignore[return-value]


def require_feasible_bracket(previous_feasible: bool) -> None:
    if not previous_feasible:
        raise RuntimeError(
            "updated geometry has no feasible scale in the current bracket"
        )


def prepare_queries(
    value: np.ndarray,
    centers: np.ndarray | None,
) -> np.ndarray:
    if centers is None:
        raise RuntimeError("fit must be called before transform or predict")
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise TypeError("X must contain real numeric values")
    if array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("X has a dtype wider than float64; cast it explicitly")
    if array.ndim != 2 or array.shape[1] != centers.shape[1]:
        raise ValueError(f"X must have shape (n, {centers.shape[1]})")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("X must contain only finite values")
    return result


def effective_config(
    X: np.ndarray,
    index_dim: int,
    N_loc: int,
    N_lin: int | None,
    N_J: int | None,
    N_phi: int | None,
    N_manifold: int | None,
    a: float | None,
    h_min: float | None,
) -> dict[str, float | int]:
    n, d = X.shape
    m = index_dim
    if n <= d + 1:
        raise ValueError("X must satisfy n > d + 1")
    if m > d:
        raise ValueError("index_dim must not exceed the number of features")
    if N_loc >= n:
        raise ValueError("N_loc must be smaller than n")

    resolved_lin = N_lin if N_lin is not None else max(2 * d, d + 2)
    resolved_J = N_J if N_J is not None else min(n, math.ceil(2 * n / N_loc))
    resolved_phi = N_phi if N_phi is not None else N_loc
    resolved_manifold = (
        N_manifold if N_manifold is not None else max(m + 1, math.ceil(resolved_J / 10))
    )
    resolved_a = a if a is not None else 2.0 ** (1.0 / m)
    resolved_h_min = (
        h_min
        if h_min is not None
        else 3.0 * float(np.std(X, axis=0).mean()) / math.sqrt(n)
    )

    if not d + 1 < resolved_lin < n:
        raise ValueError("N_lin must satisfy d + 1 < N_lin < n")
    if not m + 1 <= resolved_J <= n:
        raise ValueError("N_J must satisfy index_dim + 1 <= N_J <= n")
    if resolved_phi < m:
        raise ValueError("N_phi must be at least index_dim")
    if not m < resolved_manifold < resolved_J:
        raise ValueError("N_manifold must satisfy index_dim < N_manifold < N_J")
    if not np.isfinite(resolved_h_min) or resolved_h_min <= 0:
        raise ValueError("h_min must be finite and positive")
    return {
        "N_lin": resolved_lin,
        "N_J": resolved_J,
        "N_phi": resolved_phi,
        "N_manifold": resolved_manifold,
        "a": resolved_a,
        "h_min": resolved_h_min,
    }


def prepare_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_array = np.asarray(X)
    Y_array = np.asarray(Y)
    for name, array in (("X", X_array), ("Y", Y_array)):
        if np.issubdtype(array.dtype, np.complexfloating) or not np.issubdtype(
            array.dtype, np.number
        ):
            raise TypeError(f"{name} must contain real numeric values")
        if array.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError(
                f"{name} has a dtype wider than float64; cast it explicitly"
            )
    if X_array.ndim != 2:
        raise ValueError("X must have shape (n, d)")
    if Y_array.shape != (len(X_array),):
        raise ValueError("Y must have shape (n,)")
    X_float = np.asarray(X_array, dtype=np.float64)
    Y_float = np.asarray(Y_array, dtype=np.float64)
    if not np.all(np.isfinite(X_float)) or not np.all(np.isfinite(Y_float)):
        raise ValueError("X and Y must contain only finite values")
    return X_float, Y_float


def require_bandwidth_target(target: int, n: int) -> None:
    if not 0 < target < n:
        raise ValueError("bandwidth target must lie strictly between 0 and n")


def raise_bandwidth_bracket_failure() -> None:
    raise RuntimeError("could not bracket a feasible bandwidth")


def raise_infeasible_mass(kind: str) -> None:
    raise RuntimeError(f"{kind} mass target is infeasible even with alpha=0")


def require_positive_mass(message: str, mass: float) -> None:
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError(message)


def require_finite_directions(norms: np.ndarray) -> None:
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("failed to generate finite nonzero directions")


def require_statistics(mass: np.ndarray, I: np.ndarray, U: np.ndarray) -> None:
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise RuntimeError("function weights contain an empty neighborhood")
    if not np.all(np.isfinite(I)) or not np.all(np.isfinite(U)):
        raise RuntimeError("ADP statistics contain non-finite values")


def require_mass(mass: np.ndarray) -> None:
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise RuntimeError("function weights contain an empty neighborhood")


def require_graph_row(indices: np.ndarray) -> None:
    if len(indices) == 0:
        raise RuntimeError("manifold graph contains an empty row")


def require_full_rank(rank: int, required: int, message: str) -> None:
    if rank != required:
        raise RuntimeError(message)


def require_operator_diagonal(diagonal: np.ndarray) -> None:
    scale = max(float(diagonal.max()), 1.0)
    if not np.all(np.isfinite(diagonal)) or np.any(
        diagonal <= np.finfo(float).eps * scale
    ):
        raise RuntimeError("B normal operator is rank-deficient")


def require_cg_solution(
    info: int,
    iterations: int,
    relative_residual: float,
    solution: np.ndarray,
    limit: float,
) -> None:
    if info != 0:
        raise RuntimeError(
            "CG did not converge for B: "
            f"info={int(info)}, iterations={iterations}, "
            f"relative_residual={relative_residual:.3e}"
        )
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
        raise RuntimeError("CG returned a non-finite B solution")
    if relative_residual > limit:
        raise RuntimeError(
            "CG residual certificate failed for B: "
            f"{relative_residual:.3e} > {limit:.3e}"
        )


def require_nonnegative_objective(
    data_term: float, penalty: float, tolerance: float
) -> None:
    if not np.isfinite(data_term) or not np.isfinite(penalty):
        raise RuntimeError("manifold objective became non-finite")
    if penalty < -tolerance:
        raise RuntimeError("manifold penalty became negative")


def require_slope_matrix(
    values: np.ndarray,
    tolerance: float,
    target: int,
) -> None:
    if not np.all(np.isfinite(values)) or values[0] < -tolerance:
        raise RuntimeError(f"invalid local slope matrix at target {target}")


def require_projector(
    orthogonality: float,
    spectrum: np.ndarray,
    target: int,
) -> None:
    if not np.isfinite(orthogonality) or orthogonality > 1.0e-10:
        raise RuntimeError(
            f"non-orthonormal projector at target {target}: {orthogonality:.3e}"
        )
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
        raise RuntimeError(f"invalid local spectrum at target {target}")


def require_identified(
    singular_values: np.ndarray,
    shape: tuple[int, int],
    m: int,
    target: int,
) -> None:
    if len(singular_values) < m or not np.all(np.isfinite(singular_values)):
        raise RuntimeError(f"local EDR rank is smaller than {m} at target {target}")
    threshold = (
        np.finfo(float).eps
        * max(shape)
        * (float(singular_values[0]) if len(singular_values) else 0.0)
    )
    if singular_values[m - 1] <= threshold:
        raise RuntimeError(f"local EDR rank is smaller than {m} at target {target}")


def validate_integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def validate_float(
    name: str,
    value: float,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not np.isfinite(result) or invalid:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result
