from collections.abc import Callable, Iterator

import numpy as np

from . import utils


def pairwise_distance2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X = utils._finite_real_array(X, "X")
    centers = utils._finite_real_array(centers, "centers")
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if centers.ndim != 2 or centers.shape[1] != X.shape[1] or not len(centers):
        raise ValueError("centers must have non-empty shape (J, d)")

    distance2 = (
        np.square(centers).sum(axis=1)[:, None]
        + np.square(X).sum(axis=1)[None, :]
        - 2.0 * centers @ X.T
    )
    np.maximum(distance2, 0.0, out=distance2)
    return distance2


def search_bandwidth(
    distance2: np.ndarray,
    target: float,
    kernel: Callable,
    *,
    lower: float,
) -> float:
    distance2 = utils._finite_real_array(distance2, "distance2")
    if distance2.ndim != 2 or 0 in distance2.shape:
        raise ValueError("distance2 must have non-empty shape (J, n)")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError("lower must be finite and positive")
    if not callable(kernel):
        raise TypeError("kernel must be callable")

    def enough(h: float) -> bool:
        mass = np.sum(kernel(distance2 / h**2), axis=1)
        return bool(np.mean(mass) >= target)

    low = float(lower)
    if enough(low):
        return low

    high = max(2.0 * low, float(np.sqrt(np.max(distance2))), 1.0)
    for _ in range(100):
        if enough(high):
            break
        high *= 2.0
    else:
        raise RuntimeError("could not bracket a feasible bandwidth")

    for _ in range(60):
        middle = (low + high) / 2.0
        if enough(middle):
            high = middle
        else:
            low = middle
    return float(high)


def calculate_h0(
    X: np.ndarray,
    centers: np.ndarray,
    N_loc: int,
    h_min: float,
    kernel: Callable,
) -> float:
    return search_bandwidth(
        pairwise_distance2(X, centers),
        N_loc,
        kernel,
        lower=h_min,
    )


def initialize_beta_local(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
) -> np.ndarray:
    h_lin = search_bandwidth(
        distance2,
        N_lin,
        kernel,
        lower=np.finfo(float).eps,
    )
    weights = kernel(distance2 / h_lin**2)
    n, d = X.shape
    ridge_rows = np.zeros((d, d + 1))
    ridge_rows[:, 1:] = np.sqrt(local_ridge) * np.eye(d)
    gradients = np.empty((len(centers), d))

    for j, center in enumerate(centers):
        design = np.column_stack((np.ones(n), X - center))
        root_weight = np.sqrt(weights[j])
        augmented_design = np.vstack((design * root_weight[:, None], ridge_rows))
        augmented_Y = np.concatenate((Y * root_weight, np.zeros(d)))
        gradients[j] = np.linalg.lstsq(
            augmented_design,
            augmented_Y,
            rcond=None,
        )[
            0
        ][1:]

    _, singular_values, right_vectors = np.linalg.svd(
        gradients,
        full_matrices=False,
    )
    if singular_values[0] <= np.finfo(float).eps:
        raise RuntimeError("local gradients do not identify beta")

    beta = right_vectors[0]
    beta /= np.linalg.norm(beta)
    if beta[np.argmax(np.abs(beta))] < 0:
        beta = -beta
    return beta


def calculate_rho_k(
    X: np.ndarray,
    centers: np.ndarray,
    beta: np.ndarray,
    h_k: float,
    N_loc: int,
    kernel: Callable,
    *,
    distance2: np.ndarray | None = None,
) -> float | None:
    if not np.isfinite(h_k) or h_k <= 0:
        raise ValueError("h_k must be finite and positive")

    beta = utils._finite_real_array(beta, "beta")
    if beta.shape != (X.shape[1],):
        raise ValueError("beta must have shape (d,)")
    beta_norm = np.linalg.norm(beta)
    if beta_norm == 0:
        raise ValueError("beta must be non-zero")
    beta = beta / beta_norm

    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    projected = (centers @ beta)[:, None] - (X @ beta)[None, :]
    projection2 = np.square(projected)

    def enough(rho: float) -> bool:
        argument = (rho**2 * distance2 + projection2) / h_k**2
        mass = np.sum(kernel(argument), axis=1)
        return bool(np.mean(mass) >= N_loc)

    if enough(1.0):
        return 1.0
    if not enough(0.0):
        return None

    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if enough(middle):
            low = middle
        else:
            high = middle
    return float(low)


def generate_proj(
    rng: np.random.Generator,
    n_centers: int,
    n_directions: int,
    beta: np.ndarray,
    rho: float,
) -> np.ndarray:
    beta = utils._finite_real_array(beta, "beta")
    if beta.ndim != 1 or not len(beta):
        raise ValueError("beta must have shape (d,)")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")

    beta_norm = np.linalg.norm(beta)
    if beta_norm:
        beta = beta / beta_norm
    elif rho == 0:
        raise ValueError("rho and beta cannot both be zero")

    z = rng.standard_normal((n_centers, n_directions, len(beta)))
    xi = rng.standard_normal((n_centers, n_directions, 1))
    values = rho * z + xi * beta
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("generated a zero direction")
    return values / norms


def calculate_weight(
    X: np.ndarray,
    centers: np.ndarray,
    beta: np.ndarray,
    h: float,
    rho: float,
    kernel: Callable,
    block_size: int = 128,
) -> Iterator[tuple[int, np.ndarray]]:
    X = utils._finite_real_array(X, "X")
    centers = utils._finite_real_array(centers, "centers")
    beta = utils._finite_real_array(beta, "beta")
    utils.check_weight_block(X, centers, beta, h, rho, kernel, block_size)

    x_norm2 = np.einsum("nd,nd->n", X, X)
    x_proj = X @ beta
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        c_norm2 = np.einsum("bd,bd->b", C, C)
        distance2 = c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
        np.maximum(distance2, 0.0, out=distance2)

        projection_diff = (C @ beta)[:, None] - x_proj[None, :]
        argument = (rho**2 * distance2 + projection_diff**2) / h**2
        yield start, kernel(argument)
