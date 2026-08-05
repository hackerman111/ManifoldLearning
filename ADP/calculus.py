from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ADP import utils


@dataclass(slots=True)
class T_k:
    h: float
    rho_k: float
    beta: np.ndarray

    def unpack(self):
        return self.h, self.rho_k, self.beta

    def Get_h(self):
        return self.h

    def Get_rho_k(self):
        return self.rho_k

    def Get_beta(self):
        return self.beta


def Calculate_h0(
    X: np.ndarray, x_j: np.ndarray, N_loc: int, h_min: int, kernel: Callable
) -> float:
    X_sq = np.sum(X**2, axis=0)
    centers_sq = np.sum(x_j**2, axis=0)
    distances_sq = centers_sq[:, None] + X_sq[None, :] - 2 * x_j.T @ X
    np.maximum(distances_sq, 0, out=distances_sq)

    target = N_loc * x_j.shape[1]

    def enough(h):
        return np.sum(kernel(distances_sq / h**2)) >= target

    low = float(h_min)
    if low <= 0:
        raise ValueError("h_min must be positive")
    if enough(low):
        return low

    high = max(2 * low, np.sqrt(distances_sq.max()))
    for _ in range(100):
        if enough(high):
            break
        high *= 2
    else:
        raise ValueError("N_loc cannot be reached with the configured kernel")

    for _ in range(60):
        middle = (low + high) / 2
        if enough(middle):
            high = middle
        else:
            low = middle
    return high


def Calculate_rho_k(
    X: np.ndarray,
    x_j: np.ndarray,
    beta: np.ndarray,
    h_k: float,
    N_loc: int,
    kernel: Callable,
) -> float:
    if h_k is None or not np.isfinite(h_k) or h_k <= 0:
        raise ValueError("h_k must be finite and positive")

    beta = np.asarray(beta, dtype=float)
    beta_norm = np.linalg.norm(beta)
    if not np.isfinite(beta_norm) or beta_norm == 0:
        raise ValueError("beta must be finite and non-zero")
    beta = beta / beta_norm

    X_sq = np.sum(X**2, axis=0)
    centers_sq = np.sum(x_j**2, axis=0)
    distances_sq = centers_sq[:, None] + X_sq[None, :] - 2 * x_j.T @ X
    np.maximum(distances_sq, 0, out=distances_sq)
    projected_X = beta @ X
    projected_centers = beta @ x_j
    projection_sq = (projected_centers[:, None] - projected_X[None, :]) ** 2
    target = N_loc * x_j.shape[1]

    def enough(rho):
        argument = (rho**2 * distances_sq + projection_sq) / h_k**2
        return np.sum(kernel(argument)) >= target

    if enough(1.0):
        rho_k = 1.0
        return rho_k
    if not enough(0.0):
        raise ValueError("N_loc cannot be reached at rho_k=0")

    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if enough(middle):
            low = middle
        else:
            high = middle
    rho_k = low

    return rho_k


def Calculate_Tk_x(X: np.ndarray, center: np.ndarray, T_k: T_k) -> np.ndarray:
    """
    X:      (n, d)
    center: (d,)
    beta:   (d,), единичный вектор
    """
    h, rho, beta = T_k.unpack()

    delta = X - center
    ordinary_sq_norm = np.sum(delta * delta, axis=1)
    parallel_component = delta @ beta

    return (rho**2 * ordinary_sq_norm + parallel_component**2) / h**2


def Calculate_h_k(h_prev: float, a: float):
    return h_prev / a


def Generate_proj(rng: np.random.Generator, n_directions: int, T_k: T_k) -> np.ndarray:

    beta = T_k.Get_beta()
    rho = T_k.Get_rho_k()

    d = beta.size

    z = rng.standard_normal((n_directions, d))
    xi = rng.standard_normal((n_directions, 1))

    g = rho * z + xi * beta[None, :]
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def Calculate_weight(X, centers, Tk, kernel, block_size=128):
    X = utils._finite_real_array(X, "X")
    centers = utils._finite_real_array(centers, "centers")
    h, rho, beta = Tk.unpack
    beta = utils._finite_real_array(beta, "beta")

    utils.check_weight_block(X, centers, beta, h, rho, kernel, block_size)

    x_norm2 = np.einsum("nd,nd->n", X, X)
    x_proj = X @ beta
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        c_norm2 = np.einsum("bd,bd->b", C, C)
        D2 = c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
        np.maximum(D2, 0.0, out=D2)

        projection_diff = (C @ beta)[:, None] - x_proj[None, :]
        Q = rho**2 * D2
        Q += projection_diff**2
        Q /= h**2
        W = kernel(Q)
        del D2, projection_diff, Q
        yield start, W
