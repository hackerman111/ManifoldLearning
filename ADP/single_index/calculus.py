import numpy as np


class T_k:
    def __init__(self, h=0, rho_k=0, beta=np.zeros(1)):
        self.h = h
        self.rho_k = rho_k
        self.beta = beta

    def unpack(self):
        return self.h, self.rho_k, self.beta

    def Get_h(self):
        return self.h

    def Get_rho_k(self):
        return self.rho_k

    def Get_beta(self):
        return self.beta


def Calculate_h0(self) -> float:
    X_sq = np.sum(self.X**2, axis=0)
    centers_sq = np.sum(self.x_j**2, axis=0)
    distances_sq = centers_sq[:, None] + X_sq[None, :] - 2 * self.x_j.T @ self.X
    np.maximum(distances_sq, 0, out=distances_sq)

    target = self.N_loc * self.x_j.shape[1]

    def enough(h):
        return np.sum(self.kernel(distances_sq / h**2)) >= target

    low = float(self.h_min)
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


def Calculate_rho_k(self) -> None:
    if self.h_k is None or not np.isfinite(self.h_k) or self.h_k <= 0:
        raise ValueError("h_k must be finite and positive")

    beta = np.asarray(self.beta, dtype=float)
    beta_norm = np.linalg.norm(beta)
    if not np.isfinite(beta_norm) or beta_norm == 0:
        raise ValueError("beta must be finite and non-zero")
    beta = beta / beta_norm

    X_sq = np.sum(self.X**2, axis=0)
    centers_sq = np.sum(self.x_j**2, axis=0)
    distances_sq = centers_sq[:, None] + X_sq[None, :] - 2 * self.x_j.T @ self.X
    np.maximum(distances_sq, 0, out=distances_sq)
    projected_X = beta @ self.X
    projected_centers = beta @ self.x_j
    projection_sq = (projected_centers[:, None] - projected_X[None, :]) ** 2
    target = self.N_loc * self.x_j.shape[1]

    def enough(rho):
        argument = (rho**2 * distances_sq + projection_sq) / self.h_k**2
        return np.sum(self.kernel(argument)) >= target

    if enough(1.0):
        self.rho_k = 1.0
        return
    if not enough(0.0):
        raise ValueError("N_loc cannot be reached at rho_k=0")

    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if enough(middle):
            low = middle
        else:
            high = middle
    self.rho_k = low


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


def Calculate_weight(self, T_k: T_k, kernel) -> np.ndarray:

    transformed_X = self.T_k @ self.X
    transformed_centers = self.T_k @ self.x_j
    distances_sq = (
        np.sum(transformed_centers**2, axis=0)[:, None]
        + np.sum(transformed_X**2, axis=0)[None, :]
        - 2 * transformed_centers.T @ transformed_X
    )

    np.maximum(distances_sq, 0, out=distances_sq)
    return kernel(distances_sq)


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
