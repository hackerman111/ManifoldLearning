from collections.abc import Callable

import numpy as np

from ADP.engine.utils import _prepare_basis_eigenvalues, _prepare_multi_localization
from ADP.engine.weights import _multi_components

from . import utils


def pairwise_distance2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X, centers = utils._prepare_pairwise(X, centers)

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
    distance2 = utils._prepare_bandwidth(distance2, target, kernel, lower)

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


def calculate_h0_from_adp(config, data) -> float:
    return calculate_h0(data.X, data.x_j, config.N_loc, config.h_min, config.kernel)


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
    X, centers, beta, distance2 = utils._prepare_rho(X, centers, beta, h_k, distance2)

    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    assert distance2 is not None
    projected = (centers @ beta)[:, None] - (X @ beta)[None, :]
    inverse_h2 = 1.0 / h_k**2
    projection2 = np.square(projected) * inverse_h2
    scaled_distance2 = distance2 * inverse_h2

    def enough(rho: float) -> bool:
        argument = rho**2 * scaled_distance2 + projection2
        mass = np.sum(kernel(argument), axis=1)
        return bool(np.mean(mass) >= N_loc)

    if enough(1.0):
        return 1.0
    if not enough(0.0):
        return None

    low, high = 0.0, 1.0
    tolerance = np.sqrt(np.finfo(float).eps)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if enough(middle):
            low = middle
        else:
            high = middle
    return float(low)


def calculate_rho_k_from_adp(config, data, beta, h_k) -> float | None:
    return calculate_rho_k(data.X, data.x_j, beta, h_k, config.N_loc, config.kernel)


def generate_proj(
    rng: np.random.Generator,
    n_centers: int,
    n_directions: int,
    beta: np.ndarray,
    rho: float,
) -> np.ndarray:
    beta = utils._prepare_projection(beta, rho)

    z = rng.standard_normal((n_centers, n_directions, len(beta)))
    xi = rng.standard_normal((n_centers, n_directions, 1))
    values = rho * z + xi * beta
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("generated a zero direction")
    return values / norms


def generate_proj_from_adp(config, data, rng, beta, rho) -> np.ndarray:
    return generate_proj(rng, len(data.x_j), config.N_phi, beta, rho)


def calculate_alpha_k(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h_k: float,
    N_loc: int,
    kernel: Callable,
    *,
    distance2: np.ndarray | None = None,
) -> float | None:
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X,
        centers,
        basis,
        eigenvalues,
        h_k,
        1.0,
        kernel,
    )
    if not np.isfinite(N_loc) or N_loc <= 0:
        raise ValueError("N_loc must be finite and positive")
    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    else:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    orthogonal2, principal2 = _multi_components(
        X,
        centers,
        basis,
        eigenvalues,
        distance2,
    )
    inverse_h2 = 1.0 / h_k**2

    def enough(alpha: float) -> bool:
        argument = (alpha**2 * orthogonal2 + principal2) * inverse_h2
        return bool(np.mean(np.sum(kernel(argument), axis=1)) >= N_loc)

    if enough(1.0):
        return 1.0
    if not enough(0.0):
        return None

    low, high = 0.0, 1.0
    tolerance = np.sqrt(np.finfo(float).eps)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if enough(middle):
            low = middle
        else:
            high = middle
    return float(low)


def generate_multi_proj(
    rng: np.random.Generator,
    n_centers: int,
    n_directions: int,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    alpha: float,
) -> np.ndarray:
    basis, eigenvalues = _prepare_basis_eigenvalues(basis, eigenvalues)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")

    d, m = basis.shape
    z = rng.standard_normal((n_centers, n_directions, d))
    coordinates = z @ basis
    orthogonal = z - coordinates @ basis.T
    principal = (
        rng.standard_normal((n_centers, n_directions, m)) * np.sqrt(eigenvalues)
    ) @ basis.T
    values = alpha * orthogonal + principal
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("generated a zero direction")
    return values / norms


def generate_multi_proj_from_adp(
    config, data, rng, basis, eigenvalues, alpha
) -> np.ndarray:
    return generate_multi_proj(
        rng,
        len(data.x_j),
        config.N_phi,
        basis,
        eigenvalues,
        alpha,
    )
