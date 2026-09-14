from collections.abc import Callable

import numpy as np

from ADP.core.ADP_Config import epanechnikov
from ADP.engine.utils import _prepare_basis_eigenvalues, _prepare_multi_localization
from ADP.engine.weights import _multi_components, _projected_multi_components

from . import utils


def pairwise_distance2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X, centers = utils._prepare_pairwise(X, centers)

    # NUMERICAL: общий сдвиг сохраняет расстояния и убирает большой offset
    # до Gram identity; рабочая память O(nd+Jd), без (J,n,d).
    origin = X[0]
    X = X - origin
    centers = centers - origin

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
    if kernel is epanechnikov and target > distance2.shape[1]:
        raise ValueError("target mass cannot exceed n for the Epanechnikov kernel")
    active = distance2.ravel()

    def enough(h: float) -> bool:
        nonlocal active
        if kernel is epanechnikov:
            argument = active / h**2
            feasible = float(epanechnikov(argument).sum()) / len(distance2) >= target
            if feasible:
                # EXACT: при уменьшении h нулевые веса не оживут.
                support = argument < 1.0
                del argument
                # Копию support держим лишь после двукратного сокращения:
                # копия плюс два рабочих вектора дешевле dense scan.
                if np.count_nonzero(support) <= distance2.size // 2:
                    active = active[support]
            return feasible
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
    estimator: str = "legacy",
) -> float | None:
    X, centers, beta, distance2 = utils._prepare_rho(X, centers, beta, h_k, distance2)
    if estimator not in {"new", "legacy"}:
        raise ValueError("estimator must be 'new' or 'legacy'")
    if not np.isfinite(N_loc) or N_loc <= 0:
        raise ValueError("N_loc must be finite and positive")

    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    assert distance2 is not None
    projected = (centers @ beta)[:, None] - (X @ beta)[None, :]
    inverse_h2 = 1.0 / h_k**2
    projection2 = np.square(projected) * inverse_h2
    scaled_distance2 = distance2 * inverse_h2
    orthogonal2 = scaled_distance2 - projection2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)

    def enough(rho: float) -> bool:
        if estimator == "new":
            argument = projection2 + rho**2 * orthogonal2
        else:
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
    return calculate_rho_k(
        data.X,
        data.x_j,
        beta,
        h_k,
        config.N_loc,
        config.kernel,
        estimator=config.estimator,
    )


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


def generate_isotropic_proj(
    rng: np.random.Generator,
    n_centers: int,
    n_directions: int,
    n_features: int,
) -> np.ndarray:
    """Сгенерировать независимые равномерные направления на сфере."""
    if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)):
        raise TypeError("n_features must be an integer")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return generate_proj(
        rng,
        n_centers,
        n_directions,
        np.zeros(n_features),
        1.0,
    )


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
    tensor: str = "orthogonal",
    block_size: int = 32,
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
    if tensor not in {"orthogonal", "full"}:
        raise ValueError("tensor must be 'orthogonal' or 'full'")
    utils._check_batch_size(block_size)
    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    else:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    inverse_h2 = 1.0 / h_k**2
    if kernel is epanechnikov:
        origin = X[0]
        projected_X = (X - origin) @ basis
        projected_centers = (centers - origin) @ basis
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(centers), block_size):
            stop = start + block_size
            orthogonal2, principal2 = _projected_multi_components(
                projected_X,
                projected_centers[start:stop],
                eigenvalues,
                distance2[start:stop],
            )
            residual2 = orthogonal2 if tensor == "orthogonal" else distance2[start:stop]
            active = principal2 * inverse_h2 < 1.0
            blocks.append((residual2[active], principal2[active]))
            # Полные компоненты блока больше не нужны в бисекции.
            del orthogonal2, principal2, residual2, active
        return _compact_alpha(blocks, inverse_h2, N_loc * len(centers))

    orthogonal2, principal2 = _multi_components(
        X,
        centers,
        basis,
        eigenvalues,
        distance2,
    )

    def enough(alpha: float) -> bool:
        residual2 = orthogonal2 if tensor == "orthogonal" else distance2
        argument = (alpha**2 * residual2 + principal2) * inverse_h2
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


def _compact_alpha(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    inverse_h2: float,
    target: float,
) -> float | None:
    """EXACT: та же бисекция alpha, только по ещё возможному support.

    K(t)=max(1-t²,0), t=(alpha²*r+p)/h². После принятия low
    пары при t(low)>=1 нулевые во всём оставшемся интервале [low,high].
    target обозначает полную массу J*N_loc, не n_eff.
    Память O(E_0+B*n), E_0 — support при alpha=0.
    """

    def enough(alpha: float, *, prune: bool = False) -> bool:
        mass = 0.0
        for residual, principal in blocks:
            argument = (alpha**2 * residual + principal) * inverse_h2
            mass += float(epanechnikov(argument).sum())
        feasible = mass >= target
        if feasible and prune:
            for j, (residual, principal) in enumerate(blocks):
                support = (alpha**2 * residual + principal) * inverse_h2 < 1.0
                blocks[j] = residual[support], principal[support]
        return feasible

    if enough(1.0):
        return 1.0
    if not enough(0.0):
        return None
    low, high = 0.0, 1.0
    tolerance = np.sqrt(np.finfo(float).eps)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if enough(middle, prune=True):
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
