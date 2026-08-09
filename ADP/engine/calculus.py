from collections.abc import Callable, Iterator

import numpy as np

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


def initialize_beta_local(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
) -> np.ndarray:
    return initialize_basis_local(
        X,
        Y,
        centers,
        distance2,
        N_lin,
        kernel,
        local_ridge,
        1,
    )[:, 0]


def initialize_basis_local(
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    N_lin: int,
    kernel: Callable,
    local_ridge: float,
    index_dim: int,
) -> np.ndarray:
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if not 1 <= index_dim <= X.shape[1]:
        raise ValueError("index_dim must lie between 1 and d")

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
    threshold = (
        np.finfo(float).eps
        * max(gradients.shape)
        * (singular_values[0] if len(singular_values) else 0.0)
    )
    if len(singular_values) < index_dim or singular_values[index_dim - 1] <= threshold:
        raise RuntimeError("local gradients do not identify the requested index")

    return _orient_basis(right_vectors[:index_dim].T.copy())


def initialize_basis_random(
    rng: np.random.Generator,
    n_features: int,
    index_dim: int,
) -> np.ndarray:
    for name, value in (("n_features", n_features), ("index_dim", index_dim)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= index_dim <= n_features:
        raise ValueError("index_dim must lie between 1 and n_features")

    basis, _ = np.linalg.qr(
        rng.standard_normal((n_features, index_dim)),
        mode="reduced",
    )
    return _orient_basis(basis)


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


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
    X, centers, beta, distance2 = utils._prepare_rho(
        X, centers, beta, h_k, distance2
    )

    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
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


def calculate_weight(
    X: np.ndarray,
    centers: np.ndarray,
    beta: np.ndarray,
    h: float,
    rho: float,
    kernel: Callable,
    block_size: int = 128,
    *,
    distance2: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    X, centers, beta = utils._prepare_weight_data(
        X, centers, beta, h, rho, kernel, block_size
    )

    if distance2 is not None:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    x_norm2 = None if distance2 is not None else np.einsum("nd,nd->n", X, X)
    x_proj = X @ beta
    center_proj = centers @ beta
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        if distance2 is None:
            c_norm2 = np.einsum("bd,bd->b", C, C)
            distance2_block = (
                c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
            )
            np.maximum(distance2_block, 0.0, out=distance2_block)
        else:
            distance2_block = distance2[start : start + block_size]

        projection_diff = (
            center_proj[start : start + block_size, None] - x_proj[None, :]
        )
        argument = (rho**2 * distance2_block + projection_diff**2) / h**2
        yield start, kernel(argument)


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
        rng.standard_normal((n_centers, n_directions, m))
        * np.sqrt(eigenvalues)
    ) @ basis.T
    values = alpha * orthogonal + principal
    norms = np.linalg.norm(values, axis=2, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("generated a zero direction")
    return values / norms


def calculate_multi_weight(
    X: np.ndarray,
    centers: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    h: float,
    alpha: float,
    kernel: Callable,
    block_size: int = 128,
    *,
    distance2: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    X, centers, basis, eigenvalues = _prepare_multi_localization(
        X,
        centers,
        basis,
        eigenvalues,
        h,
        alpha,
        kernel,
    )
    utils._check_batch_size(block_size)
    if distance2 is not None:
        distance2 = utils._prepare_distance2(distance2, centers, len(X))

    x_norm2 = None if distance2 is not None else np.einsum("nd,nd->n", X, X)
    for start in range(0, len(centers), block_size):
        C = centers[start : start + block_size]
        if distance2 is None:
            c_norm2 = np.einsum("bd,bd->b", C, C)
            distance2_block = (
                c_norm2[:, None] + x_norm2[None, :] - 2.0 * C @ X.T
            )
            np.maximum(distance2_block, 0.0, out=distance2_block)
        else:
            distance2_block = distance2[start : start + block_size]

        orthogonal2, principal2 = _multi_components(
            X,
            C,
            basis,
            eigenvalues,
            distance2_block,
        )
        argument = (alpha**2 * orthogonal2 + principal2) / h**2
        yield start, kernel(argument)


def _multi_components(X, centers, basis, eigenvalues, distance2):
    orthogonal2 = distance2.copy()
    principal2 = np.zeros_like(distance2)
    for vector, eigenvalue in zip(basis.T, eigenvalues):
        difference = (centers @ vector)[:, None] - (X @ vector)[None, :]
        component2 = np.square(difference)
        orthogonal2 -= component2
        principal2 += eigenvalue * component2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)
    return orthogonal2, principal2


def _prepare_multi_localization(
    X,
    centers,
    basis,
    eigenvalues,
    h,
    alpha,
    kernel,
):
    X, centers = utils._prepare_pairwise(X, centers)
    basis, eigenvalues = _prepare_basis_eigenvalues(
        basis,
        eigenvalues,
        n_features=X.shape[1],
    )
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    return X, centers, basis, eigenvalues


def _prepare_basis_eigenvalues(basis, eigenvalues, *, n_features=None):
    basis = utils._finite_real_array(basis, "basis")
    eigenvalues = utils._finite_real_array(eigenvalues, "eigenvalues")
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError("basis must have non-empty shape (d, m)")
    if n_features is not None and basis.shape[0] != n_features:
        raise ValueError("basis must have shape (d, m)")
    if eigenvalues.shape != (basis.shape[1],):
        raise ValueError("eigenvalues must have shape (m,)")
    if np.any(eigenvalues < 0):
        raise ValueError("eigenvalues must be nonnegative")
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[1]),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("basis columns must be orthonormal")
    return basis, eigenvalues
