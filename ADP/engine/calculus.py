from collections.abc import Callable, Iterator

import numpy as np

from ..ADP_Config import epanechnikov
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


def select_optimal_alpha(
    orthogonal2: np.ndarray,
    principal2: np.ndarray,
    h: float,
    target_mass: float,
) -> float | None:
    """Largest alpha in [0, 1] meeting Epanechnikov total mass."""
    orthogonal2 = utils._finite_real_array(orthogonal2, "orthogonal2")
    principal2 = utils._finite_real_array(principal2, "principal2")
    if orthogonal2.shape != principal2.shape or orthogonal2.ndim != 2:
        raise ValueError("orthogonal2 and principal2 must have equal shape (J, n)")
    if np.any(orthogonal2 < 0) or np.any(principal2 < 0):
        raise ValueError("localization components must be nonnegative")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")

    h2 = h**2
    h4 = h2**2

    def mass(z: float) -> float:
        argument = (orthogonal2 * z + principal2) / h2
        return float(epanechnikov(argument).sum())

    if mass(1.0) >= target_mass:
        return 1.0
    if mass(0.0) < target_mass:
        return None

    active = principal2 < h2
    a = orthogonal2[active]
    b = principal2[active]
    del active
    tau = np.divide(
        h2 - b,
        a,
        out=np.full_like(a, np.inf),
        where=a > 0,
    )
    order = np.argsort(tau)
    tau = tau[order]
    a = a[order]
    b = b[order]
    del order
    count = len(a)
    suffix_ab = a * b
    np.square(a, out=a)
    np.square(b, out=b)
    np.cumsum(a[::-1], out=a[::-1])
    np.cumsum(suffix_ab[::-1], out=suffix_ab[::-1])
    np.cumsum(b[::-1], out=b[::-1])
    suffix_a2 = a
    suffix_b2 = b
    stop = int(np.searchsorted(tau, 1.0, side="right"))
    mass_tolerance = 64.0 * np.finfo(float).eps * max(target_mass, 1.0)
    low, high = 0, stop
    while low < high:
        middle = (low + high) // 2
        z = tau[middle]
        breakpoint_mass = count - middle - (
            suffix_a2[middle] * z**2
            + 2.0 * suffix_ab[middle] * z
            + suffix_b2[middle]
        ) / h4
        if breakpoint_mass <= target_mass + mass_tolerance:
            high = middle
        else:
            low = middle + 1
    if low < stop:
        right = float(tau[low])
        index = int(np.searchsorted(tau, right, side="left"))
    else:
        right = 1.0
        index = stop

    active_count = count - index
    a2 = float(suffix_a2[index]) if active_count else 0.0
    ab = float(suffix_ab[index]) if active_count else 0.0
    b2 = float(suffix_b2[index]) if active_count else 0.0
    constant = b2 - h4 * (active_count - target_mass)
    if constant == 0.0:
        root = 0.0
    elif a2:
        root = -constant / (
            ab + np.sqrt(max(ab**2 - a2 * constant, 0.0))
        )
    elif ab:
        root = -constant / (2.0 * ab)
    else:
        raise RuntimeError("could not locate the optimal alpha breakpoint")
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(right))
    if not -tolerance <= root <= right + tolerance:
        raise RuntimeError("could not locate the optimal alpha breakpoint")
    return float(np.sqrt(np.clip(root, 0.0, 1.0)))


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


def initialize_basis_pilot(
    X: np.ndarray,
    Y: np.ndarray,
    index_dim: int,
    *,
    seed: int,
) -> np.ndarray:
    try:
        from sklearn.neural_network import MLPRegressor
    except ImportError as error:
        raise ImportError(
            "pilot initialization requires scikit-learn"
        ) from error

    X, Y = utils._prepare_xy(X, Y)
    if isinstance(index_dim, bool) or not isinstance(index_dim, (int, np.integer)):
        raise TypeError("index_dim must be an integer")
    if not 1 <= index_dim <= X.shape[1]:
        raise ValueError("index_dim must lie between 1 and d")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    pilot = MLPRegressor(
        hidden_layer_sizes=(int(index_dim),),
        activation="tanh",
        solver="lbfgs",
        alpha=0.1,
        max_iter=1000,
        random_state=int(seed),
    ).fit(X, Y)
    weights = np.asarray(pilot.coefs_[0], dtype=float)
    if weights.shape != (X.shape[1], index_dim) or not np.all(
        np.isfinite(weights)
    ):
        raise RuntimeError("pilot initializer returned invalid weights")
    if np.linalg.matrix_rank(weights) != index_dim:
        raise RuntimeError("pilot initializer returned a rank-deficient basis")
    basis, _ = np.linalg.qr(weights, mode="reduced")
    return _orient_basis(basis)


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
    smart: bool = False,
) -> float | None:
    _check_smart(smart, kernel)
    X, centers, beta, distance2 = utils._prepare_rho(
        X, centers, beta, h_k, distance2
    )

    if distance2 is None:
        distance2 = pairwise_distance2(X, centers)
    projected = (centers @ beta)[:, None] - (X @ beta)[None, :]
    projection2 = np.square(projected)
    if smart:
        return select_optimal_alpha(
            distance2,
            projection2,
            h_k,
            N_loc * len(centers),
        )
    inverse_h2 = 1.0 / h_k**2
    projection2 *= inverse_h2
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


def single_nonzero_weight_mask(
    projection2: np.ndarray,
    h: float,
    rho: float,
) -> np.ndarray:
    projection2 = utils._finite_real_array(projection2, "projection2")
    if np.any(projection2 < 0):
        raise ValueError("projection2 must be nonnegative")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    return (1.0 + rho**2) * projection2 < h**2


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
    smart: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    _check_smart(smart, kernel)
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
        projection2 = projection_diff
        np.square(projection2, out=projection2)
        if smart:
            candidates = (1.0 + rho**2) * projection2 < h**2
            # Gram distances can undershoot their projection through cancellation.
            candidates |= distance2_block < projection2
            # Boolean gathers only beat dense ufuncs for very sparse support.
            if np.count_nonzero(candidates) * 64 >= candidates.size:
                projection2 += rho**2 * distance2_block
                projection2 /= h**2
                yield start, kernel(projection2)
                continue
            weights = np.zeros_like(projection2)
            argument = (
                rho**2 * distance2_block[candidates] + projection2[candidates]
            ) / h**2
            weights[candidates] = kernel(argument)
            yield start, weights
        else:
            argument = (rho**2 * distance2_block + projection2) / h**2
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
    smart: bool = False,
) -> float | None:
    _check_smart(smart, kernel)
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
    if smart:
        return select_optimal_alpha(
            orthogonal2,
            principal2,
            h_k,
            N_loc * len(centers),
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


def multi_nonzero_weight_mask(principal2: np.ndarray, h: float) -> np.ndarray:
    principal2 = utils._finite_real_array(principal2, "principal2")
    if np.any(principal2 < 0):
        raise ValueError("principal2 must be nonnegative")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    return principal2 < h**2


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
    smart: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    _check_smart(smart, kernel)
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

        if smart:
            projected2, principal2 = _multi_projected_components(
                X,
                C,
                basis,
                eigenvalues,
                distance2_block.shape,
            )
            candidates = principal2 < h**2
            # Boolean gathers only beat dense ufuncs for very sparse support.
            if np.count_nonzero(candidates) * 64 >= candidates.size:
                np.subtract(distance2_block, projected2, out=projected2)
                np.maximum(projected2, 0.0, out=projected2)
                projected2 *= alpha**2
                projected2 += principal2
                projected2 /= h**2
                yield start, kernel(projected2)
                continue
            weights = np.zeros_like(principal2)
            orthogonal2 = np.maximum(
                distance2_block[candidates] - projected2[candidates],
                0.0,
            )
            argument = (
                alpha**2 * orthogonal2 + principal2[candidates]
            ) / h**2
            weights[candidates] = kernel(argument)
            yield start, weights
        else:
            orthogonal2, principal2 = _multi_components(
                X,
                C,
                basis,
                eigenvalues,
                distance2_block,
            )
            argument = (alpha**2 * orthogonal2 + principal2) / h**2
            yield start, kernel(argument)


def _check_smart(smart, kernel):
    if not isinstance(smart, (bool, np.bool_)):
        raise TypeError("smart must be boolean")
    if smart and kernel is not epanechnikov:
        raise ValueError("smart weights require the epanechnikov kernel")


def _multi_components(X, centers, basis, eigenvalues, distance2):
    projected2, principal2 = _multi_projected_components(
        X,
        centers,
        basis,
        eigenvalues,
        distance2.shape,
    )
    orthogonal2 = distance2 - projected2
    np.maximum(orthogonal2, 0.0, out=orthogonal2)
    return orthogonal2, principal2


def _multi_projected_components(X, centers, basis, eigenvalues, shape):
    projected2 = np.zeros(shape)
    principal2 = np.zeros(shape)
    for vector, eigenvalue in zip(basis.T, eigenvalues):
        component2 = (centers @ vector)[:, None] - (X @ vector)[None, :]
        np.square(component2, out=component2)
        projected2 += component2
        principal2 += eigenvalue * component2
    return projected2, principal2


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
