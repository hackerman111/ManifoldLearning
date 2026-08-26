from collections.abc import Callable
from importlib import import_module

import numpy as np

from . import utils
from .calculus import search_bandwidth


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
        )[0][1:]

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
        MLPRegressor = import_module("sklearn.neural_network").MLPRegressor
    except ImportError as error:
        raise ImportError("pilot initialization requires scikit-learn") from error

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
    if weights.shape != (X.shape[1], index_dim) or not np.all(np.isfinite(weights)):
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
