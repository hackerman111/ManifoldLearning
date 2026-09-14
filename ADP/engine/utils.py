from collections.abc import Iterator

import numpy as np


def _finite_real_array(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must have a real numeric dtype")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _prepare_xy(X, Y, *, require_overdetermined=True):
    X, Y = (_finite_real_array(value, name) for value, name in ((X, "X"), (Y, "Y")))
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if Y.shape != (X.shape[0],):
        raise ValueError("Y must have shape (n,)")
    # EXACT: random-init не использует переопределенную локальную инициализацию.
    if require_overdetermined and X.shape[0] <= X.shape[1] + 1:
        raise ValueError("n must exceed d + 1")
    return X, Y


def _check_model_sizes(n, d, N_loc, N_lin, N_J, index_init):
    if N_loc > n:
        raise ValueError("N_loc cannot exceed n")
    if index_init in {"local", "local-cv"}:
        if N_lin > n:
            raise ValueError("N_lin cannot exceed n")
        if N_lin <= d + 1:
            raise ValueError("N_lin must exceed d + 1 for local initialization")
    if not np.ceil(n / N_loc) <= N_J <= n:
        raise ValueError("N_J must lie between ceil(n / N_loc) and n")


def _prepare_transform(X, n_features):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError("X must have shape (n, d) with the fitted d")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def _prepare_direction(beta_true, shape):
    beta_true = np.asarray(beta_true, dtype=float)
    if beta_true.shape != shape:
        raise ValueError("beta_true must have shape (d,)")
    norm = np.linalg.norm(beta_true)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("beta_true must be finite and non-zero")
    return beta_true / norm


def _prepare_pairwise(X, centers):
    X = _finite_real_array(X, "X")
    centers = _finite_real_array(centers, "centers")
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if centers.ndim != 2 or centers.shape[1] != X.shape[1] or not len(centers):
        raise ValueError("centers must have non-empty shape (J, d)")
    return X, centers


def _prepare_bandwidth(distance2, target, kernel, lower):
    distance2 = _finite_real_array(distance2, "distance2")
    if distance2.ndim != 2 or 0 in distance2.shape:
        raise ValueError("distance2 must have non-empty shape (J, n)")
    if np.any(distance2 < 0):
        raise ValueError("distance2 must be nonnegative")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError("lower must be finite and positive")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    return distance2


def _prepare_rho(X, centers, beta, h_k, distance2):
    X, centers = _prepare_pairwise(X, centers)
    if not np.isfinite(h_k) or h_k <= 0:
        raise ValueError("h_k must be finite and positive")
    beta = _finite_real_array(beta, "beta")
    if beta.shape != (X.shape[1],):
        raise ValueError("beta must have shape (d,)")
    beta_norm = np.linalg.norm(beta)
    if beta_norm == 0:
        raise ValueError("beta must be non-zero")
    if distance2 is not None:
        distance2 = _finite_real_array(distance2, "distance2")
        if distance2.shape != (len(centers), len(X)):
            raise ValueError("distance2 must have shape (J, n)")
        if np.any(distance2 < 0):
            raise ValueError("distance2 must be nonnegative")
    return X, centers, beta / beta_norm, distance2


def _prepare_projection(beta, rho):
    beta = _finite_real_array(beta, "beta")
    if beta.ndim != 1 or not len(beta):
        raise ValueError("beta must have shape (d,)")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    beta_norm = np.linalg.norm(beta)
    if beta_norm:
        beta = beta / beta_norm
    elif rho == 0:
        raise ValueError("rho and beta cannot both be zero")
    return beta


def _prepare_distance2(distance2, centers, n):
    distance2 = _finite_real_array(distance2, "distance2")
    if distance2.shape != (len(centers), n):
        raise ValueError("distance2 must have shape (J, n)")
    if np.any(distance2 < 0):
        raise ValueError("distance2 must be nonnegative")
    return distance2


def _prepare_weight_data(X, centers, beta, h, rho, kernel, block_size):
    X, centers = _prepare_pairwise(X, centers)
    beta = _finite_real_array(beta, "beta")
    check_weight_block(X, centers, beta, h, rho, kernel, block_size)
    return X, centers, beta


def _check_weight_block_order(start, expected_start, J):
    if (
        not isinstance(start, (int, np.integer))
        or isinstance(start, bool)
        or start != expected_start
    ):
        raise ValueError("weight blocks must cover centers in order")
    if start > J:
        raise ValueError("weight blocks exceed the number of directions")


def _check_batch_size(batch_size):
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")


def _prepare_data(X, Y, directions):
    arrays = []
    for value, name in (
        (X, "X"),
        (Y, "Y"),
        (directions, "directions"),
    ):
        arrays.append(_finite_real_array(value, name))

    X, Y, Phi = arrays
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if Y.shape != (X.shape[0],):
        raise ValueError("Y must have shape (n,)")
    if Phi.ndim != 3 or not Phi.shape[0] or Phi.shape[2] != X.shape[1]:
        raise ValueError("directions must have shape (J, P, d)")
    if Phi.shape[1] == 0:
        raise ValueError("directions must contain at least one direction")
    return X, Y, Phi


def _weight_blocks(weights, J, n, batch_size):
    if isinstance(weights, Iterator):
        yield from weights
        return

    W = _finite_real_array(weights, "weights")
    if W.shape != (J, n):
        raise ValueError("weights must have shape (J, n)")
    if np.any(W < 0):
        raise ValueError("weights must be nonnegative")
    for start in range(0, J, batch_size):
        yield start, W[start : start + batch_size]


def _prepare_weight_block(weights, n):
    W = _finite_real_array(weights, "weights")
    if W.ndim != 2 or not W.shape[0] or W.shape[1] != n:
        raise ValueError("weight blocks must have shape (B, n)")
    if np.any(W < 0):
        raise ValueError("weights must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        mass = W.sum(axis=1)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("every weight row must have positive finite mass")
    return W, mass


def check_weight_block(X, centers, beta, h, rho, kernel, block_size=128):
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if centers.ndim != 2 or centers.shape[1] != X.shape[1] or not len(centers):
        raise ValueError("centers must have non-empty shape (J, d)")
    if beta.shape != (X.shape[1],):
        raise ValueError("beta must have shape (d,)")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    if (
        not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")


def _prepare_multi_localization(
    X,
    centers,
    basis,
    eigenvalues,
    h,
    alpha,
    kernel,
):
    X, centers = _prepare_pairwise(X, centers)
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
    basis = _finite_real_array(basis, "basis")
    eigenvalues = _finite_real_array(eigenvalues, "eigenvalues")
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
