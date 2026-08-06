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
