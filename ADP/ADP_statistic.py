from collections.abc import Iterator

import numpy as np  # noqa: N999


def calculate_statistics(X, Y, weights, directions, batch_size=32):
    """Calculate stable ADP statistics from a matrix or iterator of weight blocks."""
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    X, Y, Phi = _prepare_data(X, Y, directions)
    J, P = Phi.shape[:2]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar
    I = np.empty((J, P))
    U = np.empty((J, P, X.shape[1]))
    mass = np.empty(J)
    mean = np.empty((J, X.shape[1]))
    n_eff = np.empty(J)
    eta = np.empty((J, P))

    expected_start = 0
    for start, W in _weight_blocks(weights, J, X.shape[0], batch_size):
        if (
            not isinstance(start, (int, np.integer))
            or isinstance(start, bool)
            or start != expected_start
        ):
            raise ValueError("weight blocks must cover centers in order")
        W, mass_block = _prepare_weight_block(W, X.shape[0])
        stop = start + W.shape[0]
        if stop > J:
            raise ValueError("weight blocks exceed the number of directions")

        batch = slice(start, stop)
        A = W / mass_block[:, None]
        Phib = Phi[batch]
        Mcb = A @ Xc
        y_bar = A @ Y
        Q = Phib @ Xc.T - Phib @ Mcb[..., None]
        residual = (Q @ A[..., None]).squeeze(-1)
        eta[batch] = _normalized_residual(A, Q, residual)
        H = (Q - residual[..., None]) * A[:, None, :]
        s = H.sum(axis=2)
        I[batch] = mass_block[:, None] * (H @ Y - s * y_bar[:, None])
        U[batch] = mass_block[:, None, None] * (H @ Xc - s[..., None] * Mcb[:, None, :])
        mass[batch] = mass_block
        mean[batch] = Mcb + x_bar
        n_eff[batch] = 1.0 / np.square(A).sum(axis=1)
        expected_start = stop

    if expected_start != J:
        raise ValueError("weight blocks do not cover all directions")

    return {
        "I": I,
        "U": U,
        "mass": mass,
        "mean": mean,
        "n_eff": n_eff,
        "eta": eta,
    }


def _weight_blocks_single(X, centers, beta, h, rho, kernel, block_size=128):
    X = _finite_real_array(X, "X")
    centers = _finite_real_array(centers, "centers")
    beta = _finite_real_array(beta, "beta")
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if centers.ndim != 2 or centers.shape[1] != X.shape[1] or not len(centers):
        raise ValueError("centers must have non-empty shape (J, d)")
    if beta.shape != (X.shape[1],):
        raise ValueError("beta must have shape (d,)")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    if (
        not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")

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


def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )


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


def Calculate_statistic(self, directions, batch_size=128) -> None:
    X = self.X.T
    weights = _weight_blocks_single(
        X,
        self.x_j.T,
        self.beta,
        self.h_k,
        self.rho_k,
        self.kernel,
        batch_size,
    )
    statistics = calculate_statistics(X, self.Y, weights, directions, batch_size)
    self.I = statistics["I"]
    self.U = statistics["U"]
    self.mass = statistics["mass"]
    self.mean = statistics["mean"]
    self.n_eff = statistics["n_eff"]
    self.eta = statistics["eta"]
