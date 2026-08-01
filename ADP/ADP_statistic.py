import numpy as np


def calculate_statistics(X, Y, weights, directions, batch_size=32):
    """Calculate stable dense ADP statistics in batches of local centers."""
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    X, Y, W, Phi, mass = _prepare_inputs(X, Y, weights, directions)
    J, P = Phi.shape[:2]
    A = W / mass[:, None]
    x_bar = X.mean(axis=0)
    Xc = X - x_bar
    Mc = A @ Xc
    mean = Mc + x_bar
    y_bar = A @ Y
    I = np.empty((J, P))
    U = np.empty((J, P, X.shape[1]))
    eta = np.empty((J, P))

    for start in range(0, J, batch_size):
        batch = slice(start, min(start + batch_size, J))
        Ab, Phib, Mcb = A[batch], Phi[batch], Mc[batch]
        Q = Phib @ Xc.T - Phib @ Mcb[..., None]
        residual = (Q @ Ab[..., None]).squeeze(-1)
        eta[batch] = _normalized_residual(Ab, Q, residual)
        H = (Q - residual[..., None]) * Ab[:, None, :]
        s = H.sum(axis=2)
        I[batch] = mass[batch, None] * (H @ Y - s * y_bar[batch, None])
        U[batch] = mass[batch, None, None] * (
            H @ Xc - s[..., None] * Mcb[:, None, :]
        )

    return {
        "I": I,
        "U": U,
        "mass": mass,
        "mean": mean,
        "n_eff": 1.0 / np.square(A).sum(axis=1),
        "eta": eta,
    }


def _normalized_residual(A, Q, residual):
    denominator = (np.abs(Q) @ A[..., None]).squeeze(-1)
    return np.divide(
        np.abs(residual),
        denominator,
        out=np.zeros_like(residual),
        where=denominator != 0,
    )


def _prepare_inputs(X, Y, weights, directions):
    arrays = []
    for value, name in (
        (X, "X"),
        (Y, "Y"),
        (weights, "weights"),
        (directions, "directions"),
    ):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError(f"{name} must have a real numeric dtype")
        array = array.astype(float, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(array)

    X, Y, W, Phi = arrays
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must have non-empty shape (n, d)")
    if Y.shape != (X.shape[0],):
        raise ValueError("Y must have shape (n,)")
    if W.ndim != 2 or W.shape[1] != X.shape[0] or W.shape[0] == 0:
        raise ValueError("weights must have non-empty shape (J, n)")
    if (
        Phi.ndim != 3
        or Phi.shape[0] != W.shape[0]
        or Phi.shape[2] != X.shape[1]
    ):
        raise ValueError("directions must have shape (J, P, d)")
    if Phi.shape[1] == 0:
        raise ValueError("directions must contain at least one direction")
    if np.any(W < 0):
        raise ValueError("weights must be nonnegative")

    with np.errstate(over="ignore", invalid="ignore"):
        mass = W.sum(axis=1)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("every weight row must have positive finite mass")
    return X, Y, W, Phi, mass
