"""Known-standard-Gaussian moments, isolated from production ADP.

Scope and independent-split variance proof: benchmarks/results/verification/weak-moment-20261030/THEORY.md.
No sample centering/whitening is silently applied.
"""

from __future__ import annotations

import numpy as np


def variance_constants(d, beta_squared, delta, sigma):
    v1 = beta_squared * (d + 1) + delta**2 * (2 * d + 8) + sigma**2 * d
    v2 = (
        beta_squared * (d**2 + 5 * d + 4)
        + delta**2 * (2 * d**2 + 18 * d + 36)
        + sigma**2 * d * (d + 1)
    )
    return v1, v2


def moments(x, y, *, block_size=256):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or len(x) < 1
        or x.shape[1] < 1
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not isinstance(block_size, (int, np.integer))
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("finite X:(n,d), y:(n,), positive integer block required")
    linear, quadratic = np.zeros(x.shape[1]), np.zeros((x.shape[1], x.shape[1]))
    total = 0.0
    for start in range(0, len(x), block_size):
        xb, yb = x[start : start + block_size], y[start : start + block_size]
        linear += yb @ xb
        quadratic += xb.T @ (yb[:, None] * xb)
        total += float(yb.sum())
    quadratic.flat[:: x.shape[1] + 1] -= total
    return linear / len(x), (quadratic + quadratic.T) / (2 * len(x))


def fit(x, y, *, arm="residual_split", block_size=256):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) < 4 or x.shape[1] < 3:
        raise ValueError("at least four observations and d>=3 required")
    if arm not in ("raw_full", "raw_split", "residual_split"):
        raise ValueError("unknown moment arm")
    # Validate before the split to retain the same input contract for every arm.
    if y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid finite X/y")
    if arm == "raw_full":
        linear, quadratic = moments(x, y, block_size=block_size)
    else:
        n0 = len(x) // 2
        linear = y[:n0] @ x[:n0] / n0
        response = y[n0:]
        if arm == "residual_split":
            response = response - x[n0:] @ linear
        _, quadratic = moments(x[n0:], response, block_size=block_size)
    matrix = np.column_stack((linear, quadratic))
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite computed moment")
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular[1] <= max(matrix.shape) * np.finfo(float).eps * singular[0]:
        raise ValueError("moment matrix has numerical rank below two")
    return {
        "basis": u[:, :2],
        "matrix": matrix,
        "singular_values": singular,
        "linear": linear,
        "quadratic": quadratic,
        "diagnostics": {"stop_reason": "direct_svd", "converged": True},
    }
