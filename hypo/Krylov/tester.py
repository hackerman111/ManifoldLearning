from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.Krylov.ADP_single_index import ADP_single_index


def projected_ridge_check():
    diagonal = np.linspace(1.0, 2.0, 6)
    subdiagonal = np.linspace(0.1, 0.6, 6)
    B = np.zeros((7, 6))
    B[np.arange(6), np.arange(6)] = diagonal
    B[np.arange(1, 7), np.arange(6)] = subdiagonal
    rho = 2.5

    model = ADP_single_index()
    selected_lambda, gcv, coordinates, _ = model._select_lambda(B, rho, None)
    rhs = np.zeros(7)
    rhs[0] = rho
    normal = B.T @ B
    identity = np.eye(B.shape[1])

    def dense_gcv(lambda_value):
        solution = np.linalg.solve(normal + lambda_value * identity, B.T @ rhs)
        residual = B @ solution - rhs
        trace = np.trace(B @ np.linalg.solve(normal + lambda_value * identity, B.T))
        return float(np.dot(residual, residual) / (B.shape[0] - trace) ** 2), solution

    reference_gcv, direct = dense_gcv(selected_lambda)
    singular_values = np.linalg.svd(B, compute_uv=False)
    scale = singular_values[0] ** 2
    grid = np.geomspace(1e-9 * scale, 1e2 * scale, 21)
    grid_scores = np.array([dense_gcv(value)[0] for value in grid])
    best = int(np.argmin(grid_scores))
    refined = minimize_scalar(
        lambda value: dense_gcv(math.exp(value))[0],
        bounds=(math.log(grid[best - 1]), math.log(grid[best + 1])),
        method="bounded",
        options={"xatol": 1e-4},
    )
    refined_lambda = math.exp(refined.x)
    return bool(
        np.isfinite(selected_lambda)
        and selected_lambda > 0
        and np.isfinite(gcv)
        and gcv >= 0
        and np.allclose(coordinates, direct, rtol=1e-9, atol=1e-11)
        and np.isclose(gcv, reference_gcv, rtol=1e-10, atol=1e-12)
        and refined.success
        and np.isclose(selected_lambda, refined_lambda, rtol=2e-5, atol=0.0)
        and np.isclose(gcv, refined.fun, rtol=1e-10, atol=1e-12)
    )


if __name__ == "__main__":
    raise SystemExit(not projected_ridge_check())
