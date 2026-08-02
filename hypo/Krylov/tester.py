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


def rank_deficient_check():
    B = np.zeros((4, 3))
    B[0, 0] = 1.0
    B[1, 1] = 0.5
    selected_lambda, gcv, coordinates, _ = ADP_single_index()._select_lambda(
        B, 1.0, None
    )
    return bool(
        np.isfinite(selected_lambda)
        and selected_lambda > 0
        and np.isfinite(gcv)
        and gcv >= 0
        and np.all(np.isfinite(coordinates))
    )


def flat_boundary_check():
    module = sys.modules[ADP_single_index.__module__]
    calls = 0

    def counted_minimize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return minimize_scalar(*args, **kwargs)

    original = module.minimize_scalar
    module.minimize_scalar = counted_minimize
    try:
        selected_lambda, gcv, coordinates, boundary = (
            ADP_single_index()._select_lambda(
                np.array([[0.1], [1.0]]), 1.0, 1e15
            )
        )
    finally:
        module.minimize_scalar = original
    return bool(
        boundary
        and calls == 0
        and np.isfinite(selected_lambda)
        and selected_lambda > 0
        and np.isfinite(gcv)
        and gcv >= 0
        and np.all(np.isfinite(coordinates))
    )


def full_span_check():
    rng = np.random.default_rng(9917)
    m, d = 30, 10
    left = np.linalg.qr(rng.normal(size=(m, d)))[0]
    right = np.linalg.qr(rng.normal(size=(d, d)))[0]
    A = left @ np.diag(np.geomspace(1.0, 1e-6, d)) @ right.T
    beta_prior = rng.normal(size=d)
    beta_prior /= np.linalg.norm(beta_prior)
    correction = right @ np.linspace(1.0, 2.0, d)
    orthogonal_noise = rng.normal(size=m)
    orthogonal_noise -= left @ (left.T @ orthogonal_noise)
    orthogonal_noise /= np.linalg.norm(orthogonal_noise)
    residual = A @ correction + 1e-5 * orthogonal_noise
    I = (A @ beta_prior + residual).reshape(1, m)

    beta, record = ADP_single_index()._hybrid_krylov(
        I, A.reshape(1, m, d), np.ones(1), beta_prior
    )
    selected_lambda = record["selected_lambda"]
    dense_left, dense_singular_values, dense_right_transpose = np.linalg.svd(
        A, full_matrices=False
    )
    dense_correction = dense_right_transpose.T @ (
        dense_singular_values
        / (np.square(dense_singular_values) + selected_lambda)
        * (dense_left.T @ residual)
    )
    dense = beta_prior + dense_correction
    dense /= np.linalg.norm(dense)
    if np.dot(dense, beta_prior) < 0:
        dense = -dense
    direction_error = min(
        np.linalg.norm(beta - dense),
        np.linalg.norm(beta + dense),
    )
    return (
        bool(
            np.isfinite(selected_lambda)
            and selected_lambda > 0
            and np.all(np.isfinite(beta))
            and np.isclose(np.linalg.norm(beta), 1.0, atol=1e-12)
            and direction_error < 1e-9
        ),
        direction_error,
    )


if __name__ == "__main__":
    full_span_valid, direction_error = full_span_check()
    checks = {
        "projected_ridge": projected_ridge_check(),
        "rank_deficient": rank_deficient_check(),
        "flat_boundary": flat_boundary_check(),
        "full_span": full_span_valid,
    }
    print(
        " ".join(f"{name}={'PASS' if valid else 'FAIL'}" for name, valid in checks.items())
        + f" direction_error={direction_error:.3e}"
    )
    raise SystemExit(not all(checks.values()))
