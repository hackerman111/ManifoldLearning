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
from hypo.single_index.tester import absolute_cosine, parse_args


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


def _positive_integer(value):
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value > 0
    )


def _finite_nonnegative(value):
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.number))
        and np.isfinite(value)
        and value >= 0
    )


def _finite_positive(value):
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.number))
        and np.isfinite(value)
        and value > 0
    )


def _valid_trace_row(row):
    status = row.get("solver_status")
    iterations = row.get("krylov_iterations")
    selected_lambda = row.get("selected_lambda")
    lambda_valid = selected_lambda is None or _finite_positive(selected_lambda)
    requires_lambda = iterations != 0 or status not in {"zero_residual", "breakdown"}
    return bool(
        status in {"stabilized", "max_iterations", "breakdown", "zero_residual"}
        and not isinstance(iterations, (bool, np.bool_))
        and isinstance(iterations, (int, np.integer))
        and iterations >= 0
        and _finite_nonnegative(row.get("projected_gcv"))
        and isinstance(row.get("lambda_at_boundary"), (bool, np.bool_))
        and lambda_valid
        and (not requires_lambda or selected_lambda is not None)
        and _positive_integer(row.get("inner_iterations"))
        and _finite_nonnegative(row.get("beta_delta"))
    )


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    X = rng.normal(size=(args.n, args.d))
    Y = np.sin(X @ beta_true) + args.noise * rng.normal(size=args.n)

    model = ADP_single_index(seed=args.seed + 1, beta_init=args.beta_init).fit(X, Y)
    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    full_span_valid, direction_error = full_span_check()
    checks = {
        "projected_ridge": projected_ridge_check(),
        "full_span": full_span_valid,
        "rank_deficient": rank_deficient_check(),
        "flat_boundary": flat_boundary_check(),
    }
    timing_names = (
        "initialization",
        "rho",
        "directions",
        "weights",
        "statistics",
        "slopes",
        "krylov",
        "total",
    )
    trace = model.trace_
    weight_fractions = [row.get("zero_weight_fraction") for row in trace]
    technical_valid = bool(
        all(checks.values())
        and np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(trace)
        and all(_valid_trace_row(row) for row in trace)
        and any(row["krylov_iterations"] > 0 for row in trace)
        and _finite_positive(model.lambda_)
        and tuple(model.timings_) == timing_names
        and all(_finite_nonnegative(model.timings_[name]) for name in timing_names)
        and model.timings_["total"] > 0
        and _finite_nonnegative(model.weight_density_)
        and model.weight_density_ <= 1
        and _finite_nonnegative(model.mean_zero_weight_fraction_)
        and model.mean_zero_weight_fraction_ <= 1
        and all(_finite_nonnegative(value) and value <= 1 for value in weight_fractions)
        and np.isclose(model.mean_zero_weight_fraction_, np.mean(weight_fractions))
    )
    quality_valid = final_cosine >= args.threshold
    overall_valid = technical_valid and quality_valid
    last = trace[-1]
    print(
        f"seed={args.seed} n={args.n} d={args.d} beta_init={args.beta_init} "
        f"cosine_init={initial_cosine:.6f} cosine_final={final_cosine:.6f} "
        f"threshold={args.threshold:.2f} outer_steps={len(trace)} "
        f"weight_density_final={model.weight_density_:.2%} "
        f"weight_zero_mean={model.mean_zero_weight_fraction_:.2%} "
        f"last_solver_status={last['solver_status']} "
        f"final_selected_lambda={model.lambda_:.6e} "
        f"technical_status={'PASS' if technical_valid else 'FAIL'} "
        f"quality_status={'PASS' if quality_valid else 'FAIL'} "
        f"status={'PASS' if overall_valid else 'FAIL'}"
    )
    print(
        "numerical_checks "
        + " ".join(f"{name}={'PASS' if valid else 'FAIL'}" for name, valid in checks.items())
        + f" full_span_direction_error={direction_error:.3e}"
    )
    print(
        "timings_sec "
        + " ".join(f"{name}={model.timings_[name]:.6f}" for name in timing_names)
    )
    print(
        "timings_pct "
        + " ".join(
            f"{name}={model.timings_[name] / model.timings_['total']:.2%}"
            for name in timing_names
        )
    )
    return int(not overall_valid)


if __name__ == "__main__":
    raise SystemExit(main())
