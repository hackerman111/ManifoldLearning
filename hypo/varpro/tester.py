from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.varpro.ADP_single_index import ADP_single_index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверка восстановления beta методом ADP VarPro / Riemannian L-BFGS."
    )
    parser.add_argument("--n", type=int, default=int(400))
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--beta-init", choices=("local", "random"), default="local")
    args = parser.parse_args(argv)
    if args.n <= args.d + 1:
        parser.error("--n must exceed --d + 1")
    if args.d < 1:
        parser.error("--d must be positive")
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise must be finite and nonnegative")
    if not np.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    return args


def absolute_cosine(left, right):
    return float(
        abs(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    )


def profiled_gradient_check():
    rng = np.random.default_rng(20260802)
    J, P, d = 4, 5, 3
    I = rng.normal(size=(J, P))
    U = rng.normal(size=(J, P, d))
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    tangent = rng.normal(size=d)
    tangent -= beta * np.dot(beta, tangent)
    tangent /= np.linalg.norm(tangent)

    model = ADP_single_index()
    value, gradient, _ = model._profiled_value_gradient(I, U, beta)
    eps = 1e-6
    plus = beta + eps * tangent
    plus /= np.linalg.norm(plus)
    minus = beta - eps * tangent
    minus /= np.linalg.norm(minus)
    plus_value = model._profiled_value_gradient(I, U, plus)[0]
    minus_value = model._profiled_value_gradient(I, U, minus)[0]
    finite_difference = (plus_value - minus_value) / (2 * eps)
    analytic = np.dot(gradient, tangent)
    return bool(
        np.all(
            np.isfinite((value, plus_value, minus_value, finite_difference, analytic))
        )
        and np.all(np.isfinite(gradient))
        and np.isclose(finite_difference, analytic, rtol=1e-5, atol=1e-7)
    )


def main(argv=None):
    args = parse_args(argv)
    gradient_check = profiled_gradient_check()
    rng = np.random.default_rng(args.seed)
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    X = rng.normal(size=(args.n, args.d))
    Y = np.sin(X @ beta_true) + args.noise * rng.normal(size=args.n)

    model = ADP_single_index(seed=args.seed + 1, beta_init=args.beta_init).fit(X, Y)
    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    timing_names = (
        "initialization",
        "rho",
        "directions",
        "weights",
        "statistics",
        "rlbfgs",
        "total",
    )
    solver_statuses = {
        "gradient",
        "objective",
        "max_iterations",
        "line_search_failed",
    }
    trace_valid = all(
        step["solver_status"] in solver_statuses
        and np.isfinite(step["objective"])
        and np.isfinite(step["gradient_norm"])
        and step["gradient_norm"] >= 0
        and all(
            np.isfinite(step[name])
            and step[name] >= 0
            and float(step[name]).is_integer()
            for name in ("solver_iterations", "line_search_steps")
        )
        for step in model.trace_
    )
    valid = (
        gradient_check
        and np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(model.trace_)
        and final_cosine >= args.threshold
        and any(step["solver_iterations"] > 0 for step in model.trace_)
        and trace_valid
        and tuple(model.timings_) == timing_names
        and all(
            np.isfinite(model.timings_[name]) and model.timings_[name] >= 0
            for name in timing_names
        )
        and model.timings_["total"] > 0
        and np.isfinite(model.weight_density_)
        and 0 <= model.weight_density_ <= 1
        and np.isfinite(model.mean_zero_weight_fraction_)
        and 0 <= model.mean_zero_weight_fraction_ <= 1
        and np.isclose(
            model.mean_zero_weight_fraction_,
            np.mean([step["zero_weight_fraction"] for step in model.trace_]),
        )
    )
    print(
        f"seed={args.seed} n={args.n} d={args.d} beta_init={args.beta_init} "
        f"cosine_init={initial_cosine:.6f} cosine_final={final_cosine:.6f} "
        f"threshold={args.threshold:.2f} outer_steps={len(model.trace_)} "
        f"weight_density_final={model.weight_density_:.2%} "
        f"weight_zero_mean={model.mean_zero_weight_fraction_:.2%} "
        f"solver_status={model.trace_[-1]['solver_status']} "
        f"gradient_check={'PASS' if gradient_check else 'FAIL'} "
        f"status={'PASS' if valid else 'FAIL'}"
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
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
