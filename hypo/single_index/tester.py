from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypo.single_index.ADP_single_index import ADP_single_index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверка восстановления beta методом ADP."
    )
    parser.add_argument("--n", type=int, default=int(1000))
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--beta-init", choices=("local", "random"), default="local")
    parser.add_argument("--lambda_penalty", type=float, default=1.0)
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


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    beta_true = rng.normal(size=args.d)
    beta_true /= np.linalg.norm(beta_true)
    X = rng.normal(size=(args.n, args.d))
    Y = np.sin(X @ beta_true) + args.noise * rng.normal(size=args.n)

    model = ADP_single_index(
        seed=args.seed + 1, beta_init=args.beta_init, lambda_penalty=args.lambda_penalty
    ).fit(X, Y)
    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    timing_names = (
        "initialization",
        "rho",
        "directions",
        "weights",
        "statistics",
        "slopes",
        "lsmr",
        "total",
    )
    valid = (
        np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(model.trace_)
        and any(step["lsmr_iterations"] > 0 for step in model.trace_)
        and final_cosine >= args.threshold
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
        f"seed={args.seed} n={args.n} d={args.d} beta_init={args.beta_init} \n"
        f"cosine_init={initial_cosine:.6f} cosine_final={final_cosine:.6f}\n "
        f"threshold={args.threshold:.2f} outer_steps={len(model.trace_)}\n "
        f"weight_density_final={model.weight_density_:.2%}\n "
        f"weight_zero_mean={model.mean_zero_weight_fraction_:.2%}\n "
        f"lambda={model.lambda_penalty}\n"
        f"status={'PASS' if valid else 'FAIL'}\n"
        "----------------------"
    )
    print(
        "timings_sec "
        + " ".join(f"{name}={model.timings_[name]:.6f}\n" for name in timing_names)
    )
    print("----------------")
    print(
        "timings_pct "
        + " ".join(
            f"{name}={model.timings_[name] / model.timings_['total']:.2%}\n"
            for name in timing_names
        )
    )
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
