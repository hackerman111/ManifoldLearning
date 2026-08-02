from __future__ import annotations

import argparse
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
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.90)
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
    Y = np.square(X @ beta_true) + args.noise * rng.normal(size=args.n)

    model = ADP_single_index(seed=args.seed + 1).fit(X, Y)
    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    valid = (
        np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0, atol=1e-10)
        and bool(model.trace_)
        and any(step["lsmr_iterations"] > 0 for step in model.trace_)
        and final_cosine >= args.threshold
    )
    print(
        f"seed={args.seed} n={args.n} d={args.d} "
        f"cosine_init={initial_cosine:.6f} cosine_final={final_cosine:.6f} "
        f"threshold={args.threshold:.2f} outer_steps={len(model.trace_)} "
        f"status={'PASS' if valid else 'FAIL'}"
    )
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
