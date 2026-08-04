from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADP.ADP_Config import ADP_Config
from ADP.ADP_Data import ADP_Data
from ADP.single_index.ADP_single_index import ADP_single_index


FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "linear": lambda value: value,
    "square": np.square,
}
KERNELS = {
    "epanechnikov": lambda value: np.maximum(1 - np.square(value), 0),
    "triangular": lambda value: np.maximum(1 - value, 0),
    "uniform": lambda value: (value <= 1).astype(float),
}


def finite_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def positive_float(value: str) -> float:
    result = finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_float(value: str) -> float:
    result = finite_float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def threshold(value: str) -> float:
    result = finite_float(value)
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must lie in [0, 1]")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверка single-index ADP на синтетических данных.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("synthetic data")
    data.add_argument("--seed", type=nonnegative_int, default=42)
    data.add_argument("--n", type=positive_int, default=1000)
    data.add_argument("--d", type=positive_int, default=100)
    data.add_argument("--mu-x", type=finite_float, default=0.0)
    data.add_argument("--sigma-x", type=positive_float, default=1.0)
    data.add_argument("--mu-eps", type=finite_float, default=0.0)
    data.add_argument("--sigma-eps", type=nonnegative_float, default=0.5)
    data.add_argument("--mu-beta", type=finite_float, default=0.0)
    data.add_argument("--sigma-beta", type=nonnegative_float, default=1.0)
    data.add_argument("--mu-phi", type=finite_float, default=0.0)
    data.add_argument("--sigma-phi", type=nonnegative_float, default=1.0)
    data.add_argument(
        "--function",
        choices=FUNCTIONS,
        default="sin",
        help="link function f",
    )

    algorithm = parser.add_argument_group("ADP configuration")
    algorithm.add_argument("--n-loc", dest="N_loc", type=positive_float, default=10)
    algorithm.add_argument(
        "--n-lin",
        dest="N_lin",
        type=positive_int,
        help="default: 2*d",
    )
    algorithm.add_argument(
        "--n-centers",
        dest="N_J",
        type=positive_int,
        default=64,
    )
    algorithm.add_argument(
        "--n-directions",
        dest="N_phi",
        type=positive_int,
        default=8,
    )
    algorithm.add_argument(
        "--lambda-penalty",
        dest="lam",
        type=positive_float,
        default=1000.0,
    )
    algorithm.add_argument("--kernel", choices=KERNELS, default="epanechnikov")
    algorithm.add_argument(
        "--bandwidth-decay",
        dest="a",
        type=positive_float,
        default=float(np.sqrt(2)),
    )
    algorithm.add_argument(
        "--h-min",
        type=positive_float,
        help="default: 10*sigma_x/n",
    )

    solver = parser.add_argument_group("model and LSMR")
    solver.add_argument("--outer-steps", type=positive_int, default=8)
    solver.add_argument("--inner-steps", type=positive_int, default=2)
    solver.add_argument("--local-ridge", type=positive_float, default=1e-8)
    solver.add_argument("--tol", type=positive_float, default=1e-6)
    solver.add_argument("--batch-size", type=positive_int, default=32)
    solver.add_argument(
        "--beta-init",
        choices=("local", "random"),
        default="local",
    )
    solver.add_argument(
        "--threshold",
        type=threshold,
        default=0.9,
        help="minimum absolute cosine for PASS",
    )
    return parser


def absolute_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(abs(np.dot(left, right)))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n <= args.d + 1:
        parser.error("--n must exceed --d + 1")

    config = ADP_Config(
        seed=args.seed,
        sigma_x=args.sigma_x,
        mu_x=args.mu_x,
        sigma_eps=args.sigma_eps,
        mu_eps=args.mu_eps,
        sigma_beta=args.sigma_beta,
        mu_beta=args.mu_beta,
        sigma_phi=args.sigma_phi,
        mu_phi=args.mu_phi,
        f=FUNCTIONS[args.function],
        d=args.d,
        n=args.n,
        N_loc=args.N_loc,
        N_lin=args.N_lin,
        N_J=args.N_J,
        N_phi=args.N_phi,
        lam=args.lam,
        kernel=KERNELS[args.kernel],
        a=args.a,
        h_min=args.h_min,
    )
    data = ADP_Data(config)
    X, noise = data.Initialize_input_data()
    beta_true = data.rng.normal(
        loc=config.mu_beta,
        scale=config.sigma_beta,
        size=config.d,
    )
    beta_norm = np.linalg.norm(beta_true)
    if not np.isfinite(beta_norm) or beta_norm == 0:
        parser.error("mu-beta and sigma-beta generated a zero beta")
    beta_true /= beta_norm
    Y = config.f(beta_true @ X) + noise

    started = perf_counter()
    model = ADP_single_index(
        config,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        local_ridge=args.local_ridge,
        tol=args.tol,
        batch_size=args.batch_size,
        beta_init=args.beta_init,
    ).fit(X.T, Y)
    elapsed = perf_counter() - started

    initial_cosine = absolute_cosine(model.beta_init_, beta_true)
    final_cosine = absolute_cosine(model.beta_, beta_true)
    valid = (
        np.all(np.isfinite(model.beta_))
        and np.isclose(np.linalg.norm(model.beta_), 1.0)
        and bool(model.trace_)
        and all(step["lsmr_iterations"] > 0 for step in model.trace_)
        and final_cosine >= args.threshold
    )
    print(
        f"status={'PASS' if valid else 'FAIL'} "
        f"seed={args.seed} n={args.n} d={args.d} "
        f"function={args.function} kernel={args.kernel}\n"
        f"cosine_init={initial_cosine:.6f} "
        f"cosine_final={final_cosine:.6f} threshold={args.threshold:.6f}\n"
        f"steps={len(model.trace_)} h0={model.h0_:.6g} "
        f"h={model.h_k:.6g} rho={model.rho_k} elapsed={elapsed:.3f}s"
    )
    for step in model.trace_:
        print(
            f"k={step['k']} h={step['h']:.6g} rho={step['rho']} "
            f"mass={step['mean_mass']:.3f} "
            f"density={step['weight_density']:.3%} "
            f"lsmr_iterations={step['lsmr_iterations']}"
        )
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
