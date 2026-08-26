from __future__ import annotations

import argparse
import importlib
import resource
import sys
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import sqrt
from time import perf_counter
from typing import cast

import numpy as np

from .core.ADP_Config import ADP_Config, epanechnikov
from .engine import utils
from .engine.calculus import (
    calculate_alpha_k,
    calculate_rho_k,
    generate_multi_proj,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)
from .engine.initialize import (
    initialize_basis_local,
    initialize_basis_pilot,
    initialize_basis_random,
    initialize_beta_local,
)
from .engine.statistic import calculate_statistics
from .engine.weights import calculate_multi_weight, calculate_weight
from .solver.LSMR import HPAOResult, solve

_STAGES = (
    "data",
    "initialization",
    "bandwidth",
    "directions",
    "statistics",
    "solver",
    "update",
    "total",
)


class _Profiler:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, float]] = {}
        self._started = perf_counter()
        self._owns_trace = not tracemalloc.is_tracing()
        if self._owns_trace:
            tracemalloc.start()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        started = perf_counter()
        try:
            yield
        finally:
            _, peak = tracemalloc.get_traced_memory()
            record = self.records.setdefault(
                name,
                {"time_seconds": 0.0, "traced_peak_bytes": 0.0},
            )
            record["time_seconds"] += perf_counter() - started
            record["traced_peak_bytes"] = max(
                record["traced_peak_bytes"],
                float(max(0, peak - current)),
            )

    def finish(self) -> None:
        peak = max(
            (record["traced_peak_bytes"] for record in self.records.values()),
            default=0.0,
        )
        self.records["total"] = {
            "time_seconds": perf_counter() - self._started,
            "traced_peak_bytes": peak,
        }
        if self._owns_trace:
            tracemalloc.stop()


def parse_kernel(value: str) -> Callable[[np.ndarray], np.ndarray]:
    if value == "epanechnikov":
        return epanechnikov
    try:
        module_name, function_name = value.rsplit(":", 1)
        kernel = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "kernel must be 'epanechnikov' or 'module:function'"
        ) from error
    if not callable(kernel):
        raise argparse.ArgumentTypeError("kernel must be callable")
    return cast(Callable[[np.ndarray], np.ndarray], kernel)


def build_parser() -> argparse.ArgumentParser:
    defaults = ADP_Config()
    parser = argparse.ArgumentParser(
        description="Minimal synthetic single/multi-index ADP runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--index-dim", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--N_loc", "--n-loc", dest="N_loc", type=int, default=defaults.N_loc
    )
    parser.add_argument("--N_lin", "--n-lin", dest="N_lin", type=int)
    parser.add_argument("--N_J", "--n-centers", dest="N_J", type=int)
    parser.add_argument("--N_phi", "--n-directions", dest="N_phi", type=int)
    parser.add_argument("--outer_steps", "--outer-steps", dest="outer_steps", type=int)
    parser.add_argument(
        "--lambda_penalty",
        "--lambda-penalty",
        dest="lambda_penalty",
        type=float,
        default=defaults.lambda_penalty,
    )
    parser.add_argument(
        "--local_ridge",
        "--local-ridge",
        dest="local_ridge",
        type=float,
        default=defaults.local_ridge,
    )
    parser.add_argument("--kernel", type=parse_kernel, default=epanechnikov)
    parser.add_argument("--a", type=float, default=defaults.a)
    parser.add_argument(
        "--h_min",
        "--h-min",
        dest="h_min",
        type=float,
        default=defaults.h_min,
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=defaults.batch_size,
    )
    parser.add_argument(
        "--index_init",
        "--index-init",
        dest="index_init",
        choices=("local", "pilot", "random"),
        default=defaults.index_init,
    )

    parser.add_argument("--solver-tol", type=float, default=1e-6)
    parser.add_argument("--solver-max-steps", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--trust-radius", type=float)
    parser.add_argument("--lsmr-maxiter", type=int)
    return parser


def _config(args: argparse.Namespace) -> ADP_Config:
    return ADP_Config(
        seed=args.seed,
        N_loc=args.N_loc,
        N_lin=args.N_lin,
        N_J=args.N_J,
        N_phi=args.N_phi,
        outer_steps=args.outer_steps,
        lambda_penalty=args.lambda_penalty,
        local_ridge=args.local_ridge,
        kernel=args.kernel,
        a=args.a,
        h_min=args.h_min,
        batch_size=args.batch_size,
        index_init=args.index_init,
    )


def _validate(args: argparse.Namespace, config: ADP_Config) -> tuple[int, int, int]:
    if args.n <= args.d + 1:
        raise ValueError("n must exceed d + 1")
    if args.mode == "single" and args.index_dim != 1:
        raise ValueError("index_dim must equal 1 in single mode")
    if args.mode == "multi" and not 1 <= args.index_dim < args.d:
        raise ValueError("index_dim must lie between 1 and d - 1 in multi mode")
    if not np.isfinite(args.noise) or args.noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if args.data_seed < 0:
        raise ValueError("data_seed must be nonnegative")
    if args.mode == "single" and config.index_init == "pilot":
        raise ValueError("pilot initialization is multi-index only")

    n_lin = config.N_lin or 2 * args.d
    n_centers = config.N_J or args.n
    n_directions = config.N_phi or min(config.N_loc, args.d)
    utils._check_model_sizes(
        args.n,
        args.d,
        config.N_loc,
        n_lin,
        n_centers,
        config.index_init,
    )
    if args.mode == "multi" and n_directions <= args.index_dim:
        raise ValueError("N_phi must exceed index_dim in multi mode")
    return n_lin, n_centers, n_directions


def _synthetic_data(
    n: int,
    d: int,
    index_dim: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seed, index_seed, noise_seed = np.random.SeedSequence(seed).spawn(3)
    X = np.random.default_rng(x_seed).standard_normal((n, d))
    basis, _ = np.linalg.qr(
        np.random.default_rng(index_seed).standard_normal((d, index_dim)),
        mode="reduced",
    )
    basis = _orient_basis(basis)
    projected = X @ basis
    signal = np.sum(np.sin(projected) + 0.25 * np.square(projected), axis=1)
    Y = signal + noise * np.random.default_rng(noise_seed).standard_normal(n)
    return X, Y, basis


def _initial_index(
    args: argparse.Namespace,
    config: ADP_Config,
    X: np.ndarray,
    Y: np.ndarray,
    centers: np.ndarray,
    distance2: np.ndarray,
    n_lin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.index_init == "random":
        basis = initialize_basis_random(rng, args.d, args.index_dim)
    elif config.index_init == "pilot":
        basis = initialize_basis_pilot(
            X,
            Y,
            args.index_dim,
            seed=config.seed,
        )
    elif args.mode == "single":
        return initialize_beta_local(
            X,
            Y,
            centers,
            distance2,
            n_lin,
            config.kernel,
            config.local_ridge,
        )
    else:
        basis = initialize_basis_local(
            X,
            Y,
            centers,
            distance2,
            n_lin,
            config.kernel,
            config.local_ridge,
            args.index_dim,
        )
    return basis[:, 0] if args.mode == "single" else basis.T


def _run(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], dict[str, object]]:
    config = _config(args)
    n_lin, n_centers, n_directions = _validate(args, config)
    profiler = _Profiler()
    try:
        with profiler.stage("data"):
            X, Y, true_basis = _synthetic_data(
                args.n,
                args.d,
                args.index_dim,
                args.noise,
                args.data_seed,
            )

        center_seed, init_seed, direction_seed = np.random.SeedSequence(
            config.seed
        ).spawn(3)
        with profiler.stage("initialization"):
            center_rng = np.random.default_rng(center_seed)
            centers = (
                X.copy()
                if n_centers == args.n
                else X[center_rng.choice(args.n, size=n_centers, replace=False)]
            )
            distance2 = pairwise_distance2(X, centers)
            index = _initial_index(
                args,
                config,
                X,
                Y,
                centers,
                distance2,
                n_lin,
                np.random.default_rng(init_seed),
            )

        with profiler.stage("bandwidth"):
            h = search_bandwidth(
                distance2,
                config.N_loc,
                config.kernel,
                lower=config.h_min,
            )

        direction_rng = np.random.default_rng(direction_seed)
        factor = 1.0
        eigenvalues = np.ones(args.index_dim)
        diagnostics: dict[str, object] = {}
        stop_reason = "h_min"
        outer_iteration = 0

        while True:
            with profiler.stage("directions"):
                if args.mode == "single":
                    directions = generate_proj(
                        direction_rng,
                        n_centers,
                        n_directions,
                        index,
                        factor,
                    )
                else:
                    directions = generate_multi_proj(
                        direction_rng,
                        n_centers,
                        n_directions,
                        index.T,
                        eigenvalues,
                        factor,
                    )

            with profiler.stage("statistics"):
                if args.mode == "single":
                    weights = calculate_weight(
                        X,
                        centers,
                        index,
                        h,
                        factor,
                        config.kernel,
                        block_size=config.batch_size,
                        distance2=distance2,
                    )
                else:
                    weights = calculate_multi_weight(
                        X,
                        centers,
                        index.T,
                        eigenvalues,
                        h,
                        factor,
                        config.kernel,
                        block_size=config.batch_size,
                        distance2=distance2,
                    )
                statistics = calculate_statistics(
                    X,
                    Y,
                    weights,
                    directions,
                    batch_size=config.batch_size,
                )

            with profiler.stage("solver"):
                result = solve(
                    index,
                    statistics.U,
                    statistics.I,
                    lambda_prox=config.lambda_penalty,
                    max_steps=args.solver_max_steps,
                    tol=args.solver_tol,
                    theta=args.theta,
                    trust_radius=args.trust_radius,
                    lsmr_maxiter=args.lsmr_maxiter,
                )
                index, eigenvalues = _solver_index(args.mode, result, index)
                diagnostics = dict(result.diagnostics)

            outer_iteration += 1
            if config.outer_steps is not None and outer_iteration >= config.outer_steps:
                stop_reason = "outer_steps"
                break
            next_h = h / config.a
            if next_h < config.h_min:
                break

            with profiler.stage("update"):
                if args.mode == "single":
                    next_factor = calculate_rho_k(
                        X,
                        centers,
                        index,
                        next_h,
                        config.N_loc,
                        config.kernel,
                        distance2=distance2,
                    )
                else:
                    next_factor = calculate_alpha_k(
                        X,
                        centers,
                        index.T,
                        eigenvalues,
                        next_h,
                        config.N_loc,
                        config.kernel,
                        distance2=distance2,
                    )
                if next_factor is None:
                    stop_reason = "local_mass_limit"
                    break
                factor = next_factor
                h = float(next_h)

        metadata: dict[str, object] = {
            "N_lin": n_lin,
            "N_J": n_centers,
            "N_phi": n_directions,
            "outer_iterations": outer_iteration,
            "stop_reason": stop_reason,
            "diagnostics": diagnostics,
        }
        return index, true_basis, profiler.records, metadata
    finally:
        profiler.finish()


def _solver_index(
    mode: str,
    result: HPAOResult,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.asarray(result.index, dtype=float)
    if mode == "single":
        norm = np.linalg.norm(index)
        if index.shape != prior.shape or not np.isfinite(norm) or norm == 0:
            raise RuntimeError("single-index solver returned an invalid index")
        index /= norm
        if np.dot(index, prior) < 0:
            index = -index
        return index, np.ones(1)

    coefficients = np.asarray(result.coefficients, dtype=float)
    if index.shape != prior.shape or coefficients.shape[1:] != (index.shape[0],):
        raise RuntimeError("multi-index solver returned incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (index, coefficients)):
        raise RuntimeError("multi-index solver returned non-finite values")
    if not np.allclose(
        index @ index.T,
        np.eye(index.shape[0]),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError("multi-index solver returned a non-orthonormal index")

    # EXACT: одновременный поворот индекса и коэффициентов сохраняет fitted values.
    values, vectors = np.linalg.eigh(coefficients.T @ coefficients)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    if values[0] <= np.finfo(float).eps:
        raise RuntimeError("local coefficients do not identify a multi-index")
    vectors = vectors[:, order]
    index = vectors.T @ index
    signs = np.sign(index[np.arange(len(index)), np.argmax(np.abs(index), axis=1)])
    index *= np.where(signs == 0, 1.0, signs)[:, None]
    return index, values / values[0]


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def _quality(mode: str, index: np.ndarray, true_basis: np.ndarray) -> float:
    if mode == "single":
        return float(abs(true_basis[:, 0] @ index))
    estimate = index.T
    return float(
        np.linalg.norm(
            true_basis @ true_basis.T - estimate @ estimate.T,
            ord="fro",
        )
        / sqrt(2.0 * true_basis.shape[1])
    )


def _rss_peak_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _print_result(
    args: argparse.Namespace,
    index: np.ndarray,
    true_basis: np.ndarray,
    profile: dict[str, dict[str, float]],
    metadata: dict[str, object],
) -> None:
    metric_name = "cosine_abs" if args.mode == "single" else "projector_distance"
    print(f"mode={args.mode} X=({args.n}, {args.d}) index={index.shape}")
    print(
        "effective: "
        f"N_lin={metadata['N_lin']} N_J={metadata['N_J']} "
        f"N_phi={metadata['N_phi']} outer_iterations={metadata['outer_iterations']}"
    )
    quality = _quality(args.mode, index, true_basis)
    print(f"stop_reason={metadata['stop_reason']} {metric_name}={quality:.6f}")
    print("\nstage statistics (tracemalloc peak excludes untraced native memory):")
    for name in _STAGES:
        record = profile.get(
            name,
            {"time_seconds": 0.0, "traced_peak_bytes": 0.0},
        )
        print(
            f"{name:>14}: {record['time_seconds']:.6f} s  "
            f"{record['traced_peak_bytes'] / 2**20:.3f} MiB"
        )
    print(f"process RSS peak: {_rss_peak_mib():.1f} MiB")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        index, true_basis, profile, metadata = _run(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    except (ImportError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"ADP failed: {error}", file=sys.stderr)
        return 1
    _print_result(args, index, true_basis, profile, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
