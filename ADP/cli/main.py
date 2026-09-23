from __future__ import annotations

import argparse
import importlib
import resource
import sys
from collections.abc import Callable
from math import sqrt
from typing import cast

import numpy as np

from ..core.ADP_Config import ADP_Config, epanechnikov
from ..core.ADP_Statistic import ADP_Statistics
from ..core.manifold.ADP_Manifold import ADP_Manifold
from ..engine.common import utils
from ..engine.common.index_fit import (
    canonical_index,
    fit_index,
    validate_sizes,
)
from ..engine.common.logger import IndexProfiler as _Profiler
from ..engine.common.statistic import calculate_statistics
from ..solver.CG import solve as solve_cg
from ..solver.HYBRID import solve as solve_hybrid
from ..solver.LSMR import HPAOResult, solve as solve_lsmr

_solver_index = canonical_index

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
        description="Minimal synthetic single/multi/manifold ADP runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode", choices=("single", "multi", "manifold"), default="single"
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--index-dim", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Compute statistics on CUDA; solver stays on CPU",
    )
    parser.add_argument(
        "--gpu-solver", action="store_true", help="With --gpu, also run LSMR on CUDA"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--N_loc", "--n-loc", dest="N_loc", type=int, default=defaults.N_loc
    )
    parser.add_argument("--N_lin", "--n-lin", dest="N_lin", type=int)
    parser.add_argument("--N_J", "--n-centers", dest="N_J", type=int)
    parser.add_argument("--N_phi", "--n-directions", dest="N_phi", type=int)
    parser.add_argument("--N_manifold", "--n-manifold", dest="N_manifold", type=int)
    parser.add_argument("--sync-steps", type=int, default=5)
    parser.add_argument("--lambda-manifold", type=float, default=1.0)
    parser.add_argument("--scale-boundary", choices=("raise", "stop"), default="raise")
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
        choices=("local", "local-cv", "pilot", "random"),
        default=defaults.index_init,
        help="local-cv: experimental weighted leave-one-out local ridge selection",
    )
    parser.add_argument(
        "--estimator",
        choices=("new", "legacy"),
        default=defaults.estimator,
    )
    parser.add_argument(
        "--directions",
        dest="direction_mode",
        choices=("auto", "isotropic", "localized"),
        default=defaults.direction_mode,
    )
    parser.add_argument(
        "--multi-tensor",
        choices=("orthogonal", "full"),
        default=defaults.multi_tensor,
    )
    parser.add_argument(
        "--select-step",
        choices=("best", "last"),
        default=defaults.select_step,
    )
    parser.add_argument(
        "--center-displacement",
        type=float,
        default=defaults.center_displacement,
    )
    parser.add_argument(
        "--training-set",
        choices=("all", "exclude_centers"),
        default=defaults.training_set,
    )
    parser.add_argument(
        "--fixed-directions",
        dest="redraw_directions",
        action="store_false",
        default=defaults.redraw_directions,
    )

    parser.add_argument("--solver", choices=("lsmr", "cg", "hybrid"), default="lsmr")
    parser.add_argument("--dense-max-unknowns", type=int, default=256)
    parser.add_argument("--dense-max-bytes", type=int, default=64 * 1024**2)
    parser.add_argument(
        "--hybrid-inner-rtol",
        type=float,
        help="opt-in relative correction error certificate for multi HYBRID",
    )
    parser.add_argument("--solver-tol", type=float, default=1e-6)
    parser.add_argument("--solver-max-steps", type=int)
    parser.add_argument("--theta", type=float, default=0.1)
    parser.add_argument("--trust-radius", type=float)
    parser.add_argument("--lsmr-maxiter", type=int)
    parser.add_argument("--cg-maxiter", type=int)
    return parser


def _config(args: argparse.Namespace) -> ADP_Config:
    return ADP_Config(
        gpu=args.gpu,
        gpu_solver=args.gpu_solver,
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
        estimator=args.estimator,
        direction_mode=args.direction_mode,
        multi_tensor=args.multi_tensor,
        select_step=args.select_step,
        center_displacement=args.center_displacement,
        training_set=args.training_set,
        redraw_directions=args.redraw_directions,
    )


def _validate_solver_mode(mode: str, solver: str) -> None:
    if mode == "single" and solver == "hybrid":
        raise ValueError(
            "hybrid solver supports multi and manifold modes; "
            "use lsmr or cg for single mode"
        )


def _validate(args: argparse.Namespace, config: ADP_Config) -> tuple[int, int, int]:
    _validate_solver_mode(args.mode, args.solver)
    if config.gpu_solver and args.solver != "lsmr":
        raise NotImplementedError(
            "--gpu-solver supports LSMR only; omit it for CPU CG/HYBRID"
        )
    if args.n < 2:
        raise ValueError("n must be at least two")
    if config.index_init != "random" and args.n <= args.d + 1:
        raise ValueError("n must exceed d + 1 unless index_init is random")
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
    if args.solver_max_steps is not None and args.solver_max_steps < 1:
        raise ValueError("solver_max_steps must be positive")
    if args.cg_maxiter is not None and args.cg_maxiter < 1:
        raise ValueError("cg_maxiter must be positive")

    return validate_sizes(config, args.mode, args.n, args.d, args.index_dim)


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


def _run(
    args: argparse.Namespace,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], dict[str, object]]:
    if args.hybrid_inner_rtol is not None:
        if args.mode != "multi" or args.solver != "hybrid":
            raise ValueError("hybrid_inner_rtol requires multi mode with hybrid solver")
        if (
            not np.isfinite(args.hybrid_inner_rtol)
            or not 0 < args.hybrid_inner_rtol <= args.theta
        ):
            raise ValueError("hybrid_inner_rtol must be finite and in (0, theta]")
    if args.mode == "manifold" and (args.gpu or args.gpu_solver):
        raise NotImplementedError("GPU mode supports single and multi only")
    if args.mode == "manifold":
        return _run_manifold(args, data=data)
    config = _config(args)
    _validate(args, config)
    profiler = _Profiler()
    try:
        with profiler.stage("data"):
            if data is None:
                X, Y, true_basis = _synthetic_data(
                    args.n,
                    args.d,
                    args.index_dim,
                    args.noise,
                    args.data_seed,
                )
            else:
                X, Y = utils._prepare_xy(
                    data[0],
                    data[1],
                    require_overdetermined=config.index_init != "random",
                )
                true_basis = utils._finite_real_array(data[2], "true_basis")
                expected = (args.d, args.index_dim)
                if X.shape != (args.n, args.d) or true_basis.shape != expected:
                    raise ValueError(
                        f"injected data must have X shape {(args.n, args.d)} "
                        f"and true_basis shape {expected}"
                    )

        solver_max_steps = args.solver_max_steps or (3 if args.mode == "single" else 5)
        normalized = config.estimator == "new"

        def solve_step(index: np.ndarray, statistics: ADP_Statistics) -> HPAOResult:
            if args.solver == "cg":
                return solve_cg(
                    index,
                    statistics.U,
                    statistics.I,
                    mass=statistics.mass if normalized else None,
                    lambda_prox=config.lambda_penalty,
                    max_steps=solver_max_steps,
                    tol=args.solver_tol,
                    cg_maxiter=args.cg_maxiter,
                )
            else:
                method = solve_hybrid if args.solver == "hybrid" else solve_lsmr
                return method(
                    index,
                    statistics.U,
                    statistics.I,
                    mass=statistics.mass if normalized else None,
                    lambda_prox=config.lambda_penalty,
                    max_steps=solver_max_steps,
                    tol=args.solver_tol,
                    theta=args.theta,
                    trust_radius=args.trust_radius,
                    lsmr_maxiter=args.lsmr_maxiter,
                    dense_max_unknowns=args.dense_max_unknowns,
                    dense_max_bytes=args.dense_max_bytes,
                    hybrid_inner_rtol=args.hybrid_inner_rtol,
                )

        fitted = fit_index(
            X,
            Y,
            config=config,
            mode=args.mode,
            index_dim=args.index_dim,
            solve=solve_step,
            profiler=profiler,
            quality=lambda index: _quality(args.mode, index, true_basis),
            displacement_scale=getattr(args, "displacement_scale", None),
            statistics_function=calculate_statistics,
        )
        metadata = fitted.metadata
        metadata.update(solver=args.solver, solver_max_steps=solver_max_steps)
        return fitted.index, true_basis, profiler.records, metadata
    finally:
        profiler.finish()


def _run_manifold(
    args: argparse.Namespace,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], dict[str, object]]:
    if args.n <= args.d + 1:
        raise ValueError("manifold mode requires n > d + 1")
    if not 1 <= args.index_dim <= args.d:
        raise ValueError("index_dim must lie between 1 and d in manifold mode")
    if not np.isfinite(args.noise) or args.noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if args.data_seed < 0:
        raise ValueError("data_seed must be nonnegative")

    profiler = _Profiler()
    try:
        with profiler.stage("data"):
            if data is None:
                X, Y, true_index = _synthetic_data(
                    args.n,
                    args.d,
                    args.index_dim,
                    args.noise,
                    args.data_seed,
                )
            else:
                X, Y = utils._prepare_xy(data[0], data[1])
                true_index = utils._finite_real_array(data[2], "true_index")
                if X.shape != (args.n, args.d):
                    raise ValueError(
                        f"injected data must have X shape {(args.n, args.d)}"
                    )

        model = ADP_Manifold(
            args.index_dim,
            N_loc=args.N_loc,
            N_lin=args.N_lin,
            N_J=args.N_J,
            N_phi=args.N_phi,
            N_manifold=args.N_manifold,
            lambda_manifold=args.lambda_manifold,
            scale_boundary=args.scale_boundary,
            sync_steps=args.sync_steps,
            a=args.a,
            h_min=args.h_min,
            batch_size=args.batch_size,
            seed=args.seed,
            cg_tol=args.solver_tol,
            cg_maxiter=args.cg_maxiter,
            solver="hybrid" if args.solver == "hybrid" else "cg",
            dense_max_unknowns=args.dense_max_unknowns,
            dense_max_bytes=args.dense_max_bytes,
        )
        with profiler.stage("solver"):
            model.fit(X, Y)
        if true_index.shape == (args.d, args.index_dim):
            true_projectors = np.broadcast_to(
                true_index.T,
                model.projectors_.shape,
            )
        elif true_index.shape == (args.n, args.d, args.index_dim):
            true_projectors = np.swapaxes(
                true_index[model.center_indices_],
                1,
                2,
            )
        else:
            raise ValueError(
                "manifold true_index must have shape "
                f"{(args.d, args.index_dim)} or {(args.n, args.d, args.index_dim)}"
            )

        with profiler.stage("update"):
            prediction = model.predict(X)
        quality = _quality("manifold", model.projectors_, true_projectors)
        prediction_error = float(np.sum(np.square(Y - prediction)))
        diagnostics: dict[str, object] = {
            "converged": True,
            "cg_iterations": sum(int(entry["cg_iterations"]) for entry in model.trace_),
            "cg_relative_residual_max": max(
                float(entry["cg_relative_residual_max"]) for entry in model.trace_
            ),
            "linear_iterations": sum(
                int(entry["linear_iterations"]) for entry in model.trace_
            ),
            "linear_relative_residual_max": max(
                float(entry["linear_relative_residual_max"]) for entry in model.trace_
            ),
            "dense_solves": sum(int(entry["dense_solves"]) for entry in model.trace_),
            "pcg_solves": sum(int(entry["pcg_solves"]) for entry in model.trace_),
            "fallback_solves": sum(
                int(entry["fallback_solves"]) for entry in model.trace_
            ),
        }
        trace = [dict(entry) for entry in model.trace_]
        effective = model.effective_config_
        metadata: dict[str, object] = {
            "N_lin": effective["N_lin"],
            "N_J": effective["N_J"],
            "N_phi": effective["N_phi"],
            "N_manifold": effective["N_manifold"],
            "sync_steps": args.sync_steps,
            "lambda_manifold": args.lambda_manifold,
            "scale_boundary": args.scale_boundary,
            "outer_iterations": len(trace),
            "stop_reason": trace[-1]["stop_reason"],
            "estimator": "manifold",
            "direction_mode": "structure-adaptive",
            "multi_tensor": None,
            "solver": model.solver,
            "solver_max_steps": args.sync_steps,
            "selection": "last",
            "selected_iteration": len(trace) - 1,
            "selected_error": prediction_error,
            "selected_eigenvalues": (),
            "initial_quality": None,
            "last_quality": quality,
            "initial_eigenvalues": (),
            "training_set": "all",
            "training_size": len(X),
            "center_displacement": 0.0,
            "center_displacement_scale": 0.0,
            "redraw_directions": True,
            "prediction_rmse": sqrt(prediction_error / len(Y)),
            "trace": trace,
            "diagnostics": diagnostics,
            "last_diagnostics": diagnostics,
        }
        return model.projectors_, true_projectors, profiler.records, metadata
    finally:
        profiler.finish()


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def _quality(mode: str, index: np.ndarray, true_basis: np.ndarray) -> float:
    if mode == "single":
        return float(abs(true_basis[:, 0] @ index))
    if mode == "manifold":
        if index.ndim != 3 or true_basis.shape != index.shape:
            raise ValueError("manifold projectors must have matching (J, m, d) shapes")
        overlap = np.einsum("jmd,jnd->jmn", index, true_basis, optimize=True)
        missed = index.shape[1] - np.square(overlap).sum(axis=(1, 2))
        return sqrt(float(np.maximum(missed, 0.0).mean()) / index.shape[1])
    # ESTIMATOR/evaluation protocol: normalized trace of projector overlap.
    overlap = index @ true_basis  # (m, m)
    return float(np.square(overlap).sum() / index.shape[0])


def _rss_peak_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _print_result(
    args: argparse.Namespace,
    index: np.ndarray,
    true_basis: np.ndarray,
    profile: dict[str, dict[str, float]],
    metadata: dict[str, object],
) -> None:
    metric_name = {
        "single": "cosine_abs",
        "multi": "trace_score",
        "manifold": "local_projector_distance",
    }[args.mode]
    print(f"mode={args.mode} X=({args.n}, {args.d}) index={index.shape}")
    print(
        "effective: "
        f"N_lin={metadata['N_lin']} N_J={metadata['N_J']} "
        f"N_phi={metadata['N_phi']} outer_iterations={metadata['outer_iterations']}"
    )
    quality = _quality(args.mode, index, true_basis)
    print(f"stop_reason={metadata['stop_reason']} {metric_name}={quality:.6f}")
    if args.mode == "manifold":
        print(
            f"estimator=manifold solver={metadata['solver']} prediction_rmse="
            f"{cast(float, metadata['prediction_rmse']):.6f}"
        )
        for step in cast(list[dict[str, object]], metadata["trace"]):
            print(
                f"step={step['iteration']} phase={step['phase']} "
                f"h={cast(float, step['h']):.6g} "
                f"alpha={cast(float, step['alpha']):.6g} "
                f"projector_change={cast(float, step['projector_change_max']):.6g}"
            )
        _print_profile(profile)
        return
    print(
        f"estimator={metadata['estimator']} solver={metadata['solver']} "
        f"directions={metadata['direction_mode']} "
        f"selection={metadata['selection']} "
        f"selected_step={metadata['selected_iteration']}"
    )
    for step in cast(list[dict[str, object]], metadata["trace"]):
        step_h = cast(float, step["h"])
        step_factor = cast(float, step["factor"])
        step_ratio = cast(float, step["h_over_factor"])
        step_error = cast(float, step["err"])
        step_quality = cast(float, step["quality"])
        print(
            f"step={step['iteration']} h={step_h:.6g} "
            f"{step['factor_name']}={step_factor:.6g} "
            f"h/{step['factor_name']}={step_ratio:.6g} "
            f"err={step_error:.6g} quality={step_quality:.6g}"
        )
    _print_profile(profile)


def _print_profile(profile: dict[str, dict[str, float]]) -> None:
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
