"""Paired Grassmann/SVD benchmark on the saved heavy Spokoiny workload.

Run from the repository root after the Grassmann solver is available:
    python -m benchmarks.grassman_benchmark --stage fullfit
    python -m benchmarks.grassman_benchmark --stage capture
    python -m benchmarks.grassman_benchmark --stage frozen

The full-fit stage is uninstrumented. Snapshot capture is a separate run so
copying frozen statistics does not contaminate full-fit time or peak RSS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

CASE = "spokoini_m2_dhigh_30s"
SEEDS = (0, 1, 2)
PARAMS: dict[str, object] = {
    "generator": "spokoini",
    "n": 800,
    "d": 50,
    "m": 2,
    "N_J": 800,
    "N_phi": 10,
    "N_lin": 100,
    "N_loc": 10,
    "a": 1.010050167084168,
    "h_min": 1.0,
    "select_step": "last",
    "solver_max_steps": 5,
    "budget_sec": 30.0,
}
BASELINE = "svd"
GRASSMAN = "core_gn"
FROZEN_VARIANTS = (
    "svd",
    "rank_one_refit",
    "rank_one_schur",
    "spectral_rank1",
    "spectral",
    "core_gn",
)
DEFAULT_TIMEOUT = 900


class _SnapshotCaptured(Exception):
    """Stop a capture-only fit immediately after the first statistic call."""


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, check=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _make_problem(seed: int):

    from ADP import ADP_Config
    from experiments.data import _generate_data, _make_seed_bundle
    from experiments.models import ExperimentPoint
    from experiments.runner import _effective_config

    point = ExperimentPoint(
        d=int(PARAMS["d"]),
        n_over_d=float(PARAMS["n"]) / float(PARAMS["d"]),
        sigma_x=1.0,
        sigma_eps=0.1,
        link="spokoini_m2",
        x_distribution="beta1_tau",
        noise_distribution="gaussian",
        mode="multi",
        index_dim=int(PARAMS["m"]),
        n_samples=int(PARAMS["n"]),
        tau=1.0,
        N_loc=int(PARAMS["N_loc"]),
        N_lin=int(PARAMS["N_lin"]),
        N_J=int(PARAMS["N_J"]),
        N_phi=int(PARAMS["N_phi"]),
        a=float(PARAMS["a"]),
        index_init="local",
        select_step="last",
        solver_max_steps=int(PARAMS["solver_max_steps"]),
    )
    seed_bundle = _make_seed_bundle("mi-spokoini-m2-dhigh", point, seed)
    generated = _generate_data("mi-spokoini-m2-dhigh", point, seed_bundle, seed)
    config = _effective_config(ADP_Config(), point, seed_bundle.init)
    # _effective_config returns a frozen config. The experiment point already
    # sets the full schedule, lambda, h_min, and selection rule.
    return generated.X, generated.Y, generated.beta, config


def _build_model(method_name: str, *, grassman_steps: int):
    from ADP import ADP_solver
    from ADP.solver.SVD import solve as solve_svd

    if method_name == BASELINE:
        return ADP_solver(
            solve_svd,
            rank=1,
            direct_max_dimension=128,
        )

    from ADP.solver.grassman import solve as solve_grassman

    if method_name == GRASSMAN:
        settings: dict[str, object] = {
            "method": "core_gn",
            "angle_backend": "schur",
            "max_steps": grassman_steps,
            "tol": 1e-6,
            "energy_tol": 0.05,
            "local_ridge": 0.0,
            "max_angle": 0.5,
            "workspace_bytes": 16 * 1024 * 1024,
        }
    elif method_name == "rank_one_refit":
        settings = {
            "method": "rank_one",
            "angle_backend": "refit",
            "rank": 1,
            "max_steps": grassman_steps,
            "tol": 1e-6,
            "energy_tol": 0.05,
            "local_ridge": 0.0,
            "max_angle": 0.5,
            "workspace_bytes": 16 * 1024 * 1024,
        }
    elif method_name == "rank_one_schur":
        settings = {
            "method": "rank_one",
            "angle_backend": "schur",
            "rank": 1,
            "max_steps": grassman_steps,
            "tol": 1e-6,
            "energy_tol": 0.05,
            "local_ridge": 0.0,
            "max_angle": 0.5,
            "workspace_bytes": 16 * 1024 * 1024,
        }
    elif method_name in {"spectral_rank1", "spectral"}:
        settings = {
            "method": "spectral",
            "angle_backend": "schur",
            "rank": 1 if method_name == "spectral_rank1" else None,
            "max_steps": grassman_steps,
            "tol": 1e-6,
            "energy_tol": 0.05,
            "local_ridge": 0.0,
            "max_angle": 0.5,
            "workspace_bytes": 16 * 1024 * 1024,
        }
    else:
        raise ValueError(f"unknown solver variant: {method_name}")
    # ADP_solver's constructor uses the name `method` for its callable; bind
    # Grassmann's algorithm selector into the callable to avoid that collision.
    return ADP_solver(partial(solve_grassman, **settings))


def _json_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fit_worker(
    seed: int,
    method_name: str,
    output: Path,
    *,
    grassman_steps: int,
    capture: bool = False,
) -> dict[str, object]:
    import numpy as np

    from ADP import ADP_multi_index

    X, y, true_basis, config = _make_problem(seed)
    solver = _build_model(method_name, grassman_steps=grassman_steps)
    captured: dict[str, np.ndarray | None] = {
        "index_init": None,
        "U": None,
        "I": None,
        "mass": None,
    }
    if capture:
        original = solver.method

        def capturing_method(index_init, U, I, *, mass=None, **kwargs):
            if captured["U"] is None:
                captured["index_init"] = np.array(index_init, copy=True)
                captured["U"] = np.array(U, copy=True)
                captured["I"] = np.array(I, copy=True)
                captured["mass"] = None if mass is None else np.array(mass, copy=True)
                raise _SnapshotCaptured
            return original(index_init, U, I, mass=mass, **kwargs)

        solver.method = capturing_method

    started = time.perf_counter()
    try:
        model = ADP_multi_index(int(PARAMS["m"]), config, solver).fit(X, y)
    except _SnapshotCaptured:
        if not capture:
            raise
        elapsed = time.perf_counter() - started
        if (
            captured["U"] is None
            or captured["I"] is None
            or captured["index_init"] is None
        ):
            raise RuntimeError(
                "capture hook stopped without complete statistics"
            ) from None
        snapshot_dir = output / "frozen"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = snapshot_dir / f"spokoini_seed{seed}_first_outer.npz"
        if snapshot_file.exists():
            raise FileExistsError(
                f"refusing to overwrite captured input {snapshot_file}"
            ) from None
        arrays = {key: value for key, value in captured.items() if value is not None}
        arrays["truth_basis"] = np.asarray(true_basis)
        np.savez(snapshot_file, **arrays)
        return {
            "case": CASE,
            "seed": seed,
            "solver": method_name,
            "status": "captured",
            "capture_time_sec": elapsed,
            "frozen_file": str(snapshot_file),
            "frozen_bytes": snapshot_file.stat().st_size,
            "frozen_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "frozen_dtype": str(arrays["U"].dtype),
            "capture_scope": "first outer solver call; fit stopped before solving",
        }
    else:
        elapsed = time.perf_counter() - started
    # Capture run RSS is retained for provenance only; full-fit runs do not
    # enable capture and are the only source for performance ratios.
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    basis = np.asarray(model.basis_)
    truth_projector = true_basis @ true_basis.T
    projector_distance = float(
        np.linalg.norm(basis @ basis.T - truth_projector, ord="fro")
        / np.sqrt(2 * int(PARAMS["m"]))
    )
    trace = model.trace_
    last_solver = trace[-1].get("solver", {}) if trace else {}
    # Each HPAOResult diagnostic's loss is the profiled local-refit objective.
    profile_loss = last_solver.get("profile_objective")
    row: dict[str, object] = {
        "case": CASE,
        "seed": seed,
        "solver": method_name,
        "status": "ok",
        "shape": {key: PARAMS[key] for key in ("n", "d", "m", "N_J", "N_phi", "N_lin")},
        "dtype": str(X.dtype),
        "fit_time_sec": elapsed,
        "peak_rss_kib": peak_rss,
        "projector_distance": projector_distance,
        "profile_objective_last_solver_call": _json_value(profile_loss),
        "outer_iterations": model.effective_parameters_.get("outer_iterations"),
        "selected_iteration": model.effective_parameters_.get("selected_iteration"),
        "stop_reason": model.result_.stop_reason,
        "solver_trace": _json_value([entry.get("solver", {}) for entry in trace]),
        "snapshot_capture": capture,
    }
    return row


def _profile_loss(index, U, I, mass) -> float:
    from ADP.solver.LSMR import _local_refit, _loss

    coefficients, _ = _local_refit(I, U, index)
    return float(_loss(I, U, index, coefficients, mass))


def _frozen_worker(
    file: Path, variant: str, *, grassman_steps: int
) -> dict[str, object]:
    import numpy as np

    from ADP.solver.LSMR import _local_refit

    with np.load(file, allow_pickle=False) as arrays:
        index_init = arrays["index_init"]
        U = arrays["U"]
        I = arrays["I"]
        mass = arrays["mass"] if "mass" in arrays.files else None
        truth_basis = arrays["truth_basis"]
    initial_loss = _profile_loss(index_init, U, I, mass)
    solver = _build_model(variant, grassman_steps=grassman_steps)
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    method_kwargs: dict[str, object] = {"mass": mass, "lambda_prox": 0.05}
    if variant == BASELINE:
        # ADP's SVD primitive requires its rank as a keyword-only argument;
        # unlike Grassmann variants it does not receive rank from a closure.
        method_kwargs["rank"] = 1
    result = solver.method(index_init, U, I, **method_kwargs)
    elapsed = time.perf_counter() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    index = np.asarray(result.index)
    loss = _profile_loss(index, U, I, mass)
    m = len(index)
    projector_distance = float(
        np.linalg.norm(index.T @ index - truth_basis @ truth_basis.T, ord="fro")
        / np.sqrt(2 * m)
    )
    orthogonality = float(np.linalg.norm(index @ index.T - np.eye(m), ord="fro"))
    coefficients, ranks = _local_refit(I, U, index)
    del coefficients
    diagnostics = _json_value(result.diagnostics)
    return {
        "case": CASE,
        "seed": int(file.stem.split("seed", 1)[1].split("_", 1)[0]),
        "variant": variant,
        "status": "ok",
        "frozen_file": str(file),
        "frozen_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "shape": {
            "J": int(U.shape[0]),
            "p": int(U.shape[1]),
            "d": int(U.shape[2]),
            "m": int(index.shape[0]),
        },
        "dtype": str(U.dtype),
        "time_sec": elapsed,
        "peak_rss_kib": peak_rss,
        "additional_peak_rss_kib": max(0, peak_rss - rss_before),
        "profile_loss_initial": initial_loss,
        "profile_loss_final": loss,
        "profile_loss_delta": loss - initial_loss,
        "projector_distance": projector_distance,
        "orthogonality_error": orthogonality,
        "local_rank_loss_count": int(np.count_nonzero(ranks < m)),
        "diagnostics": diagnostics,
    }


def _curve_worker(file: Path) -> dict[str, object]:
    """Compare Schur angle values with independent full local refits."""
    import numpy as np

    from ADP.solver.grassman import _AngleProfile, _gradient, _normalize_index, _profile

    repeats = 7
    angles = np.linspace(0.0, 0.5, 61)
    with np.load(file, allow_pickle=False) as arrays:
        index_init = arrays["index_init"]
        U = np.ascontiguousarray(arrays["U"])
        I = arrays["I"]
        mass = arrays["mass"] if "mass" in arrays.files else None
    if mass is None:
        mass = np.ones(len(I))
    Y = _normalize_index(index_init).T
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    left, _, right = np.linalg.svd(G, full_matrices=False)
    a = right[0]
    v = left[:, 0]
    if float((v @ G) @ a) > 0:
        v = -v
    w = (U @ v)[:, :, None][:, :, 0]

    def shared_projection() -> tuple[np.ndarray, np.ndarray]:
        """Common operator work required before either angle-search path."""
        return U @ Y, U @ v

    # Measure the shared U[Y,v] operator actions independently, then add their
    # median cost to each warmed 61-angle batch for a like-for-like total.
    shared_projection()
    projection_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        shared_projection()
        projection_times.append(time.perf_counter() - started)
    shared_projection_sec = float(np.median(projection_times))

    def schur_curve() -> np.ndarray:
        curve = _AngleProfile(state.M, w, a, I, mass, 0.0)
        return np.asarray([curve.value(float(t)) for t in angles])

    def refit_curve() -> np.ndarray:
        values = np.empty(len(angles))
        Ma = state.M @ a
        for i, theta in enumerate(angles):
            delta = Ma * (np.cos(theta) - 1.0) + w * np.sin(theta)
            M_new = state.M + delta[:, :, None] * a
            values[i] = _profile(M_new, I, mass, 0.0).value
        return values

    # Warm each code path without including setup in the measured repetitions.
    schur_curve()
    refit_curve()
    schur_times: list[float] = []
    refit_times: list[float] = []
    max_abs_error = 0.0
    max_relative_error = 0.0
    for repeat in range(repeats):
        if repeat % 2:
            started = time.perf_counter()
            refit = refit_curve()
            refit_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            schur = schur_curve()
            schur_times.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            schur = schur_curve()
            schur_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            refit = refit_curve()
            refit_times.append(time.perf_counter() - started)
        delta = np.abs(schur - refit)
        max_abs_error = max(max_abs_error, float(delta.max()))
        max_relative_error = max(
            max_relative_error,
            float(np.max(delta / np.maximum(1.0, np.abs(refit)))),
        )
    return {
        "case": CASE,
        "seed": int(file.stem.split("seed", 1)[1].split("_", 1)[0]),
        "status": "ok",
        "frozen_file": str(file),
        "frozen_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "shape": {
            "J": int(U.shape[0]),
            "p": int(U.shape[1]),
            "d": int(U.shape[2]),
            "m": int(Y.shape[1]),
        },
        "angle_grid": len(angles),
        "angle_range": [float(angles[0]), float(angles[-1])],
        "repeats": repeats,
        "measurement": (
            "cached angle-curve setup plus 61 evaluations; seven warmed paired "
            "repeats, alternating order. Excludes the separately timed shared "
            "U@Y and U@v operator projections. *_with_shared_projection_sec "
            "adds their median cost once to one curve batch."
        ),
        "schur_median_sec": float(np.median(schur_times)),
        "refit_median_sec": float(np.median(refit_times)),
        "shared_projection_median_sec": shared_projection_sec,
        "schur_with_shared_projection_sec": float(np.median(schur_times))
        + shared_projection_sec,
        "refit_with_shared_projection_sec": float(np.median(refit_times))
        + shared_projection_sec,
        "schur_over_refit_with_shared_projection_ratio": (
            float(np.median(schur_times)) + shared_projection_sec
        )
        / (float(np.median(refit_times)) + shared_projection_sec),
        "schur_over_refit_ratio": float(
            np.median(schur_times) / np.median(refit_times)
        ),
        "max_absolute_profile_error": max_abs_error,
        "max_scaled_profile_error": max_relative_error,
    }


def _child(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "fullfit":
        row = _fit_worker(
            args.seed, args.variant, args.output, grassman_steps=args.grassman_steps
        )
    elif args.stage == "capture":
        row = _fit_worker(
            args.seed,
            args.variant,
            args.output,
            grassman_steps=args.grassman_steps,
            capture=True,
        )
    elif args.stage == "frozen":
        row = _frozen_worker(
            args.frozen_file, args.variant, grassman_steps=args.grassman_steps
        )
    else:
        row = _curve_worker(args.frozen_file)
    row["worker_version"] = 1
    if args.worker_output:
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(row, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(row, allow_nan=False))
    return 0


def _run_process(
    command: list[str], env: dict[str, str], timeout: int
) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=timeout
        )
        if completed.returncode:
            return {"status": "process_error", "error": completed.stderr[-4000:]}
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "error": f"worker exceeded {timeout}s: {error}"}
    except (IndexError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_worker_output",
            "error": f"{type(error).__name__}: {error}",
        }


def _paired_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    import numpy as np

    successful = [row for row in rows if row.get("status") == "ok"]
    by_seed_solver = {(int(row["seed"]), str(row["solver"])): row for row in successful}
    pairs = []
    seeds = sorted({int(row["seed"]) for row in rows if "seed" in row})
    for seed in seeds:
        svd = by_seed_solver.get((seed, BASELINE))
        grassman = by_seed_solver.get((seed, GRASSMAN))
        if svd is None or grassman is None:
            continue
        pairs.append(
            {
                "seed": seed,
                "time_ratio_grassman_over_svd": float(grassman["fit_time_sec"])
                / float(svd["fit_time_sec"]),
                "projector_delta_grassman_minus_svd": float(
                    grassman["projector_distance"]
                )
                - float(svd["projector_distance"]),
                "rss_delta_kib_grassman_minus_svd": int(grassman["peak_rss_kib"])
                - int(svd["peak_rss_kib"]),
            }
        )
    return {
        "successful_pairs": len(pairs),
        "failed_runs": [row for row in rows if row.get("status") != "ok"],
        "paired": pairs,
        "median_time_ratio": float(
            np.median([x["time_ratio_grassman_over_svd"] for x in pairs])
        )
        if pairs
        else None,
        "median_projector_delta": float(
            np.median([x["projector_delta_grassman_minus_svd"] for x in pairs])
        )
        if pairs
        else None,
        "median_rss_delta_kib": float(
            np.median([x["rss_delta_kib_grassman_minus_svd"] for x in pairs])
        )
        if pairs
        else None,
    }


def _frozen_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    import numpy as np

    successful = [row for row in rows if row.get("status") == "ok"]
    by_seed_variant = {
        (int(row["seed"]), str(row["variant"])): row for row in successful
    }
    comparisons = []
    seeds = sorted({int(row["seed"]) for row in rows if "seed" in row})
    for seed in seeds:
        reference = by_seed_variant.get((seed, BASELINE))
        if reference is None:
            continue
        for variant in FROZEN_VARIANTS:
            if variant == BASELINE:
                continue
            candidate = by_seed_variant.get((seed, variant))
            if candidate is None:
                continue
            comparisons.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "time_ratio_over_svd": float(candidate["time_sec"])
                    / float(reference["time_sec"]),
                    "profile_loss_delta_vs_svd": float(candidate["profile_loss_final"])
                    - float(reference["profile_loss_final"]),
                    "projector_delta_vs_svd": float(candidate["projector_distance"])
                    - float(reference["projector_distance"]),
                    "rss_delta_kib_vs_svd": int(candidate["peak_rss_kib"])
                    - int(reference["peak_rss_kib"]),
                }
            )
    by_variant = {
        variant: [row for row in comparisons if row["variant"] == variant]
        for variant in FROZEN_VARIANTS
        if variant != BASELINE
    }
    return {
        "successful_runs": len(successful),
        "failed_runs": [row for row in rows if row.get("status") != "ok"],
        "paired_vs_svd": comparisons,
        "median_by_variant": {
            variant: {
                field: float(np.median([row[field] for row in values]))
                if values
                else None
                for field in (
                    "time_ratio_over_svd",
                    "profile_loss_delta_vs_svd",
                    "projector_delta_vs_svd",
                    "rss_delta_kib_vs_svd",
                )
            }
            for variant, values in by_variant.items()
        },
    }


def _curve_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    import numpy as np

    successful = [row for row in rows if row.get("status") == "ok"]
    return {
        "successful_seeds": [int(row["seed"]) for row in successful],
        "failed_runs": [row for row in rows if row.get("status") != "ok"],
        "median_schur_over_refit_ratio": float(
            np.median([row["schur_over_refit_ratio"] for row in successful])
        )
        if successful
        else None,
        "median_schur_over_refit_with_shared_projection_ratio": float(
            np.median(
                [
                    row["schur_over_refit_with_shared_projection_ratio"]
                    for row in successful
                ]
            )
        )
        if successful
        else None,
        "median_shared_projection_sec": float(
            np.median([row["shared_projection_median_sec"] for row in successful])
        )
        if successful
        else None,
        "max_absolute_profile_error": max(
            (float(row["max_absolute_profile_error"]) for row in successful),
            default=None,
        ),
        "max_scaled_profile_error": max(
            (float(row["max_scaled_profile_error"]) for row in successful),
            default=None,
        ),
        "per_seed": [
            {
                "seed": row["seed"],
                "ratio": row["schur_over_refit_ratio"],
                "absolute_error": row["max_absolute_profile_error"],
                "scaled_error": row["max_scaled_profile_error"],
            }
            for row in successful
        ],
    }


def _run_stage(args: argparse.Namespace) -> None:
    import numpy as np
    import scipy

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    stage_file = output / f"{args.stage}_runs.jsonl"
    existing_rows = (
        [json.loads(line) for line in stage_file.read_text().splitlines()]
        if stage_file.exists()
        else []
    )
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    variants = (
        (GRASSMAN,)
        if args.stage == "capture"
        else (BASELINE, GRASSMAN)
        if args.stage == "fullfit"
        else FROZEN_VARIANTS
    )
    seeds = tuple(args.seeds)
    tasks: list[tuple[int, str, Path | None]] = []
    if args.stage in {"frozen", "curves"}:
        frozen_files = sorted(
            (output / "frozen").glob("spokoini_seed*_first_outer.npz")
        )
        if not frozen_files:
            raise FileNotFoundError(
                "no captured frozen input; run --stage capture first"
            )
        for file in frozen_files:
            seed = int(file.stem.split("seed", 1)[1].split("_", 1)[0])
            order = (
                ("curve_check",)
                if args.stage == "curves"
                else variants
                if seed % 2
                else tuple(reversed(variants))
            )
            tasks.extend((seed, variant, file) for variant in order)
    elif args.stage == "fullfit":
        for seed in seeds:
            order = variants if seed % 2 else tuple(reversed(variants))
            tasks.extend((seed, variant, None) for variant in order)
    else:
        tasks = [(seed, variant, None) for seed in seeds for variant in variants]

    rows = list(existing_rows)
    completed = {
        (int(row.get("seed", -1)), str(row.get("variant", row.get("solver", ""))))
        for row in existing_rows
    }
    for seed, variant, frozen_file in tasks:
        if (seed, variant) in completed:
            continue
        worker_output = output / "workers" / f"{args.stage}_seed{seed}_{variant}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.grassman_benchmark",
            "--child",
            "--stage",
            args.stage,
            "--seed",
            str(seed),
            "--variant",
            variant,
            "--output",
            str(output),
            "--worker-output",
            str(worker_output),
            "--grassman-steps",
            str(args.grassman_steps),
        ]
        if frozen_file is not None:
            command.extend(("--frozen-file", str(frozen_file)))
        row = _run_process(command, env, args.timeout)
        row.setdefault("seed", seed)
        row.setdefault("variant" if args.stage == "frozen" else "solver", variant)
        rows.append(row)
        with stage_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            f"{args.stage}: seed={seed} variant={variant} status={row.get('status')}",
            flush=True,
        )

    meta = {
        "stage": args.stage,
        "case": CASE,
        "seeds": list(seeds),
        "variants": variants,
        "params": PARAMS,
        "grassman_steps": args.grassman_steps,
        "float_dtype": "float64",
        "thread_env": {
            name: env[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--porcelain"])),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "solver_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("ADP/solver/grassman.py", "ADP/solver/SVD.py")
        },
        "protocol": (
            "Full-fit stage: cold ADP_multi_index.fit in isolated processes, "
            "one process per method/seed; "
            "one BLAS thread, perf_counter around fit, process ru_maxrss, no tracing. "
            "Capture is separate and excluded from time aggregates. Frozen stage "
            "runs one HPAOResult call per variant/process on captured first-outer "
            "statistics and externally evaluates minimum-norm local-refit profile loss."
        ),
        "objective_note": (
            "Grassmann/profile methods use common profiled local-refit loss. "
            "Current SVD has a different fixed-g objective; its returned index is "
            "re-evaluated with the common profile. Full-fit recovery is descriptive."
        ),
        "memory_note": (
            "ru_maxrss KiB includes interpreter/imports and loaded data; frozen "
            "RSS also includes the input arrays."
        ),
        "timeout_sec": args.timeout,
        "command": " ".join(sys.argv),
        "results_count": len(rows),
    }
    (output / f"{args.stage}_protocol.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    if args.stage == "fullfit":
        (output / "fullfit_summary.json").write_text(
            json.dumps(_paired_summary(rows), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif args.stage == "frozen":
        (output / "frozen_summary.json").write_text(
            json.dumps(_frozen_summary(rows), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif args.stage == "curves":
        (output / "curves_summary.json").write_text(
            json.dumps(_curve_summary(rows), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("fullfit", "capture", "frozen", "curves"), required=True
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/grassman_2026_10_01/benchmark")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single worker seed; orchestrated stages use --seeds.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Fullfit/capture seed list; stages resume missing pairs safely.",
    )
    parser.add_argument("--grassman-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--frozen-file", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.child:
        if not args.variant:
            parser.error("--child requires --variant")
        try:
            status = _child(args)
        except Exception as error:
            failure = {
                "status": "error",
                "seed": args.seed,
                "variant": args.variant,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=12),
            }
            if args.worker_output:
                args.worker_output.parent.mkdir(parents=True, exist_ok=True)
                args.worker_output.write_text(
                    json.dumps(failure) + "\n", encoding="utf-8"
                )
            print(json.dumps(failure))
            status = 0
        raise SystemExit(status)
    _run_stage(args)


if __name__ == "__main__":
    main()
