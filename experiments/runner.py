from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import platform
import subprocess
import traceback
import warnings
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import asdict, fields, replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import subspace_angles
from tqdm import tqdm

from ADP.cli.experiment_utils import (
    validate_experiment_id,
    validate_run_parameters,
)
from ADP.cli.main import _run, _validate_solver_mode, build_parser
from ADP.core.ADP_Config import ADP_Config

from .data import _generate_data, _GeneratedData, _make_seed_bundle, _SeedBundle
from .models import Build, Experiment, ExperimentPoint

_RUN_COLUMNS = (
    "experiment",
    "title",
    "point",
    "point_label",
    "run",
    "seed",
    "seed_bundle",
    "model_seed",
    "build",
    "order",
    "n",
    "noise_dimension_condition",
    *tuple(field.name for field in fields(ExperimentPoint)),
    "requested_config",
    "effective_config",
    "status",
    "error",
    "quality_metric",
    "quality_direction",
    "convergence_pass",
    "quality_pass",
    "recovered",
    "failure_mode",
    "quality",
    "cosine_abs",
    "trace_score",
    "projector_distance",
    "max_principal_sine",
    "max_principal_angle_deg",
    "initial_quality",
    "last_quality",
    "initial_eigenvalues",
    "fit_time_sec",
    "max_stage_traced_peak_mib",
    "outer_iterations",
    "stop_reason",
    "selected_iteration",
    "selected_error",
    "prediction_rmse",
    "trace",
    "solver_diagnostics",
)


def run_experiment(
    experiment: Experiment,
    a: Build,
    b: Build | None = None,
    *,
    profile: str = "smoke",
    runs: int | None = None,
    seed: int = 0,
    output_dir: str | Path = "benchmark_outputs/experiments",
    plots: bool = True,
    experiment_id: str | None = None,
    progress: bool = False,
) -> Path:
    """Последовательно выполнить один ADP build или парный A/B-запуск."""
    if runs is None:
        runs = (
            5
            if profile == "overview"
            else (experiment.full_runs if profile == "full" else 1)
        )
    validate_run_parameters(a.name, None if b is None else b.name, runs, seed)
    points = experiment.points(profile)
    builds = (a,) if b is None else (a, b)
    for mode in {point.mode for point in points}:
        for build in builds:
            _validate_solver_mode(mode, build.solver)
    experiment_id = _experiment_id(experiment_id)
    series_dir = _series_directory(Path(output_dir), experiment.selector, experiment_id)
    _write_manifest(series_dir, experiment, builds, profile, runs, seed, experiment_id)
    runs_path = series_dir / "runs.csv"

    labels = _point_labels(points, experiment.report_fields)
    with tqdm(
        total=len(points) * runs * len(builds),
        desc=experiment.selector,
        unit="fit",
        disable=not progress,
    ) as progress_bar:
        for point_index, point in enumerate(points):
            for run_index in range(runs):
                run_seed = seed + run_index
                seeds = _make_seed_bundle(
                    experiment.selector,
                    point,
                    run_seed,
                    common_random_fields=experiment.common_random_fields,
                )
                order = builds
                if len(builds) == 2 and (point_index + run_index) % 2:
                    order = builds[::-1]
                try:
                    generated = _generate_data(
                        experiment.selector, point, seeds, run_seed
                    )
                except Exception as error:
                    for order_index, build in enumerate(order):
                        row = _base_row(
                            experiment,
                            point,
                            labels[point_index],
                            point_index,
                            run_index,
                            run_seed,
                            seeds,
                            build,
                            order_index,
                        )
                        row.update(
                            status="numerical_failure",
                            error=_error_text(error),
                            **_outcome_fields(
                                "numerical_failure",
                                False,
                                None,
                                experiment.quality_threshold,
                                "higher",
                            ),
                        )
                        _append_row(runs_path, row)
                        progress_bar.update()
                    continue

                for order_index, build in enumerate(order):
                    row = _base_row(
                        experiment,
                        point,
                        labels[point_index],
                        point_index,
                        run_index,
                        run_seed,
                        seeds,
                        build,
                        order_index,
                    )
                    try:
                        row.update(
                            _fit(
                                build,
                                point,
                                generated,
                                seeds.init,
                                experiment.quality_threshold,
                            )
                        )
                    except Exception as error:  # Один fit не отменяет серию.
                        row.update(
                            status="numerical_failure",
                            error=_error_text(error),
                            **_outcome_fields(
                                "numerical_failure",
                                False,
                                None,
                                experiment.quality_threshold,
                                "higher",
                            ),
                        )
                    _append_row(runs_path, row)
                    progress_bar.update()

    from ADP.cli.experiment_plots import build_report

    build_report(series_dir, plots=plots)
    return series_dir


def _fit(
    build: Build,
    point: ExperimentPoint,
    data: _GeneratedData,
    model_seed: int,
    quality_threshold: float | None,
) -> dict[str, object]:
    config = _effective_config(build.config, point, model_seed)
    args = _arguments(build, config, point)
    started = perf_counter()
    injected_basis = data.beta[:, None] if data.beta.ndim == 1 else data.beta
    index, true_basis, profile, metadata = _run(
        args,
        data=(data.X, data.Y, injected_basis),
    )
    elapsed = perf_counter() - started
    diagnostics = metadata.get("diagnostics", {})
    converged = isinstance(diagnostics, dict) and diagnostics.get("converged")
    effective = {
        **_config_spec(config),
        "solver": metadata["solver"],
        "solver_tol": build.solver_tol,
        "solver_max_steps": metadata["solver_max_steps"],
        "theta": build.theta
        if point.mode != "manifold" and metadata["solver"] in {"lsmr", "hybrid"}
        else None,
        "trust_radius": build.trust_radius
        if point.mode != "manifold" and metadata["solver"] in {"lsmr", "hybrid"}
        else None,
        "lsmr_maxiter": build.lsmr_maxiter
        if point.mode != "manifold" and metadata["solver"] in {"lsmr", "hybrid"}
        else None,
        "cg_maxiter": build.cg_maxiter
        if point.mode == "manifold" or metadata["solver"] == "cg"
        else None,
        "N_lin": metadata["N_lin"],
        "N_J": metadata["N_J"],
        "N_phi": metadata["N_phi"],
        "training_size": metadata["training_size"],
        "center_displacement_scale": metadata["center_displacement_scale"],
    }
    if point.mode == "manifold":
        effective.update(
            N_manifold=metadata["N_manifold"],
            sync_steps=metadata["sync_steps"],
            lambda_manifold=metadata["lambda_manifold"],
        )
    if point.mode == "single":
        quality_metric = "cosine_abs"
        quality_direction = "higher"
        quality = float(abs(true_basis[:, 0] @ index))
        cosine = quality
        trace_score: float | None = None
        projector_distance: float | None = None
        max_principal_sine: float | None = None
        max_principal_angle_deg: float | None = None
    elif point.mode == "multi":
        quality_metric = "trace_score"
        quality_direction = "higher"
        estimate = index.T
        (
            quality,
            projector_distance,
            max_principal_sine,
            max_principal_angle_deg,
        ) = _subspace_metrics(true_basis, estimate)
        cosine = None
        trace_score = quality
    else:
        quality_metric = "local_projector_distance"
        quality_direction = "lower"
        quality, max_principal_sine, max_principal_angle_deg = _local_subspace_metrics(
            true_basis, index
        )
        cosine = None
        trace_score = None
        projector_distance = quality
    metrics = (quality, max_principal_sine, max_principal_angle_deg)
    if any(value is not None and not np.isfinite(value) for value in metrics):
        raise RuntimeError("quality metric is not finite")
    status = "nonconverged" if converged is False else "success"
    return {
        "effective_config": _compact_json(effective),
        "status": status,
        "quality_metric": quality_metric,
        "quality_direction": quality_direction,
        **_outcome_fields(
            status,
            converged,
            quality,
            quality_threshold,
            quality_direction,
        ),
        "quality": quality,
        "cosine_abs": cosine,
        "trace_score": trace_score,
        "projector_distance": projector_distance,
        "max_principal_sine": max_principal_sine,
        "max_principal_angle_deg": max_principal_angle_deg,
        "initial_quality": metadata["initial_quality"],
        "last_quality": metadata["last_quality"],
        "initial_eigenvalues": _compact_json(metadata["initial_eigenvalues"]),
        "fit_time_sec": elapsed,
        "max_stage_traced_peak_mib": profile["total"]["traced_peak_bytes"] / 2**20,
        "outer_iterations": metadata["outer_iterations"],
        "stop_reason": metadata["stop_reason"],
        "selected_iteration": metadata["selected_iteration"],
        "selected_error": metadata["selected_error"],
        "prediction_rmse": metadata.get("prediction_rmse"),
        "trace": _compact_json(metadata["trace"]),
        "solver_diagnostics": _compact_json(diagnostics),
    }


def _subspace_metrics(
    true_basis: np.ndarray,
    estimate: np.ndarray,
) -> tuple[float, float, float, float]:
    """Вернуть trace-score, ошибку проектора, худший синус и угол.

    Требуются конечные ортонормированные базисы одинаковой формы.
    Основная метрика равна ``mean(cos(theta_j)**2)`` — нормированному
    следу произведения проекторов. Вычисление не строит проекторы ``d x d``.
    """
    if (
        true_basis.ndim != 2
        or estimate.ndim != 2
        or true_basis.shape != estimate.shape
        or true_basis.shape[1] < 1
    ):
        raise ValueError("subspace bases must have the same nonempty (d, m) shape")
    identity = np.eye(true_basis.shape[1])
    if not np.allclose(true_basis.T @ true_basis, identity, rtol=1e-7, atol=1e-8):
        raise RuntimeError("true subspace basis is not orthonormal")
    if not np.allclose(estimate.T @ estimate, identity, rtol=1e-7, atol=1e-8):
        raise RuntimeError("estimated subspace basis is not orthonormal")
    angles = subspace_angles(true_basis, estimate)
    sines = np.sin(angles)
    # ESTIMATOR/evaluation protocol: tr(P_hat P_true) / m.
    overlap = true_basis.T @ estimate
    trace_score = float(np.square(overlap).sum() / true_basis.shape[1])
    projector_distance = float(np.sum(np.square(sines)))
    max_principal_sine = float(np.max(sines))
    max_principal_angle_deg = math.degrees(float(np.max(angles)))
    return trace_score, projector_distance, max_principal_sine, max_principal_angle_deg


def _local_subspace_metrics(
    true_projectors: np.ndarray,
    estimate: np.ndarray,
) -> tuple[float, float, float]:
    """Сравнить семейства локальных row-projectors формы ``(J, m, d)``."""
    if (
        true_projectors.ndim != 3
        or estimate.shape != true_projectors.shape
        or true_projectors.shape[1] < 1
    ):
        raise ValueError("local projectors must have the same nonempty (J, m, d) shape")
    identity = np.broadcast_to(
        np.eye(true_projectors.shape[1]),
        (*true_projectors.shape[:2], true_projectors.shape[1]),
    )
    for name, projectors in (
        ("true", true_projectors),
        ("estimated", estimate),
    ):
        gram = projectors @ np.swapaxes(projectors, 1, 2)
        if not np.allclose(gram, identity, rtol=1e-7, atol=1e-8):
            raise RuntimeError(f"{name} local projectors are not orthonormal")
    overlaps = np.einsum("jmd,jnd->jmn", estimate, true_projectors, optimize=True)
    singular_values = np.linalg.svd(overlaps, compute_uv=False)
    sines = np.sqrt(np.maximum(0.0, 1.0 - np.square(singular_values)))
    quality = float(np.sqrt(np.mean(np.square(sines))))
    max_sine = float(np.max(sines))
    return quality, max_sine, math.degrees(math.asin(min(1.0, max_sine)))


def _effective_config(
    requested: ADP_Config,
    point: ExperimentPoint,
    model_seed: int,
) -> ADP_Config:
    overrides = _point_config_overrides(point)
    if point.h_min_factor is not None:
        overrides["h_min"] = point.h_min_factor * point.sigma_x / math.sqrt(point.n)
    requested = replace(requested, **overrides)
    n_loc = min(requested.N_loc, point.n)
    index_init = "random" if point.n <= point.d + 1 else requested.index_init
    n_lin = requested.N_lin or 2 * point.d
    if index_init == "local":
        n_lin = min(max(n_lin, point.d + 2), point.n)
    else:
        n_lin = min(max(n_lin, 1), point.n)
    minimum_centers = math.ceil(point.n / n_loc)
    n_centers = min(max(requested.N_J or point.n, minimum_centers), point.n)
    n_directions = requested.N_phi or max(
        point.index_dim + (point.mode == "multi"),
        min(n_loc, point.d),
    )
    return replace(
        requested,
        seed=model_seed,
        N_loc=n_loc,
        N_lin=n_lin,
        N_J=n_centers,
        N_phi=n_directions,
        index_init=index_init,
    )


def _point_config_overrides(point: ExperimentPoint) -> dict[str, object]:
    return {
        name: value
        for name in (
            "N_loc",
            "N_lin",
            "N_J",
            "N_phi",
            "outer_steps",
            "lambda_penalty",
            "a",
            "index_init",
            "direction_mode",
            "multi_tensor",
            "select_step",
            "center_displacement",
            "training_set",
            "redraw_directions",
        )
        if (value := getattr(point, name)) is not None
    }


def _arguments(
    build: Build,
    config: ADP_Config,
    point: ExperimentPoint,
) -> argparse.Namespace:
    args = build_parser().parse_args([])
    args.mode = point.mode
    args.n = point.n
    args.d = point.d
    args.index_dim = point.index_dim
    args.noise = point.sigma_eps
    args.displacement_scale = point.sigma_x
    for item in fields(config):
        setattr(args, item.name, getattr(config, item.name))
    args.solver_tol = build.solver_tol
    args.solver_max_steps = point.solver_max_steps or build.solver_max_steps
    args.solver = build.solver
    args.theta = build.theta
    args.trust_radius = build.trust_radius
    args.lsmr_maxiter = build.lsmr_maxiter
    args.cg_maxiter = build.cg_maxiter
    args.N_manifold = point.N_manifold
    if point.sync_steps is not None:
        args.sync_steps = point.sync_steps
    if point.lambda_manifold is not None:
        args.lambda_manifold = point.lambda_manifold
    return args


def _base_row(
    experiment: Experiment,
    point: ExperimentPoint,
    label: str,
    point_index: int,
    run_index: int,
    run_seed: int,
    seeds: _SeedBundle,
    build: Build,
    order: int,
) -> dict[str, object]:
    requested_config = replace(build.config, **_point_config_overrides(point))
    if point.h_min_factor is not None:
        requested_config = replace(
            requested_config,
            h_min=point.h_min_factor * point.sigma_x / math.sqrt(point.n),
        )
    requested: dict[str, object] = {
        "config": _config_spec(requested_config),
        "solver_tol": build.solver_tol,
        "solver_max_steps": point.solver_max_steps or build.solver_max_steps,
        "solver": build.solver,
        "theta": build.theta,
        "trust_radius": build.trust_radius,
        "lsmr_maxiter": build.lsmr_maxiter,
        "cg_maxiter": build.cg_maxiter,
    }
    if point.mode == "manifold":
        requested["manifold"] = {
            "N_manifold": point.N_manifold,
            "sync_steps": point.sync_steps,
            "lambda_manifold": point.lambda_manifold,
        }
    return {
        "experiment": experiment.selector,
        "title": experiment.title,
        "point": point_index,
        "point_label": label,
        "run": run_index,
        "seed": run_seed,
        "seed_bundle": _compact_json(asdict(seeds)),
        "model_seed": seeds.init,
        "build": build.name,
        "order": order,
        "n": point.n,
        "noise_dimension_condition": (
            point.sigma_eps == 0 or point.n / point.sigma_eps**2 >= 20 * point.d
        ),
        **asdict(point),
        "requested_config": _compact_json(requested),
        "status": "",
        "error": "",
        "quality_metric": (
            "cosine_abs"
            if point.mode == "single"
            else (
                "local_projector_distance"
                if point.mode == "manifold"
                else "trace_score"
            )
        ),
        "quality_direction": (
            "higher" if point.mode in {"single", "multi"} else "lower"
        ),
    }


def _append_row(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RUN_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_manifest(
    series_dir: Path,
    experiment: Experiment,
    builds: tuple[Build, ...],
    profile: str,
    runs: int,
    seed: int,
    experiment_id: str,
) -> None:
    manifest = {
        "schema_version": 9,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "experiment": experiment.selector,
        "title": experiment.title,
        "hypothesis": experiment.hypothesis,
        "profile": profile,
        "points": [asdict(point) for point in experiment.points(profile)],
        "runs": runs,
        "full_runs": experiment.full_runs,
        "report_fields": experiment.report_fields,
        "condition_field": experiment.condition_field,
        "common_random_fields": experiment.common_random_fields,
        "condition_group_fields": experiment.condition_group_fields,
        "recovery": (
            None
            if experiment.quality_threshold is None
            else {
                "metric": (
                    "cosine_abs"
                    if experiment.smoke.mode == "single"
                    else (
                        "local_projector_distance"
                        if experiment.smoke.mode == "manifold"
                        else "trace_score"
                    )
                ),
                "direction": (
                    "higher"
                    if experiment.smoke.mode in {"single", "multi"}
                    else "lower"
                ),
                "threshold": experiment.quality_threshold,
            }
        ),
        "seed": seed,
        "builds": [_build_spec(build) for build in builds],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "python": platform.python_version(),
        "dtype": "float64",
        "packages": {
            name: _package_version(name) for name in ("numpy", "scipy", "matplotlib")
        },
        "numpy_config": _numpy_config(),
        "threadpools": _threadpool_info(),
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed_design": "split-components-paired-builds-and-condition-levels-v4",
        "feature_formula": (
            "sigma_x * (tau * z0 + (1 - tau) * zi)"
            if any(point.tau is not None for point in experiment.full)
            else "catalog-specific legacy design"
        ),
        "multi_link_extension": "sum_{r=3}^m z_r^2 / r",
        "manifold_link": "0.5 * (x_1^2 + x_2^2)",
        "subspace_metrics": {
            "trace_score": "trace(P_hat @ P_true) / m = mean(cos(theta_j)^2)",
            "projector_distance": "sum(sin(theta_j)^2)",
            "max_principal_sine": "max(sin(theta_j))",
            "max_principal_angle_deg": "max(theta_j) * 180 / pi",
        },
    }
    (series_dir / "series.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_spec(build: Build) -> dict[str, object]:
    return {
        "name": build.name,
        "config": _config_spec(build.config),
        "solver": build.solver,
        "solver_tol": build.solver_tol,
        "solver_max_steps": build.solver_max_steps,
        "theta": build.theta,
        "trust_radius": build.trust_radius,
        "lsmr_maxiter": build.lsmr_maxiter,
        "cg_maxiter": build.cg_maxiter,
    }


def _config_spec(config: ADP_Config) -> dict[str, object]:
    return {
        item.name: _callable_name(value) if callable(value) else value
        for item in fields(config)
        if (value := getattr(config, item.name)) is not None
    }


def _outcome_fields(
    status: str,
    converged: object,
    quality: float | None,
    threshold: float | None,
    direction: str,
) -> dict[str, object]:
    """Классифицировать численный и статистический исход одного fit."""
    if threshold is None:
        return {
            "convergence_pass": None,
            "quality_pass": None,
            "recovered": None,
            "failure_mode": None,
        }
    convergence_pass = status != "numerical_failure" and converged is True
    quality_pass = None
    if quality is not None and np.isfinite(quality):
        quality_pass = (
            quality >= threshold if direction == "higher" else quality <= threshold
        )
    recovered = convergence_pass and quality_pass is True
    if status == "numerical_failure" or quality_pass is None:
        failure_mode = "numerical_failure"
    elif not convergence_pass:
        failure_mode = "nonconverged"
    elif not quality_pass:
        failure_mode = "converged_bad_quality"
    else:
        failure_mode = "recovered"
    return {
        "convergence_pass": convergence_pass,
        "quality_pass": quality_pass,
        "recovered": recovered,
        "failure_mode": failure_mode,
    }


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{name}"


def _experiment_id(value: str | None = None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return validate_experiment_id(value)


def _series_directory(output_dir: Path, selector: str, experiment_id: str) -> Path:
    path = output_dir / experiment_id / selector.replace(".", "_")
    path.mkdir(parents=True)
    return path


def _point_labels(
    points: tuple[ExperimentPoint, ...],
    report_fields: tuple[str, ...] = (),
) -> tuple[str, ...]:
    varying = report_fields or tuple(
        item.name
        for item in fields(ExperimentPoint)
        if len({getattr(point, item.name) for point in points}) > 1
    )
    if not varying:
        varying = ("d", "n_over_d")
    return tuple(
        ", ".join(f"{name}={getattr(point, name)}" for name in varying)
        for point in points
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _numpy_config() -> object:
    show = np.__config__.show
    try:
        return show(mode="dicts")
    except TypeError:  # NumPy 1.x не поддерживает mode="dicts".
        stream = StringIO()
        with warnings.catch_warnings(), redirect_stdout(stream):
            warnings.simplefilter("ignore")
            show()
        return stream.getvalue()


def _threadpool_info() -> object:
    try:
        threadpoolctl = importlib.import_module("threadpoolctl")
    except ImportError:
        return []
    return threadpoolctl.threadpool_info()


def _error_text(error: Exception) -> str:
    detail = "".join(traceback.format_exception_only(type(error), error)).strip()
    return detail.replace("\n", " ")


def _has_numerical_failures(series_dir: Path) -> bool:
    with (series_dir / "runs.csv").open(encoding="utf-8") as stream:
        return any(
            row["status"] == "numerical_failure" for row in csv.DictReader(stream)
        )
