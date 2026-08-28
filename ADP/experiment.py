from __future__ import annotations

import argparse
import csv
import hashlib
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
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from .cli import _run, build_parser
from .core.ADP_Config import ADP_Config

LinkName = Literal["linear", "quadratic", "square", "sin", "tanh", "oscillating"]
FeatureDistribution = Literal["gaussian", "uniform", "student_t5"]
NoiseDistribution = Literal["gaussian", "student_t5", "student_t3"]


@dataclass(frozen=True, slots=True)
class ExperimentPoint:
    """Одна точка сетки эксперимента из каталога ``main``."""

    d: int
    n_over_d: float
    sigma_x: float = 1.0
    rho_corr: float = 0.0
    sigma_eps: float = 0.5
    link: LinkName = "quadratic"
    x_distribution: FeatureDistribution = "gaussian"
    noise_distribution: NoiseDistribution = "gaussian"
    heteroscedastic: bool = False
    outlier_fraction: float = 0.0
    outlier_scale: float = 1.0
    delta: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, int) or self.d < 1:
            raise ValueError("d must be a positive integer")
        for name in ("n_over_d", "sigma_x", "outlier_scale"):
            _positive(name, getattr(self, name))
        for name in ("rho_corr", "sigma_eps", "outlier_fraction", "delta"):
            _nonnegative(name, getattr(self, name))
        if self.rho_corr >= 1:
            raise ValueError("rho_corr must be less than one")
        if self.outlier_fraction > 1:
            raise ValueError("outlier_fraction must not exceed one")
        if self.link not in {
            "linear",
            "quadratic",
            "square",
            "sin",
            "tanh",
            "oscillating",
        }:
            raise ValueError(f"unknown link: {self.link}")
        if self.x_distribution not in {"gaussian", "uniform", "student_t5"}:
            raise ValueError(f"unknown feature distribution: {self.x_distribution}")
        if self.noise_distribution not in {"gaussian", "student_t5", "student_t3"}:
            raise ValueError(f"unknown noise distribution: {self.noise_distribution}")
        if not isinstance(self.heteroscedastic, bool):
            raise ValueError("heteroscedastic must be boolean")

    @property
    def n(self) -> int:
        return math.ceil(self.d * self.n_over_d)


@dataclass(frozen=True, slots=True)
class Build:
    """Имя конфигурации ADP и настройки её внутреннего решателя."""

    name: str
    config: ADP_Config
    solver_tol: float = 1e-6
    solver_max_steps: int = 3
    theta: float = 0.1
    trust_radius: float | None = None
    lsmr_maxiter: int | None = None

    def __post_init__(self) -> None:
        if not self.name or any(character in self.name for character in "\r\n"):
            raise ValueError("build name must be non-empty and single-line")
        _positive("solver_tol", self.solver_tol)
        if (
            isinstance(self.solver_max_steps, bool)
            or not isinstance(self.solver_max_steps, int)
            or self.solver_max_steps < 1
        ):
            raise ValueError("solver_max_steps must be a positive integer")
        if not np.isfinite(self.theta) or not 0 < self.theta < 1:
            raise ValueError("theta must lie between zero and one")
        if self.trust_radius is not None:
            _positive("trust_radius", self.trust_radius)
        if self.lsmr_maxiter is not None and (
            isinstance(self.lsmr_maxiter, bool)
            or not isinstance(self.lsmr_maxiter, int)
            or self.lsmr_maxiter < 1
        ):
            raise ValueError("lsmr_maxiter must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class Experiment:
    """Минимальное описание smoke/full сетки для парного A/B-запуска."""

    selector: str
    title: str
    smoke: ExperimentPoint
    full: tuple[ExperimentPoint, ...]

    def __post_init__(self) -> None:
        if not self.selector or Path(self.selector).name != self.selector:
            raise ValueError("experiment selector must be a path-safe name")
        if not self.title:
            raise ValueError("experiment title must not be empty")
        if not self.full:
            raise ValueError("full experiment grid must not be empty")

    def points(self, profile: str) -> tuple[ExperimentPoint, ...]:
        if profile == "smoke":
            return (self.smoke,)
        if profile == "full":
            return self.full
        raise ValueError("profile must be 'smoke' or 'full'")

    def run(
        self,
        a: Build,
        b: Build,
        *,
        profile: str = "smoke",
        runs: int = 1,
        seed: int = 0,
        output_dir: str | Path = "benchmark_outputs/experiments",
        plots: bool = True,
    ) -> Path:
        return run_experiment(
            self,
            a,
            b,
            profile=profile,
            runs=runs,
            seed=seed,
            output_dir=output_dir,
            plots=plots,
        )


@dataclass(frozen=True, slots=True)
class _SeedBundle:
    beta: int
    features: int
    noise: int
    centers: int
    directions: int
    init: int
    outliers: int
    outlier_noise: int
    gamma: int
    misspecification: int


@dataclass(frozen=True, slots=True)
class _GeneratedData:
    X: np.ndarray
    Y: np.ndarray
    beta: np.ndarray


def _catalog() -> dict[str, Experiment]:
    experiments = (
        Experiment(
            "1",
            "Проверка корректности",
            ExperimentPoint(4, 5, link="linear", sigma_eps=0),
            tuple(
                ExperimentPoint(d, ratio, link=link, sigma_eps=0)
                for d, ratio, link in product(
                    (5, 25), (5.0, 10.0), ("linear", "quadratic")
                )
            ),
        ),
        Experiment(
            "2",
            "Масштабирование",
            ExperimentPoint(4, 2),
            tuple(
                ExperimentPoint(d, ratio)
                for d, ratio in product((5, 25, 50, 100), (1.0, 1.15, 2.0, 5.0, 10.0))
            ),
        ),
        Experiment(
            "3",
            "Устойчивость к шуму",
            ExperimentPoint(4, 5, sigma_eps=1),
            tuple(
                ExperimentPoint(d, ratio, sigma_eps=noise)
                for d, ratio, noise in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.0, 0.316, 0.5, 0.707, 1.0, 1.414, 2.0),
                )
            ),
        ),
        Experiment(
            "4",
            "Коррелированные признаки",
            ExperimentPoint(4, 5, rho_corr=0.5),
            tuple(
                ExperimentPoint(d, ratio, rho_corr=rho)
                for d, ratio, rho in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
                )
            ),
        ),
        Experiment(
            "5",
            "Масштаб признаков",
            ExperimentPoint(4, 5, sigma_x=2),
            tuple(
                ExperimentPoint(d, ratio, sigma_x=scale)
                for d, ratio, scale in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.25, 0.5, 1.0, 2.0, 4.0),
                )
            ),
        ),
        Experiment(
            "6",
            "Функции связи",
            ExperimentPoint(4, 5, link="sin"),
            tuple(
                ExperimentPoint(d, ratio, link=link)
                for d, ratio, link in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    ("linear", "quadratic", "square", "sin", "tanh", "oscillating"),
                )
            ),
        ),
        Experiment(
            "7.1",
            "Распределения признаков",
            ExperimentPoint(4, 5, x_distribution="uniform"),
            tuple(
                ExperimentPoint(d, ratio, x_distribution=distribution)
                for d, ratio, distribution in product(
                    (25, 100), (2.0, 5.0), ("gaussian", "uniform", "student_t5")
                )
            ),
        ),
        Experiment(
            "7.2",
            "Распределения шума",
            ExperimentPoint(4, 5, noise_distribution="student_t5"),
            tuple(
                ExperimentPoint(d, ratio, noise_distribution=distribution)
                for d, ratio, distribution in product(
                    (25, 100), (2.0, 5.0), ("gaussian", "student_t5", "student_t3")
                )
            ),
        ),
        Experiment(
            "8.1",
            "Гетероскедастичность",
            ExperimentPoint(4, 5, heteroscedastic=True),
            tuple(
                ExperimentPoint(d, ratio, heteroscedastic=value)
                for d, ratio, value in product((25, 100), (2.0, 5.0), (False, True))
            ),
        ),
        Experiment(
            "8.2",
            "Выбросы",
            ExperimentPoint(4, 5, outlier_fraction=0.01, outlier_scale=5),
            tuple(
                ExperimentPoint(
                    d,
                    ratio,
                    outlier_fraction=fraction,
                    outlier_scale=scale,
                )
                for d, ratio, (fraction, scale) in product(
                    (25, 100),
                    (2.0, 5.0),
                    (
                        (0.0, 1.0),
                        (0.01, 5.0),
                        (0.01, 10.0),
                        (0.05, 5.0),
                        (0.05, 10.0),
                    ),
                )
            ),
        ),
        Experiment(
            "8.3",
            "Нарушение одноиндексной модели",
            ExperimentPoint(4, 5, delta=0.1),
            tuple(
                ExperimentPoint(d, ratio, delta=delta)
                for d, ratio, delta in product(
                    (25, 100), (2.0, 5.0), (0.0, 0.1, 0.25, 0.5)
                )
            ),
        ),
        custom_experiment(),
    )
    return {experiment.selector: experiment for experiment in experiments}


def custom_experiment(
    n: int = 240,
    d: int = 5,
    noise: float = 0.05,
) -> Experiment:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer of at least two")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")
    _nonnegative("noise", noise)
    point = ExperimentPoint(d=d, n_over_d=n / d, sigma_eps=noise, link="sin")
    return Experiment("custom", "Пользовательский эксперимент", point, (point,))


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
    *tuple(field.name for field in fields(ExperimentPoint)),
    "requested_config",
    "effective_config",
    "status",
    "error",
    "cosine_abs",
    "fit_time_sec",
    "max_stage_traced_peak_mib",
    "outer_iterations",
    "stop_reason",
    "selected_iteration",
    "selected_error",
    "trace",
    "solver_diagnostics",
)


def run_experiment(
    experiment: Experiment,
    a: Build,
    b: Build,
    *,
    profile: str = "smoke",
    runs: int = 1,
    seed: int = 0,
    output_dir: str | Path = "benchmark_outputs/experiments",
    plots: bool = True,
) -> Path:
    """Последовательно выполнить парный A/B-тест на одинаковых данных."""
    if a.name == b.name:
        raise ValueError("A and B build names must differ")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    points = experiment.points(profile)
    series_dir = _series_directory(Path(output_dir), experiment.selector)
    _write_manifest(series_dir, experiment, (a, b), profile, runs, seed)
    runs_path = series_dir / "runs.csv"

    labels = _point_labels(points)
    for point_index, point in enumerate(points):
        for run_index in range(runs):
            run_seed = seed + run_index
            seeds = _make_seed_bundle(experiment.selector, point, run_seed)
            order = (a, b) if (point_index + run_index) % 2 == 0 else (b, a)
            try:
                generated = _generate_data(experiment.selector, point, seeds, run_seed)
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
                    )
                    _append_row(runs_path, row)
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
                    row.update(_fit(build, point, generated, seeds.init))
                except Exception as error:  # Один fit не отменяет серию.
                    row.update(
                        status="numerical_failure",
                        error=_error_text(error),
                    )
                _append_row(runs_path, row)

    if plots:
        from .experiment_plots import plot_experiment

        plot_experiment(series_dir)
    return series_dir


def _fit(
    build: Build,
    point: ExperimentPoint,
    data: _GeneratedData,
    model_seed: int,
) -> dict[str, object]:
    config = _effective_config(build.config, point, model_seed)
    args = _arguments(build, config, point)
    started = perf_counter()
    index, true_basis, profile, metadata = _run(
        args,
        data=(data.X, data.Y, data.beta[:, None]),
    )
    elapsed = perf_counter() - started
    diagnostics = metadata.get("diagnostics", {})
    converged = isinstance(diagnostics, dict) and diagnostics.get("converged")
    effective = {
        **_config_spec(config),
        "N_lin": metadata["N_lin"],
        "N_J": metadata["N_J"],
        "N_phi": metadata["N_phi"],
    }
    return {
        "effective_config": _compact_json(effective),
        "status": "nonconverged" if converged is False else "success",
        "cosine_abs": float(abs(true_basis[:, 0] @ index)),
        "fit_time_sec": elapsed,
        "max_stage_traced_peak_mib": profile["total"]["traced_peak_bytes"] / 2**20,
        "outer_iterations": metadata["outer_iterations"],
        "stop_reason": metadata["stop_reason"],
        "selected_iteration": metadata["selected_iteration"],
        "selected_error": metadata["selected_error"],
        "trace": _compact_json(metadata["trace"]),
        "solver_diagnostics": _compact_json(diagnostics),
    }


def _effective_config(
    requested: ADP_Config,
    point: ExperimentPoint,
    model_seed: int,
) -> ADP_Config:
    n_loc = min(requested.N_loc, point.n)
    index_init = "random" if point.n <= point.d + 1 else requested.index_init
    n_lin = requested.N_lin or 2 * point.d
    if index_init == "local":
        n_lin = min(max(n_lin, point.d + 2), point.n)
    else:
        n_lin = min(max(n_lin, 1), point.n)
    minimum_centers = math.ceil(point.n / n_loc)
    n_centers = min(max(requested.N_J or point.n, minimum_centers), point.n)
    n_directions = requested.N_phi or min(n_loc, point.d)
    return replace(
        requested,
        seed=model_seed,
        N_loc=n_loc,
        N_lin=n_lin,
        N_J=n_centers,
        N_phi=n_directions,
        index_init=index_init,
    )


def _arguments(
    build: Build,
    config: ADP_Config,
    point: ExperimentPoint,
) -> argparse.Namespace:
    args = build_parser().parse_args([])
    args.mode = "single"
    args.n = point.n
    args.d = point.d
    args.index_dim = 1
    args.noise = point.sigma_eps
    for item in fields(config):
        setattr(args, item.name, getattr(config, item.name))
    args.solver_tol = build.solver_tol
    args.solver_max_steps = build.solver_max_steps
    args.theta = build.theta
    args.trust_radius = build.trust_radius
    args.lsmr_maxiter = build.lsmr_maxiter
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
        **asdict(point),
        "requested_config": _compact_json(_build_spec(build)),
        "status": "",
        "error": "",
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
    builds: tuple[Build, Build],
    profile: str,
    runs: int,
    seed: int,
) -> None:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": experiment.selector,
        "title": experiment.title,
        "profile": profile,
        "runs": runs,
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
        "seed_design": "paired-within-experiment-v1",
    }
    (series_dir / "series.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_spec(build: Build) -> dict[str, object]:
    return {
        "name": build.name,
        "config": _config_spec(build.config),
        "solver_tol": build.solver_tol,
        "solver_max_steps": build.solver_max_steps,
        "theta": build.theta,
        "trust_radius": build.trust_radius,
        "lsmr_maxiter": build.lsmr_maxiter,
    }


def _config_spec(config: ADP_Config) -> dict[str, object]:
    return {
        item.name: _callable_name(value) if callable(value) else value
        for item in fields(config)
        if (value := getattr(config, item.name)) is not None
    }


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{name}"


def _series_directory(output_dir: Path, selector: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = output_dir / selector.replace(".", "_") / stamp
    path.mkdir(parents=True)
    return path


def _point_labels(points: tuple[ExperimentPoint, ...]) -> tuple[str, ...]:
    varying = tuple(
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


def _make_seed_bundle(
    selector: str,
    point: ExperimentPoint,
    seed: int,
) -> _SeedBundle:
    if selector == "custom":
        return _SeedBundle(*(seed for _ in fields(_SeedBundle)))
    payload = json.dumps(
        {
            "experiment": selector,
            "parameters": asdict(point),
            "seed": seed,
            "seed_design": "paired-within-experiment-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    entropy = int(hashlib.sha256(payload).hexdigest(), 16)
    state = np.random.SeedSequence(entropy).generate_state(10)
    return _SeedBundle(*(int(value) for value in state))


def _generate_data(
    selector: str,
    point: ExperimentPoint,
    seeds: _SeedBundle,
    run_seed: int,
) -> _GeneratedData:
    if selector == "custom":
        rng = np.random.default_rng(run_seed)
        X = rng.normal(size=(point.n, point.d))
        beta = _unit(rng.normal(size=point.d), "beta")
        Y = np.sin(X @ beta) + point.sigma_eps * rng.normal(size=point.n)
        return _GeneratedData(X, Y, beta)

    beta = _unit(np.random.default_rng(seeds.beta).normal(size=point.d), "beta")
    X = _features(point, seeds.features)
    index = X @ beta
    divisor = point.sigma_x if selector == "5" else 1.0
    signal = _standardize(_link(index / divisor, point.link), f"{point.link} link")
    noise = _noise(point, index, seeds.noise)
    noise = _outliers(point, noise, seeds.outliers, seeds.outlier_noise)
    Y = signal + noise
    if point.delta > 0:
        gamma = _gamma(beta, seeds.gamma, seeds.misspecification)
        gamma_index = X @ gamma
        Y += point.delta * _standardize(
            gamma_index + 0.5 * gamma_index**2,
            "misspecification link",
        )
    return _GeneratedData(np.asarray(X), np.asarray(Y), beta)


def _features(point: ExperimentPoint, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (point.n, point.d)
    if point.x_distribution == "gaussian":
        coordinates = np.arange(point.d)
        covariance = point.rho_corr ** np.abs(
            np.subtract.outer(coordinates, coordinates)
        )
        values = rng.normal(size=shape) @ np.linalg.cholesky(covariance).T
    elif point.x_distribution == "uniform":
        values = rng.uniform(-math.sqrt(3), math.sqrt(3), size=shape)
    else:
        values = rng.standard_t(df=5, size=shape) * math.sqrt(3 / 5)
    return np.asarray(point.sigma_x * values, dtype=float)


def _noise(point: ExperimentPoint, index: np.ndarray, seed: int) -> np.ndarray:
    if point.sigma_eps == 0:
        return np.zeros(point.n)
    rng = np.random.default_rng(seed)
    if point.heteroscedastic:
        scale = point.sigma_eps * np.sqrt((0.25 + index**2) / 1.25)
        return np.asarray(scale * rng.normal(size=point.n))
    if point.noise_distribution == "gaussian":
        values = rng.normal(size=point.n)
    elif point.noise_distribution == "student_t5":
        values = rng.standard_t(df=5, size=point.n) * math.sqrt(3 / 5)
    else:
        values = rng.standard_t(df=3, size=point.n) * math.sqrt(1 / 3)
    return np.asarray(point.sigma_eps * values)


def _outliers(
    point: ExperimentPoint,
    noise: np.ndarray,
    index_seed: int,
    noise_seed: int,
) -> np.ndarray:
    if point.outlier_fraction == 0:
        return noise
    count = min(point.n, math.ceil(point.outlier_fraction * point.n))
    indices = np.random.default_rng(index_seed).permutation(point.n)[:count]
    result = noise.copy()
    result[indices] = np.random.default_rng(noise_seed).normal(
        scale=point.outlier_scale * point.sigma_eps,
        size=count,
    )
    return result


def _gamma(beta: np.ndarray, seed: int, orientation_seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(256):
        candidate = rng.normal(size=beta.size)
        candidate -= beta * float(candidate @ beta)
        norm = np.linalg.norm(candidate)
        if np.isfinite(norm) and norm > np.finfo(float).eps:
            sign = (
                -1.0
                if np.random.default_rng(orientation_seed).integers(2) == 0
                else 1.0
            )
            return np.asarray(sign * candidate / norm)
    raise ValueError("cannot generate a direction orthogonal to beta")


def _link(index: np.ndarray, name: LinkName) -> np.ndarray:
    if name == "linear":
        return index
    if name == "quadratic":
        return index + 0.5 * index**2
    if name == "square":
        return index**2
    if name == "sin":
        return np.sin(1.5 * index)
    if name == "tanh":
        return np.tanh(2 * index)
    return index * np.sin(math.sqrt(5) * index)


def _standardize(values: np.ndarray, name: str) -> np.ndarray:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError(f"{name} has degenerate sample variance")
    return np.asarray((values - mean) / scale)


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError(f"{name} has a degenerate norm")
    return np.asarray(vector / norm)


def _positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


CATALOG: Mapping[str, Experiment] = _catalog()


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


def _selected_experiments(
    value: str,
    *,
    custom: Experiment,
) -> tuple[Experiment, ...]:
    catalog = {**CATALOG, "custom": custom}
    selectors = (
        tuple(catalog)
        if value == "all"
        else tuple(part.strip() for part in value.split(","))
    )
    if not selectors or any(not selector for selector in selectors):
        raise ValueError("experiment selectors must not be empty")
    unknown = sorted(set(selectors) - set(catalog))
    if unknown:
        raise ValueError(f"unknown experiment selector: {', '.join(unknown)}")
    return tuple(catalog[selector] for selector in selectors)


def _has_numerical_failures(series_dir: Path) -> bool:
    with (series_dir / "runs.csv").open(encoding="utf-8") as stream:
        return any(
            row["status"] == "numerical_failure" for row in csv.DictReader(stream)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal paired ADP experiments")
    parser.add_argument("--experiment", default="custom")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/experiments")
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        for experiment in CATALOG.values():
            print(f"{experiment.selector:>6}  {experiment.title}")
        return 0
    failed = False
    try:
        experiments = _selected_experiments(
            args.experiment,
            custom=custom_experiment(args.n, args.d, args.noise),
        )
        common = ADP_Config()
        builds = (
            Build("A_local", common),
            Build("B_random", replace(common, index_init="random")),
        )
        for experiment in experiments:
            path = experiment.run(
                *builds,
                profile=args.profile,
                runs=args.runs,
                seed=args.seed,
                output_dir=args.output_dir,
                plots=not args.no_plots,
            )
            print(f"{experiment.selector}: {path}")
            failed = _has_numerical_failures(path) or failed
    except (ImportError, TypeError, ValueError) as error:
        parser.error(str(error))
    return int(failed)


__all__ = [
    "CATALOG",
    "Build",
    "Experiment",
    "ExperimentPoint",
    "custom_experiment",
    "run_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
