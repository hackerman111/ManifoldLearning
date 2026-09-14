"""Проверки входных данных для CLI экспериментов."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Never, TypeVar

import numpy as np

if TYPE_CHECKING:
    from experiments.models import Build, Experiment, ExperimentPoint

_T = TypeVar("_T")


def positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def nonnegative(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate_experiment_point(point: ExperimentPoint) -> None:
    if isinstance(point.d, bool) or not isinstance(point.d, int) or point.d < 1:
        raise ValueError("d must be a positive integer")
    for name in ("n_over_d", "sigma_x", "outlier_scale"):
        positive(name, getattr(point, name))
    for name in ("rho_corr", "sigma_eps", "outlier_fraction", "delta"):
        nonnegative(name, getattr(point, name))
    if point.rho_corr >= 1:
        raise ValueError("rho_corr must be less than one")
    if point.outlier_fraction > 1:
        raise ValueError("outlier_fraction must not exceed one")
    if point.n_samples is not None and (
        isinstance(point.n_samples, bool)
        or not isinstance(point.n_samples, int)
        or point.n_samples < 2
    ):
        raise ValueError("n_samples must be an integer of at least two")
    if point.mode not in {"single", "multi", "manifold"}:
        raise ValueError("mode must be 'single', 'multi', or 'manifold'")
    if (
        isinstance(point.index_dim, bool)
        or not isinstance(point.index_dim, int)
        or not 1 <= point.index_dim <= point.d
    ):
        raise ValueError("index_dim must lie between one and d")
    if point.mode == "single" and point.index_dim != 1:
        raise ValueError("single mode requires index_dim=1")
    if point.mode == "multi" and point.index_dim >= point.d:
        raise ValueError("multi mode requires index_dim < d")
    if point.mode == "manifold" and (point.d < 2 or point.index_dim != 1):
        raise ValueError("manifold experiment data require d >= 2 and index_dim=1")
    if not isinstance(point.normalize_link_by_sigma_x, bool):
        raise ValueError("normalize_link_by_sigma_x must be boolean")
    if point.basis_pool_dim is not None:
        if (
            isinstance(point.basis_pool_dim, bool)
            or not isinstance(point.basis_pool_dim, int)
            or not point.index_dim <= point.basis_pool_dim <= point.d
        ):
            raise ValueError("basis_pool_dim must lie between index_dim and d")
        if point.mode != "multi":
            raise ValueError("basis_pool_dim is supported only in multi mode")
    if point.tau is not None and (
        not np.isfinite(point.tau) or not 0 <= point.tau <= 1
    ):
        raise ValueError("tau must lie in [0, 1] or be None")
    positive("link_scale", point.link_scale)
    links = {
        "linear",
        "quadratic",
        "square",
        "cubic",
        "quartic",
        "sin",
        "tanh",
        "oscillating",
        "sin_scaled",
        "cos_scaled",
        "x_sin",
        "tanh_scaled",
        "absolute",
        "relu",
        "gaussian_bump",
        "manifold_radial",
        "multi_additive",
        "multi_multiplicative",
    }
    if point.link not in links:
        raise ValueError(f"unknown link: {point.link}")
    allowed_links = {
        "single": links - {"multi_additive", "multi_multiplicative", "manifold_radial"},
        "multi": {"multi_additive", "multi_multiplicative"},
        "manifold": {"manifold_radial"},
    }
    if point.link not in allowed_links[point.mode]:
        raise ValueError(f"{point.link} is incompatible with {point.mode} mode")
    if point.x_distribution not in {"gaussian", "uniform", "student_t5"}:
        raise ValueError(f"unknown feature distribution: {point.x_distribution}")
    if point.noise_distribution not in {"gaussian", "student_t5", "student_t3"}:
        raise ValueError(f"unknown noise distribution: {point.noise_distribution}")
    if not isinstance(point.heteroscedastic, bool):
        raise ValueError("heteroscedastic must be boolean")
    for name in (
        "N_loc",
        "N_lin",
        "N_J",
        "N_phi",
        "N_manifold",
        "sync_steps",
        "outer_steps",
        "solver_max_steps",
    ):
        value = getattr(point, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    for name in ("lambda_penalty", "lambda_manifold", "center_displacement"):
        if (value := getattr(point, name)) is not None:
            nonnegative(name, value)
    for name in ("a", "h_min_factor"):
        if (value := getattr(point, name)) is not None:
            positive(name, value)
    if point.a is not None and point.a <= 1:
        raise ValueError("a must exceed one")
    if point.N_manifold is not None and point.N_manifold <= point.index_dim:
        raise ValueError("N_manifold must exceed index_dim")
    if (
        point.N_manifold is not None
        and point.N_J is not None
        and point.N_manifold >= point.N_J
    ):
        raise ValueError("N_manifold must be smaller than N_J")
    if point.index_init is not None and point.index_init not in {
        "local",
        "pilot",
        "random",
    }:
        raise ValueError("unknown index_init")
    if point.direction_mode is not None and point.direction_mode not in {
        "auto",
        "isotropic",
        "localized",
    }:
        raise ValueError("unknown direction_mode")
    if point.multi_tensor is not None and point.multi_tensor not in {
        "orthogonal",
        "full",
    }:
        raise ValueError("unknown multi_tensor")
    if point.select_step is not None and point.select_step not in {"best", "last"}:
        raise ValueError("unknown select_step")
    if point.training_set is not None and point.training_set not in {
        "all",
        "exclude_centers",
    }:
        raise ValueError("unknown training_set")
    if point.redraw_directions is not None and not isinstance(
        point.redraw_directions, bool
    ):
        raise ValueError("redraw_directions must be boolean or None")


def validate_build(build: Build) -> None:
    if not build.name or any(character in build.name for character in "\r\n"):
        raise ValueError("build name must be non-empty and single-line")
    positive("solver_tol", build.solver_tol)
    if (
        isinstance(build.solver_max_steps, bool)
        or not isinstance(build.solver_max_steps, int)
        or build.solver_max_steps < 1
    ):
        raise ValueError("solver_max_steps must be a positive integer")
    if not np.isfinite(build.theta) or not 0 < build.theta < 1:
        raise ValueError("theta must lie between zero and one")
    if build.trust_radius is not None:
        positive("trust_radius", build.trust_radius)
    if build.lsmr_maxiter is not None and (
        isinstance(build.lsmr_maxiter, bool)
        or not isinstance(build.lsmr_maxiter, int)
        or build.lsmr_maxiter < 1
    ):
        raise ValueError("lsmr_maxiter must be a positive integer or None")
    if build.solver not in {"lsmr", "cg", "hybrid"}:
        raise ValueError("solver must be 'lsmr', 'cg', or 'hybrid'")
    if build.cg_maxiter is not None and (
        isinstance(build.cg_maxiter, bool)
        or not isinstance(build.cg_maxiter, int)
        or build.cg_maxiter < 1
    ):
        raise ValueError("cg_maxiter must be a positive integer or None")


def validate_experiment(experiment: Experiment, point_fields: set[str]) -> None:
    if not experiment.selector or Path(experiment.selector).name != experiment.selector:
        raise ValueError("experiment selector must be a path-safe name")
    if not experiment.title:
        raise ValueError("experiment title must not be empty")
    if experiment.hypothesis is not None and (
        not experiment.hypothesis.strip()
        or "\n" in experiment.hypothesis
        or "\r" in experiment.hypothesis
    ):
        raise ValueError("hypothesis must be non-empty and single-line or None")
    if not experiment.full:
        raise ValueError("full experiment grid must not be empty")
    if any(name not in point_fields for name in experiment.report_fields):
        raise ValueError("report_fields must name ExperimentPoint fields")
    if experiment.condition_field is not None and (
        not experiment.condition_field or experiment.condition_field not in point_fields
    ):
        raise ValueError(
            "condition_field must name an ExperimentPoint field or be None"
        )
    if any(
        not name or name not in point_fields for name in experiment.common_random_fields
    ):
        raise ValueError("common_random_fields must name ExperimentPoint fields")
    group_fields = experiment.condition_group_fields
    if (
        len(set(group_fields)) != len(group_fields)
        or any(not name or name not in point_fields for name in group_fields)
        or experiment.condition_field in group_fields
    ):
        raise ValueError(
            "condition_group_fields must be unique ExperimentPoint fields "
            "distinct from condition_field"
        )
    if group_fields and experiment.condition_field is None:
        raise ValueError("condition_group_fields require condition_field")
    if (
        isinstance(experiment.full_runs, bool)
        or not isinstance(experiment.full_runs, int)
        or experiment.full_runs < 1
    ):
        raise ValueError("full_runs must be a positive integer")
    if experiment.quality_threshold is not None and (
        not np.isfinite(experiment.quality_threshold)
        or not 0 <= experiment.quality_threshold <= 1
    ):
        raise ValueError("quality_threshold must lie in [0, 1] or be None")


def points_for_profile(
    smoke: _T,
    full: tuple[_T, ...],
    profile: str,
) -> tuple[_T, ...]:
    if profile == "smoke":
        return (smoke,)
    if profile == "full":
        return full
    raise ValueError("profile must be 'smoke' or 'full'")


def validate_custom_parameters(n: int, d: int, noise: float) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer of at least two")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")
    nonnegative("noise", noise)


def validate_run_parameters(
    a_name: str,
    b_name: str | None,
    runs: int,
    seed: int,
) -> None:
    if b_name is not None and a_name == b_name:
        raise ValueError("A and B build names must differ")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def validate_experiment_id(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("experiment_id must be a path-safe name")
    return value


def fail(message: str) -> Never:
    raise ValueError(message)


def validate_single_link(name: str) -> None:
    if name in {"multi_additive", "multi_multiplicative", "manifold_radial"}:
        raise ValueError(f"{name} requires multi-index projected data")


def validate_multi_link(projected: np.ndarray, name: str) -> None:
    if projected.ndim != 2 or projected.shape[1] < 2:
        raise ValueError("multi-index links require at least two coordinates")
    if name not in {"multi_additive", "multi_multiplicative"}:
        raise ValueError(f"unknown multi-index link: {name}")


def standardize(values: np.ndarray, name: str) -> np.ndarray:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError(f"{name} has degenerate sample variance")
    return np.asarray((values - mean) / scale)


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError(f"{name} has a degenerate norm")
    return np.asarray(vector / norm)


def select_experiments(
    value: str,
    catalog: Mapping[str, _T],
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[_T, ...]:
    requested = tuple(part.strip() for part in value.split(","))
    if not requested or any(not selector for selector in requested):
        raise ValueError("experiment selectors must not be empty")
    unknown = sorted(set(requested) - set(catalog) - set(aliases))
    if unknown:
        raise ValueError(f"unknown experiment selector: {', '.join(unknown)}")
    selectors = tuple(
        dict.fromkeys(
            selector for item in requested for selector in aliases.get(item, (item,))
        )
    )
    return tuple(catalog[selector] for selector in selectors)


def require_experiment_rows(rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError("runs.csv contains no experiment rows")
