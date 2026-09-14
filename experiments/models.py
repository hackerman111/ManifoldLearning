"""Контракты конфигурации и сеток экспериментов."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from ADP.cli.experiment_utils import (
    points_for_profile,
    validate_build,
    validate_experiment,
    validate_experiment_point,
)
from ADP.core.ADP_Config import ADP_Config

LinkName = Literal[
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
]
FeatureDistribution = Literal["gaussian", "uniform", "student_t5"]
NoiseDistribution = Literal["gaussian", "student_t5", "student_t3"]
ModelMode = Literal["single", "multi", "manifold"]


@dataclass(frozen=True, slots=True)
class ExperimentPoint:
    """Одна точка воспроизводимой сетки эксперимента."""

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
    mode: ModelMode = "single"
    index_dim: int = 1
    n_samples: int | None = None
    tau: float | None = None
    link_scale: float = 1.0
    normalize_link_by_sigma_x: bool = False
    basis_pool_dim: int | None = None
    N_loc: int | None = None
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
    N_manifold: int | None = None
    sync_steps: int | None = None
    lambda_manifold: float | None = None
    outer_steps: int | None = None
    lambda_penalty: float | None = None
    a: float | None = None
    h_min_factor: float | None = None
    index_init: str | None = None
    direction_mode: str | None = None
    multi_tensor: str | None = None
    select_step: str | None = None
    center_displacement: float | None = None
    training_set: str | None = None
    redraw_directions: bool | None = None
    solver_max_steps: int | None = None

    def __post_init__(self) -> None:
        validate_experiment_point(self)

    @property
    def n(self) -> int:
        return self.n_samples or math.ceil(self.d * self.n_over_d)


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
    solver: str = "lsmr"
    cg_maxiter: int | None = None

    def __post_init__(self) -> None:
        validate_build(self)


@dataclass(frozen=True, slots=True)
class Experiment:
    """Описание воспроизводимой smoke/overview/full сетки ADP."""

    selector: str
    title: str
    smoke: ExperimentPoint
    full: tuple[ExperimentPoint, ...]
    report_fields: tuple[str, ...] = ()
    full_runs: int = 1
    quality_threshold: float | None = None
    condition_field: str | None = None
    common_random_fields: tuple[str, ...] = ()
    condition_group_fields: tuple[str, ...] = ()
    hypothesis: str | None = None

    def __post_init__(self) -> None:
        point_fields = {item.name for item in fields(ExperimentPoint)}
        validate_experiment(self, point_fields)

    def points(self, profile: str) -> tuple[ExperimentPoint, ...]:
        if profile == "overview":
            from .profiles import overview_points

            return overview_points(self)
        return points_for_profile(self.smoke, self.full, profile)

    def run(
        self,
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
        from .runner import run_experiment

        return run_experiment(
            self,
            a,
            b,
            profile=profile,
            runs=runs,
            seed=seed,
            output_dir=output_dir,
            plots=plots,
            experiment_id=experiment_id,
            progress=progress,
        )
