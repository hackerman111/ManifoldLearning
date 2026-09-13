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
from scipy.linalg import subspace_angles
from tqdm import tqdm

from ..core.ADP_Config import ADP_Config
from .experiment_utils import (
    fail as _fail,
    points_for_profile,
    select_experiments,
    standardize as _standardize,
    unit as _unit,
    validate_build,
    validate_custom_parameters,
    validate_experiment,
    validate_experiment_id,
    validate_experiment_point,
    validate_multi_link,
    validate_run_parameters,
    validate_single_link,
)
from .main import _run, build_parser

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
    """Описание воспроизводимой smoke/full сетки ADP."""

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
                for d, ratio in product(
                    (5, 25, 50, 100),
                    (1.15, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
                )
            ),
            report_fields=("d", "n_over_d"),
            full_runs=25,
            quality_threshold=0.9,
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
                    (
                        0.0,
                        0.2,
                        0.316,
                        0.4,
                        0.5,
                        0.6,
                        0.707,
                        0.8,
                        1.0,
                        1.414,
                        2.0,
                    ),
                )
            ),
            report_fields=("d", "n_over_d", "sigma_eps"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
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
                    (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99),
                )
            ),
            report_fields=("d", "n_over_d", "rho_corr"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="rho_corr",
            common_random_fields=("rho_corr",),
        ),
        Experiment(
            "5",
            "Масштаб признаков",
            ExperimentPoint(
                4,
                5,
                sigma_x=2,
                normalize_link_by_sigma_x=True,
            ),
            tuple(
                ExperimentPoint(
                    d,
                    ratio,
                    sigma_x=scale,
                    normalize_link_by_sigma_x=True,
                )
                for d, ratio, scale in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                )
            ),
            report_fields=("d", "n_over_d", "sigma_x"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="sigma_x",
            common_random_fields=("sigma_x",),
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
            condition_field="link",
            common_random_fields=("link",),
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
            condition_field="x_distribution",
            common_random_fields=("x_distribution",),
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
            condition_field="noise_distribution",
            common_random_fields=("noise_distribution",),
        ),
        Experiment(
            "8.1",
            "Гетероскедастичность",
            ExperimentPoint(4, 5, heteroscedastic=True),
            tuple(
                ExperimentPoint(d, ratio, heteroscedastic=value)
                for d, ratio, value in product((25, 100), (2.0, 5.0), (False, True))
            ),
            condition_field="heteroscedastic",
            common_random_fields=("heteroscedastic",),
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
                        (0.005, 5.0),
                        (0.01, 5.0),
                        (0.01, 10.0),
                        (0.025, 7.5),
                        (0.05, 5.0),
                        (0.05, 10.0),
                    ),
                )
            ),
            common_random_fields=("outlier_fraction", "outlier_scale"),
        ),
        Experiment(
            "8.3",
            "Нарушение одноиндексной модели",
            ExperimentPoint(4, 5, delta=0.1),
            tuple(
                ExperimentPoint(d, ratio, delta=delta)
                for d, ratio, delta in product(
                    (25, 100),
                    (2.0, 5.0),
                    (0.0, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5),
                )
            ),
            condition_field="delta",
            common_random_fields=("delta",),
        ),
        *_detailed_multi_catalog(),
        *_report_catalog(),
        *_parameter_scaling_catalog(),
        *_detailed_nd_catalog(),
        *_manifold_catalog(),
        custom_experiment(),
    )
    return {experiment.selector: experiment for experiment in experiments}


def custom_experiment(
    n: int = 240,
    d: int = 5,
    noise: float = 0.05,
) -> Experiment:
    validate_custom_parameters(n, d, noise)
    point = ExperimentPoint(d=d, n_over_d=n / d, sigma_eps=noise, link="sin")
    return Experiment("custom", "Пользовательский эксперимент", point, (point,))


def _manifold_catalog() -> tuple[Experiment, ...]:
    """Базовые однофакторные ESTIMATOR-сетки manifold ADP."""
    smoke = ExperimentPoint(
        d=3,
        n_over_d=80 / 3,
        n_samples=80,
        sigma_eps=0.02,
        link="manifold_radial",
        mode="manifold",
        index_dim=1,
        N_loc=15,
        N_lin=30,
        N_J=12,
        N_phi=6,
        N_manifold=5,
        sync_steps=1,
        lambda_manifold=0.5,
        a=2.0,
        h_min_factor=10.0,
    )
    base = replace(
        smoke,
        d=4,
        n_over_d=60,
        n_samples=240,
        sigma_eps=0.05,
        N_loc=20,
        N_lin=60,
        N_J=24,
        N_phi=10,
        N_manifold=6,
    )

    def sweep(
        selector: str,
        title: str,
        field: str,
        values: tuple[int | float, ...],
        hypothesis: str,
        *,
        common_random_fields: tuple[str, ...] | None = None,
    ) -> Experiment:
        def point(value: int | float) -> ExperimentPoint:
            updates = {field: value}
            if field in {"n_samples", "d"}:
                n = int(value) if field == "n_samples" else base.n
                d = int(value) if field == "d" else base.d
                updates["n_over_d"] = n / d
            return replace(base, **updates)

        return Experiment(
            selector,
            title,
            smoke,
            tuple(point(value) for value in values),
            report_fields=(field,),
            full_runs=5,
            quality_threshold=0.2,
            condition_field=field,
            common_random_fields=common_random_fields or (field,),
            hypothesis=hypothesis,
        )

    joint = Experiment(
        "manifold",
        "Manifold ADP: локально меняющееся радиальное направление",
        smoke,
        tuple(
            replace(
                smoke,
                d=d,
                n_over_d=n / d,
                n_samples=n,
                N_lin=linear_mass,
                N_J=centers,
                N_manifold=neighbors,
                N_loc=local_mass,
                N_phi=directions,
                sigma_eps=noise,
            )
            for (
                n,
                d,
                centers,
                neighbors,
                local_mass,
                directions,
                linear_mass,
                noise,
            ) in (
                (120, 3, 16, 6, 15, 6, 30, 0.02),
                (240, 4, 24, 8, 15, 6, 30, 0.05),
                (400, 6, 32, 10, 25, 12, 60, 0.10),
            )
        ),
        report_fields=("n_samples", "d", "sigma_eps"),
        full_runs=5,
        quality_threshold=0.2,
        hypothesis=(
            "Локальные проекторы восстанавливают радиальное направление при "
            "росте n и умеренном шуме."
        ),
    )
    return (
        joint,
        sweep(
            "manifold-n",
            "Manifold ADP: объём выборки",
            "n_samples",
            (120, 240, 360, 480),
            "Ошибка локальных проекторов уменьшается при росте n.",
            common_random_fields=("n_samples", "n_over_d"),
        ),
        sweep(
            "manifold-d",
            "Manifold ADP: размерность пространства",
            "d",
            (2, 3, 4, 6, 8, 10),
            "Восстановление ухудшается при росте внешней размерности d.",
            common_random_fields=("d", "n_over_d"),
        ),
        sweep(
            "manifold-noise",
            "Manifold ADP: шум отклика",
            "sigma_eps",
            (0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
            "При увеличении шума отклика ошибка локальных проекторов растёт.",
        ),
        sweep(
            "manifold-scale",
            "Manifold ADP: масштаб признаков",
            "sigma_x",
            (0.5, 1.0, 2.0, 4.0),
            "Масштаб признаков не меняет качество после адаптации bandwidth.",
        ),
        sweep(
            "manifold-corr",
            "Manifold ADP: корреляция признаков",
            "rho_corr",
            (0.0, 0.25, 0.5, 0.75, 0.9),
            "Сильная корреляция признаков ухудшает локальную идентификацию.",
        ),
        sweep(
            "manifold-nlin",
            "Manifold ADP: масса пилотной регрессии",
            "N_lin",
            (10, 20, 40, 60, 80),
            "Слишком малая или большая N_lin ухудшает пилотные градиенты.",
        ),
        sweep(
            "manifold-nloc",
            "Manifold ADP: локальная масса",
            "N_loc",
            (10, 15, 20, 30, 40),
            "N_loc задаёт компромисс между вариативностью и локальным смещением.",
        ),
        sweep(
            "manifold-centers",
            "Manifold ADP: число центров",
            "N_J",
            (12, 18, 24, 36, 48),
            "После достаточного покрытия рост N_J даёт убывающий выигрыш.",
        ),
        sweep(
            "manifold-nphi",
            "Manifold ADP: число случайных направлений",
            "N_phi",
            (2, 4, 6, 10, 16, 24),
            "Рост N_phi стабилизирует локальные статистики до насыщения.",
        ),
        sweep(
            "manifold-neighbors",
            "Manifold ADP: соседи manifold-графа",
            "N_manifold",
            (2, 4, 6, 8, 10),
            "N_manifold задаёт компромисс между связностью и локальностью графа.",
        ),
        sweep(
            "manifold-lambda",
            "Manifold ADP: регуляризация согласованности",
            "lambda_manifold",
            (0.0, 0.1, 0.5, 1.0, 5.0),
            "Умеренная lambda_manifold улучшает согласованность локальных chart.",
        ),
        sweep(
            "manifold-sync",
            "Manifold ADP: шаги синхронизации",
            "sync_steps",
            (1, 2, 3, 5),
            "Дополнительные шаги синхронизации улучшают качество до насыщения.",
        ),
    )


def _scaling_point(mode: Literal["single", "multi"], n: int, d: int) -> ExperimentPoint:
    """Фиксированная конфигурация для сопоставимых SI/MI scaling-сеток."""
    multi = mode == "multi"
    return ExperimentPoint(
        d=d,
        n_over_d=n / d,
        n_samples=n,
        sigma_eps=0.2,
        tau=0.0,
        link="multi_additive" if multi else "sin_scaled",
        mode=mode,
        index_dim=2 if multi else 1,
        N_loc=15,
        N_lin=d + 60,
        N_J=math.ceil(n / 5),
        N_phi=20,
        outer_steps=5,
        lambda_penalty=0.05,
        a=math.sqrt(2),
        h_min_factor=3,
        index_init="pilot" if multi else "local",
        direction_mode="isotropic" if multi else "localized",
        select_step="best",
        solver_max_steps=8,
    )


def _parameter_scaling_catalog() -> tuple[Experiment, ...]:
    """Парные SI/MI-сетки зависимости N_phi, N_loc и N_J от (n, d)."""
    shapes = tuple(product((500, 1000, 2000), (10, 25, 50)))
    n_phi_values = {
        "single": (5, 10, 20, 30, 40, 60),
        "multi": (3, 5, 10, 20, 30, 40),
    }
    n_loc_values = (7, 10, 15, 20, 30, 45)
    center_multipliers = (1, 2, 4, 8, 12)
    baseline_n_loc = 15

    def make(
        mode: ModelMode,
        parameter: str,
        points: tuple[ExperimentPoint, ...],
        hypothesis: str,
    ) -> Experiment:
        prefix = "mi" if mode == "multi" else "si"
        parameter_slug = parameter.lower().replace("_", "")
        return Experiment(
            f"scale-{prefix}-{parameter_slug}",
            f"{prefix.upper()}: зависимость {parameter} от n и d",
            _smoke_point(points[0]),
            points,
            report_fields=("n_samples", "d", parameter),
            full_runs=5,
            quality_threshold=0.1 if mode == "multi" else 0.9,
            condition_field=parameter,
            common_random_fields=(parameter,),
            hypothesis=hypothesis,
        )

    experiments: list[Experiment] = []
    for mode in ("single", "multi"):
        bases = tuple(_scaling_point(mode, n, d) for n, d in shapes)
        multi = mode == "multi"
        experiments.extend(
            (
                make(
                    mode,
                    "N_phi",
                    tuple(
                        replace(point, N_phi=value)
                        for point in bases
                        for value in n_phi_values[mode]
                    ),
                    (
                        "N_phi > m. "
                        "Это обязательная граница; качество насыщается около 20--30 "
                        "направлений. Зависимость от объёма выборки слабая, "
                        "стоимость при этом продолжает расти."
                        if multi
                        else "N_phi. "
                        "Плато качества ожидается около 20--30 "
                        "направлений. Большая размерность может сдвинуть это плато "
                        "вправо, объём выборки влияет слабее."
                    ),
                ),
                make(
                    mode,
                    "N_loc",
                    tuple(
                        replace(point, N_loc=value)
                        for point in bases
                        for value in n_loc_values
                    ),
                    "Эффективная локальная масса растёт медленнее объёма выборки "
                    "и слабо зависит от размерности; малые значения "
                    "нестабильны, большие увеличивают смещение.",
                ),
                make(
                    mode,
                    "N_J",
                    tuple(
                        replace(
                            point,
                            N_J=math.ceil(multiplier * point.n / baseline_n_loc),
                        )
                        for point in bases
                        for multiplier in center_multipliers
                    ),
                    "N_J = c*n/N_loc. "
                    "Коэффициент покрытия почти постоянен; "
                    "после значений около 4--8 стоимость растёт быстрее качества.",
                ),
            )
        )
    return tuple(experiments)


def _detailed_nd_catalog() -> tuple[Experiment, ...]:
    """Уточнение MI-границы и зависимости SI от N_loc по (n, d)."""
    mi_points = tuple(
        _scaling_point("multi", n, d)
        for n, d in product(
            (400, 600, 800, 1000, 1400, 1800, 2400),
            (10, 15, 20, 25, 30, 35, 40, 50, 60),
        )
    )
    si_points = tuple(
        replace(_scaling_point("single", n, d), N_loc=n_loc)
        for n, d, n_loc in product(
            (400, 700, 1000, 1500, 2200),
            (10, 20, 30, 40, 50, 60),
            (10, 15, 20, 25, 30, 40, 50),
        )
    )
    return (
        Experiment(
            "mi-boundary-nd",
            "Multi-index: подробная граница по n и d",
            _smoke_point(mi_points[0]),
            mi_points,
            report_fields=("n_samples", "d", "n_over_d"),
            full_runs=10,
            quality_threshold=0.1,
            hypothesis=(
                "При фиксированном estimator граница projector_distance <= 0.1 "
                "зависит и от n/d, и от абсолютных n и d."
            ),
        ),
        Experiment(
            "si-nloc-nd",
            "Single-index: N_loc в зависимости от n и d",
            _smoke_point(si_points[0]),
            si_points,
            report_fields=("n_samples", "d", "N_loc"),
            full_runs=10,
            quality_threshold=0.9,
            condition_field="N_loc",
            common_random_fields=("N_loc",),
            hypothesis=(
                "Минимальный N_loc для устойчивой сходимости растёт при росте d; "
                "после насыщения качества дальнейший рост оплачивается временем."
            ),
        ),
    )


def _detailed_multi_catalog() -> tuple[Experiment, ...]:
    """Крупные MI-сетки границы восстановления."""
    base = ExperimentPoint(
        d=25,
        n_over_d=2,
        mode="multi",
        index_dim=2,
        link="multi_additive",
        link_scale=3,
        solver_max_steps=5,
    )
    dimensions = (25, 50)
    ratios = (2.0, 5.0, 10.0)

    def make(
        selector: str,
        title: str,
        points: tuple[ExperimentPoint, ...],
        report_fields: tuple[str, ...],
        *,
        condition_field: str | None = None,
        common_random_fields: tuple[str, ...] = (),
        condition_group_fields: tuple[str, ...] = (),
    ) -> Experiment:
        return Experiment(
            selector,
            title,
            _smoke_point(points[0]),
            points,
            report_fields=report_fields,
            full_runs=25,
            quality_threshold=0.1,
            condition_field=condition_field,
            common_random_fields=common_random_fields,
            condition_group_fields=condition_group_fields,
        )

    scaling = tuple(
        replace(base, d=d, n_over_d=ratio)
        for d, ratio in product(
            (5, 25, 50),
            (1.15, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        )
    )
    noise = tuple(
        replace(base, d=d, n_over_d=ratio, sigma_eps=value)
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.0, 0.2, 0.316, 0.4, 0.5, 0.6, 0.707, 0.8, 1.0, 1.414, 2.0),
        )
    )
    correlation = tuple(
        replace(base, d=d, n_over_d=ratio, rho_corr=value)
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99),
        )
    )
    scale = tuple(
        replace(
            base,
            d=d,
            n_over_d=ratio,
            sigma_x=value,
            normalize_link_by_sigma_x=True,
        )
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
    )
    index_dimension = tuple(
        replace(
            base,
            d=d,
            n_over_d=ratio,
            index_dim=index_dim,
            link=link,
            link_scale=link_scale,
            basis_pool_dim=10,
        )
        for d, ratio, index_dim, link, link_scale in product(
            (25, 50),
            ratios,
            (2, 3, 5, 7, 10),
            ("multi_additive", "multi_multiplicative"),
            (1.0, 2.0, 3.0, 4.0),
        )
    )
    return (
        make(
            "mi-1",
            "Multi-index: граница по размеру задачи",
            scaling,
            ("d", "n_over_d"),
        ),
        make(
            "mi-2",
            "Multi-index: устойчивость к шуму",
            noise,
            ("d", "n_over_d", "sigma_eps"),
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
        ),
        make(
            "mi-3",
            "Multi-index: коррелированные признаки",
            correlation,
            ("d", "n_over_d", "rho_corr"),
            condition_field="rho_corr",
            common_random_fields=("rho_corr",),
        ),
        make(
            "mi-4",
            "Multi-index: масштаб признаков",
            scale,
            ("d", "n_over_d", "sigma_x"),
            condition_field="sigma_x",
            common_random_fields=("sigma_x",),
        ),
        make(
            "mi-5",
            "Multi-index: размер подпространства и link",
            index_dimension,
            ("d", "n_over_d", "index_dim", "link", "link_scale"),
            condition_field="index_dim",
            common_random_fields=("index_dim", "link", "link_scale"),
            condition_group_fields=("link", "link_scale"),
        ),
    )


def _report_catalog() -> tuple[Experiment, ...]:
    """Полные SI/MI-сетки из ``task.md`` без внешних методов."""
    si = ExperimentPoint(
        d=100,
        n_over_d=10,
        n_samples=1000,
        sigma_eps=0.2,
        tau=0.4,
        link="x_sin",
        link_scale=3,
        N_loc=20,
        N_lin=160,
        N_J=500,
        N_phi=20,
        outer_steps=3,
        lambda_penalty=0.05,
        a=math.sqrt(2),
        h_min_factor=3,
        index_init="local",
        direction_mode="localized",
        center_displacement=0.1,
        training_set="all",
        redraw_directions=True,
        solver_max_steps=3,
    )
    mi = replace(
        si,
        mode="multi",
        index_dim=2,
        link="multi_additive",
        N_phi=40,
        direction_mode="isotropic",
        multi_tensor="orthogonal",
        solver_max_steps=5,
    )
    n_values = (800, 1000, 1200, 2000)
    n_sweep_values = (800, 900, 1000, 1200, 1400, 1600, 2000)
    d_values = tuple(range(10, 101, 10))
    noise_values = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    tau_values = (0.0, 0.2, 0.4, 0.8)
    tau_sweep_values = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
    scales = (1.0, 2.0, 3.0, 4.0)
    scale_sweep_values = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    sigma_x_values = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    n_lin_scales = tuple(range(1, 8))
    n_loc_values = (7, 9, 10, 12, 15, 17, 20)
    lambdas = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    displacements = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    ao_steps = tuple(range(1, 8))
    single_directions = (10, 15, 20, 30, 40, 50, 60)
    multi_directions = (3, 5, 10, 20, 30, 40, 60)
    single_a = tuple(2 ** (1 / 4 + step / 8) for step in range(7))
    multi_a = tuple(2 ** (1 / 4 + step / 24) for step in range(7))
    h_min_factors = (1.0, 4 / 3, 5 / 3, 2.0, 7 / 3, 8 / 3, 3.0)
    center_counts = tuple(
        scale * si.n // (si.N_loc or 1) for scale in (1, 2, 3, 5, 10, 15, 20)
    )

    def make(
        selector: str,
        title: str,
        points: tuple[ExperimentPoint, ...],
        *report_fields: str,
        full_runs: int = 1,
    ) -> Experiment:
        condition_field = report_fields[0] if len(report_fields) == 1 else None
        return Experiment(
            selector,
            title,
            _smoke_point(points[0]),
            points,
            report_fields,
            full_runs,
            condition_field=condition_field,
            common_random_fields=(condition_field,) if condition_field else (),
        )

    single = (
        make(
            "si-n",
            "Single-index: объём выборки",
            tuple(replace(si, n_samples=n, N_J=n // 2) for n in n_sweep_values),
            "n_samples",
        ),
        make(
            "si-d",
            "Single-index: размерность",
            tuple(replace(si, d=d, N_lin=d + 60) for d in d_values),
            "d",
        ),
        make(
            "si-noise",
            "Single-index: шум",
            tuple(replace(si, sigma_eps=value) for value in noise_values),
            "sigma_eps",
        ),
        make(
            "si-tau",
            "Single-index: общий фактор",
            tuple(replace(si, tau=value) for value in tau_sweep_values),
            "tau",
        ),
        make(
            "si-link",
            "Single-index: функция связи",
            tuple(replace(si, link=link) for link in ("sin_scaled", "x_sin")),
            "link",
        ),
        make(
            "si-frequency-sin",
            "Single-index: частота sin(sx)",
            tuple(
                replace(si, link="sin_scaled", link_scale=s) for s in scale_sweep_values
            ),
            "link_scale",
        ),
        make(
            "si-frequency-xsin",
            "Single-index: частота x sin(sx)",
            tuple(replace(si, link="x_sin", link_scale=s) for s in scale_sweep_values),
            "link_scale",
        ),
        make(
            "si-scale",
            "Single-index: масштаб признаков при d=10",
            tuple(
                replace(si, d=10, N_lin=70, sigma_x=value) for value in sigma_x_values
            ),
            "sigma_x",
        ),
        make(
            "si-nlin",
            "Single-index: локальная линейная масса",
            tuple(replace(si, N_lin=si.d + s * 20) for s in n_lin_scales),
            "N_lin",
        ),
        make(
            "si-centers",
            "Single-index: число центров",
            tuple(replace(si, N_J=value) for value in center_counts),
            "N_J",
        ),
        make(
            "si-displacement",
            "Single-index: смещение центров",
            tuple(replace(si, center_displacement=value) for value in displacements),
            "center_displacement",
        ),
        make(
            "si-training",
            "Single-index: обучающая выборка",
            tuple(
                replace(si, training_set=value) for value in ("all", "exclude_centers")
            ),
            "training_set",
        ),
        make(
            "si-kmax",
            "Single-index: число внутренних AO-шагов",
            tuple(replace(si, solver_max_steps=value) for value in ao_steps),
            "solver_max_steps",
        ),
        make(
            "si-nloc",
            "Single-index: локальная масса",
            tuple(replace(si, N_loc=value) for value in n_loc_values),
            "N_loc",
        ),
        make(
            "si-nphi",
            "Single-index: число направлений",
            tuple(replace(si, N_phi=value) for value in single_directions),
            "N_phi",
        ),
        make(
            "si-lambda",
            "Single-index: регуляризация",
            tuple(replace(si, lambda_penalty=value) for value in lambdas),
            "lambda_penalty",
        ),
        make(
            "si-a",
            "Single-index: коэффициент уменьшения h",
            tuple(replace(si, a=value) for value in single_a),
            "a",
        ),
        make(
            "si-hmin",
            "Single-index: нижняя граница h",
            tuple(replace(si, h_min_factor=value) for value in h_min_factors),
            "h_min_factor",
        ),
    )

    hypothesis_base = replace(
        si,
        d=30,
        n_over_d=20,
        n_samples=600,
        N_lin=90,
        N_J=200,
        N_phi=20,
        outer_steps=5,
        solver_max_steps=8,
        link_scale=1.0,
    )
    function_classes = (
        "linear",
        "quadratic",
        "square",
        "cubic",
        "quartic",
        "sin_scaled",
        "cos_scaled",
        "x_sin",
        "tanh_scaled",
        "absolute",
        "relu",
        "gaussian_bump",
    )
    parity_scales = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    support_scales = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    init_links = (
        "linear",
        "square",
        "cos_scaled",
        "tanh_scaled",
        "relu",
        "gaussian_bump",
    )
    single_hypotheses = (
        Experiment(
            "si-function-classes",
            "Single-index: классы функций связи",
            _smoke_point(replace(hypothesis_base, link=function_classes[0])),
            tuple(replace(hypothesis_base, link=link) for link in function_classes),
            report_fields=("link",),
            full_runs=3,
            common_random_fields=("link",),
            hypothesis=(
                "Функции, имеющие информативную производную на широкой области "
                "значений "
                "индекса восстанавливаются лучше локализованных, насыщаемых и "
                "осциллирующих функций."
            ),
        ),
        Experiment(
            "si-parity-frequency",
            "Single-index: чётность и частота",
            _smoke_point(replace(hypothesis_base, link="sin_scaled", link_scale=0.5)),
            tuple(
                replace(hypothesis_base, link=link, link_scale=scale)
                for link, scale in product(("sin_scaled", "cos_scaled"), parity_scales)
            ),
            report_fields=("link", "link_scale"),
            full_runs=3,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "При одинаковой частоте чётность функции не должна существенно "
                "влиять на восстановление направления; рост частоты должен ухудшать "
                "локальную оценку обеих функций."
            ),
        ),
        Experiment(
            "si-gradient-support",
            "Single-index: ширина информативной области производной",
            _smoke_point(replace(hypothesis_base, link="tanh_scaled", link_scale=0.25)),
            tuple(
                replace(hypothesis_base, link=link, link_scale=scale)
                for link, scale in product(
                    ("tanh_scaled", "gaussian_bump"), support_scales
                )
            ),
            report_fields=("link", "link_scale"),
            full_runs=3,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "Увеличение масштаба сосредоточивает производную tanh и gaussian bump "
                "на меньшей доле выборки, поэтому качество и вероятность сходимости "
                "должны снижаться."
            ),
        ),
        Experiment(
            "si-init-by-function",
            "Single-index: инициализация по классам функций",
            _smoke_point(
                replace(hypothesis_base, link=init_links[0], index_init="local")
            ),
            tuple(
                replace(hypothesis_base, link=link, index_init=index_init)
                for link, index_init in product(init_links, ("local", "random"))
            ),
            report_fields=("link", "index_init"),
            full_runs=3,
            common_random_fields=("link", "index_init"),
            hypothesis=(
                "Локальная градиентная инициализация превосходит случайную, причём "
                "её преимущество сильнее для чётных, негладких и локализованных "
                "функций."
            ),
        ),
    )

    focused_dimensions = (40, 44, 48, 50, 52, 56, 60)
    focused_frequencies = (2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0)
    focused_tau = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    focused_displacements = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
    single_focus = (
        Experiment(
            "si-focus-d",
            "Single-index: повтор границы размерности",
            _smoke_point(
                replace(
                    si,
                    d=focused_dimensions[0],
                    N_lin=focused_dimensions[0] + 60,
                )
            ),
            tuple(replace(si, d=d, N_lin=d + 60) for d in focused_dimensions),
            report_fields=("d",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="d",
            common_random_fields=("d", "N_lin"),
            hypothesis=(
                "Граница устойчивого восстановления проходит внутри диапазона d=40--60."
            ),
        ),
        Experiment(
            "si-focus-frequency",
            "Single-index: повтор границы частоты",
            _smoke_point(
                replace(si, link="sin_scaled", link_scale=focused_frequencies[0])
            ),
            tuple(
                replace(si, link=link, link_scale=scale)
                for link, scale in product(("sin_scaled", "x_sin"), focused_frequencies)
            ),
            report_fields=("link", "link_scale"),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "Для sin(sx) качество резко падает около s=3--3.5, тогда как "
                "x sin(sx) остаётся более трудной и немонотонной функцией."
            ),
        ),
        Experiment(
            "si-focus-tau",
            "Single-index: повтор границы общего фактора",
            _smoke_point(replace(si, tau=focused_tau[0])),
            tuple(replace(si, tau=value) for value in focused_tau),
            report_fields=("tau",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="tau",
            common_random_fields=("tau",),
            hypothesis=(
                "Резкий переход к качественному восстановлению происходит между "
                "tau=0.7 и tau=0.8."
            ),
        ),
        Experiment(
            "si-focus-displacement",
            "Single-index: повтор границы смещения центров",
            _smoke_point(replace(si, center_displacement=focused_displacements[0])),
            tuple(
                replace(si, center_displacement=value)
                for value in focused_displacements
            ),
            report_fields=("center_displacement",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="center_displacement",
            common_random_fields=("center_displacement",),
            hypothesis=(
                "Умеренное смещение улучшает восстановление, тогда как приближение "
                "к единице повышает риск пустой компактной поддержки."
            ),
        ),
    )

    multi = (
        make(
            "mi-n",
            "Multi-index: объём выборки",
            tuple(replace(mi, n_samples=n, N_J=n // 2) for n in n_sweep_values),
            "n_samples",
        ),
        make(
            "mi-d",
            "Multi-index: размерность",
            tuple(replace(mi, d=d, N_lin=d + 60) for d in d_values),
            "d",
        ),
        make(
            "mi-noise",
            "Multi-index: шум",
            tuple(replace(mi, sigma_eps=value) for value in noise_values),
            "sigma_eps",
        ),
        make(
            "mi-tau",
            "Multi-index: общий фактор",
            tuple(replace(mi, tau=value) for value in tau_sweep_values),
            "tau",
        ),
        make(
            "mi-link",
            "Multi-index: функция связи",
            tuple(
                replace(mi, link=link)
                for link in ("multi_additive", "multi_multiplicative")
            ),
            "link",
        ),
        make(
            "mi-frequency-additive",
            "Multi-index: частота additive link",
            tuple(replace(mi, link_scale=s) for s in scale_sweep_values),
            "link_scale",
        ),
        make(
            "mi-frequency-multiplicative",
            "Multi-index: частота multiplicative link",
            tuple(
                replace(mi, link="multi_multiplicative", link_scale=s)
                for s in scale_sweep_values
            ),
            "link_scale",
        ),
        make(
            "mi-scale",
            "Multi-index: масштаб признаков при d=10",
            tuple(
                replace(mi, d=10, N_lin=70, sigma_x=value) for value in sigma_x_values
            ),
            "sigma_x",
        ),
        make(
            "mi-nlin",
            "Multi-index: локальная линейная масса",
            tuple(replace(mi, N_lin=mi.d + s * 20) for s in n_lin_scales),
            "N_lin",
        ),
        make(
            "mi-centers",
            "Multi-index: число центров",
            tuple(replace(mi, N_J=value) for value in center_counts),
            "N_J",
        ),
        make(
            "mi-displacement",
            "Multi-index: смещение центров",
            tuple(replace(mi, center_displacement=value) for value in displacements),
            "center_displacement",
        ),
        make(
            "mi-training",
            "Multi-index: обучающая выборка",
            tuple(
                replace(mi, training_set=value) for value in ("all", "exclude_centers")
            ),
            "training_set",
        ),
        make(
            "mi-init",
            "Multi-index: инициализация",
            tuple(replace(mi, index_init=value) for value in ("local", "random")),
            "index_init",
        ),
        make(
            "mi-kmax",
            "Multi-index: число внутренних AO-шагов",
            tuple(replace(mi, solver_max_steps=value) for value in ao_steps),
            "solver_max_steps",
        ),
        make(
            "mi-nloc",
            "Multi-index: локальная масса",
            tuple(replace(mi, N_loc=value) for value in n_loc_values),
            "N_loc",
        ),
        make(
            "mi-nphi",
            "Multi-index: число направлений",
            tuple(replace(mi, N_phi=value) for value in multi_directions),
            "N_phi",
        ),
        make(
            "mi-lambda",
            "Multi-index: регуляризация",
            tuple(replace(mi, lambda_penalty=value) for value in lambdas),
            "lambda_penalty",
        ),
        make(
            "mi-a",
            "Multi-index: коэффициент уменьшения h",
            tuple(replace(mi, a=value) for value in multi_a),
            "a",
        ),
        make(
            "mi-hmin",
            "Multi-index: нижняя граница h",
            tuple(replace(mi, h_min_factor=value) for value in h_min_factors),
            "h_min_factor",
        ),
        make(
            "mi-tensor",
            "Multi-index: тензор локализации",
            tuple(replace(mi, multi_tensor=value) for value in ("orthogonal", "full")),
            "multi_tensor",
        ),
        make(
            "mi-direction-law",
            "Multi-index: закон направлений",
            tuple(
                replace(mi, direction_mode=value)
                for value in ("isotropic", "localized")
            ),
            "direction_mode",
        ),
        make(
            "mi-direction-refresh",
            "Multi-index: обновление направлений",
            tuple(replace(mi, redraw_directions=value) for value in (False, True)),
            "redraw_directions",
        ),
    )

    multi_focus_base = replace(
        mi,
        d=25,
        n_over_d=40,
        n_samples=1000,
        N_loc=12,
        N_lin=85,
        N_J=500,
        N_phi=30,
        link_scale=3.0,
        multi_tensor="full",
    )
    d_boundary = (20, 22, 24, 25, 26, 28, 30)
    additive_frequencies = (2.0, 2.15, 2.3, 2.5, 2.7, 2.85, 3.0)
    focused_directions = (20, 24, 28, 30, 32, 36, 40)
    high_noise = (0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)
    multi_focus = (
        Experiment(
            "mi-focus-d",
            "Multi-index: уточнение границы размерности",
            _smoke_point(
                replace(
                    multi_focus_base,
                    d=d_boundary[0],
                    n_over_d=1000 / d_boundary[0],
                    N_lin=d_boundary[0] + 60,
                )
            ),
            tuple(
                replace(
                    multi_focus_base,
                    d=d,
                    n_over_d=1000 / d,
                    N_lin=d + 60,
                )
                for d in d_boundary
            ),
            report_fields=("d",),
            full_runs=3,
            quality_threshold=0.1,
            condition_field="d",
            common_random_fields=("d", "n_over_d", "N_lin"),
            hypothesis=(
                "После настройки остальных параметров граница восстановления "
                "остаётся внутри диапазона d=20--30."
            ),
        ),
        Experiment(
            "mi-focus-frequency-additive",
            "Multi-index: уточнение частоты additive-link",
            _smoke_point(replace(multi_focus_base, link_scale=additive_frequencies[0])),
            tuple(
                replace(multi_focus_base, link_scale=scale)
                for scale in additive_frequencies
            ),
            report_fields=("link_scale",),
            full_runs=3,
            quality_threshold=0.1,
            condition_field="link_scale",
            common_random_fields=("link_scale",),
            hypothesis=(
                "Увеличение частоты additive-link от 2 до 3 монотонно ухудшает "
                "восстановление подпространства."
            ),
        ),
        Experiment(
            "mi-focus-nphi",
            "Multi-index: уточнение числа направлений",
            _smoke_point(replace(multi_focus_base, N_phi=focused_directions[0])),
            tuple(
                replace(multi_focus_base, N_phi=value) for value in focused_directions
            ),
            report_fields=("N_phi",),
            full_runs=3,
            quality_threshold=0.1,
            condition_field="N_phi",
            common_random_fields=("N_phi",),
            hypothesis=(
                "Рост N_phi улучшает качество примерно до 30 направлений, после "
                "чего дополнительная память не даёт заметного выигрыша."
            ),
        ),
        Experiment(
            "mi-focus-tensor",
            "Multi-index: повторная проверка тензора",
            _smoke_point(replace(multi_focus_base, multi_tensor="orthogonal")),
            tuple(
                replace(multi_focus_base, multi_tensor=value)
                for value in ("orthogonal", "full")
            ),
            report_fields=("multi_tensor",),
            full_runs=3,
            common_random_fields=("multi_tensor",),
            hypothesis=(
                "Полный тензор локализации даёт меньшую ошибку проектора, чем "
                "ортогональный вариант."
            ),
        ),
        Experiment(
            "mi-focus-init",
            "Multi-index: повторная проверка инициализации",
            _smoke_point(replace(multi_focus_base, index_init="local")),
            tuple(
                replace(multi_focus_base, index_init=value)
                for value in ("local", "random")
            ),
            report_fields=("index_init",),
            full_runs=3,
            common_random_fields=("index_init",),
            hypothesis=(
                "Локальная градиентная инициализация устойчивее случайной при "
                "одинаковых данных и алгоритмической случайности."
            ),
        ),
        Experiment(
            "mi-focus-noise",
            "Multi-index: высокая шумовая граница",
            _smoke_point(replace(multi_focus_base, sigma_eps=high_noise[0])),
            tuple(replace(multi_focus_base, sigma_eps=value) for value in high_noise),
            report_fields=("sigma_eps",),
            full_runs=3,
            quality_threshold=0.1,
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
            hypothesis=(
                "При sigma_eps от 0.6 до 1.0 ошибка проектора возрастает, причём "
                "ADP перестаёт улучшать начальное подпространство."
            ),
        ),
    )

    link_cases = tuple(
        (link, scale) for link in ("sin_scaled", "x_sin") for scale in scales
    )
    multi_link_cases = tuple(
        (link, scale)
        for link in ("multi_additive", "multi_multiplicative")
        for scale in scales
    )
    single_breaking = tuple(
        replace(
            si,
            n_samples=n,
            N_J=n // 2,
            d=d,
            N_lin=d + 60,
            sigma_eps=noise,
            tau=tau,
            link=link,
            link_scale=scale,
        )
        for n, d, noise, tau, (link, scale) in product(
            n_values,
            d_values,
            noise_values,
            tau_values,
            link_cases,
        )
    )
    multi_breaking = tuple(
        replace(
            mi,
            n_samples=n,
            N_J=n // 2,
            d=d,
            N_lin=d + 60,
            sigma_eps=noise,
            tau=tau,
            link=link,
            link_scale=scale,
        )
        for n, d, noise, tau, (link, scale) in product(
            n_values,
            d_values,
            noise_values,
            tau_values,
            multi_link_cases,
        )
    )
    breaking = (
        make(
            "si-breaking",
            "Single-index: полная breaking-dimension сетка",
            single_breaking,
            "n_samples",
            "d",
            "sigma_eps",
            "tau",
            "link",
            "link_scale",
            full_runs=100,
        ),
        make(
            "mi-breaking",
            "Multi-index: полная breaking-dimension сетка",
            multi_breaking,
            "n_samples",
            "d",
            "sigma_eps",
            "tau",
            "link",
            "link_scale",
            full_runs=100,
        ),
    )
    return (
        *single,
        *single_hypotheses,
        *single_focus,
        *multi,
        *multi_focus,
        *breaking,
    )


def _smoke_point(point: ExperimentPoint) -> ExperimentPoint:
    """Уменьшить одну точку, сохранив исследуемый estimator-вариант."""
    multi = point.mode == "multi"
    return replace(
        point,
        d=4,
        n_over_d=12 if multi else 10,
        n_samples=48 if multi else 40,
        index_dim=2 if multi else 1,
        basis_pool_dim=3 if point.basis_pool_dim is not None else None,
        N_loc=6,
        N_lin=10,
        N_J=8,
        N_phi=4 if multi else 3,
        outer_steps=1,
        h_min_factor=3,
        solver_max_steps=2,
    )


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
        runs = experiment.full_runs if profile == "full" else 1
    validate_run_parameters(a.name, None if b is None else b.name, runs, seed)
    points = experiment.points(profile)
    builds = (a,) if b is None else (a, b)
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

    from .experiment_plots import build_report

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
        "theta": build.theta if metadata["solver"] == "lsmr" else None,
        "trust_radius": (build.trust_radius if metadata["solver"] == "lsmr" else None),
        "lsmr_maxiter": (build.lsmr_maxiter if metadata["solver"] == "lsmr" else None),
        "cg_maxiter": build.cg_maxiter if metadata["solver"] == "cg" else None,
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
        projector_distance: float | None = None
        max_principal_sine: float | None = None
        max_principal_angle_deg: float | None = None
    elif point.mode == "multi":
        quality_metric = "projector_distance"
        quality_direction = "lower"
        estimate = index.T
        quality, max_principal_sine, max_principal_angle_deg = _subspace_metrics(
            true_basis,
            estimate,
        )
        cosine = None
        projector_distance = quality
    else:
        quality_metric = "local_projector_distance"
        quality_direction = "lower"
        quality, max_principal_sine, max_principal_angle_deg = _local_subspace_metrics(
            true_basis, index
        )
        cosine = None
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
) -> tuple[float, float, float]:
    """Вернуть ошибку из multiindex.tex, худший синус и угол.

    Требуются конечные ортонормированные базисы одинаковой формы.
    Основная метрика равна ``sum(sin(theta_j)**2)``. Вычисление работает
    через главные углы без проекторов размера ``d x d``.
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
    # ESTIMATOR/evaluation protocol: основная ошибка из (SEDRqua) multiindex.tex.
    projector_distance = float(np.sum(np.square(sines)))
    max_principal_sine = float(np.max(sines))
    max_principal_angle_deg = math.degrees(float(np.max(angles)))
    return projector_distance, max_principal_sine, max_principal_angle_deg


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
                else "projector_distance"
            )
        ),
        "quality_direction": "higher" if point.mode == "single" else "lower",
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
        "schema_version": 8,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "experiment": experiment.selector,
        "title": experiment.title,
        "hypothesis": experiment.hypothesis,
        "profile": profile,
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
                        else "projector_distance"
                    )
                ),
                "direction": (
                    "higher" if experiment.smoke.mode == "single" else "lower"
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


def _make_seed_bundle(
    selector: str,
    point: ExperimentPoint,
    seed: int,
    *,
    common_random_fields: tuple[str, ...] = (),
) -> _SeedBundle:
    if selector == "custom":
        return _SeedBundle(*(seed for _ in fields(_SeedBundle)))
    parameters = asdict(point)
    for name in (
        "normalize_link_by_sigma_x",
        "basis_pool_dim",
        *common_random_fields,
    ):
        parameters.pop(name, None)
    payload = json.dumps(
        {
            "experiment": selector,
            "parameters": parameters,
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

    X = _features(point, seeds.features)
    link_divisor = point.sigma_x if point.normalize_link_by_sigma_x else 1.0
    if point.mode == "single":
        beta = _unit(
            np.random.default_rng(seeds.beta).normal(size=point.d),
            "beta",
        )
        index = X @ beta
        signal_values = _link(
            index / link_divisor,
            point.link,
            scale=point.link_scale,
        )
        noise_index = index
    elif point.mode == "multi":
        basis_pool_dim = point.basis_pool_dim or point.index_dim
        basis_pool, _ = np.linalg.qr(
            np.random.default_rng(seeds.beta).normal(size=(point.d, basis_pool_dim)),
            mode="reduced",
        )
        beta = _orient_columns(basis_pool)[:, : point.index_dim]
        projected = X @ beta
        signal_values = _multi_link(
            projected / link_divisor,
            point.link,
            point.link_scale,
        )
        noise_index = np.linalg.norm(projected, axis=1)
    else:
        radial = X[:, :2]
        radii = np.linalg.norm(radial, axis=1)
        if np.any(radii <= np.finfo(float).eps):
            raise RuntimeError("radial manifold data contain an undefined direction")
        beta = np.zeros((point.n, point.d, 1))
        beta[:, :2, 0] = radial / radii[:, None]
        signal_values = 0.5 * point.link_scale * np.square(radii)
        noise_index = radii
    signal = _standardize(signal_values, f"{point.link} link")
    noise = _noise(point, noise_index, seeds.noise)
    noise = _outliers(point, noise, seeds.outliers, seeds.outlier_noise)
    Y = signal + noise
    if point.delta > 0 and point.mode == "single":
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
    if point.tau is not None:
        # ESTIMATOR/data design: точная формула из TeX, без переименования tau в corr.
        common = rng.normal(size=(point.n, 1))
        values = point.tau * common + (1 - point.tau) * rng.normal(size=shape)
    elif point.x_distribution == "gaussian":
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
    return _fail("cannot generate a direction orthogonal to beta")


def _link(index: np.ndarray, name: LinkName, *, scale: float = 1.0) -> np.ndarray:
    if name == "linear":
        return index
    if name == "quadratic":
        return index + 0.5 * index**2
    if name == "square":
        return index**2
    if name == "cubic":
        return index**3
    if name == "quartic":
        return index**4
    if name == "sin":
        return np.sin(1.5 * index)
    if name == "tanh":
        return np.tanh(2 * index)
    if name == "sin_scaled":
        return np.sin(scale * index)
    if name == "cos_scaled":
        return np.cos(scale * index)
    if name == "x_sin":
        return index * np.sin(scale * index)
    if name == "tanh_scaled":
        return np.tanh(scale * index)
    if name == "absolute":
        return np.abs(index)
    if name == "relu":
        return np.maximum(index, 0.0)
    if name == "gaussian_bump":
        return np.exp(-0.5 * np.square(scale * index))
    validate_single_link(name)
    return index * np.sin(math.sqrt(5) * index)


def _multi_link(
    projected: np.ndarray,
    name: LinkName,
    scale: float,
) -> np.ndarray:
    validate_multi_link(projected, name)
    if name == "multi_additive":
        values = projected[:, 0] ** 2 + np.sin(scale * projected[:, 1])
    else:
        values = projected[:, 0] * np.sin(scale * projected[:, 1])
    if projected.shape[1] > 2:
        # ESTIMATOR/data design: каждая дополнительная координата участвует явно.
        denominators = np.arange(3, projected.shape[1] + 1)
        values = values + np.sum(projected[:, 2:] ** 2 / denominators, axis=1)
    return np.asarray(values)


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    return np.asarray(basis * np.where(signs == 0, 1.0, signs))


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
    single_hypotheses = (
        "si-function-classes",
        "si-parity-frequency",
        "si-gradient-support",
        "si-init-by-function",
    )
    focused_single = (
        "si-focus-d",
        "si-focus-frequency",
        "si-focus-tau",
        "si-focus-displacement",
    )
    focused_multi = (
        "mi-focus-d",
        "mi-focus-frequency-additive",
        "mi-focus-nphi",
        "mi-focus-tensor",
        "mi-focus-init",
        "mi-focus-noise",
    )
    detailed_nd = ("mi-boundary-nd", "si-nloc-nd")
    parameter_scaling = tuple(name for name in catalog if name.startswith("scale-"))
    manifold_basic = tuple(
        name for name in catalog if name == "manifold" or name.startswith("manifold-")
    )
    aliases = {
        "all": tuple(catalog),
        "tex-all": tuple(
            name
            for name in catalog
            if name.startswith(("si-", "mi-"))
            and name not in {"mi-1", "mi-2", "mi-3", "mi-4", "mi-5"}
            and name not in single_hypotheses
            and name not in focused_single
            and name not in focused_multi
            and name not in detailed_nd
            and not name.endswith("-breaking")
        ),
        "si-hypotheses": single_hypotheses,
        "batch-2": (*focused_multi, *single_hypotheses),
        "batch-3": focused_single,
        "parameter-scaling": parameter_scaling,
        "parameter-scaling-si": tuple(
            name for name in parameter_scaling if name.startswith("scale-si-")
        ),
        "parameter-scaling-mi": tuple(
            name for name in parameter_scaling if name.startswith("scale-mi-")
        ),
        "nd-detail": detailed_nd,
        "manifold-basic": manifold_basic,
        "tex-tuning": (
            "si-nlin",
            "si-centers",
            "si-displacement",
            "si-training",
            "si-kmax",
            "si-nloc",
            "si-nphi",
            "si-lambda",
            "si-a",
            "si-hmin",
            "mi-nlin",
            "mi-centers",
            "mi-displacement",
            "mi-training",
            "mi-init",
            "mi-kmax",
            "mi-nloc",
            "mi-nphi",
            "mi-lambda",
            "mi-a",
            "mi-hmin",
            "mi-tensor",
            "mi-direction-law",
            "mi-direction-refresh",
        ),
        "report": tuple(
            name
            for name in catalog
            if name.startswith(("si-", "mi-")) and not name.endswith("-breaking")
        ),
        "si": tuple(
            name
            for name in catalog
            if name.startswith("si-") and not name.endswith("-breaking")
        ),
        "mi": tuple(
            name
            for name in catalog
            if name.startswith("mi-") and not name.endswith("-breaking")
        ),
    }
    return select_experiments(value, catalog, aliases)


def _has_numerical_failures(series_dir: Path) -> bool:
    with (series_dir / "runs.csv").open(encoding="utf-8") as stream:
        return any(
            row["status"] == "numerical_failure" for row in csv.DictReader(stream)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal paired ADP experiments")
    parser.add_argument("--experiment", default="custom")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/experiments")
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--solver", choices=("lsmr", "cg"), default="lsmr")
    parser.add_argument("--solver-max-steps", type=int, default=3)
    parser.add_argument("--cg-maxiter", type=int)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        for experiment in CATALOG.values():
            print(
                f"{experiment.selector:>24}  "
                f"points={len(experiment.full):>5} runs={experiment.full_runs:>3}  "
                f"{experiment.title}"
            )
        return 0
    failed = False
    try:
        experiments = _selected_experiments(
            args.experiment,
            custom=custom_experiment(args.n, args.d, args.noise),
        )
        experiment_id = _experiment_id()
        build = Build(
            "ADP",
            ADP_Config(),
            solver=args.solver,
            solver_max_steps=args.solver_max_steps,
            cg_maxiter=args.cg_maxiter,
        )
        for experiment in experiments:
            path = experiment.run(
                build,
                profile=args.profile,
                runs=args.runs,
                seed=args.seed,
                output_dir=args.output_dir,
                plots=not args.no_plots,
                experiment_id=experiment_id,
                progress=True,
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
