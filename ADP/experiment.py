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
from tqdm import tqdm

from .cli import _run, build_parser
from .core.ADP_Config import ADP_Config

LinkName = Literal[
    "linear",
    "quadratic",
    "square",
    "sin",
    "tanh",
    "oscillating",
    "sin_scaled",
    "x_sin",
    "multi_additive",
    "multi_multiplicative",
]
FeatureDistribution = Literal["gaussian", "uniform", "student_t5"]
NoiseDistribution = Literal["gaussian", "student_t5", "student_t3"]
ModelMode = Literal["single", "multi"]


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
    N_loc: int | None = None
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
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
        if self.n_samples is not None and (
            isinstance(self.n_samples, bool)
            or not isinstance(self.n_samples, int)
            or self.n_samples < 2
        ):
            raise ValueError("n_samples must be an integer of at least two")
        if self.mode not in {"single", "multi"}:
            raise ValueError("mode must be 'single' or 'multi'")
        if (
            isinstance(self.index_dim, bool)
            or not isinstance(self.index_dim, int)
            or not 1 <= self.index_dim <= self.d
        ):
            raise ValueError("index_dim must lie between one and d")
        if self.mode == "single" and self.index_dim != 1:
            raise ValueError("single mode requires index_dim=1")
        if self.mode == "multi" and self.index_dim >= self.d:
            raise ValueError("multi mode requires index_dim < d")
        if self.tau is not None and (
            not np.isfinite(self.tau) or not 0 <= self.tau <= 1
        ):
            raise ValueError("tau must lie in [0, 1] or be None")
        _positive("link_scale", self.link_scale)
        if self.link not in {
            "linear",
            "quadratic",
            "square",
            "sin",
            "tanh",
            "oscillating",
            "sin_scaled",
            "x_sin",
            "multi_additive",
            "multi_multiplicative",
        }:
            raise ValueError(f"unknown link: {self.link}")
        multi_links = {"multi_additive", "multi_multiplicative"}
        if (self.mode == "multi") != (self.link in multi_links):
            raise ValueError("link and mode must both be single-index or multi-index")
        if self.x_distribution not in {"gaussian", "uniform", "student_t5"}:
            raise ValueError(f"unknown feature distribution: {self.x_distribution}")
        if self.noise_distribution not in {"gaussian", "student_t5", "student_t3"}:
            raise ValueError(f"unknown noise distribution: {self.noise_distribution}")
        if not isinstance(self.heteroscedastic, bool):
            raise ValueError("heteroscedastic must be boolean")
        for name in (
            "N_loc",
            "N_lin",
            "N_J",
            "N_phi",
            "outer_steps",
            "solver_max_steps",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in ("lambda_penalty", "center_displacement"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(name, value)
        for name in ("a", "h_min_factor"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.a is not None and self.a <= 1:
            raise ValueError("a must exceed one")
        if self.index_init is not None and self.index_init not in {
            "local",
            "pilot",
            "random",
        }:
            raise ValueError("unknown index_init")
        if self.direction_mode is not None and self.direction_mode not in {
            "auto",
            "isotropic",
            "localized",
        }:
            raise ValueError("unknown direction_mode")
        if self.multi_tensor is not None and self.multi_tensor not in {
            "orthogonal",
            "full",
        }:
            raise ValueError("unknown multi_tensor")
        if self.select_step is not None and self.select_step not in {"best", "last"}:
            raise ValueError("unknown select_step")
        if self.training_set is not None and self.training_set not in {
            "all",
            "exclude_centers",
        }:
            raise ValueError("unknown training_set")
        if self.redraw_directions is not None and not isinstance(
            self.redraw_directions, bool
        ):
            raise ValueError("redraw_directions must be boolean or None")

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
        if self.solver not in {"lsmr", "cg"}:
            raise ValueError("solver must be 'lsmr' or 'cg'")
        if self.cg_maxiter is not None and (
            isinstance(self.cg_maxiter, bool)
            or not isinstance(self.cg_maxiter, int)
            or self.cg_maxiter < 1
        ):
            raise ValueError("cg_maxiter must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class Experiment:
    """Описание воспроизводимой smoke/full сетки ADP."""

    selector: str
    title: str
    smoke: ExperimentPoint
    full: tuple[ExperimentPoint, ...]
    report_fields: tuple[str, ...] = ()
    full_runs: int = 1

    def __post_init__(self) -> None:
        if not self.selector or Path(self.selector).name != self.selector:
            raise ValueError("experiment selector must be a path-safe name")
        if not self.title:
            raise ValueError("experiment title must not be empty")
        if not self.full:
            raise ValueError("full experiment grid must not be empty")
        point_fields = {item.name for item in fields(ExperimentPoint)}
        if any(name not in point_fields for name in self.report_fields):
            raise ValueError("report_fields must name ExperimentPoint fields")
        if (
            isinstance(self.full_runs, bool)
            or not isinstance(self.full_runs, int)
            or self.full_runs < 1
        ):
            raise ValueError("full_runs must be a positive integer")

    def points(self, profile: str) -> tuple[ExperimentPoint, ...]:
        if profile == "smoke":
            return (self.smoke,)
        if profile == "full":
            return self.full
        raise ValueError("profile must be 'smoke' or 'full'")

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
        *_report_catalog(),
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
    d_values = tuple(range(10, 101, 10))
    noise_values = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    tau_values = (0.0, 0.2, 0.4, 0.8)
    scales = (1.0, 2.0, 3.0, 4.0)
    n_loc_values = (7, 10, 15, 20)
    lambdas = (0.0, 0.05, 0.1, 0.5, 1.0)
    center_counts = tuple(
        scale * si.n // (si.N_loc or 1) for scale in range(1, (si.N_loc or 1) + 1)
    )

    def make(
        selector: str,
        title: str,
        points: tuple[ExperimentPoint, ...],
        *report_fields: str,
        full_runs: int = 1,
    ) -> Experiment:
        return Experiment(
            selector,
            title,
            _smoke_point(points[0]),
            points,
            report_fields,
            full_runs,
        )

    single = (
        make(
            "si-n",
            "Single-index: объём выборки",
            tuple(replace(si, n_samples=n, N_J=n // 2) for n in n_values),
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
            tuple(replace(si, tau=value) for value in tau_values),
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
            tuple(replace(si, link="sin_scaled", link_scale=s) for s in scales),
            "link_scale",
        ),
        make(
            "si-frequency-xsin",
            "Single-index: частота x sin(sx)",
            tuple(replace(si, link="x_sin", link_scale=s) for s in scales),
            "link_scale",
        ),
        make(
            "si-scale",
            "Single-index: масштаб признаков при d=10",
            tuple(
                replace(si, d=10, N_lin=70, sigma_x=value)
                for value in (0.25, 0.5, 1.0, 2.0, 4.0)
            ),
            "sigma_x",
        ),
        make(
            "si-nlin",
            "Single-index: локальная линейная масса",
            tuple(replace(si, N_lin=si.d + s * 20) for s in (1, 2, 3)),
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
            tuple(replace(si, center_displacement=value) for value in (0, 0.1, 0.5, 1)),
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
            tuple(replace(si, solver_max_steps=value) for value in (3, 5, 7)),
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
            tuple(replace(si, N_phi=value) for value in (20, 40, 60)),
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
            tuple(replace(si, a=value) for value in (2**0.25, math.sqrt(2), 2.0)),
            "a",
        ),
        make(
            "si-hmin",
            "Single-index: нижняя граница h",
            tuple(replace(si, h_min_factor=value) for value in (1.0, 2.0, 3.0)),
            "h_min_factor",
        ),
    )

    multi = (
        make(
            "mi-n",
            "Multi-index: объём выборки",
            tuple(replace(mi, n_samples=n, N_J=n // 2) for n in n_values),
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
            tuple(replace(mi, tau=value) for value in tau_values),
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
            tuple(replace(mi, link_scale=s) for s in scales),
            "link_scale",
        ),
        make(
            "mi-frequency-multiplicative",
            "Multi-index: частота multiplicative link",
            tuple(
                replace(mi, link="multi_multiplicative", link_scale=s) for s in scales
            ),
            "link_scale",
        ),
        make(
            "mi-scale",
            "Multi-index: масштаб признаков при d=10",
            tuple(
                replace(mi, d=10, N_lin=70, sigma_x=value)
                for value in (0.25, 0.5, 1.0, 2.0, 4.0)
            ),
            "sigma_x",
        ),
        make(
            "mi-nlin",
            "Multi-index: локальная линейная масса",
            tuple(replace(mi, N_lin=mi.d + s * 20) for s in (1, 2, 3)),
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
            tuple(replace(mi, center_displacement=value) for value in (0, 0.1, 0.5, 1)),
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
            tuple(replace(mi, solver_max_steps=value) for value in (3, 5, 7)),
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
            tuple(replace(mi, N_phi=value) for value in (3, 20, 40, 60)),
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
            tuple(
                replace(mi, a=value)
                for value in (2 ** (1 / 4), 2 ** (1 / 3), math.sqrt(2))
            ),
            "a",
        ),
        make(
            "mi-hmin",
            "Multi-index: нижняя граница h",
            tuple(replace(mi, h_min_factor=value) for value in (1.0, 2.0, 3.0)),
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
    return (*single, *multi, *breaking)


def _smoke_point(point: ExperimentPoint) -> ExperimentPoint:
    """Уменьшить одну точку, сохранив исследуемый estimator-вариант."""
    multi = point.mode == "multi"
    return replace(
        point,
        d=4,
        n_over_d=12 if multi else 10,
        n_samples=48 if multi else 40,
        index_dim=2 if multi else 1,
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
    "quality",
    "cosine_abs",
    "projector_distance",
    "initial_quality",
    "last_quality",
    "initial_eigenvalues",
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
    if b is not None and a.name == b.name:
        raise ValueError("A and B build names must differ")
    if runs is None:
        runs = experiment.full_runs if profile == "full" else 1
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
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
                seeds = _make_seed_bundle(experiment.selector, point, run_seed)
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
                        row.update(_fit(build, point, generated, seeds.init))
                    except Exception as error:  # Один fit не отменяет серию.
                        row.update(
                            status="numerical_failure",
                            error=_error_text(error),
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
        "theta": build.theta if build.solver == "lsmr" else None,
        "trust_radius": build.trust_radius if build.solver == "lsmr" else None,
        "lsmr_maxiter": build.lsmr_maxiter if build.solver == "lsmr" else None,
        "cg_maxiter": build.cg_maxiter if build.solver == "cg" else None,
        "N_lin": metadata["N_lin"],
        "N_J": metadata["N_J"],
        "N_phi": metadata["N_phi"],
        "training_size": metadata["training_size"],
        "center_displacement_scale": metadata["center_displacement_scale"],
    }
    if point.mode == "single":
        quality_metric = "cosine_abs"
        quality_direction = "higher"
        quality = float(abs(true_basis[:, 0] @ index))
        cosine = quality
        projector_distance: float | None = None
    else:
        quality_metric = "projector_distance"
        quality_direction = "lower"
        estimate = index.T
        quality = float(
            np.linalg.norm(
                true_basis @ true_basis.T - estimate @ estimate.T,
                ord="fro",
            )
            / math.sqrt(2 * point.index_dim)
        )
        cosine = None
        projector_distance = quality
    return {
        "effective_config": _compact_json(effective),
        "status": "nonconverged" if converged is False else "success",
        "quality_metric": quality_metric,
        "quality_direction": quality_direction,
        "quality": quality,
        "cosine_abs": cosine,
        "projector_distance": projector_distance,
        "initial_quality": metadata["initial_quality"],
        "last_quality": metadata["last_quality"],
        "initial_eigenvalues": _compact_json(metadata["initial_eigenvalues"]),
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
        "requested_config": _compact_json(
            {
                "config": _config_spec(requested_config),
                "solver_tol": build.solver_tol,
                "solver_max_steps": point.solver_max_steps or build.solver_max_steps,
                "solver": build.solver,
                "theta": build.theta,
                "trust_radius": build.trust_radius,
                "lsmr_maxiter": build.lsmr_maxiter,
                "cg_maxiter": build.cg_maxiter,
            }
        ),
        "status": "",
        "error": "",
        "quality_metric": (
            "cosine_abs" if point.mode == "single" else "projector_distance"
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
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "experiment": experiment.selector,
        "title": experiment.title,
        "profile": profile,
        "runs": runs,
        "full_runs": experiment.full_runs,
        "report_fields": experiment.report_fields,
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
        "seed_design": "split-data-components-and-paired-build-model-seed-v2",
        "feature_formula": (
            "sigma_x * (tau * z0 + (1 - tau) * zi)"
            if any(point.tau is not None for point in experiment.full)
            else "catalog-specific legacy design"
        ),
        "multi_link_extension": "sum_{r=3}^m z_r^2 / r",
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


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{name}"


def _experiment_id(value: str | None = None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%dT%H%M%S%f")
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("experiment_id must be a path-safe name")
    return value


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

    X = _features(point, seeds.features)
    if point.mode == "single":
        beta = _unit(
            np.random.default_rng(seeds.beta).normal(size=point.d),
            "beta",
        )
        index = X @ beta
        divisor = point.sigma_x if selector == "5" else 1.0
        signal_values = _link(
            index / divisor,
            point.link,
            scale=point.link_scale,
        )
        noise_index = index
    else:
        basis, _ = np.linalg.qr(
            np.random.default_rng(seeds.beta).normal(size=(point.d, point.index_dim)),
            mode="reduced",
        )
        beta = _orient_columns(basis)
        projected = X @ beta
        signal_values = _multi_link(projected, point.link, point.link_scale)
        noise_index = np.linalg.norm(projected, axis=1)
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
    raise ValueError("cannot generate a direction orthogonal to beta")


def _link(index: np.ndarray, name: LinkName, *, scale: float = 1.0) -> np.ndarray:
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
    if name == "sin_scaled":
        return np.sin(scale * index)
    if name == "x_sin":
        return index * np.sin(scale * index)
    if name in {"multi_additive", "multi_multiplicative"}:
        raise ValueError(f"{name} requires multi-index projected data")
    return index * np.sin(math.sqrt(5) * index)


def _multi_link(
    projected: np.ndarray,
    name: LinkName,
    scale: float,
) -> np.ndarray:
    if projected.ndim != 2 or projected.shape[1] < 2:
        raise ValueError("multi-index links require at least two coordinates")
    if name == "multi_additive":
        values = projected[:, 0] ** 2 + np.sin(scale * projected[:, 1])
    elif name == "multi_multiplicative":
        values = projected[:, 0] * np.sin(scale * projected[:, 1])
    else:
        raise ValueError(f"unknown multi-index link: {name}")
    if projected.shape[1] > 2:
        # ESTIMATOR/data design: каждая дополнительная координата участвует явно.
        denominators = np.arange(3, projected.shape[1] + 1)
        values = values + np.sum(projected[:, 2:] ** 2 / denominators, axis=1)
    return np.asarray(values)


def _orient_columns(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    return np.asarray(basis * np.where(signs == 0, 1.0, signs))


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
    requested = tuple(part.strip() for part in value.split(","))
    if not requested or any(not selector for selector in requested):
        raise ValueError("experiment selectors must not be empty")
    aliases = {
        "all": tuple(catalog),
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
    unknown = sorted(set(requested) - set(catalog) - set(aliases))
    if unknown:
        raise ValueError(f"unknown experiment selector: {', '.join(unknown)}")
    selectors = tuple(
        dict.fromkeys(
            selector for item in requested for selector in aliases.get(item, (item,))
        )
    )
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
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/experiments")
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--solver", choices=("lsmr", "cg"), default="lsmr")
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
