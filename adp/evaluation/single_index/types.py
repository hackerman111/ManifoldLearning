from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Literal

from ...common.experiment_log import Scalar


EXPERIMENT_SELECTORS = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7.1",
    "7.2",
    "8.1",
    "8.2",
    "8.3",
)

LinkName = Literal[
    "linear",
    "quadratic",
    "square",
    "sin",
    "tanh",
    "oscillating",
]
FeatureDistribution = Literal["gaussian", "uniform", "student_t5"]
NoiseDistribution = Literal["gaussian", "student_t5", "student_t3"]

_LINKS = frozenset(
    {"linear", "quadratic", "square", "sin", "tanh", "oscillating"}
)
_FEATURE_DISTRIBUTIONS = frozenset({"gaussian", "uniform", "student_t5"})
_NOISE_DISTRIBUTIONS = frozenset({"gaussian", "student_t5", "student_t3"})


@dataclass(frozen=True, slots=True)
class ExperimentParameters:
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
    center_fraction: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, int) or self.d < 1:
            raise ValueError("d must be a positive integer")
        _require_positive_finite("n_over_d", self.n_over_d)
        _require_positive_finite("sigma_x", self.sigma_x)
        _require_nonnegative_finite("rho_corr", self.rho_corr)
        if float(self.rho_corr) >= 1.0:
            raise ValueError("rho_corr must be less than 1")
        _require_nonnegative_finite("sigma_eps", self.sigma_eps)
        if not isinstance(self.link, str) or self.link not in _LINKS:
            raise ValueError(f"unknown link: {self.link}")
        if (
            not isinstance(self.x_distribution, str)
            or self.x_distribution not in _FEATURE_DISTRIBUTIONS
        ):
            raise ValueError(f"unknown feature distribution: {self.x_distribution}")
        if (
            not isinstance(self.noise_distribution, str)
            or self.noise_distribution not in _NOISE_DISTRIBUTIONS
        ):
            raise ValueError(f"unknown noise distribution: {self.noise_distribution}")
        if not isinstance(self.heteroscedastic, bool):
            raise ValueError("heteroscedastic must be boolean")
        _require_nonnegative_finite("outlier_fraction", self.outlier_fraction)
        if float(self.outlier_fraction) > 1.0:
            raise ValueError("outlier_fraction must not exceed 1")
        _require_positive_finite("outlier_scale", self.outlier_scale)
        _require_nonnegative_finite("delta", self.delta)
        _require_center_fraction(self.center_fraction)
        for name in (
            "n_over_d",
            "sigma_x",
            "rho_corr",
            "sigma_eps",
            "outlier_fraction",
            "outlier_scale",
            "delta",
            "center_fraction",
        ):
            object.__setattr__(self, name, _canonical_float(getattr(self, name)))

    @property
    def n(self) -> int:
        return math.ceil(self.d * self.n_over_d)

    @property
    def n_centers(self) -> int:
        return min(self.n, math.ceil(self.center_fraction * self.n))


@dataclass(frozen=True, slots=True)
class SeedBundle:
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

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{item.name} seed must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SingleIndexJob:
    experiment: str
    parameters: ExperimentParameters
    seed: int
    seeds: SeedBundle
    run_id: str
    diagnostic: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.experiment, str)
            or self.experiment not in EXPERIMENT_SELECTORS
        ):
            raise ValueError(f"unknown experiment selector: {self.experiment}")
        if not isinstance(self.parameters, ExperimentParameters):
            raise ValueError("parameters must be ExperimentParameters")
        _require_seed("seed", self.seed)
        if not isinstance(self.seeds, SeedBundle):
            raise ValueError("seeds must be SeedBundle")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not isinstance(self.diagnostic, bool):
            raise ValueError("diagnostic must be boolean")


@dataclass(frozen=True, slots=True)
class SingleIndexSeriesConfig:
    profile: Literal["smoke", "full"] = "smoke"
    experiments: tuple[str, ...] = EXPERIMENT_SELECTORS
    jobs: int | Literal["auto"] = "auto"
    seeds: tuple[int, ...] | None = None
    diagnostic_seeds: tuple[int, ...] = (0, 1, 2)
    center_fraction: float = 1.0
    retry_failed: bool = False
    max_runs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or self.profile not in {"smoke", "full"}:
            raise ValueError(f"unknown single-index profile: {self.profile}")
        if not isinstance(self.experiments, tuple) or not self.experiments:
            raise ValueError("experiments must be a nonempty tuple")
        if any(not isinstance(selector, str) for selector in self.experiments):
            raise ValueError("experiments must contain selector strings")
        unknown = sorted(set(self.experiments) - set(EXPERIMENT_SELECTORS))
        if unknown:
            raise ValueError(f"unknown experiment selector: {', '.join(unknown)}")
        selected = set(self.experiments)
        object.__setattr__(
            self,
            "experiments",
            tuple(selector for selector in EXPERIMENT_SELECTORS if selector in selected),
        )
        if self.jobs != "auto" and (
            isinstance(self.jobs, bool)
            or not isinstance(self.jobs, int)
            or self.jobs < 1
        ):
            raise ValueError("jobs must be 'auto' or a positive integer")
        if self.seeds is not None:
            object.__setattr__(self, "seeds", _canonical_seeds("seeds", self.seeds))
        object.__setattr__(
            self,
            "diagnostic_seeds",
            _canonical_seeds(
                "diagnostic_seeds",
                self.diagnostic_seeds,
                allow_empty=True,
            ),
        )
        _require_center_fraction(self.center_fraction)
        object.__setattr__(self, "center_fraction", float(self.center_fraction))
        if not isinstance(self.retry_failed, bool):
            raise ValueError("retry_failed must be boolean")
        if self.max_runs is not None and (
            isinstance(self.max_runs, bool)
            or not isinstance(self.max_runs, int)
            or self.max_runs < 1
        ):
            raise ValueError("max_runs must be a positive integer")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    metrics: dict[str, Scalar]
    iterations: tuple[dict[str, Scalar], ...]
    solver_iterations: tuple[dict[str, Scalar], ...]
    stop_reason: str
    algorithm_usage: dict[str, Scalar] = field(default_factory=dict)


def _require_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_center_fraction(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("center_fraction must be in (0, 1]")
    if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
        raise ValueError("center_fraction must be in (0, 1]")


def _require_seed(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must contain nonnegative integers")


def _canonical_float(value: object) -> float:
    canonical = float(value)
    return 0.0 if canonical == 0.0 else canonical


def _canonical_seeds(
    name: str,
    values: tuple[int, ...],
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _require_seed(name, value)
    return tuple(sorted(set(values)))


__all__ = [
    "EXPERIMENT_SELECTORS",
    "ExperimentParameters",
    "RunOutcome",
    "SeedBundle",
    "SingleIndexJob",
    "SingleIndexSeriesConfig",
]
