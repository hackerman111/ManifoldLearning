from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from ...common.experiment_log import Scalar


ScenarioExecutor = Literal["correctness", "recovery", "scaling", "real_data"]
ScenarioFamily = Literal["C", "S", "T", "R", "M", "I", "B", "A", "D"]

_SCENARIO_ID = re.compile(r"^[CSTRMIBAD]\d{2}$")
_POSITIVE_KEYS = {
    "n",
    "d",
    "n_centers",
    "n_directions",
    "min_neighbors",
    "outer_steps",
    "inner_steps",
    "chunk_size",
}


@dataclass(frozen=True, slots=True)
class SingleIndexScenario:
    scenario_id: str
    family: ScenarioFamily
    executor: ScenarioExecutor
    hypothesis: str
    data: dict[str, Scalar]
    algorithm: dict[str, Scalar]
    solver: dict[str, Scalar]
    repeats: int
    methods: tuple[str, ...]
    record_solver_trace: bool = False

    def __post_init__(self) -> None:
        if not _SCENARIO_ID.fullmatch(self.scenario_id):
            raise ValueError("scenario_id must match FAMILY plus two digits")
        if self.family != self.scenario_id[0]:
            raise ValueError("scenario family must match scenario_id")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if not self.methods or any(not method for method in self.methods):
            raise ValueError("methods must not be empty")
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        for group_name, values in (
            ("data", self.data),
            ("algorithm", self.algorithm),
            ("solver", self.solver),
        ):
            _validate_scalar_mapping(group_name, values)


def _validate_scalar_mapping(group_name: str, values: dict[str, Scalar]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"{group_name}.{key} must be finite")
            if key in _POSITIVE_KEYS and value <= 0:
                raise ValueError(f"{group_name}.{key} must be positive")
        if key == "corr" and isinstance(value, (int, float)):
            if not -1.0 < float(value) < 1.0:
                raise ValueError(f"{group_name}.corr must be in (-1, 1)")
        if key == "noise" and isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"{group_name}.noise must be nonnegative")
        if key == "sigma_x" and isinstance(value, (int, float)) and value <= 0:
            raise ValueError(f"{group_name}.sigma_x must be positive")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    data: int
    beta: int
    centers: int
    directions: int
    init: int


@dataclass(frozen=True, slots=True)
class SingleIndexJob:
    scenario: SingleIndexScenario
    method: str
    repeat: int
    seeds: SeedBundle
    run_id: str

    def __post_init__(self) -> None:
        if self.repeat < 0:
            raise ValueError("repeat must be nonnegative")
        if self.method not in self.scenario.methods:
            raise ValueError("job method must be enabled by scenario")
        if not self.run_id:
            raise ValueError("run_id must not be empty")


@dataclass(frozen=True, slots=True)
class SingleIndexSeriesConfig:
    profile: str
    base_seed: int
    jobs: int
    statistics_workers: int
    retry_failed: bool = False
    max_scenarios: int | None = None
    data_dir: str | None = None
    allow_download: bool = False

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile must not be empty")
        if isinstance(self.jobs, bool) or not isinstance(self.jobs, int) or self.jobs < 1:
            raise ValueError("jobs must be positive")
        if (
            isinstance(self.statistics_workers, bool)
            or not isinstance(self.statistics_workers, int)
            or self.statistics_workers < 1
        ):
            raise ValueError("statistics_workers must be positive")
        if self.max_scenarios is not None and self.max_scenarios < 1:
            raise ValueError("max_scenarios must be positive")

