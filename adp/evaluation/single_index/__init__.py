"""Воспроизводимый benchmark single-index ADP."""

from .scenarios import PROFILE_IDS, scenario_registry, scenarios_for_profile
from .runner import build_single_index_jobs, run_single_index_benchmark
from .storage import SingleIndexSeriesStore
from .types import (
    RunOutcome,
    SeedBundle,
    SingleIndexJob,
    SingleIndexScenario,
    SingleIndexSeriesConfig,
)

__all__ = [
    "PROFILE_IDS",
    "RunOutcome",
    "SeedBundle",
    "SingleIndexJob",
    "SingleIndexScenario",
    "SingleIndexSeriesConfig",
    "SingleIndexSeriesStore",
    "build_single_index_jobs",
    "run_single_index_benchmark",
    "scenario_registry",
    "scenarios_for_profile",
]
