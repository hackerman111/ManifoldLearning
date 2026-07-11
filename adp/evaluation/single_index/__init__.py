"""Воспроизводимый benchmark single-index ADP."""

from .scenarios import PROFILE_IDS, scenario_registry, scenarios_for_profile
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
    "scenario_registry",
    "scenarios_for_profile",
]
