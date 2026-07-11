"""Воспроизводимый benchmark single-index ADP."""

from .scenarios import PROFILE_IDS, scenario_registry, scenarios_for_profile
from .runner import build_single_index_jobs, run_single_index_benchmark
from .reports import (
    build_single_index_summary,
    fit_scaling_exponents,
    paired_method_differences,
    select_worst_five,
    write_single_index_reports,
)
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
    "build_single_index_summary",
    "fit_scaling_exponents",
    "paired_method_differences",
    "run_single_index_benchmark",
    "select_worst_five",
    "scenario_registry",
    "scenarios_for_profile",
    "write_single_index_reports",
]
