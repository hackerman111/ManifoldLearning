"""Воспроизводимый benchmark single-index ADP."""

from .scenarios import (
    EXPERIMENT_COUNTS,
    EXPERIMENT_SELECTORS,
    PROFILE_IDS,
    full_parameter_grid,
    parse_experiment_selectors,
    parse_seed_selection,
    smoke_parameter_grid,
)
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
    ExperimentParameters,
    RunOutcome,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)

__all__ = [
    "EXPERIMENT_COUNTS",
    "EXPERIMENT_SELECTORS",
    "ExperimentParameters",
    "PROFILE_IDS",
    "RunOutcome",
    "SeedBundle",
    "SingleIndexJob",
    "SingleIndexSeriesConfig",
    "SingleIndexSeriesStore",
    "build_single_index_jobs",
    "build_single_index_summary",
    "fit_scaling_exponents",
    "full_parameter_grid",
    "paired_method_differences",
    "parse_experiment_selectors",
    "parse_seed_selection",
    "run_single_index_benchmark",
    "select_worst_five",
    "smoke_parameter_grid",
    "write_single_index_reports",
]
