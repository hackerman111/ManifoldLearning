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
    add_report_metrics,
    prepare_quantile_band,
    write_single_index_reports,
)
from .storage import SingleIndexSeriesStore
from .types import (
    ExperimentParameters,
    RunOutcome,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
    parse_local_solver_selection,
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
    "add_report_metrics",
    "build_single_index_jobs",
    "full_parameter_grid",
    "parse_experiment_selectors",
    "parse_local_solver_selection",
    "parse_seed_selection",
    "prepare_quantile_band",
    "run_single_index_benchmark",
    "smoke_parameter_grid",
    "write_single_index_reports",
]
