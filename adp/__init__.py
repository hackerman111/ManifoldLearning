"""Average Derivative Procedure.

Основная точка входа:

    model = ADP.create("new", ADPConfig(...))
    data = model.generate_data(...)
    result = model.fit(data.X, data.y)
"""

from .benchmarks import BenchmarkScenario, benchmark_summary, default_scenarios, grid_scenarios, run_benchmark_suite, save_benchmark_report
from .engine.algorithm import ADPAlgorithm
from .core import (
    ADP,
    ADPConfig,
    ADPData,
    ADPResult,
    InitialBetaMode,
    LocalMassMode,
    LocalStatistics,
    RandomProjectionADP,
    TrainingStep,
)
from .stages import ADPState, StageContext, StageExecutionError, StageFactory, StageRegistry

__all__ = [
    "ADP",
    "ADPAlgorithm",
    "ADPConfig",
    "ADPData",
    "ADPResult",
    "ADPState",
    "InitialBetaMode",
    "LocalMassMode",
    "LocalStatistics",
    "RandomProjectionADP",
    "TrainingStep",
    "StageContext",
    "StageExecutionError",
    "StageFactory",
    "StageRegistry",
    "BenchmarkScenario",
    "benchmark_summary",
    "default_scenarios",
    "grid_scenarios",
    "run_benchmark_suite",
    "save_benchmark_report",
]
