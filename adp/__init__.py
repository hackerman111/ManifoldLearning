"""Average Derivative Procedure.

Основная точка входа:

    model = ADP.create("new", ADPConfig(...))
    data = model.generate_data(...)
    result = model.fit(data.X, data.y)
"""

from .benchmarks import BenchmarkScenario, benchmark_summary, default_scenarios, grid_scenarios, run_benchmark_suite, save_benchmark_report
from .core import (
    ADP,
    ADPConfig,
    ADPData,
    ADPResult,
    FullMomentADP,
    LocalStatistics,
    RandomProjectionADP,
    TrainingStep,
)

__all__ = [
    "ADP",
    "ADPConfig",
    "ADPData",
    "ADPResult",
    "FullMomentADP",
    "LocalStatistics",
    "RandomProjectionADP",
    "TrainingStep",
    "BenchmarkScenario",
    "benchmark_summary",
    "default_scenarios",
    "grid_scenarios",
    "run_benchmark_suite",
    "save_benchmark_report",
]
