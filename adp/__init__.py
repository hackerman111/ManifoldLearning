"""Average Derivative Procedure.

Основная точка входа:

    model = ADP.create("new", ADPConfig(...))
    data = model.generate_data(...)
    result = model.fit(data.X, data.y)
"""

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
from .benchmarks import BenchmarkScenario, benchmark_summary, default_scenarios, run_benchmark_suite, save_benchmark_report

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
    "run_benchmark_suite",
    "save_benchmark_report",
]
