from data.generation import FunctionValue, GenerateNoise, GenerateX, MakeData
from algorithm.step0 import (
    ChooseH0,
    ChooseJ,
    ComputeWeight,
    ComputeWeigth,
    GenerateDirection,
    GenerateDirectionsForCenters,
    Kernel,
    NormVector,
    NormalizeRows,
    PrepareADPInitialState,
)
from algorithm.stepk import (
    CalculateRho,
    CosineSimilarity,
    EstimateLocalGradients,
    LocalLinearGradient,
    StandardizeFeatures,
)
from runtime.monitoring import (
    CreateRuntimeMonitor,
    IterateWithProgress,
    LogRuntimeEvent,
    RuntimeStage,
    RuntimeSummary,
)
from diagnostics.trace import (
    CreateTrace,
    GetTraceTable,
    PlotADPDiagnostics,
    SaveADPDiagnostics,
    SaveTraceSummary,
    TraceStep,
)
from pipeline.main import AlteringOptimisation, AverageDerivativeProcedure, FitADP, RunADP
from models.single_index import ADPSingleIndex, ADP_single_index


# Этот файл остается общей точкой входа для учебного проекта.
# Генерация данных лежит в data/generation.py.
# Шаг 0 лежит в algorithm/step0.py.
# Шаг k лежит в algorithm/stepk.py.
# Runtime-monitoring лежит в runtime/monitoring.py.
# Трассировка и графики лежат в diagnostics/trace.py.
# Оркестрация пайплайна лежит в pipeline/main.py.
# Единый объектный интерфейс лежит в models/single_index.py.
