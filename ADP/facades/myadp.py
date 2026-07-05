from ADP_Data_Gen import FunctionValue, GenerateNoise, GenerateX, MakeData
from ADP_step0 import (
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
from ADP_stepk import (
    CalculateRho,
    CosineSimilarity,
    EstimateLocalGradients,
    LocalLinearGradient,
    StandardizeFeatures,
)
from ADP_Runtime import (
    CreateRuntimeMonitor,
    IterateWithProgress,
    LogRuntimeEvent,
    RuntimeStage,
    RuntimeSummary,
)
from ADP_Trace import (
    CreateTrace,
    GetTraceTable,
    PlotADPDiagnostics,
    SaveADPDiagnostics,
    SaveTraceSummary,
    TraceStep,
)
from Main_ADP import AlteringOptimisation, AverageDerivativeProcedure, FitADP, RunADP
from ADP_single_index import ADPSingleIndex, ADP_single_index


# Этот файл остается общей точкой входа для учебного проекта.
# Генерация данных лежит в data/generation.py.
# Шаг 0 лежит в algorithm/step0.py.
# Шаг k лежит в algorithm/stepk.py.
# Runtime-monitoring лежит в runtime/monitoring.py.
# Трассировка и графики лежат в diagnostics/trace.py.
# Оркестрация пайплайна лежит в pipeline/main.py.
# Единый объектный интерфейс лежит в models/single_index.py.
