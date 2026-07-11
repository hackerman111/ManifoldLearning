from __future__ import annotations

from .backends.neighbors import NeighborIndex
from .backends.numpy_backend import NumpyBackend
from .common.progress import format_progress_postfix
from .common.types import (
    ADPConfig,
    ADPData,
    ADPResult,
    BackendName,
    InitialBetaMode,
    KernelName,
    LocalMassMode,
    LocalStatistics,
    TrainingStep,
    VariantName,
)
from .common.utils import (
    as_1d_float,
    as_2d_float,
    average_kernel_weight,
    kernel_np,
    link_function,
    normalize_rows,
    pairwise_norm2,
    pairwise_projection2,
    unit_vector,
)
from .engine.base import ADP, ADPBase, tqdm
from .engine.algorithm import ADPAlgorithm
from .stages import ADPState, StageContext, StageExecutionError, StageFactory, StageRegistry
from .variants import RandomProjectionADP

_NumpyBackend = NumpyBackend
_NeighborIndex = NeighborIndex
_as_1d_float = as_1d_float
_as_2d_float = as_2d_float
_average_kernel_weight = average_kernel_weight
_format_progress_postfix = format_progress_postfix
_kernel_np = kernel_np
_link_function = link_function
_normalize_rows = normalize_rows
_pairwise_norm2 = pairwise_norm2
_pairwise_projection2 = pairwise_projection2
_unit_vector = unit_vector

__all__ = [
    "ADP",
    "ADPAlgorithm",
    "ADPBase",
    "ADPConfig",
    "ADPData",
    "ADPResult",
    "ADPState",
    "BackendName",
    "InitialBetaMode",
    "KernelName",
    "LocalMassMode",
    "LocalStatistics",
    "NeighborIndex",
    "NumpyBackend",
    "RandomProjectionADP",
    "StageContext",
    "StageExecutionError",
    "StageFactory",
    "StageRegistry",
    "TrainingStep",
    "VariantName",
    "as_1d_float",
    "as_2d_float",
    "average_kernel_weight",
    "format_progress_postfix",
    "kernel_np",
    "link_function",
    "normalize_rows",
    "pairwise_norm2",
    "pairwise_projection2",
    "tqdm",
    "unit_vector",
    "_NumpyBackend",
    "_NeighborIndex",
    "_as_1d_float",
    "_as_2d_float",
    "_average_kernel_weight",
    "_format_progress_postfix",
    "_kernel_np",
    "_link_function",
    "_normalize_rows",
    "_pairwise_norm2",
    "_pairwise_projection2",
    "_unit_vector",
]
