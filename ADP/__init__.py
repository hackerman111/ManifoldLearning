from importlib import import_module as _import_module
from sys import modules as _modules

from .core.ADP_Config import ADP_Config
from .core.ADP_Data import ADP_Data
from .core.ADP_Solver import ADP_Solver, ADP_solver, ADP_SolverResult
from .core.ADP_Statistic import ADP_Statistics
from .core.manifold.ADP_Manifold import ADP_Manifold, ADP_manifold
from .core.manifold.ADP_Manifold_result import ADP_Manifold_result
from .core.multi.ADP_multi_index import ADP_multi_index
from .core.multi.ADP_multi_index_result import ADP_multi_index_result
from .core.single.ADP_single_index import ADP_single_index
from .core.single.ADP_single_index_result import ADP_single_index_result
from .engine.common.calculus import (
    calculate_alpha_k,
    calculate_h0,
    calculate_rho_k,
    generate_isotropic_proj,
    generate_multi_proj,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)
from .engine.common.statistic import calculate_statistics
from .engine.common.weights import calculate_multi_weight, calculate_weight

# Старые модульные пути остаются только как aliases, без файлов-дубликатов.
for _old_name, _new_name in {
    "ADP_Config": "core.ADP_Config",
    "ADP_Data": "core.ADP_Data",
    "ADP_Manifold": "core.manifold.ADP_Manifold",
    "ADP_Manifold_result": "core.manifold.ADP_Manifold_result",
    "ADP_Solver": "core.ADP_Solver",
    "ADP_Statistic": "engine.common.ADP_Statistic_engine",
}.items():
    _modules.setdefault(
        f"{__name__}.{_old_name}",
        _import_module(f"{__name__}.{_new_name}"),
    )

del _import_module, _modules

__all__ = [
    "ADP_Config",
    "ADP_Data",
    "ADP_Manifold",
    "ADP_Manifold_result",
    "ADP_Solver",
    "ADP_SolverResult",
    "ADP_Statistics",
    "ADP_manifold",
    "ADP_multi_index",
    "ADP_multi_index_result",
    "ADP_single_index",
    "ADP_single_index_result",
    "ADP_solver",
    "calculate_alpha_k",
    "calculate_h0",
    "calculate_multi_weight",
    "calculate_rho_k",
    "calculate_statistics",
    "calculate_weight",
    "generate_isotropic_proj",
    "generate_multi_proj",
    "generate_proj",
    "pairwise_distance2",
    "search_bandwidth",
]
