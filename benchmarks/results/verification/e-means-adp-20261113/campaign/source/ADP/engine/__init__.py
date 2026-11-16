"""Совместимый фасад движков ADP.

Реализации хранятся в ``common``, ``single_index``, ``multi_index`` и
``manifol_engine``. Этот модуль оставляет верхнеуровневые функции и старые
имена подмодулей доступными для существующих клиентов.
"""

from importlib import import_module as _import_module
from sys import modules as _modules

from .common.calculus import (
    calculate_alpha_k,
    calculate_h0,
    calculate_rho_k,
    generate_isotropic_proj,
    generate_multi_proj,
    generate_proj,
    pairwise_distance2,
    search_bandwidth,
)
from .common.initialize import (
    initialize_basis_local,
    initialize_basis_pilot,
    initialize_basis_random,
    initialize_beta_local,
)
from .common.statistic import calculate_statistics
from .common.weights import calculate_multi_weight, calculate_weight

for _old_name, _new_name in {
    "ADP_Statistic_engine": "common.ADP_Statistic_engine",
    "box_kernel": "common.box_kernel",
    "calculus": "common.calculus",
    "initialize": "common.initialize",
    "logger": "common.logger",
    "statistic": "common.statistic",
    "utils": "common.utils",
    "weights": "common.weights",
}.items():
    _module = _import_module(f"{__name__}.{_new_name}")
    _modules.setdefault(f"{__name__}.{_old_name}", _module)
    globals()[_old_name] = _module

_LEGACY_INDEX_MODULES = {
    "ADP_multi_index_engine": "multi_index.ADP_multi_index_engine",
    "ADP_single_index_engine": "single_index.ADP_single_index_engine",
}


def __getattr__(name: str):
    """Лениво разрешить старые имена index-specific модулей.

    Импорт выполняется только при обращении к старому атрибуту, чтобы загрузка
    общих функций не втягивала ``core.multi``/``core.single`` и не создавала
    циклический импорт при инициализации пакета.
    """
    new_name = _LEGACY_INDEX_MODULES.get(name)
    if new_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{new_name}")
    _modules.setdefault(f"{__name__}.{name}", module)
    globals()[name] = module
    return module


__all__ = [
    "calculate_alpha_k",
    "calculate_h0",
    "calculate_multi_weight",
    "calculate_rho_k",
    "calculate_statistics",
    "calculate_weight",
    "generate_isotropic_proj",
    "generate_multi_proj",
    "generate_proj",
    "initialize_basis_local",
    "initialize_basis_pilot",
    "initialize_basis_random",
    "initialize_beta_local",
    "pairwise_distance2",
    "search_bandwidth",
]
