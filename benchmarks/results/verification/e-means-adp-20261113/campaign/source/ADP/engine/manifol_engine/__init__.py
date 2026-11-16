"""Декомпозированный manifold-движок ADP."""

from ...core.ADP_Config import epanechnikov as kernel
from ..common.calculus import pairwise_distance2
from .fit import ManifoldFitState, fit, predict, transform
from .graphs import build_manifold_graph, initialize_projectors
from .optimisation import (
    build_B_system,
    local_slopes,
    objective,
    one_step,
    penalty_action,
    projector_distance,
    recover_projector,
    solve_B,
)
from .utils import (
    bandwidth_floor,
    feasible_scale,
    local_coordinates,
    nearest_center_indices,
    orient_rows,
    trace_entry,
)
from .weights import (
    calculate_statistics,
    local_gradients,
    mean_mass,
    random_directions,
    search_anisotropy,
    search_bandwidth,
    weight_block,
)

__all__ = [
    "ManifoldFitState",
    "bandwidth_floor",
    "build_B_system",
    "build_manifold_graph",
    "calculate_statistics",
    "feasible_scale",
    "fit",
    "initialize_projectors",
    "kernel",
    "local_coordinates",
    "local_gradients",
    "local_slopes",
    "mean_mass",
    "nearest_center_indices",
    "objective",
    "one_step",
    "orient_rows",
    "pairwise_distance2",
    "penalty_action",
    "predict",
    "projector_distance",
    "random_directions",
    "recover_projector",
    "search_anisotropy",
    "search_bandwidth",
    "solve_B",
    "trace_entry",
    "transform",
    "weight_block",
]
