from . import HYBRID, HYBRID_manifold, HYBRID_multi, HYBRID_single
from .HYBRID import (
    DENSE_MAX_BYTES,
    DENSE_MAX_UNKNOWNS,
    LinearResult,
    design_matrix,
    solve_augmented,
    use_dense,
)
from .HYBRID_manifold import PenaltyRoot, solve_manifold
from .HYBRID_multi import RidgeWorkspace, solve

solve_single = HYBRID_single.solve

__all__ = [
    "DENSE_MAX_BYTES",
    "DENSE_MAX_UNKNOWNS",
    "HYBRID",
    "HYBRID_manifold",
    "HYBRID_multi",
    "HYBRID_single",
    "LinearResult",
    "PenaltyRoot",
    "RidgeWorkspace",
    "design_matrix",
    "solve",
    "solve_augmented",
    "solve_manifold",
    "solve_single",
    "use_dense",
]
