"""APPROXIMATE regular-center GN proposal, full original profile acceptance."""

from .backtracking import solve as _solve


def solve(index_init, U, I, *, max_steps=5, **kwargs):
    return _solve(index_init, U, I, max_steps=max_steps, intrinsic_gn=True, **kwargs)
