"""APPROXIMATE rank-q spectral fallback at nonsmooth profile states."""

from .backtracking import solve as _solve


def solve(index_init, U, I, *, max_steps=10, **kwargs):
    return _solve(
        index_init,
        U,
        I,
        max_steps=max_steps,
        spectral_guard=True,
        **kwargs,
    )
