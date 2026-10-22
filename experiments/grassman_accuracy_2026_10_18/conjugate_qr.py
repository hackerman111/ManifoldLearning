"""APPROXIMATE conjugate spectral solver with per-center QR/SVD profiles."""

from .backtracking import solve as _solve


def solve(index_init, U, I, *, max_steps=10, **kwargs):
    return _solve(
        index_init,
        U,
        I,
        max_steps=max_steps,
        spectral_guard=True,
        conjugate_guard=True,
        per_center_profile=True,
        **kwargs,
    )
