"""APPROXIMATE: three true-profile quadratic angle proposals, smooth5/rank10."""

from .backtracking import solve as _solve


def solve(index_init, U, I, *, max_steps=10, **kwargs):
    return _solve(index_init, U, I, max_steps=max_steps, quadratic_trials=3, **kwargs)
