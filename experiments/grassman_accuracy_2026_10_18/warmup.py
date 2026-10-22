"""APPROXIMATE intrinsic GN after one original nonsmooth scalar step."""

from .backtracking import solve as _solve


def solve(index_init, U, I, *, max_steps=10, **kwargs):
    return _solve(
        index_init,
        U,
        I,
        max_steps=max_steps,
        intrinsic_gn=True,
        intrinsic_warmup=True,
        **kwargs,
    )
