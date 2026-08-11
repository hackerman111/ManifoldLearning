from functools import partial

import numpy as np


def _validated_tau(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau) or not 0.0 < tau < 1.0:
        raise ValueError("tau must be finite and lie in (0, 1)")
    return tau


def _kernel_input(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("kernel input must be finite")
    return value


def box_kernel(value) -> np.ndarray:
    return np.asarray(_kernel_input(value) < 1.0, dtype=float)


def plateau_kernel(value, *, tau: float = 0.5) -> np.ndarray:
    value = _kernel_input(value)
    tau = _validated_tau(tau)
    result = np.zeros_like(value)
    result[value <= tau] = 1.0
    boundary = (value > tau) & (value < 1.0)
    s = (value[boundary] - tau) / (1.0 - tau)
    result[boundary] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
    return result


def make_plateau_kernel(tau: float = 0.5):
    return partial(plateau_kernel, tau=_validated_tau(tau))


def sparse_kernel_parameters(kernel) -> tuple[str, float | None] | None:
    if kernel is box_kernel:
        return "box", None
    if kernel is plateau_kernel:
        return "plateau", 0.5
    if isinstance(kernel, partial) and kernel.func is plateau_kernel:
        if kernel.args or set(kernel.keywords or {}) - {"tau"}:
            raise ValueError("plateau kernel accepts only keyword tau")
        return "plateau", _validated_tau((kernel.keywords or {}).get("tau", 0.5))
    return None
