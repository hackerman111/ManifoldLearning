"""Профильные центры U_j=0 имеют постоянную цель и не ограничивают GN."""

from __future__ import annotations

import numpy as np

from ADP.solver.LSMR import HPAOResult, _validate_inputs

from .candidate import solve as local_solve


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    tol: float = 1e-6,
    **kwargs,
) -> HPAOResult:
    """Исключить только точно нулевые операторы, вернуть полные coefficients."""
    B, U, I, mass = _validate_inputs(index_init, U, I, mass)
    active = np.any(U != 0, axis=(1, 2))
    if np.all(active) or not np.any(active):
        return local_solve(B, U, I, mass=mass, tol=tol, **kwargs)
    assert mass is not None
    original_scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    active_scale = max(1.0, float(np.sum(mass[active, None] * I[active] ** 2)))
    constant = float(np.sum(mass[~active, None] * I[~active] ** 2))
    result = local_solve(
        B,
        U[active],
        I[active],
        mass=mass[active],
        tol=tol * original_scale / active_scale,
        **kwargs,
    )
    coefficients = np.zeros((len(I), B.shape[0]))
    coefficients[active] = result.coefficients
    diagnostics = dict(result.diagnostics)
    diagnostics["method"] = "core_gn_constant_centers"
    diagnostics["constant_centers"] = int(np.count_nonzero(~active))
    diagnostics["profile_objective"] += constant
    diagnostics["loss"] += constant / 2
    diagnostics["profile_history"] = tuple(
        v + constant for v in diagnostics["profile_history"]
    )
    diagnostics["loss_history"] = tuple(
        v + constant / 2 for v in diagnostics["loss_history"]
    )
    if not kwargs.get("local_ridge", 0.0):
        diagnostics["local_rank_loss"] += int(np.count_nonzero(~active))
    diagnostics["riemannian_gradient"] = diagnostics["gradient_norm"] / original_scale
    return HPAOResult(result.index, coefficients, diagnostics)
