"""Full fixed-g proximal step; isolated recovery experiment."""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import lsmr

from ADP.solver.LSMR import (
    HPAOResult,
    _linear_operator,
    _local_refit,
    _loss,
    _normalize_index,
    _validate_inputs,
)


def fixed(P, U, I, g, mass, ridge, maxiter=None):
    operator = _linear_operator(U, g, np.sqrt(mass), P.shape)
    target = (np.sqrt(mass)[:, None] * I).ravel() - operator @ P.ravel()
    result = lsmr(
        operator,
        target,
        damp=math.sqrt(ridge),
        atol=1e-10,
        btol=1e-10,
        maxiter=maxiter or max(50, 5 * P.size),
    )
    delta = result[0]
    residual = operator @ delta - target
    rhs = operator.rmatvec(target)
    gradient = operator.rmatvec(residual) + ridge * delta
    certificate = float(np.linalg.norm(gradient) / max(1.0, np.linalg.norm(rhs)))
    objective = float(residual @ residual + ridge * (delta @ delta))
    initial = float(target @ target)
    if not np.all(np.isfinite(delta)) or certificate > 1e-7:
        raise RuntimeError(f"full fixed-g certificate failed: {certificate}")
    if objective > initial + 64 * np.finfo(float).eps * max(1.0, initial):
        raise RuntimeError("full fixed-g objective increased")
    return P + delta.reshape(P.shape), {
        "normal_residual": certificate,
        "normal_residual_norm": float(np.linalg.norm(gradient)),
        "objective": objective,
        "initial_objective": initial,
        "lsmr_iterations_total": int(result[2]),
        "lsmr_stop": int(result[1]),
    }


def solve(index_init, U, I, *, mass=None, lambda_prox=1.0, **kwargs):
    P, U, I, mass = _validate_inputs(index_init, U, I, mass)
    P = _normalize_index(P)
    g, _ = _local_refit(I, U, P)
    raw, diagnostics = fixed(P, U, I, g, mass, lambda_prox)
    _, values, basis = np.linalg.svd(raw, full_matrices=False)
    if values[-1] <= np.finfo(float).eps * max(raw.shape) * values[0]:
        raise RuntimeError("full fixed-g step lost row rank")
    coefficients, ranks = _local_refit(I, U, basis)
    diagnostics.update(
        linear_solver="full_fixed_g",
        low_rank_target="full",
        loss=_loss(I, U, basis, coefficients, mass),
        local_rank_loss=int(np.count_nonzero(ranks < len(P))),
        basis_singular_values=values.tolist(),
        inner_iterations=(1,),
        inner_converged=(True,),
    )
    return HPAOResult(basis, coefficients, diagnostics)
