"""Experimental one-step HPAO warm start followed by guarded reduced L-BFGS."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from ADP.gpu import array_module
from ADP.solver.LSMR import HPAOResult, _validate_inputs, solve as solve_lsmr

from .multi_solver_certificate import evaluate
from .reduced_lbfgs import solve as solve_reduced


def solve(
    index_init: np.ndarray,
    U: np.ndarray,
    I: np.ndarray,
    *,
    mass: np.ndarray | None = None,
    lambda_prox: float = 0.05,
    max_steps: int = 320,
    tol: float = 1e-6,
    theta: float = 0.1,
    trust_radius: float | None = None,
    lsmr_maxiter: int | None = None,
) -> HPAOResult:
    """Run exactly one certified HPAO AO step, then reduced optimization."""
    if array_module(U) is not np:
        raise NotImplementedError(
            "experimental warm reduced solver requires CPU arrays"
        )
    if not np.isfinite(lambda_prox) or lambda_prox <= 0:
        raise ValueError("warm reduced solver requires positive lambda_prox")
    index_init, U, I, mass = _validate_inputs(index_init, U, I, mass)
    started = perf_counter()
    warm = solve_lsmr(
        index_init,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=1,
        tol=tol,
        theta=theta,
        trust_radius=trust_radius,
        lsmr_maxiter=lsmr_maxiter,
    )
    warm_time = perf_counter() - started
    warm_diag = warm.diagnostics
    if warm_diag["accepted_steps"] != 1 or warm_diag["normal_residual_ratio"] > theta:
        raise RuntimeError("HPAO warm step lacks a certified correction")
    initial_loss, warm_loss = warm_diag["loss_history"]
    rounding = 64 * np.finfo(float).eps * max(1.0, initial_loss)
    if warm_loss > initial_loss + rounding:
        raise RuntimeError("HPAO warm step increased the objective")
    reference = evaluate(warm.index, U, I, mass)
    if not math.isclose(reference.objective, warm_loss, rel_tol=1e-8, abs_tol=1e-10):
        raise RuntimeError("HPAO warm objective differs from common reference")
    if reference.orthogonality >= tol:
        raise RuntimeError("HPAO warm basis lost orthogonality")

    reduced = solve_reduced(
        warm.index,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=max_steps,
        tol=tol,
    )
    diagnostics = {
        **reduced.diagnostics,
        "solver": "warm-reduced",
        "warm_steps": 1,
        "warm_time_sec": warm_time,
        "warm_loss_initial": initial_loss,
        "warm_loss_final": warm_loss,
        "warm_normal_residual_ratio": warm_diag["normal_residual_ratio"],
        "warm_lsmr_iterations": warm_diag["lsmr_iterations_total"],
        "warm_rejected_trials": warm_diag["rejected_trials"],
        "total_accepted_steps": 1 + reduced.diagnostics["accepted_steps"],
    }
    return HPAOResult(reduced.index, reduced.coefficients, diagnostics)
