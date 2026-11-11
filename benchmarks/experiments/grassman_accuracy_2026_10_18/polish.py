"""APPROXIMATE intrinsic GN refinement after the complete original five steps."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ADP.solver import grassman
from ADP.solver.LSMR import HPAOResult, _validate_inputs

from . import backtracking


def solve(index_init, U, I, *, max_steps=8, **kwargs):
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError("max_steps must be an integer")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    baseline_settings = {
        "relative_gn": False,
        "quadratic_trials": 0,
        "constant_centers": False,
        "smooth_max_steps": 5,
    }
    first = grassman.solve(
        index_init, U, I, max_steps=min(max_steps, 5), **(baseline_settings | kwargs)
    )
    phases: list[dict[str, Any]] = [cast(dict[str, Any], first.diagnostics)]
    skip = (
        "budget"
        if max_steps <= 5
        else "converged"
        if first.diagnostics["converged"]
        else "local_ridge"
        if kwargs.get("local_ridge", 0.0)
        else ""
    )
    if not skip:
        _, checked_U, checked_I, mass = _validate_inputs(
            first.index, U, I, kwargs.get("mass")
        )
        state = grassman._profile(
            grassman._project(checked_U, first.index.T), checked_I, mass, 0.0
        )
        if (
            not state.smooth
            and backtracking._intrinsic_model(
                checked_U, checked_I, mass, state, np.full(len(I), -1, dtype=int)
            )
            is None
        ):
            skip = "rank_guard"
    result = first
    if not skip:
        result = backtracking.solve(
            first.index,
            U,
            I,
            max_steps=max_steps - 5,
            intrinsic_gn=True,
            smooth_max_steps=None,
            **kwargs,
        )
        phases.append(result.diagnostics)
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        method="core_gn_intrinsic_polish",
        angle_backend="schur_then_refit_armijo",
        effective_max_steps=max_steps,
        polish_skip_reason=skip or None,
        phase_diagnostics=tuple(phases),
        phase_profile_history=tuple(d["profile_history"] for d in phases),
        baseline_profile_objective=first.diagnostics["profile_objective"],
        polish_boundary_gap=(
            phases[1]["profile_history"][0] - phases[0]["profile_history"][-1]
            if len(phases) > 1
            else 0.0
        ),
    )
    for key in (
        "accepted_steps",
        "iterations",
        "profile_evaluations",
        "gn_solves",
        "rank_guard_fallbacks",
        "workspace_fallbacks",
        "intrinsic_gn_proposals",
    ):
        diagnostics[key] = sum(int(d.get(key, 0)) for d in phases)
    for key in ("profile_history", "loss_history", "step_ranks"):
        diagnostics[key] = tuple(item for phase in phases for item in phase[key])
    return HPAOResult(result.index, result.coefficients, diagnostics)
