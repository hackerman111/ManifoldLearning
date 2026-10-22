"""Checks for exact-five-step baseline plus guarded intrinsic refinement."""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np
import pytest

from ADP.solver.grassman import solve as current_solve
from experiments.grassman_accuracy_2026_10_18.backtracking import (
    solve as intrinsic_solve,
)
from experiments.grassman_accuracy_2026_10_18.polish import solve

production_solve = partial(
    current_solve,
    relative_gn=False,
    quadratic_trials=0,
    constant_centers=False,
    smooth_max_steps=5,
)


def _case():
    rng = np.random.default_rng(417)
    U = rng.normal(size=(7, 6, 5))
    U[0] = 0.0
    I = rng.normal(size=(7, 6))
    mass = np.geomspace(0.2, 2.5, len(I))
    return np.eye(5)[:2], U, I, mass


def _direct_fit(U, Y, I, mass):
    coefficients, ranks, value = [], [], 0.0
    for matrix, target, weight in zip(U @ Y, I, mass, strict=True):
        coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ coef
        coefficients.append(coef)
        ranks.append(rank)
        value += weight * float(residual @ residual)
    return np.asarray(coefficients), np.asarray(ranks), value


def _same_baseline_diagnostics(result, baseline):
    for key, value in baseline.diagnostics.items():
        if key not in {"method", "angle_backend"}:
            assert result.diagnostics[key] == value
    assert result.diagnostics["method"] == "core_gn_intrinsic_polish"
    assert result.diagnostics["angle_backend"] == "schur_then_refit_armijo"


def test_five_step_budget_is_exact_production_result():
    index, U, I, mass = _case()
    baseline = production_solve(index, U, I, mass=mass, max_steps=5)
    result = solve(index, U, I, mass=mass, max_steps=5)
    assert np.array_equal(result.index, baseline.index)
    assert np.array_equal(result.coefficients, baseline.coefficients)
    _same_baseline_diagnostics(result, baseline)
    assert result.diagnostics["polish_skip_reason"] == "budget"


def test_two_step_refinement_matches_explicit_composition_and_live_refit():
    index, U, I, mass = _case()
    baseline = production_solve(index, U, I, mass=mass, max_steps=5, tol=1e-14)
    refine = intrinsic_solve(
        baseline.index,
        U,
        I,
        mass=mass,
        max_steps=2,
        smooth_max_steps=None,
        intrinsic_gn=True,
        tol=1e-14,
    )
    result = solve(index, U, I, mass=mass, max_steps=7, tol=1e-14)
    coef, ranks, value = _direct_fit(U, result.index.T, I, mass)
    diag = result.diagnostics

    assert result.index.T @ result.index == pytest.approx(
        refine.index.T @ refine.index, abs=2e-10
    )
    assert result.coefficients == pytest.approx(coef, abs=2e-10)
    assert result.coefficients == pytest.approx(refine.coefficients, abs=2e-10)
    assert cast(float, diag["profile_objective"]) == pytest.approx(value, abs=2e-10)
    assert diag["local_rank_loss"] == np.count_nonzero(ranks < index.shape[0])
    assert cast(float, diag["orthogonality_error"]) < 1e-12
    assert diag["polish_skip_reason"] is None

    phases = cast(tuple[tuple[float, ...], ...], diag["phase_profile_history"])
    assert len(phases) == 2
    assert np.all(np.diff(phases[0]) <= 1e-12)
    assert np.all(np.diff(phases[1]) <= 1e-12)
    assert diag["polish_boundary_gap"] == pytest.approx(phases[1][0] - phases[0][-1])
    assert cast(int, diag["accepted_steps"]) == cast(
        int, baseline.diagnostics["accepted_steps"]
    ) + cast(int, refine.diagnostics["accepted_steps"])


def test_unavailable_projected_rank_model_returns_exact_baseline():
    index = np.eye(4)[:2]
    U = np.zeros((3, 4, 4))
    U[0, 0, 0] = U[0, 1, 2] = 1.0
    U[1:] = np.random.default_rng(19).normal(size=(2, 4, 4))
    I, mass = np.zeros((3, 4)), np.ones(3)
    baseline = production_solve(index, U, I, mass=mass, max_steps=5)
    assert cast(int, baseline.diagnostics["local_rank_loss"]) > 0
    result = solve(index, U, I, mass=mass, max_steps=7)
    assert np.array_equal(result.index, baseline.index)
    assert np.array_equal(result.coefficients, baseline.coefficients)
    _same_baseline_diagnostics(result, baseline)
    assert result.diagnostics["polish_skip_reason"] == "rank_guard"


@pytest.mark.parametrize("kwargs", [{"local_ridge": 0.2}, {"tol": 1e-6}])
def test_ridge_or_converged_baseline_skips_refinement(kwargs):
    index, U, I, mass = _case()
    if "tol" in kwargs:
        U = U.copy()
        U[0] = np.random.default_rng(7).normal(size=U[0].shape)
        I = np.zeros_like(I)
    baseline = production_solve(index, U, I, mass=mass, max_steps=5, **kwargs)
    result = solve(index, U, I, mass=mass, max_steps=7, **kwargs)
    assert np.array_equal(result.index, baseline.index)
    assert np.array_equal(result.coefficients, baseline.coefficients)
    _same_baseline_diagnostics(result, baseline)
    expected = "local_ridge" if "local_ridge" in kwargs else "converged"
    assert result.diagnostics["polish_skip_reason"] == expected


@pytest.mark.parametrize("budget", [True, -1])
def test_invalid_total_budget_is_rejected(budget):
    index, U, I, mass = _case()
    with pytest.raises((TypeError, ValueError)):
        solve(index, U, I, mass=mass, max_steps=budget)
