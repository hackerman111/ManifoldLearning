"""Checks for one original rank-one step before intrinsic GN proposals."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ADP.solver.grassman import solve as production_solve
from experiments.grassman_accuracy_2026_10_18.backtracking import solve as core_solve
from experiments.grassman_accuracy_2026_10_18.warmup import solve as warmup_solve


def direct_fit(U, Y, I, mass):
    coefficients, ranks, value = [], [], 0.0
    for matrix, target, weight in zip(U @ Y, I, mass, strict=True):
        coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ coef
        coefficients.append(coef)
        ranks.append(rank)
        value += weight * float(residual @ residual)
    return np.asarray(coefficients), np.asarray(ranks), value


def cases():
    rng = np.random.default_rng(417)
    U = rng.normal(size=(7, 6, 5))
    U[0] = 0.0
    U[0, 0, :2] = (1.0, 0.7)
    I, mass, index = rng.normal(size=(7, 6)), np.geomspace(0.2, 2.5, 7), np.eye(5)[:2]
    yield index, U, I, mass

    U = np.zeros((3, 4, 4))
    U[0, 0, 0] = U[0, 1, 2] = 1.0  # U has rank 2, but U @ index.T has rank 1.
    U[1:] = np.random.default_rng(19).normal(size=(2, 4, 4))
    yield np.eye(4)[:2], U, np.random.default_rng(20).normal(size=(3, 4)), np.ones(3)


@pytest.mark.parametrize("index,U,I,mass", list(cases()))
def test_first_warmup_step_matches_original_production_rank_one(index, U, I, mass):
    expected = production_solve(
        index,
        U,
        I,
        mass=mass,
        max_steps=1,
        tol=1e-12,
        relative_gn=False,
        quadratic_trials=0,
        constant_centers=False,
    )
    result = warmup_solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    coef, ranks, value = direct_fit(U, result.index.T, I, mass)
    diag = result.diagnostics
    assert result.index.T @ result.index == pytest.approx(
        expected.index.T @ expected.index, abs=2e-10
    )
    assert diag["intrinsic_warmup_steps"] == 1
    assert diag["rank_guard_fallbacks"] == 1
    assert diag["gn_solves"] == 0
    assert result.coefficients == pytest.approx(coef, abs=2e-10)
    assert cast(float, diag["profile_objective"]) == pytest.approx(value, abs=2e-10)
    assert diag["local_rank_loss"] == np.count_nonzero(ranks < index.shape[0])
    assert (
        diag["stationarity_applicable"]
        == expected.diagnostics["stationarity_applicable"]
    )
    assert cast(float, diag["orthogonality_error"]) < 1e-12


def test_only_first_nonsmooth_step_warms_then_intrinsic_gn_is_used():
    index, U, I, mass = next(cases())
    result = warmup_solve(index, U, I, mass=mass, max_steps=2, tol=1e-14)
    diag = result.diagnostics
    assert diag["intrinsic_warmup_steps"] == 1
    assert cast(int, diag["intrinsic_gn_proposals"]) >= 1
    assert np.all(np.diff(diag["profile_history"]) <= 1e-12)

    bounded = warmup_solve(
        index, U, I, mass=mass, max_steps=2, tol=1e-14, workspace_bytes=1
    )
    assert cast(int, bounded.diagnostics["workspace_fallbacks"]) >= 1


def test_smooth_budget_and_intrinsic_false_path_are_unchanged():
    index, U, I, mass = next(cases())
    U = U.copy()
    U[0] = np.random.default_rng(7).normal(size=U[0].shape)
    base = core_solve(index, U, I, mass=mass, max_steps=10)
    result = warmup_solve(index, U, I, mass=mass, max_steps=10)
    assert result.diagnostics["effective_max_steps"] == 5
    assert result.diagnostics["intrinsic_warmup_steps"] == 0
    assert result.index == pytest.approx(base.index, abs=2e-12)
    assert result.coefficients == pytest.approx(base.coefficients, abs=2e-12)
