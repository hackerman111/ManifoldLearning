"""Checks for rank-q spectral steps at nonsmooth profile rank guards."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ADP.solver.grassman import (
    _gradient,
    _profile,
    solve as production_solve,
)
from ADP.solver.grassman_exp import solve as experimental_solve
from experiments.grassman_accuracy_2026_10_18.spectral_guard import solve


def _case(m=2):
    rng = np.random.default_rng(417)
    centers, p, d = 7, 6, 5 if m == 2 else 4
    U = rng.normal(size=(centers, p, d))
    U[0] = 0.0
    if m == 2:
        U[0, 0, 0] = 1.0
    I = rng.normal(size=(centers, p))
    mass = np.geomspace(0.2, 2.5, centers)
    return np.eye(d)[:m], U, I, mass


def _direct_fit(U, Y, I, mass):
    coefficients, ranks, value = [], [], 0.0
    for matrix, target, weight in zip(U @ Y, I, mass, strict=True):
        coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ coef
        coefficients.append(coef)
        ranks.append(rank)
        value += weight * float(residual @ residual)
    return np.asarray(coefficients), np.asarray(ranks), value


def test_rank_guard_joint_geodesic_matches_energy_zero_spectral_reference():
    index, U, I, mass = _case()
    result = solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    reference = experimental_solve(
        index,
        U,
        I,
        mass=mass,
        method="spectral",
        energy_tol=0.0,
        max_steps=1,
        tol=1e-12,
    )
    coef, ranks, value = _direct_fit(U, result.index.T, I, mass)
    diag = result.diagnostics

    assert np.count_nonzero(ranks < index.shape[0]) == 1
    assert diag["rank_guard_fallbacks"] == 1
    assert diag["step_ranks"] == (min(index.shape[0], U.shape[2] - index.shape[0]),)
    assert result.index.T @ result.index == pytest.approx(
        reference.index.T @ reference.index, abs=2e-10
    )
    assert result.coefficients == pytest.approx(coef, abs=2e-10)
    assert result.coefficients == pytest.approx(reference.coefficients, abs=2e-10)
    assert cast(float, diag["profile_objective"]) == pytest.approx(value, abs=2e-10)
    assert diag["local_rank_loss"] == np.count_nonzero(ranks < index.shape[0])
    assert cast(float, diag["orthogonality_error"]) < 1e-12
    assert np.all(np.diff(diag["profile_history"]) <= 1e-12)


def test_rank_one_case_uses_same_geodesic_direction_as_rank_one_armijo():
    index, U, I, mass = _case(m=1)
    spectral = solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    rank_one = experimental_solve(
        index, U, I, mass=mass, method="rank_one", max_steps=1, tol=1e-12
    )
    Y = index.T
    tangents = []
    for result in (spectral, rank_one):
        moved = result.index.T
        tangent = moved - Y @ (Y.T @ moved)
        tangents.append(tangent / np.linalg.norm(tangent))
    assert abs(float(np.vdot(tangents[0], tangents[1]))) == pytest.approx(
        1.0, abs=2e-10
    )
    assert spectral.diagnostics["step_ranks"] == (1,)
    assert spectral.diagnostics["rank_guard_fallbacks"] == 1


def test_smooth_guard_path_matches_original_five_step_core():
    rng = np.random.default_rng(99)
    index = np.eye(5)[:2]
    U = rng.normal(size=(6, 7, 5))
    I, mass = rng.normal(size=(6, 7)), np.geomspace(0.3, 2.0, 6)
    baseline = production_solve(index, U, I, mass=mass, max_steps=5, tol=1e-12)
    result = solve(index, U, I, mass=mass, max_steps=10, tol=1e-12)
    assert result.index == pytest.approx(baseline.index, abs=2e-12)
    assert result.coefficients == pytest.approx(baseline.coefficients, abs=2e-12)
    for key in ("profile_history", "step_ranks", "accepted_steps", "stop_reason"):
        assert result.diagnostics[key] == baseline.diagnostics[key]


def test_workspace_fallback_and_explicit_rank_remain_bounded():
    rng = np.random.default_rng(99)
    index = np.eye(5)[:2]
    U = rng.normal(size=(6, 7, 5))
    I, mass = rng.normal(size=(6, 7)), np.ones(6)
    workspace = solve(
        index, U, I, mass=mass, max_steps=1, workspace_bytes=1, tol=1e-12
    )
    assert workspace.diagnostics["workspace_fallbacks"] == 1
    assert workspace.diagnostics["rank_guard_fallbacks"] == 0

    index, U, I, mass = _case()
    limited = solve(index, U, I, mass=mass, rank=1, max_steps=1, tol=1e-12)
    assert limited.diagnostics["rank_guard_fallbacks"] == 1
    assert limited.diagnostics["step_ranks"] == (1,)


def test_fixed_intrinsic_rank_center_contributes_zero_directional_slope():
    rng = np.random.default_rng(731)
    Y = np.eye(4)[:, :2]
    U = rng.normal(size=(5, 5, 4))
    U[0] = 0.0
    U[0, 0, 0] = 1.0  # rank(U0) = rank(U0 @ Y) = 1 locally
    I, mass = rng.normal(size=(5, 5)), np.geomspace(0.4, 2.0, 5)
    state = _profile(U @ Y, I, mass, 0.0)
    direction = np.zeros_like(Y)
    direction[2, 1] = 1.0
    assert state.ranks[0] == 1
    assert np.linalg.matrix_rank(U[0]) == 1
    assert np.allclose(Y.T @ direction, 0.0)

    def objective(theta):
        moved = Y.copy()
        moved[:, 1] = np.cos(theta) * Y[:, 1] + np.sin(theta) * direction[:, 1]
        return _direct_fit(U, moved, I, mass)[2]

    step = 1e-6
    finite_difference = (objective(step) - objective(-step)) / (2 * step)
    predicted = float(np.sum(_gradient(Y, U, state, mass) * direction))
    assert finite_difference == pytest.approx(predicted, rel=2e-6, abs=2e-7)
