"""Independent checks for transported PR+ directions at spectral rank guards."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from ADP.solver.grassman import _gradient, _profile, solve as production_solve
from experiments.grassman_accuracy_2026_10_18 import backtracking
from experiments.grassman_accuracy_2026_10_18.conjugate_guard import solve
from experiments.grassman_accuracy_2026_10_18.spectral_guard import (
    solve as spectral_solve,
)


def _case():
    rng = np.random.default_rng(417)
    U = rng.normal(size=(7, 6, 5))
    U[0] = 0.0
    U[0, 0, 0] = 1.0
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


def test_pr_plus_projection_transport_descent_and_zero_denominator():
    Y = np.eye(3)[:, :1]
    G = np.array([[0.0], [2.0], [0.0]])
    previous_gradient = np.array([[5.0], [1.0], [0.0]])
    previous_direction = np.array([[7.0], [-1.0], [0.0]])
    search, beta = backtracking._conjugate_gradient(
        Y, G, previous_gradient, previous_direction
    )
    assert beta == pytest.approx(2.0 / 26.0)
    assert search == pytest.approx(np.array([[0.0], [2.0 + beta], [0.0]]))

    uphill, beta = backtracking._conjugate_gradient(
        Y, G, previous_gradient, np.array([[0.0], [100.0], [0.0]])
    )
    assert beta == 0.0
    assert uphill == pytest.approx(G)
    zero, beta = backtracking._conjugate_gradient(
        Y, G, np.zeros_like(G), previous_direction
    )
    assert beta == 0.0
    assert zero == pytest.approx(G)


def test_first_step_matches_spectral_and_multistep_result_has_live_refit():
    index, U, I, mass = _case()
    first = solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    spectral = spectral_solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    assert first.index.T @ first.index == pytest.approx(
        spectral.index.T @ spectral.index, abs=2e-10
    )
    assert first.coefficients == pytest.approx(spectral.coefficients, abs=2e-10)
    assert first.diagnostics["conjugate_betas"] == (0.0,)

    result = solve(index, U, I, mass=mass, max_steps=4, tol=1e-12)
    coef, ranks, value = _direct_fit(U, result.index.T, I, mass)
    diag = result.diagnostics
    assert any(beta > 0 for beta in diag["conjugate_betas"])
    assert np.all(np.diff(diag["profile_history"]) <= 1e-12)
    assert result.coefficients == pytest.approx(coef, abs=2e-10)
    assert cast(float, diag["profile_objective"]) == pytest.approx(value, abs=2e-10)
    assert diag["local_rank_loss"] == np.count_nonzero(ranks < index.shape[0])
    assert cast(float, diag["orthogonality_error"]) < 1e-12


def test_rank_pattern_change_resets_previous_pair(monkeypatch):
    rng = np.random.default_rng(233)
    index = np.eye(4)[:2]
    U = rng.normal(size=(5, 5, 4))
    U[0] = 0.0
    U[0, 0, 0] = U[0, 1, 2] = 1.0
    I, mass = rng.normal(size=(5, 5)), np.ones(5)
    calls = []
    profile_ranks = []
    original_profile = backtracking._profile
    original = backtracking._conjugate_gradient

    def rank_changing_profile(*args, **kwargs):
        state = original_profile(*args, **kwargs)
        ranks = state.ranks.copy()
        if profile_ranks:
            ranks[0] = 2
            state = replace(state, ranks=ranks, smooth=False)
        profile_ranks.append(state.ranks.copy())
        return state

    def recording(Y, G, previous_gradient, previous_direction):
        calls.append((previous_gradient, previous_direction))
        return original(Y, G, previous_gradient, previous_direction)

    monkeypatch.setattr(backtracking, "_profile", rank_changing_profile)
    monkeypatch.setattr(backtracking, "_conjugate_gradient", recording)
    solve(index, U, I, mass=mass, max_steps=2, tol=1e-12)
    assert len(calls) == 2
    assert profile_ranks[0][0] != profile_ranks[1][0]
    assert calls[0][0] is None and calls[0][1] is None
    assert calls[1][0] is None and calls[1][1] is None


def test_smooth_path_is_numerically_unchanged():
    rng = np.random.default_rng(99)
    index = np.eye(5)[:2]
    U = rng.normal(size=(6, 7, 5))
    I, mass = rng.normal(size=(6, 7)), np.ones(6)
    baseline = production_solve(index, U, I, mass=mass, max_steps=5, tol=1e-12)
    result = solve(index, U, I, mass=mass, max_steps=10, tol=1e-12)
    assert result.index == pytest.approx(baseline.index, abs=2e-12)
    assert result.coefficients == pytest.approx(baseline.coefficients, abs=2e-12)
    for key in ("profile_history", "step_ranks", "accepted_steps", "stop_reason"):
        assert result.diagnostics[key] == baseline.diagnostics[key]


def test_conjugate_spectral_armijo_uses_actual_full_gradient_slope():
    index, U, I, mass = _case()
    Y = index.T
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    previous_gradient = 0.8 * G
    random = np.random.default_rng(65).normal(size=Y.shape)
    random -= Y @ (Y.T @ random)
    random -= G * (np.sum(G * random) / np.sum(G**2))
    random *= 0.1 * np.linalg.norm(G) / np.linalg.norm(random)
    search, beta = backtracking._conjugate_gradient(
        Y, G, previous_gradient, -previous_gradient + random
    )
    assert beta > 0

    left, singular, _right = np.linalg.svd(search, full_matrices=False)
    available = min(2, Y.shape[0] - 2, np.count_nonzero(singular > 1e-12 * singular[0]))
    V = left[:, :available]
    V -= Y @ (Y.T @ V)
    V, _ = np.linalg.qr(V, mode="reduced")
    reduced_gradient = V.T @ G
    L, values, R = np.linalg.svd(V.T @ search, full_matrices=False)
    angular_speed = values / values[0]
    slope = -float(np.sum(reduced_gradient * ((L * angular_speed) @ R)))
    rotation_V = V @ L

    def objective(step):
        angles = step * angular_speed
        delta = (Y @ R.T) * (np.cos(angles) - 1) - rotation_V * np.sin(angles)
        moved = Y + delta @ R
        return _direct_fit(U, moved, I, mass)[2]

    step = 1e-6
    finite_difference = (objective(step) - objective(-step)) / (2 * step)
    assert slope < 0
    assert finite_difference == pytest.approx(slope, rel=3e-6, abs=3e-7)
