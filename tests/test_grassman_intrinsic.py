"""Independent checks for the intrinsic-rank approximate GN model."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ADP.solver import grassman
from ADP.solver.grassman import _gradient, _profile, _project, solve as production_solve
from experiments.grassman_accuracy_2026_10_18 import backtracking
from experiments.grassman_accuracy_2026_10_18.intrinsic import solve


def direct_fit(U: np.ndarray, Y: np.ndarray, I: np.ndarray, mass: np.ndarray):
    coefficients, ranks, value = [], [], 0.0
    for matrix, target, weight in zip(U @ Y, I, mass, strict=True):
        coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ coef
        coefficients.append(coef)
        ranks.append(rank)
        value += weight * float(residual @ residual)
    return np.asarray(coefficients), np.asarray(ranks), value


def low_rank_case():
    rng = np.random.default_rng(417)
    U = rng.normal(size=(7, 6, 5))
    U[0] = 0.0
    U[0, 0, :2] = (1.0, 0.7)
    I = rng.normal(size=(7, 6))
    mass = np.geomspace(0.2, 2.5, len(I))
    Y = np.eye(5)[:, :2]
    return Y.T, U, I, mass


def test_intrinsic_model_matches_regular_center_profile_and_full_refit():
    index, U, I, mass = low_rank_case()
    Y = index.T
    full = _profile(_project(U, Y), I, mass, 0.0)
    cached = np.full(len(I), -1, dtype=int)
    model = backtracking._intrinsic_model(U, I, mass, full, cached)
    assert model is not None
    regular, regular_state = model
    assert np.flatnonzero(~regular).tolist() == [0]
    assert cached[0] == full.ranks[0] == 1
    assert full.value == pytest.approx(
        regular_state.value + mass[0] * np.linalg.norm(full.residual[0]) ** 2,
        abs=2e-12,
    )
    full_gradient = _gradient(Y, U, full, mass)
    regular_gradient = _gradient(Y, U[regular], regular_state, mass[regular])
    assert full_gradient == pytest.approx(regular_gradient, abs=2e-11)

    result = solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    coef, ranks, value = direct_fit(U, result.index.T, I, mass)
    reference = production_solve(
        index, U[regular], I[regular], mass=mass[regular], max_steps=1, tol=1e-12
    )
    reference_value = direct_fit(
        U[regular], reference.index.T, I[regular], mass[regular]
    )[2]
    constant_offset = direct_fit(
        U[~regular], reference.index.T, I[~regular], mass[~regular]
    )[2]
    assert result.diagnostics["intrinsic_gn_proposals"] == 1
    assert result.diagnostics["rank_guard_fallbacks"] == 0
    assert result.index.T @ result.index == pytest.approx(
        reference.index.T @ reference.index, abs=2e-10
    )
    assert value == pytest.approx(reference_value + constant_offset, abs=2e-10)
    assert result.diagnostics["profile_objective"] == pytest.approx(value, abs=2e-11)
    assert result.coefficients == pytest.approx(coef, abs=2e-11)
    assert result.diagnostics["local_rank_loss"] == np.count_nonzero(ranks < 2) == 1
    assert not result.diagnostics["stationarity_applicable"]
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 1e-12)
    assert cast(float, result.diagnostics["orthogonality_error"]) < 1e-12


@pytest.mark.parametrize(("m", "intrinsic_rank"), [(1, 0), (3, 2)])
def test_intrinsic_general_rank_matches_regular_reference(m: int, intrinsic_rank: int):
    rng = np.random.default_rng(700 + m)
    p, d, centers = m + 3, m + 2, 6
    U = rng.normal(size=(centers, p, d))
    U[0] = 0.0
    if intrinsic_rank:
        U[0, :intrinsic_rank, :intrinsic_rank] = np.eye(intrinsic_rank)
    I, mass = rng.normal(size=(centers, p)), np.geomspace(0.3, 2.0, centers)
    index = np.eye(d)[:m]
    initial = _profile(U @ index.T, I, mass, 0.0)
    assert initial.ranks[0] == intrinsic_rank
    assert np.linalg.matrix_rank(U[0]) == intrinsic_rank

    result = solve(index, U, I, mass=mass, max_steps=1, tol=1e-12)
    reference = production_solve(
        index, U[1:], I[1:], mass=mass[1:], max_steps=1, tol=1e-12
    )
    coef, ranks, value = direct_fit(U, result.index.T, I, mass)
    regular_value = direct_fit(U[1:], reference.index.T, I[1:], mass[1:])[2]
    offset = direct_fit(U[:1], reference.index.T, I[:1], mass[:1])[2]
    assert result.diagnostics["intrinsic_gn_proposals"] == 1
    assert result.index.T @ result.index == pytest.approx(
        reference.index.T @ reference.index, abs=3e-9
    )
    assert value == pytest.approx(regular_value + offset, abs=3e-9)
    assert result.diagnostics["profile_objective"] == pytest.approx(value, abs=3e-10)
    assert result.coefficients == pytest.approx(coef, abs=3e-9)
    assert result.diagnostics["local_rank_loss"] == np.count_nonzero(ranks < m)
    assert cast(float, result.diagnostics["orthogonality_error"]) < 2e-12


def test_projected_rank_loss_with_full_rank_U_falls_back():
    U = np.zeros((3, 4, 4))
    U[0, 0, 0] = U[0, 1, 2] = 1.0  # rank(U0)=2, rank(U0 @ Y)=1
    U[1:] = np.random.default_rng(19).normal(size=(2, 4, 4))
    I = np.random.default_rng(20).normal(size=(3, 4))
    index = np.eye(4)[:2]
    mass = np.ones(3)
    initial = _profile(U @ index.T, I, mass, 0.0)
    assert initial.ranks[0] == 1
    assert np.linalg.matrix_rank(U[0]) == 2
    result = solve(index, U, I, mass=mass, max_steps=1)
    assert result.diagnostics["intrinsic_gn_proposals"] == 0
    assert result.diagnostics["rank_guard_fallbacks"] == 1


def test_sensitive_regular_subset_and_ridge_bypass_intrinsic_model():
    index = np.eye(4)[:2]
    U = np.zeros((3, 4, 4))
    U[0, 0, :2] = (1.0, 1.0)
    U[1, 0, 0], U[1, 1, 1], U[1, 2, 2] = 1.0, 1e-8, 1.0
    U[2] = np.random.default_rng(27).normal(size=(4, 4))
    I = np.random.default_rng(28).normal(size=(3, 4))
    mass = np.ones(3)
    state = _profile(U @ index.T, I, mass, 0.0)
    assert not state.smooth
    assert backtracking._intrinsic_model(U, I, mass, state, np.full(3, -1)) is None
    fallback = solve(index, U, I, mass=mass, max_steps=1)
    assert fallback.diagnostics["intrinsic_gn_proposals"] == 0
    assert fallback.diagnostics["rank_guard_fallbacks"] == 1

    ridge = solve(index, U, I, mass=mass, max_steps=1, local_ridge=0.2)
    assert ridge.diagnostics["intrinsic_gn_proposals"] == 0
    assert ridge.diagnostics["local_ridge"] == 0.2


@pytest.mark.parametrize("module", [backtracking, grassman])
def test_intrinsic_rank_svd_batches_stay_within_four_mib(monkeypatch, module):
    rng = np.random.default_rng(31)
    centers, p, d = 130, 64, 64
    U = np.zeros((centers, p, d))
    U[:, 0, :] = rng.normal(size=(centers, d))
    U[-1, :2, :2] = np.eye(2)
    Y = np.eye(d)[:, :2]
    I, mass = rng.normal(size=(centers, p)), np.ones(centers)
    state = _profile(U @ Y, I, mass, 0.0)
    original_svd = np.linalg.svd
    seen = []

    def recording_svd(array, *args, **kwargs):
        seen.append(array.nbytes)
        return original_svd(array, *args, **kwargs)

    monkeypatch.setattr(module.np.linalg, "svd", recording_svd)
    model = module._intrinsic_model(U, I, mass, state, np.full(centers, -1))
    assert model is not None
    assert max(seen) <= 4 * 1024**2
    assert len(seen) >= 2
