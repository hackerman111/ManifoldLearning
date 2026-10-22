from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import grassman as g
from experiments.grassman_accuracy_2026_10_18.candidate import AngleProfile, solve


@pytest.mark.parametrize("m,tau", [(1, 0.0), (2, 0.0), (3, 0.0), (2, 0.03)])
def test_local_scalar_profiles_match_independent_lstsq(m: int, tau: float) -> None:
    rng = np.random.default_rng(330 + m)
    J, p = 13, 7
    M = rng.normal(size=(J, p, m))
    w = rng.normal(size=(J, p))
    I = rng.normal(size=(J, p))
    M[0] = 0
    w[0] = 0
    if m > 1:
        M[1, :, 1:] = 0
    M[2] *= 1e-13
    w[2] *= 1e-13
    mass = np.geomspace(1e-4, 1e4, J)
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    curve = AngleProfile(M, w, a, I, mass, tau)
    for t in (0.0, 1e-8, 0.13, 0.5, -0.3):
        designs = M + ((np.cos(t) - 1) * (M @ a) + np.sin(t) * w)[..., None] * a
        expected = 0.0
        for j, design in enumerate(designs):
            augmented = np.vstack((design, np.sqrt(tau) * np.eye(m)))
            target = np.r_[I[j], np.zeros(m)]
            coeff = np.linalg.lstsq(augmented, target, rcond=None)[0]
            residual = I[j] - design @ coeff
            expected += mass[j] * (residual @ residual + tau * (coeff @ coeff))
        assert curve.value(t) == pytest.approx(expected, rel=2e-12, abs=2e-10)


def test_unsafe_center_does_not_trigger_whole_batch_refit() -> None:
    rng = np.random.default_rng(912)
    M = rng.normal(size=(20, 6, 2))
    M[0] = 0
    w = rng.normal(size=(20, 6))
    w[0] = 0
    curve = AngleProfile(M, w, np.array([1.0, 0.0]), np.ones((20, 6)), np.ones(20), 0)
    curve.value(0.1)
    assert curve.refit_centers == 1


def test_scalar_profile_honors_full_design_rank_cutoff() -> None:
    M = np.zeros((1, 5, 2))
    M[0, 0, 0], M[0, 1, 1] = 1.0, 1e-15
    I = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    w = M[:, :, 1].copy()
    curve = AngleProfile(M, w, np.array([0.0, 1.0]), I, np.ones(1), 0)
    assert curve.value(0.0) == pytest.approx(1.0)


def test_candidate_preserves_rank_guard_true_objective_and_orthogonality() -> None:
    rng = np.random.default_rng(961)
    U = rng.normal(size=(30, 7, 8))
    U[0] = 0
    U[1, :, 1:] = 0
    I = rng.normal(size=(30, 7))
    mass = np.geomspace(0.1, 10, 30)
    Y = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    result = solve(Y.T, U, I, mass=mass, max_steps=10)
    diag = result.diagnostics
    assert diag["rank_guard_fallbacks"] > 0
    assert diag["gn_solves"] == 0
    assert not diag["stationarity_applicable"]
    assert diag["orthogonality_error"] < 1e-12
    assert np.max(np.diff(diag["profile_history"])) <= 1e-10
    assert diag["profile_objective"] == pytest.approx(
        g._profile(U @ result.index.T, I, mass, 0).value, rel=2e-13
    )


def test_direction_search_includes_original_rank_one_plane() -> None:
    from experiments.grassman_accuracy_2026_10_18.directions import solve as search

    rng = np.random.default_rng(731)
    for _ in range(6):
        U = rng.normal(size=(20, 6, 7))
        U[0] = 0
        I = rng.normal(size=(20, 6))
        Y = np.linalg.qr(rng.normal(size=(7, 2)))[0]
        original = solve(Y.T, U, I, max_steps=1, energy_tol=0)
        result = search(Y.T, U, I, max_steps=1, energy_tol=0)
        assert result.diagnostics["profile_objective"] <= (
            original.diagnostics["profile_objective"] + 1e-10
        )
        assert result.diagnostics["angle_directions_tested"] == 2
        assert result.diagnostics["step_ranks"] == (1,)
        assert result.diagnostics["gn_solves"] == 0
