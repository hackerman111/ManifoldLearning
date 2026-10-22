"""Independent checks for structurally constant rank-one curve centers."""

from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import grassman as baseline
from experiments.grassman_accuracy_2026_10_18.constant_angle import AngleProfile, solve


def direct_value(M, w, a, I, mass, tau, t):
    u = M @ a
    designs = M + (u * (np.cos(t) - 1) + w * np.sin(t))[:, :, None] * a
    total = 0.0
    for j, design in enumerate(designs):
        if tau:
            augmented = np.vstack((design, np.sqrt(tau) * np.eye(M.shape[2])))
            target = np.r_[I[j], np.zeros(M.shape[2])]
        else:
            augmented, target = design, I[j]
        coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        total += mass[j] * (residual @ residual + tau * (coefficient @ coefficient))
    return float(total), designs


def test_cached_columnspaces_match_direct_profiles_across_angles() -> None:
    rng = np.random.default_rng(96800)
    M, w, I = np.zeros((5, 4, 2)), np.zeros((5, 4)), rng.normal(size=(5, 4))
    M[1, 0] = [1.2, 0.8]       # one supported row, robust complement
    w[1, 0] = 0.3
    M[2, 0], M[2, 2] = [1.0, 0.4], [0.2, 0.5]  # multiple rows
    M[3, 0] = [1.0, 1e-5]      # near-zero complement: do not cache
    w[3, 0] = 0.4
    M[4, 0] = [1.0, 0.0]
    w[4, 0] = -2.0             # cancellation at atan(1/2)
    mass = np.array([0.0, 2.3, 1.1, 0.8, 1.5])
    curve = AngleProfile(M, w, np.array([1.0, 0.0]), I, mass, 0.0)
    assert curve.constant_centers == 2
    assert curve.curve is not None
    assert curve.constant_value == pytest.approx(
        mass[0] * (I[0] @ I[0]) + mass[1] * (I[1, 1:] @ I[1, 1:])
    )
    for t in (-0.5, 0.0, 0.5, np.arctan(0.5), np.arctan(0.5) + 1e-8):
        expected, _ = direct_value(M, w, np.array([1.0, 0.0]), I, mass, 0.0, t)
        assert curve.value(t) == pytest.approx(expected, rel=3e-12, abs=3e-12)


@pytest.mark.parametrize("small_c, cached", [(1e-4, False), (2e-4, True)])
def test_complement_cutoff_guards_near_cancellation(
    small_c: float, cached: bool
) -> None:
    M, w = np.zeros((1, 4, 2)), np.zeros((1, 4))
    I = np.array([[1.0, 0.2, -0.4, 0.7]])
    M[0, 0] = [1.0, small_c]
    curve = AngleProfile(M, w, np.array([1.0, 0.0]), I, np.ones(1), 0.0)
    assert curve.constant_centers == int(cached)
    for t in (-0.5, 0.5):
        expected, _ = direct_value(M, w, np.array([1.0, 0.0]), I, np.ones(1), 0.0, t)
        assert curve.value(t) == pytest.approx(expected, rel=3e-12, abs=3e-12)


@pytest.mark.parametrize(("m", "p", "tau"), [(1, 4, 0.0), (2, 1, 0.0),
                                                  (3, 2, 0.0), (2, 4, 0.2)])
def test_shape_and_ridge_cases_match_augmented_lstsq(
    m: int, p: int, tau: float
) -> None:
    rng = np.random.default_rng(79 + m + p)
    M = rng.normal(size=(4, p, m))
    w, I = rng.normal(size=(4, p)), rng.normal(size=(4, p))
    M[0], w[0] = 0.0, 0.0
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    curve = AngleProfile(M, w, a, I, np.array([0.0, 0.3, 1.2, 2.0]), tau)
    assert curve.constant_centers >= 1
    for t in (-0.5, 0.0, 0.5):
        expected, _ = direct_value(M, w, a, I, np.array([0.0, 0.3, 1.2, 2.0]), tau, t)
        assert curve.value(t) == pytest.approx(expected, rel=3e-12, abs=2e-11)


def test_solver_acceptance_history_and_live_certificate_use_all_centers() -> None:
    rng = np.random.default_rng(96800)
    U = rng.normal(size=(18, 6, 5))
    U[0] = 0.0
    I, mass = rng.normal(size=(18, 6)), np.geomspace(0.2, 2.0, 18)
    index = np.linalg.qr(rng.normal(size=(5, 2)))[0].T
    result = solve(index, U, I, mass=mass, max_steps=3, tol=1e-10)
    live = baseline._profile(U @ result.index.T, I, mass, 0.0)
    diag = result.diagnostics
    assert diag["profile_objective"] == pytest.approx(live.value, abs=2e-10)
    assert np.all(np.diff(diag["profile_history"]) <= 1e-10)
    assert diag["local_rank_loss"] == np.count_nonzero(
        live.ranks < result.index.shape[0]
    )
    assert diag["orthogonality_error"] < 1e-12
