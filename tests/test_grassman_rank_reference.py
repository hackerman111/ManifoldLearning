from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import grassman
from experiments.grassman_rank_k.reference import direct_profile, reference


def test_independent_reference_profile_and_derivative() -> None:
    rng = np.random.default_rng(81698)
    Y = np.linalg.qr(rng.normal(size=(9, 2)))[0]
    U, I = rng.normal(size=(7, 8, 9)), rng.normal(size=(7, 8))
    mass = np.geomspace(0.1, 10, 7)
    f, g = direct_profile(Y, U, I, mass)
    state = grassman._profile(U @ Y, I, mass, 0.0)
    assert f == pytest.approx(state.value, rel=3e-13)
    np.testing.assert_allclose(g, grassman._gradient(Y, U, state, mass), atol=2e-11)
    H = rng.normal(size=Y.shape)
    H -= Y @ (Y.T @ H)
    eps = 1e-6
    plus = np.linalg.qr(Y + eps * H)[0]
    minus = np.linalg.qr(Y - eps * H)[0]
    fd = (
        direct_profile(plus, U, I, mass)[0] - direct_profile(minus, U, I, mass)[0]
    ) / (2 * eps)
    assert fd == pytest.approx(float(np.sum(g * H)), rel=2e-6, abs=2e-7)


def test_independent_multistart_reference_reports_each_start() -> None:
    rng = np.random.default_rng(81699)
    truth = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    init = np.linalg.qr(truth + 0.08 * rng.normal(size=(7, 2)))[0]
    U = rng.normal(size=(12, 8, 7))
    I = np.einsum("jpm,jm->jp", U @ truth, rng.normal(size=(12, 2)))
    mass = np.ones(12)
    row = reference(
        {"index_init": init.T, "U": U, "I": I, "mass": mass, "truth": truth},
        max_steps=200,
    )
    assert len(row["starts"]) == 3
    assert row["objective"] < 1e-15
    assert row["orthogonality_error"] < 2e-12
    assert row["converged"]


def test_smooth_nonoptimal_stationary_point_has_no_universal_rank_bound() -> None:
    U = np.eye(4)[None]
    I = np.array([[1.0, 0.0, 0.0, 0.0]])
    initial = np.eye(4)[:, 2:]
    truth = np.eye(4)[:, :2]
    f, gradient = direct_profile(initial, U, I, np.ones(1))
    minimum, _ = direct_profile(truth, U, I, np.ones(1))
    result = grassman.block_rank_k(initial.T, U, I, rank_k=2)
    assert f == 1.0
    assert minimum == 0.0
    np.testing.assert_array_equal(gradient, np.zeros((4, 2)))
    assert result.diagnostics["stationarity_applicable"]
    assert result.diagnostics["converged"]
    assert result.diagnostics["profile_objective"] == 1.0
