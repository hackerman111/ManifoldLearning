"""Catch incorrect chain-rule gradients and accidental changes of the profile."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from ADP.core.ADP_Statistic import ADP_Statistics
from experiments import full_ridge_batched as baseline
from experiments import light_metric as candidate


def problem(kind="full"):
    rng = np.random.default_rng(1262026)
    d, m, j = 6, 2, 7
    raw = rng.normal(size=(j, d, d))
    covariance = raw @ raw.swapaxes(1, 2) + np.eye(d)
    cross = rng.normal(size=(j, d))
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    if kind == "singular":
        covariance[:, 2:, :] = 0
        covariance[:, :, 2:] = 0
        cross[:, 2:] = 0
    if kind == "zero":
        covariance[:] = 0
        cross[:] = 0
    if kind == "nullspace":
        covariance[:] = np.diag([2, 1, 0, 0, 0, 0])
        cross[:] = np.array([1, 0, 0, 0, 0, 0])
        basis = np.eye(d)[:, 2:4] + 1e-9 * np.eye(d)[:, :2]
        basis = np.linalg.qr(basis)[0]
    return basis, ADP_Statistics(
        cross,
        covariance,
        np.ones(j),
        rng.normal(size=(j, d)),
        np.full(j, 20),
        np.empty((j, 0)),
        rng.normal(size=j),
    )


@pytest.mark.parametrize("arm", ["identity", "right", "feature", "both"])
def test_transformed_chart_gradient_matches_finite_differences(arm):
    basis, stats = problem()
    p = baseline.build(stats, basis)
    chart = candidate.Chart(basis, p, arm)
    point = np.random.default_rng(3).normal(size=8) * 0.1
    value, gradient, _, state = chart.evaluate(point)
    numeric = approx_derivative(lambda z: chart.evaluate(z)[0], point).ravel()
    np.testing.assert_allclose(gradient, numeric, rtol=2e-6, atol=1e-8)
    assert value == pytest.approx(state.value / p.scale)


def test_identity_solver_reproduces_existing_solver():
    basis, stats = problem()
    old = baseline.solve(basis.T, stats, max_steps=6)
    new = candidate.solve(basis.T, stats, "identity", max_steps=6)
    np.testing.assert_allclose(
        new.index.T @ new.index, old.index.T @ old.index, atol=1e-11
    )
    assert new.diagnostics["profile_objective"] == pytest.approx(
        old.diagnostics["profile_objective"], abs=1e-12
    )
    assert (
        new.diagnostics["profile_evaluations"] == old.diagnostics["profile_evaluations"]
    )


@pytest.mark.parametrize("kind", ["full", "singular", "zero", "nullspace"])
@pytest.mark.parametrize("arm", ["right", "feature", "both"])
def test_metric_keeps_profile_and_accepts_only_descent(kind, arm):
    basis, stats = problem(kind)
    p = baseline.build(stats, basis)
    result = candidate.solve(basis.T, stats, arm, max_steps=5)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    state = baseline.profile(result.index.T, p)
    assert result.diagnostics["profile_objective"] == pytest.approx(
        state.value, abs=1e-10
    )
    history = result.diagnostics["profile_history"]
    assert all(b <= a + 1e-10 * max(1, abs(a)) for a, b in pairwise(history))
    assert np.isfinite(result.diagnostics["riemannian_gradient"])


def test_coordinate_factors_regularize_only_the_map():
    basis, stats = problem()
    p = baseline.build(stats, basis)
    tau = p.tau
    candidate.Chart(basis, p, "both")
    assert p.tau == tau
    q = np.linalg.qr(np.random.default_rng(1).normal(size=(2, 2)))[0]
    left = candidate.regularized_inverse_sqrt(np.diag([0.0, 2.0]))
    right = candidate.regularized_inverse_sqrt(q.T @ np.diag([0.0, 2.0]) @ q)
    np.testing.assert_allclose(right, q.T @ left @ q, atol=1e-12)
    np.testing.assert_allclose(
        candidate.regularized_inverse_sqrt(np.zeros((2, 2))), np.eye(2)
    )
    with pytest.raises(ValueError):
        candidate.regularized_inverse_sqrt(np.diag([-1.0, 1.0]))
    with pytest.raises(ValueError):
        candidate.solve(basis.T * 2, stats, "right")
    with pytest.raises(ValueError):
        candidate.solve(basis.T, replace(stats, U=-stats.U), "right")
