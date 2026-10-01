from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import grassman as g
from ADP.solver.LSMR import _local_refit


def problem(m=2, p=7):
    rng = np.random.default_rng(9181 + m + p)
    frame = np.linalg.qr(rng.normal(size=(9, m + 3)))[0]
    Y, V = frame[:, :m], frame[:, m:]
    U = rng.normal(size=(8, p, 9))
    I = rng.normal(size=(8, p))
    mass = np.geomspace(1e-3, 1e3, 8)
    return Y, V, U, I, mass


@pytest.mark.parametrize("tau", [0.0, 0.2, 1e-9])
@pytest.mark.parametrize("m", [1, 2, 4])
def test_local_profile_and_schur_against_independent_lstsq(tau, m):
    Y, V, U, I, mass = problem(m)
    M = U @ Y
    rng = np.random.default_rng(144)
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    w = U @ V[:, 0]
    curve = g._AngleProfile(M, w, a, I, mass, tau)
    for angle in np.linspace(-1.55, 1.55, 17):
        Yn = Y + (Y @ a * (np.cos(angle) - 1) + V[:, 0] * np.sin(angle))[:, None] * a
        Mn = U @ Yn
        coeff = []
        loss = 0.0
        for j in range(len(I)):
            c = np.linalg.lstsq(
                np.vstack((Mn[j], np.sqrt(tau) * np.eye(m))),
                np.concatenate((I[j], np.zeros(m))),
                rcond=None,
            )[0]
            coeff.append(c)
            loss += mass[j] * (np.sum((I[j] - Mn[j] @ c) ** 2) + tau * np.sum(c**2))
        state = g._profile(Mn, I, mass, tau)
        np.testing.assert_allclose(state.coefficients, coeff, rtol=2e-11, atol=1e-11)
        assert state.value == pytest.approx(loss, rel=2e-13, abs=1e-10)
        assert curve.value(angle) == pytest.approx(loss, rel=2e-13, abs=1e-10)


@pytest.mark.parametrize("tau", [0.0, 0.3])
def test_full_core_jacobian_finite_difference_adjoint_and_dense_ridge(tau):
    Y, V, U, I, mass = problem()
    W = U @ V
    state = g._profile(U @ Y, I, mass, tau)
    jac, residual = g._core_jacobian(state, W, mass, tau)
    rng = np.random.default_rng(631)
    K = rng.normal(size=(3, 2))
    K /= np.linalg.norm(K)
    expected_derivative = jac @ K.ravel()
    errors = []

    def packed(M):
        s = g._profile(M, I, mass, tau)
        r = (
            np.concatenate((s.residual, -np.sqrt(tau) * s.coefficients), axis=1)
            if tau
            else s.residual
        )
        return (np.sqrt(mass)[:, None] * r).ravel()

    for h in [1e-3, 1e-4, 1e-5]:
        _, plus = g._polar(Y, state.M, V, W, h * K)
        _, minus = g._polar(Y, state.M, V, W, -h * K)
        errors.append(
            np.linalg.norm(
                (packed(plus) - packed(minus)) / (2 * h) - expected_derivative
            )
        )
    assert errors[-1] < 2e-7 and errors[-1] < errors[0] / 100
    grad = g._gradient(Y, U, state, mass)
    np.testing.assert_allclose(
        2 * jac.T @ residual, (V.T @ grad).ravel(), rtol=1e-11, atol=1e-9
    )
    # Independent coordinate FD Jacobian, not a duplicate of eq:dg.
    dense = np.empty_like(jac)
    for i, e in enumerate(np.eye(K.size)):
        _, plus = g._polar(Y, state.M, V, W, 1e-5 * e.reshape(K.shape))
        _, minus = g._polar(Y, state.M, V, W, -1e-5 * e.reshape(K.shape))
        dense[:, i] = (packed(plus) - packed(minus)) / 2e-5
    damping = 0.05

    def solve(A):
        return np.linalg.lstsq(
            np.vstack((A, np.sqrt(damping) * np.eye(K.size))),
            np.concatenate((-residual, np.zeros(K.size))),
            rcond=None,
        )[0]

    np.testing.assert_allclose(solve(jac), solve(dense), rtol=1e-7, atol=1e-8)
    x, dual = rng.normal(size=K.size), rng.normal(size=jac.shape[0])
    assert (jac @ x) @ dual == pytest.approx(x @ (jac.T @ dual), rel=1e-13, abs=1e-11)


@pytest.mark.parametrize("method", ["rank_one", "spectral", "core_gn"])
def test_solver_monotonicity_gauge_invariance_and_reference(method):
    Y, _, U, I, mass = problem()
    result = g.solve(Y.T, U, I, mass=mass, method=method, max_steps=8)
    rotated = g.solve(Y[:, ::-1].T, U, I, mass=mass, method=method, max_steps=8)
    np.testing.assert_allclose(
        result.index.T @ result.index, rotated.index.T @ rotated.index, atol=2e-8
    )
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=2e-13)
    coeff, _ = _local_refit(I, U, result.index)
    np.testing.assert_allclose(result.coefficients, coeff, atol=1e-10)
    loss = 0.5 * np.sum(
        mass[:, None] * (I - (U @ result.index.T @ coeff[..., None]).squeeze(-1)) ** 2
    )
    assert result.diagnostics["loss"] == pytest.approx(loss, rel=1e-12)
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 0)
    assert np.all(np.isfinite(result.index))
    assert not result.diagnostics["converged"]
    assert result.diagnostics["stop_reason"] == "max_steps"


def test_schur_refit_solver_pair_and_rank_guard_workspace_fallback():
    Y, _, U, I, mass = problem()
    fast = g.solve(Y.T, U, I, mass=mass, method="rank_one", max_steps=5)
    slow = g.solve(
        Y.T, U, I, mass=mass, method="rank_one", angle_backend="refit", max_steps=5
    )
    np.testing.assert_allclose(
        fast.index.T @ fast.index, slow.index.T @ slow.index, atol=2e-7
    )
    U[0] = 0
    U[1] = np.outer(np.arange(7), np.arange(9))
    result = g.solve(Y.T, U, I, mass=mass, max_steps=3)
    assert result.diagnostics["rank_guard_fallbacks"] > 0
    assert result.diagnostics["local_rank_loss"] >= 2
    state = g._profile(U @ Y, I, mass, 0)
    with pytest.raises(ValueError, match="full-rank"):
        g._core_jacobian(state, U @ Y, mass, 0)
    result = g.solve(
        Y.T, U, I, mass=mass, local_ridge=0.2, workspace_bytes=1, max_steps=3
    )
    assert result.diagnostics["workspace_fallbacks"] > 0
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 0)


def test_recovery_stationary_full_space_and_input_errors():
    Y, _, U, _, mass = problem()
    rng = np.random.default_rng(551)
    I = (U @ Y @ rng.normal(size=(8, 2))[..., None]).squeeze(-1)
    result = g.solve(Y.T, U, I, mass=mass)
    assert result.diagnostics["converged"]
    assert result.diagnostics["accepted_steps"] == 0
    result = g.solve(np.eye(9), U, I, mass=mass)
    assert result.diagnostics["converged"]
    with pytest.raises(ValueError, match="multi-index"):
        g.solve(Y[:, 0], U, I)
    for kw in [
        {"lambda_prox": 0},
        {"max_steps": 0},
        {"tol": np.nan},
        {"local_ridge": -1},
        {"rank": 8},
        {"energy_tol": 1},
        {"method": "bad"},
        {"angle_backend": "bad"},
        {"max_angle": 2},
    ]:
        with pytest.raises(ValueError):
            g.solve(Y.T, U, I, **kw)
    with pytest.raises(TypeError):
        g.solve(Y.T, U, I, max_steps=True)
    with pytest.raises(ValueError, match="finite"):
        g.solve(Y.T, U, np.full_like(I, np.nan))


def test_noiseless_recovery_from_perturbed_subspace():
    rng = np.random.default_rng(728)
    Y = np.linalg.qr(rng.normal(size=(8, 2)))[0]
    B = np.linalg.qr(Y + 0.1 * rng.normal(size=Y.shape))[0].T
    U = rng.normal(size=(100, 6, 8))
    c = rng.normal(size=(100, 2))
    I = (U @ Y @ c[..., None]).squeeze(-1)
    result = g.solve(B, U, I, max_steps=80, tol=1e-8)
    assert result.diagnostics["converged"]
    assert result.diagnostics["loss"] < 1e-11
    assert np.linalg.norm(result.index - (result.index @ Y) @ Y.T) < 1e-7


def test_nearly_collinear_ridge_and_uncertified_rank_boundary():
    Y, _, U, I, mass = problem()
    U[:, :, 1:] = U[:, :, :1] + 1e-12 * U[:, :, 1:]
    result = g.solve(Y.T, U, I, mass=mass, local_ridge=1e-4, max_steps=5)
    assert np.isfinite(result.diagnostics["profile_objective"])
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 0)
    zero = g.solve(Y.T, np.zeros_like(U), I, mass=mass)
    assert not zero.diagnostics["converged"]
    assert not zero.diagnostics["stationarity_applicable"]
    assert zero.diagnostics["stop_reason"] == "rank_boundary_stationary"
