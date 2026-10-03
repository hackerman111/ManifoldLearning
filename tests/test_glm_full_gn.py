from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import lsmr

from ADP.solver import grassman as base
from ADP.solver.grassman_exp import (
    ProfileJacobian,
    horizontal,
    packed,
    polar,
    solve_full_gn as solve,
)


@pytest.mark.parametrize("tau", [0.0, 0.2, 1e-9])
def test_full_operator_independent_fd_adjoint_and_dense_ridge(tau):
    rng = np.random.default_rng(73002)
    d, m = 7, 2
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0]
    U, I = rng.normal(size=(8, 6, d)), rng.normal(size=(8, 6))
    mass = np.geomspace(1e-3, 1e3, 8)
    state = base._profile(U @ Y, I, mass, tau)
    operator = ProfileJacobian(Y, U, state, mass, tau)
    x, dual = rng.normal(size=Y.size), rng.normal(size=operator.shape[0])
    np.testing.assert_allclose(
        operator.matvec(x) @ dual, x @ operator.rmatvec(dual), rtol=2e-13
    )
    np.testing.assert_allclose(
        (2 * operator.rmatvec(packed(state, mass, tau))).reshape(Y.shape),
        base._gradient(Y, U, state, mass),
        rtol=2e-12,
        atol=1e-10,
    )

    # Independent reference: local augmented lstsq, not prototype dg formula.
    def reference(Z):
        eigen, vectors = np.linalg.eigh(np.eye(m) + Z.T @ Z)
        Q = (Y + Z) @ (vectors * (1 / np.sqrt(eigen))) @ vectors.T
        M = U @ Q
        coefficients = np.stack(
            [
                np.linalg.lstsq(
                    np.vstack((M[j], np.sqrt(tau) * np.eye(m))),
                    np.concatenate((I[j], np.zeros(m))),
                    rcond=None,
                )[0]
                for j in range(len(I))
            ]
        )
        residual = I - (M @ coefficients[..., None]).squeeze(-1)
        if tau:
            residual = np.concatenate((residual, -np.sqrt(tau) * coefficients), axis=1)
        return (np.sqrt(mass)[:, None] * residual).ravel()

    h = 1e-5
    # FD in independent horizontal coordinates avoids numerical null-gauge noise.
    V = np.linalg.qr(Y, mode="complete")[0][:, m:]
    coordinate_shape = (d - m, m)
    coordinate_size = (d - m) * m
    dense = np.empty((operator.shape[0], coordinate_size))
    directions = [V @ e.reshape(coordinate_shape) for e in np.eye(coordinate_size)]
    for k, Z in enumerate(directions):
        dense[:, k] = (reference(h * Z) - reference(-h * Z)) / (2 * h)
    action = np.column_stack([operator.matvec(Z.ravel()) for Z in directions])
    np.testing.assert_allclose(action, dense, rtol=2e-6, atol=3e-8)
    damping = 0.05
    rhs = packed(state, mass, tau)
    expected = np.linalg.lstsq(
        np.vstack((dense, np.sqrt(damping) * np.eye(coordinate_size))),
        np.concatenate((-rhs, np.zeros(coordinate_size))),
        rcond=None,
    )[0]
    result = lsmr(
        operator, -rhs, damp=np.sqrt(damping), atol=1e-13, btol=1e-13, maxiter=100
    )
    np.testing.assert_allclose(
        result[0].reshape(Y.shape),
        V @ expected.reshape(coordinate_shape),
        rtol=3e-7,
        atol=2e-8,
    )
    Z = horizontal(Y, result[0].reshape(Y.shape))
    np.testing.assert_allclose(Y.T @ Z, 0, atol=1e-12)
    Yn, Mn = polar(Y, state.M, Z, U @ Z, 0.3)
    np.testing.assert_allclose(Yn.T @ Yn, np.eye(m), atol=1e-12)
    np.testing.assert_allclose(Mn, U @ Yn, atol=1e-12)


def test_solver_monotonicity_gauge_cap_and_rank_guard():
    rng = np.random.default_rng(73003)
    Y = np.linalg.qr(rng.normal(size=(12, 2)))[0]
    U, I = rng.normal(size=(12, 8, 12)), rng.normal(size=(12, 8))
    result = solve(Y.T, U, I, max_steps=4)
    rotated = solve(Y[:, ::-1].T, U, I, max_steps=4)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        result.index.T @ result.index, rotated.index.T @ rotated.index, atol=1e-9
    )
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 1e-10)
    assert result.diagnostics["forward_calls"] > 0
    capped = solve(Y.T, U, I, max_steps=1, lsmr_maxiter=1)
    assert capped.diagnostics["lsmr_stop"] == (7,)
    assert capped.diagnostics["fallbacks"] == ("inner_certificate",)
    assert not capped.diagnostics["converged"]
    Y = np.eye(12)[:, :2]
    U[:, :, 1] = 1e-10 * U[:, :, 0]
    unsafe = solve(Y.T, U, I, max_steps=1)
    assert unsafe.diagnostics["fallbacks"] == ("rank_guard",)
    zero = solve(Y.T, np.zeros_like(U), I, max_steps=1)
    assert zero.diagnostics["stop_reason"] == "rank_boundary_stationary"
    assert not zero.diagnostics["stationarity_applicable"]
    assert not zero.diagnostics["converged"]
    with pytest.raises(ValueError, match="safe full local rank"):
        ProfileJacobian(Y, U, base._profile(U @ Y, I, np.ones(12), 0), np.ones(12), 0)
    with pytest.raises(ValueError, match="positive"):
        solve(Y.T, U, I, max_steps=0)
    with pytest.raises(TypeError, match="integer"):
        solve(Y.T, U, I, lsmr_maxiter=True)


def test_manuscript_kyfan_and_scalar_retraction_counterexamples():
    # k=1 Ky Fan = spectral norm. Full-rank I beats every rank-one atom for G=I.
    X = np.eye(2)
    assert np.linalg.norm(X, ord=2) == 1
    assert np.trace(X) == 2
    polar_angles = np.arctan(np.array([1.0, 2.0]))
    assert abs(polar_angles[1] / polar_angles[0] - 2) > 0.5
    # Rational derivative of one center is the factored polynomial of degree 2.
    alpha, beta, d0, d1, d2 = 0.7, -0.3, 1.2, 0.1, 2.0
    for eta in [-3.0, 0.0, 2.0]:
        denominator = d2 * eta**2 + 2 * d1 * eta + d0
        expected = (
            -2
            * (alpha + beta * eta)
            * (beta * d0 - alpha * d1 + (beta * d1 - alpha * d2) * eta)
            / denominator**2
        )

        def f(t):
            return -((alpha + beta * t) ** 2) / (d2 * t**2 + 2 * d1 * t + d0)

        assert expected == pytest.approx(
            (f(eta + 1e-5) - f(eta - 1e-5)) / 2e-5, abs=1e-10
        )
