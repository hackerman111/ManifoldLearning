"""The projected Jacobian proposal is a projection, not the full dr."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import qr

from experiments.grassman_accuracy_2026_10_22 import candidates as c


@pytest.mark.parametrize(
    "m,tau", [(1, 0.0), (1, 0.03), (2, 0.0), (2, 0.03), (4, 0.0), (4, 0.03)]
)
def test_projected_jacobian_matches_explicit_augmented_qr_projection(m, tau):
    rng = np.random.default_rng(1300 + 10 * m + int(100 * tau))
    J, p, q = 5, 10, 3
    M = rng.normal(size=(J, p, m))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.03, 30.0, J)
    W = rng.normal(size=(J, p, q))
    state = c.reference._profile(M, I, mass, tau)
    assert state.smooth
    projected, residual = c.projected_jacobian(state, W, mass, tau)
    projected = projected.reshape(J, p + (m if tau else 0), q * m)
    residual = residual.reshape(J, p + (m if tau else 0))
    expected_projected = np.empty_like(projected)
    expected_full = np.empty_like(projected)
    expected_residual = np.empty_like(residual)

    for j in range(J):
        design = (
            np.vstack((M[j], np.sqrt(tau) * np.eye(m)))
            if tau else M[j]
        )
        r_aug = (
            np.r_[state.residual[j], -np.sqrt(tau) * state.coefficients[j]]
            if tau else state.residual[j]
        )
        Q, R = qr(design, mode="economic")
        assert np.linalg.norm(design.T @ r_aug) <= 2e-12 * max(
            1.0, np.linalg.norm(design) * np.linalg.norm(r_aug)
        )
        expected_residual[j] = np.sqrt(mass[j]) * r_aug
        for a in range(q):
            for b in range(m):
                ddesign = np.zeros_like(design)
                ddesign[:p, b] = W[j, :, a]
                direct = ddesign @ state.coefficients[j]
                perpendicular = direct - Q @ (Q.T @ direct)
                col = a * m + b
                omitted = Q @ np.linalg.solve(R.T, ddesign.T @ r_aug)
                expected_projected[j, :, col] = -np.sqrt(mass[j]) * perpendicular
                expected_full[j, :, col] = -np.sqrt(mass[j]) * (perpendicular + omitted)

    np.testing.assert_allclose(projected, expected_projected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(residual, expected_residual, rtol=2e-12, atol=2e-12)
    omitted = expected_full - expected_projected
    np.testing.assert_allclose(projected.transpose(0, 2, 1) @ omitted, 0, atol=2e-9)
    gram_delta = expected_full.reshape(-1, q * m).T @ expected_full.reshape(-1, q * m)
    gram_proj = projected.reshape(-1, q * m).T @ projected.reshape(-1, q * m)
    assert np.linalg.eigvalsh(gram_delta - gram_proj).min() >= -2e-9
    np.testing.assert_allclose(
        projected.reshape(-1, q * m).T @ expected_residual.ravel(),
        expected_full.reshape(-1, q * m).T @ expected_residual.ravel(),
        rtol=2e-9, atol=2e-9,
    )


def test_projected_jacobian_rejects_local_rank_loss():
    M = np.zeros((2, 4, 1))
    state = c.reference._profile(M, np.ones((2, 4)), np.ones(2), 0.0)
    assert not state.smooth
    with pytest.raises(ValueError, match="safe full local rank"):
        c.projected_jacobian(state, np.ones((2, 4, 1)), np.ones(2), 0.0)

    rng = np.random.default_rng(1389)
    v, w = rng.normal(size=(2, 8)), rng.normal(size=(2, 8))
    nearly_dependent = np.stack((v, v + 1e-8 * w), axis=-1)
    state = c.reference._profile(
        nearly_dependent, rng.normal(size=(2, 8)), np.ones(2), 0.0
    )
    assert not state.smooth
    with pytest.raises(ValueError, match="safe full local rank"):
        c.projected_jacobian(
            state, np.ones((2, 8, 1)), np.ones(2), 0.0
        )


def test_projected_solver_variants_return_live_monotone_certificates():
    rng = np.random.default_rng(1411)
    J, p, d, m = 28, 6, 9, 2
    B, _ = np.linalg.qr(rng.normal(size=(d, m)))
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    mass = np.geomspace(0.1, 10.0, J)
    for name in ("projected10", "projected20"):
        result = c.solver_for(name)(B.T, U, I, mass=mass, tol=1e-12)
        Y = result.index.T
        assert np.allclose(Y.T @ Y, np.eye(m), atol=2e-8)
        state = c.reference._profile(U @ Y, I, mass, 0.0)
        G = c.reference._gradient(Y, U, state, mass)
        history = result.diagnostics["profile_history"]
        assert np.all(np.diff(history) <= 1e-10)
        assert np.isclose(history[-1], state.value, rtol=2e-8, atol=2e-9)
        assert np.isclose(
            result.diagnostics["gradient_norm"], np.linalg.norm(G), rtol=2e-7, atol=2e-9
        )
