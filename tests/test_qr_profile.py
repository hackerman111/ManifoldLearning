"""Guarded general QR must preserve the profiled problem and Jacobian."""

import numpy as np
import pytest

from ADP.solver import grassman as base
from experiments import qr_profile as qr


@pytest.mark.parametrize("m", [3, 4, 5, 8])
@pytest.mark.parametrize("tau", [0.0, 1e-8, 0.2])
def test_qr_profile_matches_independent_augmented_lstsq_and_normal_solve(m, tau):
    rng = np.random.default_rng(1840 + m)
    M = rng.normal(size=(7, m + 5, m))
    I = rng.normal(size=M.shape[:2])
    mass = np.geomspace(0.01, 100, len(M))
    state = qr.profile(M, I, mass, tau)
    assert state.smooth and np.all(state.qr_mask)
    expected = []
    normals = []
    rhs = rng.normal(size=(len(M), m, 4))
    for j in range(len(M)):
        design = np.vstack((M[j], np.sqrt(tau) * np.eye(m)))
        target = np.r_[I[j], np.zeros(m)]
        expected.append(np.linalg.lstsq(design, target, rcond=None)[0])
        normals.append(np.linalg.solve(design.T @ design, rhs[j]))
    np.testing.assert_allclose(state.coefficients, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(state.normal_solve(rhs), normals, rtol=2e-11, atol=2e-11)
    residual = I - (M @ np.asarray(expected)[..., None]).squeeze(-1)
    value = np.sum(
        mass
        * (
            np.sum(residual**2, axis=1)
            + tau * np.sum(np.asarray(expected) ** 2, axis=1)
        )
    )
    assert state.value == pytest.approx(value, rel=2e-13, abs=1e-11)


def test_mixed_rank_loss_uses_svd_without_losing_safe_centers():
    rng = np.random.default_rng(152)
    M = rng.normal(size=(4, 12, 5))
    M[1] = 0
    M[2, :, -1] = M[2, :, 0]
    M[3, :, -1] *= 1e-6
    I = rng.normal(size=M.shape[:2])
    mass = np.ones(4)
    state = qr.profile(M, I, mass, 0.0)
    reference = base._profile_svd(M, I, mass, 0.0)
    np.testing.assert_array_equal(state.qr_mask, [True, False, False, False])
    np.testing.assert_array_equal(state.ranks, reference.ranks)
    np.testing.assert_allclose(
        state.coefficients, reference.coefficients, rtol=1e-10, atol=1e-10
    )
    assert not state.smooth
    assert state.value == pytest.approx(reference.value, rel=1e-13)
    rhs = rng.normal(size=(4, 5, 3))
    np.testing.assert_array_equal(state.normal_solve(rhs), reference.normal_solve(rhs))


def test_unsafe_but_smooth_svd_part_keeps_normal_action():
    rng = np.random.default_rng(19)
    frame = np.linalg.qr(rng.normal(size=(12, 5)))[0]
    M = np.stack((frame, frame @ np.diag([1, 1, 1, 1, 1e-5])))
    I = rng.normal(size=M.shape[:2])
    state = qr.profile(M, I, np.ones(2), 0.0)
    assert state.smooth
    np.testing.assert_array_equal(state.qr_mask, [True, False])
    rhs = rng.normal(size=(2, 5, 3))
    reference = base._profile_svd(M, I, np.ones(2), 0.0)
    np.testing.assert_allclose(
        state.normal_solve(rhs), reference.normal_solve(rhs), rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize("m,p", [(2, 8), (5, 3)])
def test_small_or_wide_uses_current_profile(m, p):
    rng = np.random.default_rng(m + p)
    M = rng.normal(size=(5, p, m))
    I = rng.normal(size=(5, p))
    result = qr.profile(M, I, np.ones(5), 0.0)
    reference = base._profile(M, I, np.ones(5), 0.0)
    np.testing.assert_array_equal(result.coefficients, reference.coefficients)
    assert result.smooth == reference.smooth


@pytest.mark.parametrize("tau", [0.0, 0.3])
def test_general_qr_core_jacobian_matches_finite_difference(tau):
    rng = np.random.default_rng(621)
    Y = np.linalg.qr(rng.normal(size=(11, 4)))[0]
    V = rng.normal(size=(11, 3))
    V -= Y @ (Y.T @ V)
    V = np.linalg.qr(V)[0]
    U = rng.normal(size=(9, 8, 11))
    I = rng.normal(size=(9, 8))
    mass = np.geomspace(0.1, 10, 9)
    state = qr.profile(U @ Y, I, mass, tau)
    jac, _ = base._core_jacobian(state, U @ V, mass, tau)
    K = rng.normal(size=(3, 4))
    K /= np.linalg.norm(K)

    def packed(step):
        _, M = base._polar(Y, state.M, V, U @ V, step * K)
        trial = qr.profile(M, I, mass, tau)
        residual = (
            np.concatenate((trial.residual, -np.sqrt(tau) * trial.coefficients), axis=1)
            if tau
            else trial.residual
        )
        return (np.sqrt(mass)[:, None] * residual).ravel()

    derivative = (packed(1e-5) - packed(-1e-5)) / 2e-5
    np.testing.assert_allclose(jac @ K.ravel(), derivative, rtol=2e-8, atol=2e-8)
