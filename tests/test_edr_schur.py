from __future__ import annotations

import numpy as np
import pytest

from ADP.solver._grassman_schur import SchurCore, metric_direction, trust_step


def _data(seed=741, *, J=7, p=6, m=3, q=2):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(J, p, m))
    W = rng.normal(size=(J, p, q))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.2, 3.0, J)
    return rng, M, W, I, mass


def _reference(M, W, I, mass, tau, A, K):
    J, _, m = M.shape
    # Coordinates of Y+V K A.T, followed by its polar factor.
    raw = np.concatenate(
        (
            np.broadcast_to(np.eye(m), (J, m, m)),
            np.zeros((J, W.shape[2], m)),
        ),
        axis=1,
    )
    raw[:, m:, :] = K @ A.T
    gram = raw.swapaxes(1, 2) @ raw
    eig, vec = np.linalg.eigh(gram)
    polar = raw @ ((vec * (1.0 / np.sqrt(eig))[:, None, :]) @ vec.swapaxes(1, 2))
    design = np.concatenate((M, W), axis=2) @ polar
    loss = 0.0
    for j in range(J):
        aug = np.vstack((design[j], np.sqrt(tau) * np.eye(m)))
        target = np.r_[I[j], np.zeros(m)]
        coef = np.linalg.lstsq(aug, target, rcond=None)[0]
        residual = I[j] - design[j] @ coef
        loss += mass[j] * (residual @ residual + tau * (coef @ coef))
    return float(loss)


@pytest.mark.parametrize("tau", [0.0, 0.4, 1e-9])
def test_schur_profile_gradient_hvp_match_polar_augmented_lstsq(tau):
    rng, M, W, I, mass = _data()
    m, q, a = 3, 2, 2
    A = np.linalg.qr(rng.normal(size=(m, a)))[0]
    core = SchurCore(M, W, I, mass, tau, A=A)
    K = rng.normal(scale=0.12, size=(q, a))
    value, gradient = core.evaluate(K)
    reference = _reference(M, W, I, mass, tau, A, K)
    np.testing.assert_allclose(value, reference, rtol=2e-12, atol=2e-12)
    E = rng.normal(size=K.shape)
    E /= np.linalg.norm(E)
    h = 2e-5
    plus = core.evaluate(K + h * E)[0]
    minus = core.evaluate(K - h * E)[0]
    np.testing.assert_allclose(
        np.sum(gradient * E),
        (plus - minus) / (2 * h),
        rtol=2e-6,
        atol=2e-7,
    )
    hv = core.hvp(K, E)
    gp = core.evaluate(K + h * E)[1]
    gm = core.evaluate(K - h * E)[1]
    np.testing.assert_allclose(hv, (gp - gm) / (2 * h), rtol=3e-5, atol=2e-6)
    F = rng.normal(size=K.shape)
    np.testing.assert_allclose(
        np.sum(F * core.hvp(K, E)),
        np.sum(E * core.hvp(K, F)),
        rtol=2e-10,
        atol=2e-10,
    )
    apply_inverse = core.precondition(K, damping=0.1)
    probe = rng.normal(size=K.shape)
    preconditioned = apply_inverse(probe)
    assert preconditioned.shape == K.shape
    assert np.all(np.isfinite(preconditioned))
    assert np.sum(probe * preconditioned) > 0


def test_schur_full_active_and_tau_zero_rank_guard():
    _, M, W, I, mass = _data(seed=742, m=2, q=2)
    K = np.zeros((2, 2))
    full = SchurCore(M, W, I, mass, 0.2)
    value, grad = full.evaluate(K)
    reference = _reference(M, W, I, mass, 0.2, np.eye(2), K)
    np.testing.assert_allclose(value, reference, rtol=2e-12, atol=2e-12)
    assert grad.shape == K.shape
    M[:, :, 1] = M[:, :, 0]
    singular = SchurCore(M, W, I, mass, 0.0)
    with pytest.raises(ValueError):
        singular.evaluate(K)


def test_schur_hvp_indefinite_scalar_counterexample():
    M = np.array([[[1.0], [0.0]]])
    W = np.array([[[0.0], [1.0]]])
    I = np.array([[1.0, 2.0]])
    core = SchurCore(M, W, I, np.ones(1), 1.0)
    actual = core.hvp(np.zeros((1, 1)), np.ones((1, 1)))
    np.testing.assert_allclose(actual, [[-3.0]], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.full((2, 3), np.nan)])
def test_schur_rejects_bad_core_shapes_and_nonfinite_K(bad):
    _, M, W, I, mass = _data()
    core = SchurCore(M, W, I, mass, 0.3)
    with pytest.raises((ValueError, TypeError)):
        core.evaluate(bad)


def test_metric_direction_is_whitened_rank_limited_horizontal_descent():
    rng = np.random.default_rng(743)
    Y = np.linalg.qr(rng.normal(size=(9, 3)))[0]
    V = np.linalg.qr(np.eye(9) - Y @ Y.T)[0][:, :6]
    G = V @ rng.normal(size=(6, 3))
    coefficients = rng.normal(size=(8, 3)) @ np.diag([2.0, 0.7, 0.15])
    mass = np.geomspace(0.1, 4.0, len(coefficients))
    floor = 0.05
    metric = (coefficients.T * mass) @ coefficients
    eig, vec = np.linalg.eigh(metric)
    clipped = np.maximum(eig, floor * eig[-1])
    invroot = (vec * (1.0 / np.sqrt(clipped))[None, :]) @ vec.T
    whitened = G @ invroot
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    rank = 2
    whitened_rank = (left[:, :rank] * singular[:rank]) @ right[:rank]
    expected_raw = whitened_rank @ invroot
    expected = -expected_raw / np.linalg.norm(whitened_rank)
    direction = metric_direction(G, coefficients, mass, rank, metric_floor=floor)
    np.testing.assert_allclose(direction, expected, rtol=3e-10, atol=3e-11)
    assert np.linalg.matrix_rank(direction, tol=1e-10) <= rank
    np.testing.assert_allclose(Y.T @ direction, 0.0, atol=2e-12)
    assert np.sum(G * direction) < 0
    root = (vec * np.sqrt(clipped)[None, :]) @ vec.T
    assert np.linalg.norm(direction @ root, ord="fro") == pytest.approx(1.0, abs=2e-12)


def test_steihaug_trust_step_spd_solution_boundary_and_negative_curvature():
    H = np.diag([1.0, 4.0, 9.0])
    g = np.array([1.0, -2.0, 3.0])

    def apply(x):
        return H @ x

    def precondition(x):
        return x / np.diag(H)

    step, iterations, negative = trust_step(g, apply, precondition, 10.0, tol=1e-12)
    np.testing.assert_allclose(step, -np.linalg.solve(H, g), rtol=2e-12, atol=2e-12)
    assert iterations > 0 and not negative
    bounded, _, _ = trust_step(g, apply, precondition, 0.2, tol=1e-12)
    assert np.linalg.norm(bounded) == pytest.approx(0.2, abs=2e-12)

    def saddle(x):
        return np.array([-x[0], 2.0 * x[1]])

    step_nc, _, negative = trust_step(np.array([1.0, 0.0]), saddle, lambda x: x, 0.3)
    assert negative and np.linalg.norm(step_nc) == pytest.approx(0.3, abs=2e-12)
    assert np.dot(np.array([1.0, 0.0]), step_nc) < 0
    diagonal = np.array([1.0, 2.0, 4.0, 8.0])
    matrix_gradient = np.array([[1.0, -2.0], [3.0, -4.0]])

    def matrix_hvp(x):
        return (diagonal * x.ravel()).reshape(2, 2)

    def matrix_precondition(x):
        return (x.ravel() / diagonal).reshape(2, 2)

    matrix_step, _, _ = trust_step(
        matrix_gradient, matrix_hvp, matrix_precondition, 10.0, tol=1e-12
    )
    np.testing.assert_allclose(
        matrix_step,
        -matrix_gradient / diagonal.reshape(2, 2),
        rtol=2e-12,
        atol=2e-12,
    )
