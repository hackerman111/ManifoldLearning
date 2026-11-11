"""Independent references for the shared nonlinear subspace profile."""

import importlib.util
from itertools import pairwise

import numpy as np
import pytest
from scipy.special import eval_hermitenorm, factorial


def module():
    assert importlib.util.find_spec("experiments.shared_polynomial") is not None
    from experiments import shared_polynomial

    return shared_polynomial


def sample(seed=8, n=180, d=6, m=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    b = np.linalg.qr(rng.normal(size=(d, m)))[0]
    z = x @ b
    y = np.sin(z[:, 0]) + 0.2 * z[:, 1] ** 3 + 0.02 * rng.normal(size=n)
    return rng, x, y, b


def test_features_match_scalar_hermite_reference():
    lib = module()
    _, x, _, b = sample()
    for degree in (1, 3, 5):
        powers = lib.powers(2, degree)
        actual = lib.features(x @ b, powers)
        expected = np.column_stack(
            [
                np.prod(
                    [
                        eval_hermitenorm(k, (x @ b)[:, j]) / np.sqrt(factorial(k))
                        for j, k in enumerate(alpha)
                    ],
                    axis=0,
                )
                for alpha in powers
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=2e-13)


def test_profile_matches_dense_least_squares_and_rotation():
    lib = module()
    rng, x, y, b = sample()
    profile = lib.PolynomialProfile(x, y, 3)
    state = profile(b)
    design = lib.features(x @ b, lib.powers(2, 3))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(state.coefficients, coef, atol=1e-11)
    assert state.value == pytest.approx(np.mean((design @ coef - y) ** 2))
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = profile(b @ rotation)
    np.testing.assert_allclose(rotated.prediction, state.prediction, atol=2e-12)
    np.testing.assert_allclose(rotated.gradient, state.gradient @ rotation, atol=2e-11)
    assert state.normal_residual < 1e-12


def test_envelope_gradient_matches_finite_difference():
    lib = module()
    rng, x, y, b = sample()
    profile = lib.PolynomialProfile(x, y, 3)
    state = profile(b)
    direction = rng.normal(size=b.shape)
    direction -= b @ (b.T @ direction)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    finite = (
        profile(lib.polar(b + h * direction)).value
        - profile(lib.polar(b - h * direction)).value
    ) / (2 * h)
    assert finite == pytest.approx(np.sum(state.gradient * direction), abs=1e-8)


def test_schur_matrix_equals_explicit_projected_jacobian():
    lib = module()
    _, x, y, b = sample()
    profile = lib.PolynomialProfile(x, y, 3)
    state = profile(b)
    v = np.linalg.qr(b, mode="complete")[0][:, 2:]
    jac = np.einsum("na,nk->nak", x @ v, state.link_gradient).reshape(len(x), -1)
    design = lib.features(x @ b, lib.powers(2, 3))
    reference = jac - design @ np.linalg.lstsq(design, jac, rcond=None)[0]
    hessian, score = lib.schur_system(x @ v, state)
    np.testing.assert_allclose(hessian, reference.T @ reference / len(x), atol=2e-12)
    np.testing.assert_allclose(score, v.T @ state.gradient / 2, atol=2e-12)


def test_recovers_exact_polynomial_with_weak_direction():
    lib = module()
    rng, x, _, truth = sample(n=600)
    z = x @ truth
    y = z[:, 0] + 0.08 * (z[:, 1] ** 3 - 3 * z[:, 1])
    start = lib.polar(truth + 0.2 * rng.normal(size=truth.shape))
    result = lib.refine(x, y, start, degree=3, max_steps=60)
    edr = np.linalg.norm(result.basis @ result.basis.T - truth @ truth.T) / 2
    assert edr < 1e-5
    assert result.diagnostics["normal_residual_max"] < 1e-10
    history = result.diagnostics["history"]
    assert all(v <= u + 1e-13 for u, v in pairwise(history))


def test_rank_failure_and_invalid_inputs_are_explicit():
    lib = module()
    with pytest.raises(ValueError, match="rank"):
        lib.PolynomialProfile(np.ones((100, 5)), np.arange(100.0), 3)(np.eye(5)[:, :2])
    with pytest.raises(ValueError, match="finite"):
        lib.PolynomialProfile(np.full((100, 5), np.nan), np.ones(100), 3)
    with pytest.raises(ValueError, match="degree"):
        lib.PolynomialProfile(np.ones((100, 5)), np.ones(100), 0)
    _, x, y, b = sample(n=8)
    with pytest.raises(ValueError, match="observations"):
        lib.PolynomialProfile(x, y, 5)(b)


def test_zero_signal_reports_gradient_rank_failure():
    lib = module()
    _, x, _, b = sample()
    result = lib.refine(x, np.zeros(len(x)), b)
    assert result.diagnostics["gradient_rank_ok"] is False
    assert result.diagnostics["stop_reason"] == "gradient_rank_deficient"


def test_nonzero_linear_signal_does_not_identify_two_dimensions():
    lib = module()
    _, x, _, _ = sample()
    result = lib.refine(x, x[:, 0], np.eye(6)[:, :2])
    assert not result.diagnostics["gradient_rank_ok"]
    assert not result.diagnostics["converged"]


def test_full_coefficient_rank_is_not_enough_for_tangent_information():
    lib = module()
    _, x, y, b = sample(n=16)
    assert lib.PolynomialProfile(x, y, 3)(b).design_condition < 1e10
    with pytest.raises(ValueError, match="tangent"):
        lib.refine(x, y, b, degree=3)


def test_near_rank_deficient_design_is_rejected():
    lib = module()
    rng, x, y, _ = sample()
    x[:, 1] = x[:, 0] + 1e-13 * rng.normal(size=len(x))
    with pytest.raises(lib.PolynomialRankError):
        lib.PolynomialProfile(x, y, 3)(np.eye(6)[:, :2])


def test_weak_nonzero_direction_is_not_dropped():
    lib = module()
    rng, x, _, truth = sample(n=600)
    z = x @ truth
    y = z[:, 0] + 0.002 * (z[:, 1] ** 3 - 3 * z[:, 1])
    start = lib.polar(truth + 0.1 * rng.normal(size=truth.shape))
    result = lib.refine(x, y, start, tol=1e-10)
    assert result.diagnostics["gradient_rank_ok"]
    assert result.diagnostics["tangent_rank_ok"]
    assert np.linalg.norm(result.basis @ result.basis.T - truth @ truth.T) < 1e-4


def test_complete_fit_accepts_only_data_and_reproduces_with_seed():
    lib = module()
    _, x, _, truth = sample(n=300)
    z = x @ truth
    y = z[:, 0] ** 2 + 0.5 * (z[:, 1] ** 3 - 3 * z[:, 1])
    one = lib.fit(x, y, 2, seed=71)
    two = lib.fit(x, y, 2, seed=71)
    np.testing.assert_allclose(
        one.basis @ one.basis.T, two.basis @ two.basis.T, atol=1e-12
    )
    assert np.linalg.norm(one.basis @ one.basis.T - truth @ truth.T) < 1e-4
    assert one.diagnostics["fit_seconds"] >= one.diagnostics["refine_seconds"] > 0
