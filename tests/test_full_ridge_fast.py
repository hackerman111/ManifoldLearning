"""Independent whitened and raw-data references for the direct ridge profile."""

from dataclasses import replace

import numpy as np
import pytest

from ADP.core.ADP_Statistic import ADP_Statistics
from experiments import coupled_penalties as reference
from experiments import full_ridge_fast as fast


def fixture(spectrum=None):
    rng = np.random.default_rng(973)
    j, d, m = 7, 6, 2
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    factors = rng.normal(size=(j, d, d))
    covariance = factors @ factors.swapaxes(1, 2) + np.eye(d)
    if spectrum is not None:
        rotations = np.linalg.qr(factors)[0]
        covariance = (
            rotations * np.array(spectrum)[None, None, :]
        ) @ rotations.swapaxes(1, 2)
    cross = rng.normal(size=(j, d))
    stats = ADP_Statistics(
        cross,
        covariance,
        rng.uniform(1, 3, j),
        rng.normal(size=(j, d)),
        np.full(j, 20),
        np.empty((j, 0)),
        rng.normal(size=j),
    )
    return rng, stats, basis


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        [2, 1, 1e-14, 0, 0, 0],
        [0] * 6,
        [2, 1, 2.1e-12, 1.9e-12, 0, 0],
        [2, 1, 1, 0.1, 0.01, 1e-11],
    ],
)
def test_direct_profile_matches_whitened_reference(spectrum):
    rng, stats, incoming = fixture(spectrum)
    white, _ = reference.full_moments(stats, stats.U, stats.I)
    problem = reference.build(white, incoming, "full")
    direct = fast.build(stats, incoming)
    assert direct.tau == pytest.approx(problem.tau, rel=2e-11, abs=1e-15)
    assert direct.scale == pytest.approx(problem.scale, rel=2e-10)
    for _ in range(3):
        basis = np.linalg.qr(rng.normal(size=incoming.shape))[0]
        actual = fast.profile(basis, direct)
        expected = reference.profile(basis, problem)
        np.testing.assert_allclose(
            actual.coefficients, expected.coefficients, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            actual.gradient, expected.gradient, rtol=1e-9, atol=1e-9
        )
        assert actual.value == pytest.approx(expected.value, rel=1e-9, abs=1e-9)
        assert actual.normal_residual < 1e-10
    assert direct.info["spectral_fallbacks"] == (0 if spectrum is None else 7)


def test_gradient_rotation_and_zero_response():
    rng, stats, basis = fixture()
    problem = fast.build(stats, basis)
    direction = rng.normal(size=basis.shape)
    step = 1e-6
    state = fast.profile(basis, problem)
    derivative = (
        fast.profile(basis + step * direction, problem).value
        - fast.profile(basis - step * direction, problem).value
    ) / (2 * step)
    assert derivative == pytest.approx(
        np.sum(state.gradient * direction), rel=1e-6, abs=1e-7
    )
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = fast.profile(basis @ rotation, problem)
    assert rotated.value == pytest.approx(state.value, abs=1e-12)
    np.testing.assert_allclose(
        rotated.coefficients, state.coefficients @ rotation, atol=1e-12
    )
    zero = fast.profile(
        basis, fast.build(replace(stats, I=np.zeros_like(stats.I)), basis)
    )
    assert zero.value == 0
    np.testing.assert_array_equal(zero.coefficients, 0)


def test_indefinite_covariance_rejected():
    _, stats, basis = fixture([2, 1, 1, 1, 1, -0.1])
    with pytest.raises(ValueError, match="positive semidefinite"):
        fast.build(stats, basis)


def test_subnormal_covariance_uses_reference_cutoff_floor():
    _, stats, basis = fixture()
    covariance = np.broadcast_to(
        np.diag([1e-308, 2.1e-320, 0, 0, 0, 0]), stats.U.shape
    ).copy()
    stats = replace(stats, U=covariance, I=np.zeros_like(stats.I))
    white, info = reference.full_moments(stats, stats.U, stats.I)
    actual = fast.build(stats, basis)
    assert info["rank_max"] == actual.info["rank_max"] == 1
    np.testing.assert_allclose(
        actual.covariance, white.U @ white.U.swapaxes(1, 2), atol=0, rtol=1e-10
    )


def test_all_positive_subnormal_certificate_does_not_keep_discarded_mode():
    _, stats, _ = fixture()
    covariance = np.broadcast_to(np.diag([1e-308, 2.1e-320]), (7, 2, 2)).copy()
    stats = replace(stats, U=covariance, I=np.zeros((7, 2)))
    actual = fast.build(stats, np.array([[1.0], [0.0]]))
    assert actual.info["rank_max"] == 1


def test_singular_nullspace_incoming_scale_matches_reference():
    rng, stats, _ = fixture()
    q = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    covariance = np.broadcast_to(
        (q[:, :2] * [2.0, 1.0]) @ q[:, :2].T, stats.U.shape
    ).copy()
    stats = replace(stats, U=covariance)
    incoming = q[:, 2:4]
    white, _ = reference.full_moments(stats, stats.U, stats.I)
    expected = reference.build(white, incoming, "full")
    actual = fast.build(stats, incoming)
    assert actual.tau == pytest.approx(expected.tau, rel=1e-12, abs=0)


def test_mixed_zero_and_rank_one_windows_with_positive_ridge():
    _, stats, basis = fixture()
    covariance = np.zeros_like(stats.U)
    covariance[:5, 0, 0] = 1
    cross = np.zeros_like(stats.I)
    cross[:5, 0] = 1
    problem = fast.build(replace(stats, U=covariance, I=cross), basis)
    assert problem.tau > 0
    state = fast.profile(basis, problem)
    assert np.isfinite(state.value)
    assert np.isfinite(state.gradient).all()
    assert state.normal_residual < 1e-10


def test_streamed_statistics_match_raw_centered_moments():
    rng = np.random.default_rng(974)
    x, y = rng.normal(size=(30, 6)), rng.normal(size=30)
    weights = rng.uniform(size=(7, 30))
    weights[weights < 0.7] = 0
    directions = rng.normal(size=(7, 4, 6))
    blocks = ((i, weights[i : i + 2]) for i in range(0, 7, 2))
    stats = fast.calculate_statistics(x, y, blocks, directions)
    for j, row in enumerate(weights):
        a = row / row.sum()
        z, dy = x - a @ x, y - a @ y
        np.testing.assert_allclose(stats.U[j], z.T @ (a[:, None] * z), atol=1e-14)
        np.testing.assert_allclose(stats.I[j], z.T @ (a * dy), atol=1e-14)
        assert stats.S[j] == pytest.approx(a @ y)
        assert stats.n_eff[j] == pytest.approx(1 / (a @ a))
    assert list(blocks) == []


@pytest.mark.parametrize(
    "kind", ["negative", "empty", "nan", "incomplete", "unordered"]
)
def test_invalid_weights_rejected(kind):
    rng, _, _ = fixture()
    weights = np.ones((7, 30))
    if kind == "negative":
        weights[0, 0] = -1
    elif kind == "empty":
        weights[0] = 0
    elif kind == "nan":
        weights[0, 0] = np.nan
    blocks = [(0, weights)]
    if kind == "incomplete":
        blocks = [(0, weights[:3])]
    if kind == "unordered":
        blocks = [(1, weights[:3])]
    with pytest.raises(ValueError):
        fast.calculate_statistics(
            rng.normal(size=(30, 6)), np.ones(30), iter(blocks), np.ones((7, 4, 6))
        )


def test_solver_matches_reference_subspace():
    _, stats, basis = fixture()
    actual = fast.solve(basis.T, stats, max_steps=5)
    expected = reference.solve(
        basis.T, stats, "full", moments=(stats.U, stats.I), max_steps=5
    )
    np.testing.assert_allclose(
        actual.index.T @ actual.index, expected.index.T @ expected.index, atol=2e-9
    )
    np.testing.assert_allclose(actual.index @ actual.index.T, np.eye(2), atol=1e-12)


def test_full_dimension_chart_rejected():
    _, stats, _ = fixture()
    with pytest.raises(ValueError, match="dimension"):
        fast.solve(np.eye(6), stats)
