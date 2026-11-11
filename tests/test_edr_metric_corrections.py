"""Independent dense checks for the proposed moment metrics and certificate."""

import numpy as np
import pytest

from ADP.engine.common.statistic import calculate_statistics
from experiments.edr_metric_corrections import (
    moment_covariances,
    projector_certificate,
    reweight_statistics,
)


def fixture():
    rng = np.random.default_rng(721)
    x = rng.normal(size=(40, 5))
    y = rng.normal(size=40)
    w = rng.uniform(0.1, 1, size=(3, 40))
    phi = rng.normal(size=(3, 3, 5))
    stats = calculate_statistics(x, y, w, phi, normalized=True)
    return x, y, w, phi, stats


def test_covariance_includes_weighted_response_centering():
    x, _y, w, phi, _ = fixture()
    cov = moment_covariances(x, [(0, w)], phi)
    # The exact noise operator is obtained by applying the real moment routine
    # to each coordinate of the response; this also catches lost centering.
    operators = np.stack(
        [calculate_statistics(x, e, w, phi, normalized=True).I for e in np.eye(40)],
        axis=-1,
    )
    np.testing.assert_allclose(cov, operators @ operators.swapaxes(1, 2), atol=1e-13)
    shifted = moment_covariances(x + 17, [(0, 7 * w)], phi)
    np.testing.assert_allclose(cov, shifted, atol=1e-13)


@pytest.mark.parametrize("kind", ["gls", "row", "neff"])
def test_metric_preserves_the_stated_profile_and_original_mass(kind):
    x, _, w, phi, stats = fixture()
    transformed, info = reweight_statistics(x, [(0, w)], phi, stats, kind)
    basis = np.eye(5)[:, :2]
    actual = expected = 0.0
    for j in range(3):
        design = transformed.U[j] @ basis
        g = np.linalg.lstsq(design, transformed.I[j], rcond=None)[0]
        actual += stats.mass[j] * np.linalg.norm(transformed.I[j] - design @ g) ** 2
        a = w[j] / w[j].sum()
        dx = x - a @ x
        if kind == "gls":
            h = a[:, None] * (dx @ phi[j].T)
            precision = np.linalg.inv(h.T @ h)
        elif kind == "row":
            precision = stats.mass[j] * np.linalg.inv(stats.U[j] @ stats.U[j].T)
        else:
            precision = np.eye(3) / (a @ a)
        design = stats.U[j] @ basis
        g = np.linalg.solve(
            design.T @ precision @ design, design.T @ precision @ stats.I[j]
        )
        residual = stats.I[j] - design @ g
        expected += residual @ precision @ residual
    assert actual == pytest.approx(expected, rel=1e-11)
    np.testing.assert_array_equal(transformed.mass, stats.mass)
    np.testing.assert_array_equal(transformed.S, stats.S)
    assert info["clipped_eigenvalues"] == 0


def test_singular_covariance_is_regularized_and_reported():
    x, y, w, phi, _ = fixture()
    phi[:, 2] = phi[:, 1]
    stats = calculate_statistics(x, y, w, phi, normalized=True)
    transformed, info = reweight_statistics(x, [(0, w)], phi, stats, "gls")
    assert info["clipped_eigenvalues"] == 3
    assert np.isfinite(transformed.U).all()


@pytest.mark.parametrize("kind", ["gls", "row"])
def test_single_observation_centers_contribute_zero_information(kind):
    x, y, w, phi, _ = fixture()
    w[0] = 0
    w[0, 0] = 1
    stats = calculate_statistics(x, y, w, phi, normalized=True)
    transformed, info = reweight_statistics(x, [(0, w)], phi, stats, kind)
    np.testing.assert_array_equal(transformed.I[0], np.zeros(3))
    np.testing.assert_array_equal(transformed.U[0], np.zeros((3, 5)))
    assert info["zero_information_centers"] == 1


def test_certificate_bounds_true_subspace_and_uniform_surrogate_error():
    rng = np.random.default_rng(91)
    truth = np.eye(6)[:, :2]
    v = truth @ rng.normal(size=(2, 15))
    estimated = v + 0.005 * rng.normal(size=v.shape)
    t = np.linalg.norm(estimated - v, 2)
    basis, info = projector_certificate(estimated, t, 2)
    true_edr = np.linalg.norm(basis - truth @ (truth.T @ basis)) / np.sqrt(2)
    assert true_edr <= info["edr_bound"]
    for _ in range(10):
        candidate = np.linalg.qr(rng.normal(size=(6, 2)))[0]
        surrogate = 2 - np.linalg.norm(basis.T @ candidate) ** 2
        exact = np.linalg.norm(truth - candidate @ (candidate.T @ truth)) ** 2
        assert abs(surrogate - exact) <= 2 * info["edr_bound"]


def test_exact_balance_reproduces_the_counterexample():
    v = np.diag([10.0, 1.0, 0.0, 0.0])[:, :2]
    basis, info = projector_certificate(v, 0, 2)
    b1 = np.array([[np.sqrt(0.96), 0], [0, 1], [0.2, 0], [0, 0]])
    b2 = np.array([[1, 0], [0, np.sqrt(0.2)], [0, 0], [0, np.sqrt(0.8)]])
    assert 2 - np.linalg.norm(basis.T @ b1) ** 2 == pytest.approx(0.04)
    assert 2 - np.linalg.norm(basis.T @ b2) ** 2 == pytest.approx(0.8)
    assert info["edr_bound"] < 1e-12
    with pytest.raises(ValueError):
        projector_certificate(np.zeros((4, 2)), 0, 2)
