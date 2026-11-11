"""Independent algebra/finite-precision references for weak moments."""

import itertools

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from experiments.weak_moments import fit, moments, variance_constants


def test_exact_gaussian_variances():
    # Degree <= 8: five-point 1D quadrature integrates all terms exactly.
    z, w = hermegauss(5)
    w /= np.sqrt(2 * np.pi)
    for d in (3, 4):
        idx = np.array(list(itertools.product(range(5), repeat=d)))
        x, weights = z[idx], w[idx].prod(axis=1)
        beta = np.zeros(d)
        beta[1] = 1.7
        delta, sigma = -0.3, 0.4
        g = x @ beta + delta * (x[:, 0] ** 2 - 1)
        h = x[:, :, None] * x[:, None, :] - np.eye(d)
        t1, t2 = (weights * g) @ x, np.einsum("n,nij->ij", weights * g, h)
        expected = np.zeros((d, d))
        expected[0, 0] = 2 * delta
        np.testing.assert_allclose(t1, beta, atol=1e-13)
        np.testing.assert_allclose(t2, expected, atol=1e-13)
        v1, v2 = variance_constants(d, beta @ beta, delta, sigma)
        e1 = np.sum(weights * (g**2 + sigma**2) * (x * x).sum(axis=1))
        e2 = np.sum(weights * (g**2 + sigma**2) * (h * h).sum(axis=(1, 2)))
        np.testing.assert_allclose(e1 - beta @ beta, v1, rtol=1e-13)
        np.testing.assert_allclose(e2 - (t2 * t2).sum(), v2, rtol=1e-13)
        # c can have a nonzero component along the quadratic direction.
        c = np.arange(1, d + 1) / 3
        residual = x @ c + delta * (x[:, 0] ** 2 - 1)
        e = np.sum(weights * (residual**2 + sigma**2) * (h * h).sum((1, 2)))
        _, vc = variance_constants(d, c @ c, delta, sigma)
        np.testing.assert_allclose(e - 4 * delta**2, vc, rtol=1e-13)


def test_streaming_and_rectangular_reference():
    rng = np.random.default_rng(713)
    x, y = rng.normal(size=(123, 7)), rng.normal(size=123)
    t1, t2 = moments(x, y, block_size=11)
    h = x[:, :, None] * x[:, None, :] - np.eye(7)
    np.testing.assert_allclose(t1, y @ x / len(x), atol=1e-15)
    np.testing.assert_allclose(t2, np.einsum("n,nij->ij", y, h) / len(x), atol=1e-15)
    for arm in ("raw_full", "raw_split", "residual_split"):
        result = fit(x, y, arm=arm, block_size=11)
        m = result["matrix"]
        _, vectors = np.linalg.eigh(m @ m.T)
        reference = vectors[:, -2:] @ vectors[:, -2:].T
        basis = result["basis"]
        np.testing.assert_allclose(basis @ basis.T, reference, atol=2e-14)
        other = fit(x, y, arm=arm, block_size=37)["basis"]
        np.testing.assert_allclose(basis @ basis.T, other @ other.T, atol=2e-14)


def test_wls_identity_singular_and_unequal_weights():
    rng = np.random.default_rng(714)
    x = rng.normal(size=(7, 12)) + 1e5
    y, weights = rng.normal(size=7), np.geomspace(1e-4, 1, 7)
    weights /= weights.sum()
    r, yc = x - weights @ x, y - weights @ y
    a, t = np.sqrt(weights)[:, None] * r, np.sqrt(weights) * yc
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > 1e-9 * s[0]
    c, d = a.T @ a, a.T @ t
    cinv = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    constant = t @ t - d @ cinv @ d
    for _ in range(5):
        h = rng.normal(size=12)
        np.testing.assert_allclose(
            (t - a @ h) @ (t - a @ h),
            (d - c @ h) @ cinv @ (d - c @ h) + constant,
            atol=2e-8,
        )


def test_wls_conditioning_and_rectangular_subspace_bound():
    c = np.diag([1.0, 1e-3])
    assert np.linalg.cond(c) == pytest.approx(1e3)
    assert np.linalg.cond(c @ c) == pytest.approx(1e6)
    rng = np.random.default_rng(715)
    basis = np.linalg.qr(rng.normal(size=(9, 2)))[0]
    right = np.linalg.qr(rng.normal(size=(10, 2)))[0]
    matrix = (basis * [1, 0.2]) @ right.T
    error = rng.normal(size=matrix.shape)
    error *= 0.03 / np.linalg.norm(error, 2)
    vectors = np.linalg.svd(matrix + error, full_matrices=False)[0][:, :2]
    observed = np.linalg.norm(vectors @ vectors.T - basis @ basis.T) ** 2
    assert observed <= 4 * (0.03 / (0.2 - 0.03)) ** 2


@pytest.mark.parametrize("arm", ["raw_full", "raw_split", "residual_split"])
def test_rank_and_input_guards(arm):
    with pytest.raises(ValueError):
        fit(np.ones((10, 4)), np.zeros(10), arm=arm)
    with pytest.raises(ValueError):
        fit(np.ones((10, 4)), np.full(10, np.nan), arm=arm)
    with pytest.raises(ValueError):
        fit(np.ones((1, 4)), np.ones(1), arm=arm)
    with pytest.raises(ValueError):
        fit(np.ones((10, 4)), np.ones(10), arm=arm, block_size=0)
