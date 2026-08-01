import numpy as np

from ADP_statistic import calculate_statistics


def test_matrix_statistics_match_direct_sums_and_are_shift_invariant():
    rng = np.random.default_rng(42)
    n, d, J, P = 11, 4, 5, 3
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=n)
    weights = rng.uniform(0.1, 1.0, size=(J, n))
    directions = rng.normal(size=(J, P, d))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    result = calculate_statistics(X, Y, weights, directions, batch_size=2)

    mass = weights.sum(axis=1)
    mean = weights @ X / mass[:, None]
    centered = X[None, :, :] - mean[:, None, :]
    projections = np.einsum("jpd,jnd->jpn", directions, centered)
    weighted = weights[:, None, :] * projections

    np.testing.assert_allclose(result["I"], weighted @ Y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["U"], weighted @ centered, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result["mass"], mass)
    np.testing.assert_allclose(result["mean"], mean)
    normalized = weights / mass[:, None]
    np.testing.assert_allclose(result["n_eff"], 1 / np.square(normalized).sum(1))
    assert np.all(np.isfinite(result["eta"]))
    assert np.all((0 <= result["eta"]) & (result["eta"] < 1e-12))

    shifted = calculate_statistics(
        X + 1e6, Y - 1e6, weights, directions, batch_size=3
    )
    np.testing.assert_allclose(shifted["I"], result["I"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(shifted["U"], result["U"], rtol=1e-9, atol=1e-9)
