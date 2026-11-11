import numpy as np

from ADP import _weight_blocks_single, calculate_statistics


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


def test_single_index_weights_are_consumed_by_center_blocks():
    rng = np.random.default_rng(7)
    n, d, J, P = 13, 3, 5, 2
    X = rng.normal(size=(n, d))
    Y = rng.normal(size=n)
    centers = rng.normal(size=(J, d))
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    directions = rng.normal(size=(J, P, d))
    h, rho = 1.7, 0.4

    D2 = (
        np.square(centers).sum(axis=1)[:, None]
        + np.square(X).sum(axis=1)[None, :]
        - 2 * centers @ X.T
    )
    np.maximum(D2, 0.0, out=D2)
    P2 = np.square((centers @ beta)[:, None] - (X @ beta)[None, :])
    kernel = lambda q: np.exp(-q)
    weights = kernel((rho**2 * D2 + P2) / h**2)
    expected = calculate_statistics(X, Y, weights, directions, batch_size=2)

    block_shapes = []

    def recording_kernel(q):
        block_shapes.append(q.shape)
        return kernel(q)

    blocks = _weight_blocks_single(
        X, centers, beta, h, rho, recording_kernel, block_size=2
    )
    actual = calculate_statistics(X, Y, blocks, directions, batch_size=2)

    assert block_shapes == [(2, n), (2, n), (1, n)]
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)
