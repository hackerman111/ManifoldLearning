import numpy as np

from ADP.engine.calculus import (
    calculate_multi_weight,
    initialize_basis_local,
    initialize_basis_random,
)


def test_multi_initializers_are_orthonormal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    true_basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    Y = np.sin(X @ true_basis[:, 0]) + np.square(X @ true_basis[:, 1])
    centers = X[:16]
    distance2 = np.maximum(
        np.square(centers).sum(1)[:, None]
        + np.square(X).sum(1)[None, :]
        - 2 * centers @ X.T,
        0,
    )

    local = initialize_basis_local(
        X,
        Y,
        centers,
        distance2,
        20,
        lambda q: np.maximum(1 - q, 0),
        1e-8,
        2,
    )
    random = initialize_basis_random(rng, 5, 2)

    assert local.shape == random.shape == (5, 2)
    np.testing.assert_allclose(local.T @ local, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(random.T @ random, np.eye(2), atol=1e-12)


def test_multi_weight_matches_explicit_low_rank_tensor():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 5))
    centers = rng.normal(size=(4, 5))
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    eigenvalues = np.array([1.7, 0.4])
    alpha, h = 0.35, 1.2
    kernel = lambda q: np.exp(-q)

    actual = np.vstack(
        [
            block
            for _, block in calculate_multi_weight(
                X,
                centers,
                basis,
                eigenvalues,
                h,
                alpha,
                kernel,
                block_size=2,
            )
        ]
    )
    projector = basis @ basis.T
    tensor2 = (
        alpha**2 * (np.eye(5) - projector)
        + basis @ np.diag(eigenvalues) @ basis.T
    ) / h**2
    differences = centers[:, None, :] - X[None, :, :]
    argument = np.einsum("jnd,de,jne->jn", differences, tensor2, differences)
    np.testing.assert_allclose(actual, kernel(argument), rtol=1e-12, atol=1e-12)
