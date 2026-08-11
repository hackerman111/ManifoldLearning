from functools import partial

import numpy as np
import pytest

from ADP.engine.box_kernel import (
    NeighborhoodEngine,
    box_kernel,
    make_plateau_kernel,
    plateau_kernel,
    sparse_kernel_parameters,
)


def test_box_and_plateau_kernel_boundaries():
    q = np.array([0.0, 0.5, 0.75, 1.0, 2.0])
    np.testing.assert_array_equal(
        box_kernel(q),
        [1.0, 1.0, 1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        plateau_kernel(q, tau=0.5),
        [1.0, 1.0, 0.5, 0.0, 0.0],
        atol=1e-15,
    )


def test_plateau_tau_and_kernel_identity_are_validated():
    kernel = make_plateau_kernel(0.4)
    assert isinstance(kernel, partial)
    assert sparse_kernel_parameters(box_kernel) == ("box", None)
    assert sparse_kernel_parameters(kernel) == ("plateau", 0.4)
    with pytest.raises(ValueError, match="tau"):
        make_plateau_kernel(1.0)
    with pytest.raises(ValueError, match="finite"):
        plateau_kernel(np.array([np.nan]))


def _dense_blocks(blocks, rows, columns):
    result = np.zeros((rows, columns))
    for block in blocks:
        for local_row in range(block.rows):
            indices, weights = block.row(local_row)
            result[block.start + local_row, indices] = weights
    return result


def test_single_neighborhood_matches_dense_box_reference():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(11, 4))
    centers = X[[0, 4, 8]]
    beta = rng.normal(size=4)
    beta /= np.linalg.norm(beta)
    h, rho = 1.6, 0.45

    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(engine.single_blocks(beta, h, rho, box_kernel))
    actual = _dense_blocks(blocks, len(centers), len(X))

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projection2 = np.square(differences @ beta)
    expected = box_kernel((rho**2 * distance2 + projection2) / h**2)

    np.testing.assert_array_equal(actual, expected)
    assert sum(block.edge_count for block in blocks) == np.count_nonzero(expected)
    assert all(block.boundary_count == 0 for block in blocks)


def test_multi_neighborhood_matches_dense_plateau_reference():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(13, 5))
    centers = X[[1, 6, 10]]
    basis, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    eigenvalues = np.array([1.0, 0.35])
    h, alpha, tau = 1.8, 0.4, 0.5
    kernel = make_plateau_kernel(tau)

    engine = NeighborhoodEngine(X, centers, block_size=2)
    blocks = list(
        engine.multi_blocks(basis, eigenvalues, h, alpha, kernel)
    )
    actual = _dense_blocks(blocks, len(centers), len(X))

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projected = differences @ basis
    projected2 = np.einsum("jnm,jnm->jn", projected, projected)
    principal2 = np.einsum(
        "jnm,m,jnm->jn",
        projected,
        eigenvalues,
        projected,
    )
    q = (alpha**2 * np.maximum(distance2 - projected2, 0.0) + principal2) / h**2
    expected = plateau_kernel(q, tau=tau)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert sum(block.edge_count for block in blocks) == np.count_nonzero(q < 1.0)
    assert sum(block.boundary_count for block in blocks) == np.count_nonzero(
        (q > tau) & (q < 1.0)
    )
