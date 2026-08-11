from functools import partial

import numpy as np
import pytest

from ADP.engine.box_kernel import (
    NeighborhoodEngine,
    box_kernel,
    initialize_basis_local_sparse,
    make_plateau_kernel,
    plateau_kernel,
    search_sparse_bandwidth,
    search_sparse_scale,
    sparse_kernel_parameters,
)
from ADP.engine.calculus import initialize_basis_local, pairwise_distance2


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


def _mean_mass(blocks):
    masses = [block.mass for block in blocks]
    return float(np.concatenate(masses).mean())


def test_sparse_bandwidth_returns_smallest_feasible_box_radius():
    X = np.array([[0.0], [1.0], [3.0]])
    engine = NeighborhoodEngine(X, X[[0]], block_size=1)

    h = search_sparse_bandwidth(engine, 2.0, box_kernel, lower=0.1)

    assert _mean_mass(engine.isotropic_blocks(h, box_kernel)) >= 2.0
    assert _mean_mass(
        engine.isotropic_blocks(np.nextafter(h, 0.0), box_kernel)
    ) < 2.0


@pytest.mark.parametrize(
    "kernel",
    [box_kernel, make_plateau_kernel(0.5)],
    ids=["box", "plateau"],
)
def test_sparse_scale_matches_dense_bisection(kernel):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 3))
    centers = X[[0, 7, 15, 20]]
    beta = rng.normal(size=3)
    beta /= np.linalg.norm(beta)
    h = 1.5
    engine = NeighborhoodEngine(X, centers, block_size=2)

    differences = centers[:, None, :] - X[None, :, :]
    distance2 = np.einsum("jnd,jnd->jn", differences, differences)
    projection2 = np.square(differences @ beta)

    def dense_mass(scale):
        return float(
            kernel((scale**2 * distance2 + projection2) / h**2).sum(axis=1).mean()
        )

    target = (dense_mass(0.0) + dense_mass(1.0)) / 2.0
    assert dense_mass(0.0) > dense_mass(1.0)
    actual = search_sparse_scale(
        lambda scale: engine.single_blocks(
            beta,
            h,
            scale,
            kernel,
            record=False,
        ),
        target,
    )
    low, high = 0.0, 1.0
    while high - low > np.sqrt(np.finfo(float).eps):
        middle = (low + high) / 2.0
        if dense_mass(middle) >= target:
            low = middle
        else:
            high = middle

    np.testing.assert_allclose(actual, low, rtol=0, atol=1e-14)


def test_sparse_local_initialization_matches_dense_reference():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 3))
    true_beta = np.array([0.8, -0.5, 0.3])
    true_beta /= np.linalg.norm(true_beta)
    Y = X @ true_beta + 0.01 * rng.normal(size=len(X))
    centers = X[np.arange(0, len(X), 3)]
    ridge = 1e-6
    engine = NeighborhoodEngine(X, centers, block_size=4)

    actual = initialize_basis_local_sparse(
        X,
        Y,
        engine,
        6,
        box_kernel,
        ridge,
        1,
    )
    expected = initialize_basis_local(
        X,
        Y,
        centers,
        pairwise_distance2(X, centers),
        6,
        box_kernel,
        ridge,
        1,
    )

    np.testing.assert_allclose(actual.T @ actual, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        rtol=1e-9,
        atol=1e-10,
    )
