from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ADP.engine.common.statistic import calculate_statistics
from ADP.solver.statistic_operator import (
    DenseOperator,
    FactorizedOperator,
    factorized_statistics,
)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_factorized_statistics_matches_dense_reference(normalized, streamed):
    rng = np.random.default_rng(6107)
    X = 1e8 + rng.normal(size=(31, 6))
    y = rng.normal(size=31)
    directions = rng.normal(size=(5, 4, 6))
    weights = rng.random((5, 31))
    weights[weights < 0.78] = 0.0
    weights *= np.geomspace(0.03, 17.0, 5)[:, None]
    source = iter((j, weights[j : j + 1]) for j in range(5)) if streamed else weights
    op, I, mass = factorized_statistics(
        X, y, source, directions, batch_size=2, normalized=normalized
    )
    dense = calculate_statistics(X, y, weights, directions, normalized=normalized)
    np.testing.assert_allclose(I, dense.I, rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(mass, dense.mass, rtol=0, atol=0)
    np.testing.assert_allclose(
        op.project(np.eye(X.shape[1])), dense.U, rtol=3e-11, atol=2e-12
    )
    assert op.shape == dense.U.shape
    assert op.nbytes >= op.C.data.nbytes + op.C.indices.nbytes + op.C.indptr.nbytes
    assert op.nbytes < dense.U.nbytes + 5 * 31 * 6 * 8


def test_dense_and_factorized_operator_adjoint_and_composition():
    rng = np.random.default_rng(6108)
    U = rng.normal(size=(7, 3, 8))
    dense = DenseOperator(U)
    C = csr_matrix(rng.normal(size=(21, 11)))
    X = rng.normal(size=(11, 8))
    factored = FactorizedOperator(C, X, n_centers=7, n_directions=3)
    V = rng.normal(size=(8, 2))
    P = rng.normal(size=(7, 3, 2))
    for op in (dense, factored):
        np.testing.assert_allclose(
            np.vdot(op.project(V), P), np.vdot(V, op.adjoint(P)), rtol=2e-13, atol=2e-13
        )
    np.testing.assert_allclose(factored.project(V), (C @ (X @ V)).reshape(7, 3, 2))
    np.testing.assert_allclose(
        factored.adjoint(P), X.T @ (C.T @ P.reshape(21, 2)), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DenseOperator(np.ones((2, 3))),
        lambda: DenseOperator(np.ones((2, 3, 4))).project(np.ones((3, 2))),
        lambda: FactorizedOperator(
            csr_matrix((6, 5)), np.ones((4, 3)), n_centers=2, n_directions=3
        ),
        lambda: FactorizedOperator(
            csr_matrix((6, 5)), np.ones((5, 3)), n_centers=2, n_directions=3
        ).project(np.ones((4, 1))),
    ],
)
def test_operator_rejects_invalid_shapes(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_factorized_builder_rejects_invalid_inputs():
    X = np.ones((8, 3))
    y = np.ones(8)
    directions = np.ones((2, 4, 3))
    weights = np.ones((2, 8))
    cases = [
        (X[:, :2], y, weights, directions),
        (X, y[:-1], weights, directions),
        (X, y, np.zeros_like(weights), directions),
        (X, y, weights, directions[:, :, :2]),
    ]
    for args in cases:
        with pytest.raises((ValueError, TypeError)):
            factorized_statistics(*args)


def test_factorized_operator_rejects_nonfinite_coefficients_and_bad_stream_order():
    bad_C = csr_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        FactorizedOperator(bad_C, np.ones((2, 3)), n_centers=1, n_directions=2)
    rng = np.random.default_rng(6109)
    X = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    weights = rng.random((3, 12))
    directions = rng.normal(size=(3, 2, 3))
    out_of_order = iter(((1, weights[1:2]), (0, weights[:1]), (2, weights[2:])))
    with pytest.raises(ValueError):
        factorized_statistics(X, y, out_of_order, directions)
    incomplete = iter(((0, weights[:1]),))
    with pytest.raises(ValueError):
        factorized_statistics(X, y, incomplete, directions)
