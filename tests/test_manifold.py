from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ADP import ADP_Manifold, ADP_manifold


def _projectors(rng: np.random.Generator, count: int, m: int, d: int) -> np.ndarray:
    result = np.empty((count, m, d))
    for j in range(count):
        basis, _ = np.linalg.qr(rng.normal(size=(d, m)), mode="reduced")
        result[j] = basis.T
    return result


def test_chunked_weights_and_sparse_graph_match_dense_reference() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 4))
    X -= X.mean(axis=0)
    centers = X[:5]
    projectors = _projectors(rng, len(centers), 2, X.shape[1])
    eigenvalues = rng.uniform(0.2, 1.0, size=(len(centers), 2))
    eigenvalues[:, 0] = 1.0
    h = 1.7
    alpha = 0.45
    model = ADP_Manifold(2, batch_size=2)

    distance2 = model._pairwise_distance2(X, centers)
    explicit_difference = X[None, :, :] - centers[:, None, :]
    np.testing.assert_allclose(
        distance2,
        np.einsum(
            "jnd,jnd->jn", explicit_difference, explicit_difference, optimize=True
        ),
        rtol=1e-14,
        atol=1e-14,
    )

    coordinates = np.einsum(
        "jmd,jnd->jmn", projectors, explicit_difference, optimize=True
    )
    projected2 = np.square(coordinates).sum(axis=1)
    principal2 = np.einsum(
        "jm,jmn->jn", eigenvalues, np.square(coordinates), optimize=True
    )
    argument = (alpha**2 * (distance2 - projected2) + principal2) / h**2
    expected = np.maximum(1.0 - np.square(argument), 0.0)
    actual = np.vstack(
        [
            model._weight_block(
                X,
                centers,
                projectors,
                eigenvalues,
                h,
                alpha,
                start,
            )
            for start in range(0, len(centers), model.batch_size)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)

    graph = model._build_manifold_graph(centers, projectors, eigenvalues, h, alpha)
    np.testing.assert_allclose(graph.toarray(), actual[:, : len(centers)])
    assert graph.nnz == np.count_nonzero(actual[:, : len(centers)])

    Y = rng.normal(size=len(X))
    directions = rng.normal(size=(len(centers), 3, X.shape[1]))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    I, U, mass, _, edges = model._calculate_statistics(
        X,
        Y,
        centers,
        directions,
        projectors,
        eigenvalues,
        h,
        alpha,
    )
    expected_I = np.empty_like(I)
    expected_U = np.empty_like(U)
    for j, weights_j in enumerate(expected):
        normalized = weights_j / weights_j.sum()
        mean = normalized @ X
        projected = (X - mean) @ directions[j].T
        projected -= normalized @ projected
        moments = normalized[:, None] * projected
        expected_I[j] = moments.T @ Y
        expected_U[j] = moments.T @ X
    np.testing.assert_allclose(mass, expected.sum(axis=1), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(I, expected_I, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(U, expected_U, rtol=2e-14, atol=2e-14)
    assert edges == np.count_nonzero(expected)


def test_penalty_operator_and_cg_match_dense_reference() -> None:
    rng = np.random.default_rng(5)
    K, P, m, d = 7, 5, 2, 3
    U = rng.normal(size=(K, P, d))
    I = rng.normal(size=(K, P))
    mass = rng.uniform(0.7, 2.0, size=K)
    weights = rng.uniform(0.2, 1.0, size=K)
    source_projectors = _projectors(rng, K, m, d)
    slopes = rng.normal(size=(K, m))
    B = rng.normal(size=(m, d))
    model = ADP_Manifold(2, lambda_manifold=0.6, cg_tol=1e-11)

    normalized = weights / weights.sum()
    dense_average = np.einsum(
        "j,jrd,jre->de",
        normalized,
        source_projectors,
        source_projectors,
        optimize=True,
    )
    np.testing.assert_allclose(
        model._penalty_action(B, source_projectors, normalized),
        B @ (np.eye(d) - dense_average),
        rtol=2e-14,
        atol=2e-14,
    )

    rotations = _projectors(rng, K, m, m)
    rotated = np.einsum("jab,jbd->jad", rotations, source_projectors, optimize=True)
    np.testing.assert_allclose(
        model._penalty_action(B, rotated, normalized),
        model._penalty_action(B, source_projectors, normalized),
        rtol=2e-14,
        atol=2e-14,
    )

    operator, preconditioner, rhs = model._build_B_system(
        U, I, mass, weights, source_projectors, slopes
    )
    left = rng.normal(size=m * d)
    right = rng.normal(size=m * d)
    np.testing.assert_allclose(
        np.dot(operator @ left, right),
        np.dot(left, operator.rmatvec(right)),
        rtol=2e-14,
        atol=2e-14,
    )

    identity = np.eye(m * d)
    dense_operator = np.column_stack([operator @ column for column in identity])
    expected = np.linalg.solve(dense_operator, rhs).reshape(m, d)
    initial = rng.normal(size=(m, d))
    actual, _, residual = model._solve_B(operator, preconditioner, rhs, initial)
    unpreconditioned, _, _ = model._solve_B(operator, None, rhs, np.zeros_like(initial))
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(unpreconditioned, expected, rtol=2e-10, atol=2e-11)
    assert residual <= 1e-10

    failing = ADP_Manifold(2, lambda_manifold=0.6, cg_tol=1e-14, cg_maxiter=1)
    with pytest.raises(RuntimeError, match="CG did not converge"):
        failing._solve_B(operator, preconditioner, rhs, initial)


def test_low_rank_recovery_matches_dense_edr_matrix() -> None:
    rng = np.random.default_rng(8)
    K, m, d = 9, 2, 6
    B = rng.normal(size=(m, d))
    slopes = rng.normal(size=(K, m))
    gamma = rng.uniform(0.5, 2.0, size=K)

    projector, spectrum = ADP_Manifold._recover_projector(B, slopes, gamma, 0)
    M = np.einsum("j,ja,jb->ab", gamma, slopes, slopes, optimize=True)
    dense_edr = B.T @ M @ B
    values, vectors = np.linalg.eigh(dense_edr)
    order = np.argsort(values)[::-1][:m]
    expected_basis = vectors[:, order].T
    expected_spectrum = values[order] / values[order[0]]

    np.testing.assert_allclose(
        projector.T @ projector,
        expected_basis.T @ expected_basis,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(spectrum, expected_spectrum, rtol=1e-12, atol=1e-12)


def test_fit_is_reproducible_chunk_invariant_and_recovers_subspace() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(120, 4))
    Y = np.sin(2.0 * X[:, 0]) + np.square(X[:, 1])
    Y += 0.02 * rng.normal(size=len(X))
    settings: dict[str, Any] = {
        "N_loc": 28,
        "N_lin": 50,
        "N_J": 14,
        "N_phi": 12,
        "N_manifold": 8,
        "sync_steps": 1,
        "a": 2.0,
        "h_min": 0.5,
        "seed": 11,
    }

    first = ADP_Manifold(2, batch_size=5, **settings).fit(X, Y)
    shifted = ADP_manifold(2, batch_size=3, **settings).fit(X + 1.0e6, Y + 1.0e5)

    assert first.projectors_.shape == (14, 2, 4)
    assert first.eigenvalues_.shape == (14, 2)
    assert first.gradients_.shape == (14, 4)
    assert first.n_scales_ == 1
    assert [entry["phase"] for entry in first.trace_] == ["sync", "scale"]
    assert first.trace_[-1]["stop_reason"] == "h_min"
    np.testing.assert_allclose(
        first.projectors_ @ np.swapaxes(first.projectors_, 1, 2),
        np.broadcast_to(np.eye(2), (14, 2, 2)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(first.eigenvalues_[:, 0], 1.0, atol=1e-14)
    np.testing.assert_allclose(first.projectors_, shifted.projectors_, atol=2e-9)
    np.testing.assert_allclose(first.eigenvalues_, shifted.eigenvalues_, atol=2e-9)
    np.testing.assert_allclose(first.centers_ + 1.0e6, shifted.centers_, atol=0.0)

    true_basis = np.eye(4)[:2]
    captured = np.square(first.projectors_ @ true_basis.T).sum(axis=(1, 2)) / 2.0
    assert float(np.median(captured)) > 0.98
    assert first.trace_[-1]["cg_relative_residual_max"] <= 1e-7


def test_invalid_and_degenerate_inputs_fail_explicitly() -> None:
    with pytest.raises(TypeError, match="index_dim"):
        ADP_Manifold(True)
    with pytest.raises(ValueError, match="N_loc must be smaller"):
        ADP_Manifold(1, N_loc=20, N_lin=5, N_J=6, N_manifold=2).fit(
            np.ones((20, 2)), np.ones(20)
        )

    rng = np.random.default_rng(13)
    coordinate = rng.normal(size=40)
    X = np.column_stack((coordinate, coordinate))
    with pytest.raises(RuntimeError, match="rank-deficient local-linear fit"):
        ADP_Manifold(
            1,
            N_loc=8,
            N_lin=15,
            N_J=8,
            N_phi=4,
            N_manifold=3,
            sync_steps=1,
            a=100.0,
            h_min=0.1,
        ).fit(X, coordinate)

    model = ADP_Manifold(1, batch_size=2)
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    projectors = np.broadcast_to(np.array([[[1.0, 0.0]]]), (3, 1, 2)).copy()
    eigenvalues = np.ones((3, 1))
    with pytest.raises(RuntimeError, match="infeasible even with alpha=0"):
        model._search_anisotropy(
            points,
            points,
            projectors,
            eigenvalues,
            0.1,
            2,
            kind="manifold",
        )
