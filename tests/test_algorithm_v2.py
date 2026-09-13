from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ADP.cli.main import _solver_index
from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Data import ADP_Data
from ADP.core.ADP_Solver import ADP_Solver
from ADP.engine.calculus import calculate_rho_k, generate_isotropic_proj
from ADP.engine.initialize import _principal_gradient_basis
from ADP.engine.statistic import calculate_statistics
from ADP.engine.weights import calculate_multi_weight, calculate_weight
from ADP.solver.LSMR import HPAOResult


def _identity(value: np.ndarray) -> np.ndarray:
    return value


def _box(value: np.ndarray) -> np.ndarray:
    return (value <= 1.0).astype(float)


def test_single_weight_matches_new_tensor_and_keeps_legacy_variant() -> None:
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    centers = np.array([[0.0, 0.0]])
    beta = np.array([1.0, 0.0])
    distance2 = np.array([[0.0, 5.0]])

    isotropic = next(
        calculate_weight(
            X,
            centers,
            beta,
            1.0,
            1.0,
            _identity,
            distance2=distance2,
            estimator="new",
        )
    )[1]
    index_only = next(
        calculate_weight(
            X,
            centers,
            beta,
            1.0,
            0.0,
            _identity,
            distance2=distance2,
            estimator="new",
        )
    )[1]
    legacy = next(
        calculate_weight(
            X,
            centers,
            beta,
            1.0,
            1.0,
            _identity,
            distance2=distance2,
            estimator="legacy",
        )
    )[1]

    np.testing.assert_allclose(isotropic, [[0.0, 5.0]])
    np.testing.assert_allclose(index_only, [[0.0, 1.0]])
    np.testing.assert_allclose(legacy, [[0.0, 6.0]])


def test_rho_search_uses_the_same_new_single_tensor() -> None:
    X = np.array([[0.0, 0.0], [0.8, 0.0], [0.0, 2.0]])
    centers = np.array([[0.0, 0.0]])
    beta = np.array([1.0, 0.0])

    rho = calculate_rho_k(
        X,
        centers,
        beta,
        1.0,
        2,
        _box,
        estimator="new",
    )

    assert rho is not None
    np.testing.assert_allclose(rho, 1.0, rtol=0, atol=2e-8)
    legacy = calculate_rho_k(
        X,
        centers,
        beta,
        1.0,
        2,
        _box,
        estimator="legacy",
    )
    assert legacy is not None and legacy < 0.8


def test_both_multi_tensor_variants_have_dense_reference_values() -> None:
    X = np.array([[1.0, 2.0]])
    centers = np.array([[0.0, 0.0]])
    basis = np.array([[1.0], [0.0]])
    eigenvalues = np.ones(1)
    distance2 = np.array([[5.0]])

    orthogonal = next(
        calculate_multi_weight(
            X,
            centers,
            basis,
            eigenvalues,
            1.0,
            0.5,
            _identity,
            distance2=distance2,
            tensor="orthogonal",
        )
    )[1]
    full = next(
        calculate_multi_weight(
            X,
            centers,
            basis,
            eigenvalues,
            1.0,
            0.5,
            _identity,
            distance2=distance2,
            tensor="full",
        )
    )[1]

    np.testing.assert_allclose(orthogonal, [[2.0]])
    np.testing.assert_allclose(full, [[2.25]])


def test_normalized_statistics_match_dense_definition() -> None:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]
    )
    Y = np.array([1.0, -1.0, 2.0, 5.0])
    weights = np.array([[1.0, 2.0, 1.0, 0.5], [0.5, 1.0, 3.0, 2.0]])
    directions = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])

    normalized = calculate_statistics(
        X,
        Y,
        weights,
        directions,
        normalized=True,
    )
    legacy = calculate_statistics(X, Y, weights, directions, normalized=False)

    mass = weights.sum(axis=1)
    A = weights / mass[:, None]
    mean = A @ X
    centered = X[None, :, :] - mean[:, None, :]
    projected = np.einsum("jnd,jpd->jpn", centered, directions, optimize=True)
    expected_I = np.einsum("jn,n,jpn->jp", A, Y, projected, optimize=True)
    expected_U = np.einsum("jn,jnd,jpn->jpd", A, centered, projected, optimize=True)

    np.testing.assert_allclose(normalized.I, expected_I, atol=1e-14)
    np.testing.assert_allclose(normalized.U, expected_U, atol=1e-14)
    np.testing.assert_allclose(normalized.S, A @ Y, atol=1e-14)
    np.testing.assert_allclose(normalized.mass, mass)
    np.testing.assert_allclose(legacy.I, mass[:, None] * normalized.I)
    np.testing.assert_allclose(legacy.U, mass[:, None, None] * normalized.U)

    shifted = calculate_statistics(
        X + 1e12,
        Y,
        weights,
        directions,
        normalized=True,
    )
    np.testing.assert_allclose(shifted.I, normalized.I, atol=1e-12)
    np.testing.assert_allclose(shifted.U, normalized.U, atol=1e-12)


def test_sparse_statistics_match_direct_local_sums() -> None:
    X = np.arange(16.0).reshape(8, 2)
    Y = np.linspace(-1.0, 2.0, 8)
    weights = np.array(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0],
        ]
    )
    directions = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])

    result = calculate_statistics(
        X,
        Y,
        weights,
        directions,
        normalized=True,
    )
    mass = weights.sum(axis=1)
    A = weights / mass[:, None]
    mean = A @ X
    centered = X[None, :, :] - mean[:, None, :]
    projected = np.einsum("jnd,jpd->jpn", centered, directions, optimize=True)

    np.testing.assert_allclose(
        result.I,
        np.einsum("jn,n,jpn->jp", A, Y, projected, optimize=True),
    )
    np.testing.assert_allclose(
        result.U,
        np.einsum("jn,jnd,jpn->jpd", A, centered, projected, optimize=True),
    )


def test_initialization_pca_and_multi_svd_use_local_mass() -> None:
    gradients = np.array([[10.0, 0.0], [0.0, 2.0]])
    mass = np.array([0.01, 100.0])

    basis = _principal_gradient_basis(gradients, 1, mass=mass)

    np.testing.assert_allclose(np.abs(basis[:, 0]), [0.0, 1.0])

    result = HPAOResult(
        index=np.eye(2),
        coefficients=gradients,
        diagnostics={},
    )
    index, eigenvalues = _solver_index("multi", result, np.eye(2), mass=mass)

    np.testing.assert_allclose(np.abs(index[0]), [0.0, 1.0])
    np.testing.assert_allclose(eigenvalues, [1.0, 0.0025])

    rank_deficient = HPAOResult(
        index=np.eye(2),
        coefficients=np.array([[1.0, 0.0], [2.0, 0.0]]),
        diagnostics={},
    )
    with pytest.raises(RuntimeError, match="do not identify"):
        _solver_index("multi", rank_deficient, np.eye(2), mass=np.ones(2))


def test_isotropic_directions_are_reproducible_unit_vectors() -> None:
    first = generate_isotropic_proj(np.random.default_rng(4), 3, 5, 7)
    second = generate_isotropic_proj(np.random.default_rng(4), 3, 5, 7)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=2), 1.0, atol=1e-15)


def test_adp_data_multi_metric_matches_multiindex_tex() -> None:
    true_basis = np.eye(4)[:, :2]
    angle = np.deg2rad(60.0)
    estimate = np.column_stack(
        (
            np.eye(4)[:, 0],
            np.cos(angle) * np.eye(4)[:, 1] + np.sin(angle) * np.eye(4)[:, 2],
        )
    )
    data = ADP_Data(
        X=np.zeros((5, 4)),
        Y=np.zeros(5),
        beta_true=true_basis,
        beta_k=np.empty(0),
        beta_init=true_basis,
        a_k=np.empty(0),
        rho_k=np.empty(0),
        x_j=np.zeros((1, 4)),
        n=5,
        d=4,
    )

    assert data.Calculate_metric(estimate) == pytest.approx(0.75, abs=1e-14)


def test_public_single_solver_runs_the_new_mass_weighted_path() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    beta = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    data = SimpleNamespace(
        X=X,
        Y=np.sin(X @ beta),
        beta_init=np.array([1.0, 0.0, 0.0]),
        x_j=X[:8],
    )
    config = ADP_Config(
        N_loc=6,
        N_phi=3,
        outer_steps=1,
        h_min=1e6,
        batch_size=4,
    )

    estimate = ADP_Solver(config, data).fit()

    assert estimate.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(estimate), 1.0, atol=1e-12)
