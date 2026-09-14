from __future__ import annotations

import numpy as np
import pytest

from ADP import ADP_Manifold
from ADP.solver import HYBRID
from ADP.solver.HYBRID import PenaltyRoot, RidgeWorkspace, solve, solve_manifold


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("ridge", [0.0, 1e-8, 1e2])
def test_ridge_matches_augmented_reference(dense: bool, ridge: float) -> None:
    rng = np.random.default_rng(931)
    U = rng.normal(size=(13, 6, 7))
    index = np.linalg.qr(rng.normal(size=(7, 2)))[0].T
    coefficients = rng.normal(size=(13, 2))
    I = rng.normal(size=(13, 6))
    mass = np.geomspace(1e-3, 1e3, 13)
    workspace = RidgeWorkspace(
        U, I, index, coefficients, mass, max_unknowns=100 if dense else 0
    )
    design = HYBRID.design_matrix(U, coefficients, np.sqrt(mass))
    augmented = np.vstack((design, np.sqrt(ridge) * np.eye(index.size)))
    rhs = np.concatenate((workspace.residual, np.zeros(index.size)))
    expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    result = workspace.correction(ridge, 1e-8, 2000)
    np.testing.assert_allclose(result[0], expected, rtol=1e-7, atol=1e-8)
    assert np.all(np.isfinite(result[0]))
    # Проверяем исходный градиент независимо от преобразованных координат.
    normal = augmented.T @ (augmented @ result[0] - rhs)
    assert np.linalg.norm(normal) / np.linalg.norm(augmented.T @ rhs) < 1e-8
    assert workspace.calls == 1


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_manifold_augmented_matches_dense_and_adjoint(
    dense: bool, fallback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(622)
    K, P, m, d = 9, 7, 2, 5
    U = rng.normal(size=(K, P, d))
    I = rng.normal(size=(K, P))
    slopes = rng.normal(size=(K, m))
    projectors = np.array(
        [np.linalg.qr(rng.normal(size=(d, m)))[0].T for _ in range(K)]
    )
    weights = rng.uniform(0.1, 1, K)
    mass = np.geomspace(0.01, 100, K)
    initial = projectors[0]
    ridge = 0.7
    C = np.eye(d) - np.einsum(
        "j,jad,jae->de", weights / weights.sum(), projectors, projectors
    )
    values, vectors = np.linalg.eigh(C)
    root = (vectors * np.sqrt(values)) @ vectors.T
    actual_root = PenaltyRoot(projectors, weights / weights.sum()).apply(np.eye(d))
    np.testing.assert_allclose(actual_root @ actual_root, C, atol=2e-14)
    data = HYBRID.design_matrix(U, slopes, np.sqrt(mass * weights))
    design = np.vstack((data, np.sqrt(ridge) * np.kron(np.eye(m), root)))
    rhs = np.concatenate(
        ((np.sqrt(mass * weights)[:, None] * I).ravel(), np.zeros(m * d))
    )
    expected = np.linalg.lstsq(design, rhs, rcond=None)[0]
    original = HYBRID.solve_augmented
    original_cg = HYBRID._cg_method

    def checked_cg(operator, target, **settings):
        x = rng.normal(size=m * d)
        np.testing.assert_allclose(
            operator @ x, design.T @ (design @ x), rtol=1e-12, atol=1e-10
        )
        assert np.vdot(x, settings["M"] @ x) > 0
        if fallback:
            return np.zeros_like(target), 1
        return original_cg(operator, target, **settings)

    monkeypatch.setattr(HYBRID, "_cg_method", checked_cg)

    def checked(operator, target, start, diagonal, **settings):
        x = rng.normal(size=m * d)
        y = rng.normal(size=design.shape[0])
        np.testing.assert_allclose(operator @ x, design @ x, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvec(y), design.T @ y, atol=1e-12)
        assert np.vdot(operator @ x, y) == pytest.approx(
            np.vdot(x, operator.rmatvec(y)), rel=1e-12
        )
        return original(operator, target, start, diagonal, **settings)

    monkeypatch.setattr(HYBRID, "solve_augmented", checked)
    result = solve_manifold(
        U,
        I,
        mass,
        weights,
        projectors,
        slopes,
        initial,
        ridge=ridge,
        tol=1e-9,
        maxiter=1000,
        dense_max_unknowns=100 if dense else 0,
    )
    np.testing.assert_allclose(result.solution, expected, rtol=1e-8, atol=1e-9)
    assert result.relative_residual < 1e-9
    expected_backend = (
        "dense-svd"
        if dense
        else ("block-pcg->scaled-lsmr" if fallback else "block-pcg")
    )
    assert result.backend == expected_backend


def test_hybrid_iteration_limit_and_invalid_inputs() -> None:
    rng = np.random.default_rng(751)
    U = rng.normal(size=(15, 7, 8))
    I = rng.normal(size=(15, 7))
    initial = np.linalg.qr(rng.normal(size=(8, 2)))[0].T
    coefficients = rng.normal(size=(15, 2))
    workspace = RidgeWorkspace(U, I, initial, coefficients, np.ones(15), max_unknowns=0)
    result = workspace.correction(1, 1e-8, 1)
    assert result[1] == 7 and np.isinf(result[3])
    with pytest.raises(ValueError, match="nonnegative"):
        solve(initial, U, I, dense_max_bytes=-1)
    with pytest.raises(ValueError):
        solve(initial, U * np.nan, I)


def test_manifold_hybrid_fit_matches_cg() -> None:
    rng = np.random.default_rng(115)
    X = rng.normal(size=(100, 5))
    Y = np.sin(X[:, 0]) + X[:, 1] ** 2
    settings = {
        "N_loc": 20,
        "N_lin": 30,
        "N_J": 12,
        "N_phi": 8,
        "N_manifold": 5,
        "sync_steps": 1,
        "h_min": 1e6,
        "cg_tol": 1e-9,
        "cg_maxiter": 1000,
    }
    old = ADP_Manifold(2, **settings).fit(X, Y)
    new = ADP_Manifold(2, solver="hybrid", **settings).fit(X, Y)
    np.testing.assert_allclose(new.projectors_, old.projectors_, rtol=1e-6, atol=1e-7)
    assert new.trace_[0]["linear_relative_residual_max"] < 1e-8


def test_ill_conditioned_ridge_rotation_and_memory_gate() -> None:
    rng = np.random.default_rng(871)
    U = rng.normal(size=(20, 8, 6))
    U[..., -1] = U[..., 0] + 1e-10 * U[..., -1]
    I = rng.normal(size=(20, 8))
    index = np.linalg.qr(rng.normal(size=(6, 2)))[0].T
    coefficients = rng.normal(size=(20, 2))
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    mass = np.geomspace(1e-3, 1e3, 20)
    dense = RidgeWorkspace(U, I, index, coefficients, mass)
    streamed = RidgeWorkspace(
        U, I, rotation @ index, coefficients @ rotation.T, mass, max_bytes=0
    )
    assert streamed.backend == "scaled-lsmr"
    correction = dense.correction(1.0, 1e-8, 1000)[0].reshape(index.shape)
    rotated = streamed.correction(1.0, 1e-8, 1000)[0].reshape(index.shape)
    np.testing.assert_allclose(rotated, rotation @ correction, rtol=1e-7, atol=1e-7)


def test_manifold_both_iterative_methods_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(429)
    U = rng.normal(size=(8, 5, 4))
    I = rng.normal(size=(8, 5))
    projectors = np.array(
        [np.linalg.qr(rng.normal(size=(4, 2)))[0].T for _ in range(8)]
    )
    slopes = rng.normal(size=(8, 2))
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_manifold(
            U,
            I,
            np.ones(8),
            np.ones(8),
            projectors,
            slopes,
            projectors[0],
            ridge=1,
            tol=1e-12,
            maxiter=1,
            dense_max_unknowns=0,
        )
