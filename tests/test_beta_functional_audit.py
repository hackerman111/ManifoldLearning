# ruff: noqa: RUF003
"""Независимые малые проверки формул, сжатия и NumPy-кандидатов."""

from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import HYBRID, LSMR
from benchmarks.beta_functional_audit import (
    compress_rows,
    design_without_temporary,
    stationarity_matmul,
)


def test_joint_hessian_gradient_and_cross_terms() -> None:
    rng = np.random.default_rng(72000)
    U, I, C = (
        rng.normal(size=(5, 6, 4)),
        rng.normal(size=(5, 6)),
        rng.normal(size=(5, 2)),
    )
    B = rng.normal(size=(2, 4))
    mass = np.geomspace(0.01, 100, 5)
    A = np.vstack([np.kron(c[None, :], u) for c, u in zip(C, U, strict=True)])
    A *= np.repeat(np.sqrt(mass), 6)[:, None]
    y = (np.sqrt(mass)[:, None] * I).ravel()
    gradient = np.sum(
        [
            w * np.outer(c, u.T @ (u @ B.T @ c - i))
            for w, c, u, i in zip(mass, C, U, I, strict=True)
        ],
        axis=0,
    )
    np.testing.assert_allclose(gradient.ravel(), A.T @ (A @ B.ravel() - y))
    H = A.T @ A
    np.testing.assert_allclose(
        H[:4, 4:],
        sum(w * c[0] * c[1] * u.T @ u for w, c, u in zip(mass, C, U, strict=True)),
    )
    assert np.linalg.norm(H[:4, 4:]) > 1
    delta = rng.normal(size=B.shape)
    step = 1e-5

    def objective(x):
        return 0.5 * np.linalg.norm(A @ x.ravel() - y) ** 2

    difference = (objective(B + step * delta) - objective(B - step * delta)) / (
        2 * step
    )
    np.testing.assert_allclose(difference, np.sum(gradient * delta), rtol=1e-8)


@pytest.mark.parametrize("degenerate", [False, True])
def test_augmented_row_qr_preserves_objective_local_cutoff_and_ridge(
    degenerate: bool,
) -> None:
    rng = np.random.default_rng(72001)
    U, I = rng.normal(size=(7, 12, 4)), rng.normal(size=(7, 12))
    if degenerate:
        U[0] = 0
        U[1, :, 1:] = 0
    B = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
    C = rng.normal(size=(7, 2))
    mass = np.geomspace(1e-5, 1e5, 7)
    V, y = compress_rows(U, I)
    r, compact_r = (
        I - (U @ (C @ B)[..., None]).squeeze(-1),
        y - (V @ (C @ B)[..., None]).squeeze(-1),
    )
    np.testing.assert_allclose(
        np.sum(r**2, axis=1), np.sum(compact_r**2, axis=1), rtol=1e-12
    )
    # Исходный P, а не сжатый r, определяет численный cutoff local refit.
    cutoff = np.finfo(float).eps * max(U.shape[1], len(B))
    for u, v, rhs, small_rhs in zip(U, V, I, y, strict=True):
        old = np.linalg.lstsq(u @ B.T, rhs, rcond=cutoff)[0]
        new = np.linalg.lstsq(v @ B.T, small_rhs, rcond=cutoff)[0]
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-11)
    A = HYBRID.design_matrix(U, C, np.sqrt(mass))
    compact_A = HYBRID.design_matrix(V, C, np.sqrt(mass))
    np.testing.assert_allclose(compact_A.T @ compact_A, A.T @ A, rtol=1e-12, atol=1e-8)
    for ridge in (0.01, 100.0):

        def ridge_solution(a, rhs, ridge=ridge):
            return np.linalg.lstsq(
                np.vstack((a, np.sqrt(ridge) * np.eye(B.size))),
                np.concatenate((rhs, np.zeros(B.size))),
                rcond=None,
            )[0]

        old = ridge_solution(A, (np.sqrt(mass)[:, None] * r).ravel())
        new = ridge_solution(compact_A, (np.sqrt(mass)[:, None] * compact_r).ravel())
        np.testing.assert_allclose(new, old, rtol=1e-8, atol=1e-10)


def test_common_metric_spectral_ridge_matches_joint_dense_reference() -> None:
    rng = np.random.default_rng(72002)
    J, d, m = 6, 4, 2
    L = rng.normal(size=(7, d))
    alpha, mass = np.geomspace(0.1, 10, J), np.geomspace(0.01, 10, J)
    U = np.sqrt(alpha)[:, None, None] * np.broadcast_to(L, (J, *L.shape))
    C, B0, I = rng.normal(size=(J, m)), rng.normal(size=(m, d)), rng.normal(size=(J, 7))
    ridge = 0.3
    M, H = (C.T * (mass * alpha)) @ C, L.T @ L
    rhs = (
        sum(w * np.outer(c, i @ u) for w, c, i, u in zip(mass, C, I, U, strict=True))
        + ridge * B0
    )
    mu, S = np.linalg.eigh(M)
    eta, T = np.linalg.eigh(H)
    B = S @ ((S.T @ rhs @ T) / (mu[:, None] * eta[None, :] + ridge)) @ T.T
    A = HYBRID.design_matrix(U, C, np.sqrt(mass))
    expected = (
        B0.ravel()
        + np.linalg.lstsq(
            np.vstack((A, np.sqrt(ridge) * np.eye(m * d))),
            np.concatenate(
                ((np.sqrt(mass)[:, None] * I).ravel() - A @ B0.ravel(), np.zeros(m * d))
            ),
            rcond=None,
        )[0]
    )
    np.testing.assert_allclose(B.ravel(), expected, rtol=1e-11, atol=1e-11)


def test_gradient_averaging_loses_metric_and_can_cancel() -> None:
    gradients = np.eye(2)
    metric1, metric2 = np.diag([100.0, 1.0]), np.eye(2)
    beta = np.linalg.solve(
        metric1 + metric2, metric1 @ gradients[0] + metric2 @ gradients[1]
    )
    assert np.linalg.norm(beta - gradients.mean(axis=0)) > 0.4
    # f(x)=x_1² на симметричных X: mean gradient=0, OPG has rank 1.
    symmetric = np.array([[2.0, 0.0], [-2.0, 0.0]])
    np.testing.assert_array_equal(symmetric.mean(axis=0), np.zeros(2))
    assert np.linalg.matrix_rank(symmetric.T @ symmetric) == 1


@pytest.mark.parametrize("degenerate", [False, True])
def test_stationarity_identity_and_strided_design(degenerate: bool) -> None:
    rng = np.random.default_rng(72003)
    U = rng.normal(size=(8, 9, 12))[:, :, ::2]
    I, mass = rng.normal(size=(8, 9)), np.geomspace(1e-8, 1e8, 8)
    B = np.linalg.qr(rng.normal(size=(6, 2)))[0].T
    if degenerate:
        U[0] = 0
        U[1, :, 1:] = 0
    C, _ = LSMR._local_refit(I, U, B)
    loss = LSMR._loss(I, U, B, C, mass)
    actual = stationarity_matmul(I, U, B, C, mass, loss)
    expected = LSMR._stationarity(I, U, B, C, mass, loss)
    np.testing.assert_allclose(
        np.array(actual)[[0, 2]], np.array(expected)[[0, 2]], rtol=1e-9
    )
    # Local gradient почти нулевой после LS; reassociation усиливается mass.
    # Проверяем масштабированный forward-error bound, а не относительную
    # ошибку около нуля. Это НЕ ослабление production certificate.
    residual = I - LSMR._predict(U, B, C)
    projected = U @ B.T
    scale = np.maximum(
        1.0, np.linalg.norm(projected, axis=(1, 2)) * np.linalg.norm(I, axis=1)
    )
    gamma = (U.shape[1] + U.shape[2]) * np.finfo(float).eps
    bound = (
        8
        * gamma
        * np.max(
            mass
            * np.linalg.norm(U, axis=(1, 2))
            * np.linalg.norm(B)
            * np.linalg.norm(residual, axis=1)
            / scale
        )
    )
    assert abs(actual[1] - expected[1]) < bound
    # Этот adversarial пример явно запрещает обещать ту же AO trajectory
    # при любом tol: порог между двумя local scores меняет certificate.
    assert actual[1] != expected[1]
    threshold = (actual[1] + expected[1]) / 2
    assert (actual[1] < threshold) != (expected[1] < threshold)
    np.testing.assert_array_equal(
        design_without_temporary(U, C, np.sqrt(mass)),
        HYBRID.design_matrix(U, C, np.sqrt(mass)),
    )


def test_compression_rejects_nonreducing_shape() -> None:
    with pytest.raises(ValueError, match="P>d\\+1"):
        compress_rows(np.zeros((3, 5, 6)), np.zeros((3, 5)))
