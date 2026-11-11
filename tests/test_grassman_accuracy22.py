"""Independent checks for isolated accuracy candidates."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from experiments.grassman_accuracy_2026_10_22 import candidates as c


def decimal_residual(M, I, W, column, step):
    """80-digit two-column reference; independent central differences."""
    with localcontext() as context:
        context.prec = 80
        a = [[Decimal(float(v)) for v in row] for row in M]
        y = [Decimal(float(v)) for v in I]
        for row, w in zip(a, W, strict=True):
            row[column] += step * Decimal(float(w))
        aa = sum(row[0] ** 2 for row in a)
        bb = sum(row[1] ** 2 for row in a)
        ab = sum(row[0] * row[1] for row in a)
        ay = sum(row[0] * v for row, v in zip(a, y, strict=True))
        by = sum(row[1] * v for row, v in zip(a, y, strict=True))
        determinant = aa * bb - ab**2
        g = ((ay * bb - by * ab) / determinant, (by * aa - ay * ab) / determinant)
        return [v - row[0] * g[0] - row[1] * g[1] for row, v in zip(a, y, strict=True)]


def test_orthogonal_jacobian_against_high_precision_difference():
    rng = np.random.default_rng(498)
    frame = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    M = np.stack((frame[:, 0], frame[:, 0] + 1e-7 * frame[:, 1]), axis=1)[None]
    I, W = rng.normal(size=(1, 6)), rng.normal(size=(1, 6, 1))
    state = c.reference._profile(M, I, np.ones(1), 0.0)
    assert state.smooth
    with localcontext() as context:
        context.prec = 80
        h = Decimal("1e-27")
        expected = []
        for column in range(2):
            plus = decimal_residual(M[0], I[0], W[0, :, 0], column, h)
            minus = decimal_residual(M[0], I[0], W[0, :, 0], column, -h)
            expected.append(
                [float((p - n) / (2 * h)) for p, n in zip(plus, minus, strict=True)]
            )
    expected = np.array(expected).T
    jac, _ = c.orthogonal_jacobian(state, W, np.ones(1), 0.0)
    relative_error = np.linalg.norm(jac - expected) / np.linalg.norm(expected)
    # A backward-stable factorization still has O(eps*cond(M)) error.
    assert relative_error < 10 * np.finfo(float).eps * np.linalg.cond(M[0])


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("ridge", [0.0, 0.2])
def test_orthogonal_jacobian_matches_independent_profile_fd(m, ridge):
    rng = np.random.default_rng(815 + m)
    M, W, I = (
        rng.normal(size=(5, 8, m)),
        rng.normal(size=(5, 8, 2)),
        rng.normal(size=(5, 8)),
    )
    mass = np.geomspace(1e-3, 1e3, 5)
    state = c.reference._profile(M, I, mass, ridge)
    jac, residual = c.orthogonal_jacobian(state, W, mass, ridge)

    def packed(design):
        result = []
        for a, y, weight in zip(design, I, mass, strict=True):
            augmented = np.vstack((a, np.sqrt(ridge) * np.eye(m))) if ridge else a
            target = np.r_[y, np.zeros(m)] if ridge else y
            g = np.linalg.lstsq(augmented, target, rcond=None)[0]
            result.extend(np.sqrt(weight) * (target - augmented @ g))
        return np.array(result)

    np.testing.assert_allclose(residual, packed(M), atol=2e-13)
    for index in range(2 * m):
        delta = np.zeros((2, m))
        delta.flat[index] = 1e-5
        expected = (packed(M + W @ delta) - packed(M - W @ delta)) / 2e-5
        np.testing.assert_allclose(jac[:, index], expected, rtol=1e-7, atol=2e-8)


def test_smooth_relative_damping_is_scale_invariant():
    rng = np.random.default_rng(182)
    U, I = rng.normal(size=(7, 6, 8)), rng.normal(size=(7, 6))
    B = np.eye(8)[:2]
    solve = c.solver_for("smooth_relative")
    first = solve(B, U, I, max_steps=1, tol=1e-14)
    scaled = solve(B, U, I * 1e-3, max_steps=1, tol=1e-14)
    np.testing.assert_allclose(
        first.index.T @ first.index, scaled.index.T @ scaled.index, atol=2e-11
    )
    assert first.diagnostics["normalized_gn_proposals"] == 1
    old = c.reference.solve(B, U, I, max_steps=1, tol=1e-14)
    assert old.diagnostics["normalized_gn_proposals"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "baseline",
        "smooth_relative",
        "stable_jacobian",
        "diagonal",
        "gn_quadratic",
        "krylov10",
        "krylov20",
    ],
)
@pytest.mark.parametrize("ridge", [0.0, 0.1])
def test_candidate_true_profile_and_certificate(name, ridge):
    rng = np.random.default_rng(785)
    U, I = rng.normal(size=(8, 7, 9)), rng.normal(size=(8, 7))
    U[0] = 0
    mass = np.geomspace(1e-3, 1e3, 8)
    result = c.solver_for(name)(
        np.eye(9)[:2], U, I, mass=mass, local_ridge=ridge, max_steps=3
    )
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=2e-14)
    assert np.all(np.diff(result.diagnostics["profile_history"]) <= 0)
    state = c.reference._profile(U @ result.index.T, I, mass, ridge)
    assert result.diagnostics["profile_objective"] == pytest.approx(
        state.value, rel=1e-12
    )
    G = c.reference._gradient(result.index.T, U, state, mass)
    assert result.diagnostics["gradient_norm"] == pytest.approx(
        np.linalg.norm(G), rel=1e-11
    )


def test_diagonal_plane_is_horizontal_and_descent_with_bounded_chunks():
    rng = np.random.default_rng(45)
    Y = np.linalg.qr(rng.normal(size=(9, 2)))[0]
    U, I = rng.normal(size=(7, 6, 9)), rng.normal(size=(7, 6))
    mass = np.geomspace(0.1, 10, 7)
    state = c.reference._profile(U @ Y, I, mass, 0.0)
    G = c.reference._gradient(Y, U, state, mass)
    direction = c.diagonal_direction(Y, U, state, mass, G, workspace_bytes=128)
    np.testing.assert_allclose(Y.T @ direction, 0, atol=1e-11)
    assert np.sum(G * direction) > 0
    weights = mass * np.sum(state.coefficients**2, axis=1)
    diagonal = np.einsum("jpd,jpd,j->d", U, U, weights)
    expected = G / (diagonal[:, None] + 0.05 * diagonal.mean())
    expected -= Y @ (Y.T @ expected)
    np.testing.assert_allclose(direction, expected, rtol=1e-13, atol=1e-13)


def test_extra_core_length_trial_preserves_accepted_step_and_can_improve_it():
    gains = []
    for seed in range(11, 16):
        rng = np.random.default_rng(seed)
        U, I = rng.normal(size=(10, 8, 10)), rng.normal(size=(10, 8))
        B = np.eye(10)[:2]
        old = c.solver_for("baseline")(B, U, I, max_steps=1)
        new = c.solver_for("gn_quadratic")(B, U, I, max_steps=1)
        gains.append(
            old.diagnostics["profile_objective"] - new.diagnostics["profile_objective"]
        )
        assert gains[-1] >= -1e-12
    assert max(gains) > 1e-6


def test_intrinsic_krylov_plane_ignores_constant_rows_and_is_descent(monkeypatch):
    rng = np.random.default_rng(824)
    Y = np.eye(9)[:, :2]
    U, I = rng.normal(size=(8, 7, 9)), rng.normal(size=(8, 7))
    U[0] = np.outer(rng.normal(size=7), rng.normal(size=9))
    mass = np.geomspace(0.1, 10, 8)
    state = c.reference._profile(U @ Y, I, mass, 0.0)
    G = c.reference._gradient(Y, U, state, mass)
    operators = []
    original = c.ProfileJacobian

    def observed(*args):
        operator = original(*args)
        operators.append(operator)
        return operator

    monkeypatch.setattr(c, "ProfileJacobian", observed)
    stops = []
    Z = c.krylov_direction(Y, U, I, state, mass, G, 0.0, np.full(8, -1), 0.05, stops)
    np.testing.assert_allclose(Y.T @ Z, 0, atol=1e-12)
    assert np.sum(G * Z) > 0
    assert stops and stops[0][1] <= 2
    assert np.linalg.norm(Z / np.linalg.norm(Z) - G / np.linalg.norm(G)) > 1e-3
    padded = operators[0]
    assert np.shares_memory(padded.U, U)
    regular = original(
        Y, U[1:], c.reference._profile((U @ Y)[1:], I[1:], mass[1:], 0.0), mass[1:], 0.0
    )
    v, dual = rng.normal(size=Y.size), rng.normal(size=I.shape)
    output = padded.matvec(v).reshape(I.shape)
    np.testing.assert_allclose(output[0], 0, atol=0)
    np.testing.assert_allclose(output[1:].ravel(), regular.matvec(v), atol=1e-12)
    np.testing.assert_allclose(
        padded.rmatvec(dual.ravel()), regular.rmatvec(dual[1:].ravel()), atol=1e-12
    )
