from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import grassman as reference, grassman_optim as optimized


def _profile_reference(M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float):
    J, _, m = M.shape
    coefficients = np.empty((J, m))
    ranks = np.empty(J, dtype=int)
    value = 0.0
    for j in range(J):
        if tau:
            design = np.vstack((M[j], np.sqrt(tau) * np.eye(m)))
            target = np.concatenate((I[j], np.zeros(m)))
        else:
            design, target = M[j], I[j]
        coefficients[j], _, ranks[j], _ = np.linalg.lstsq(design, target, rcond=None)
        residual = I[j] - M[j] @ coefficients[j]
        value += mass[j] * (
            residual @ residual + tau * (coefficients[j] @ coefficients[j])
        )
    return coefficients, ranks, float(value)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("tau", [0.0, 1e-9, 0.2])
def test_qr_profile_matches_independent_augmented_lstsq(m: int, tau: float) -> None:
    rng = np.random.default_rng(4100 + m)
    J, p = 7, 9
    M = rng.normal(size=(J, p, m))
    if m == 2:
        M[:, :, 1] *= 1e-2
    I = rng.normal(size=(J, p))
    mass = np.geomspace(1e-3, 1e3, J)

    got = optimized._profile(M, I, mass, tau)
    coeff, ranks, value = _profile_reference(M, I, mass, tau)
    np.testing.assert_allclose(got.coefficients, coeff, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(got.ranks, ranks)
    assert got.value == pytest.approx(value, rel=2e-12, abs=2e-11)

    baseline = reference._profile(M, I, mass, tau)
    np.testing.assert_allclose(
        got.coefficients, baseline.coefficients, rtol=2e-10, atol=2e-11
    )
    assert got.value == pytest.approx(baseline.value, rel=2e-12, abs=2e-11)


@pytest.mark.parametrize("tau", [0.0, 0.2])
def test_qr_profile_guard_fallbacks_match_svd_reference(tau: float) -> None:
    rng = np.random.default_rng(414)
    mass = np.ones(4)
    I = rng.normal(size=(4, 3))
    cases = [
        np.stack([rng.normal(size=(3, 2)) for _ in range(4)]),
        np.stack([rng.normal(size=(3, 2)) @ np.diag([1.0, 1e-10]) for _ in range(4)]),
        np.stack([rng.normal(size=(3, 2)) @ np.diag([1e-30, 2e-30]) for _ in range(4)]),
        np.stack(
            [
                np.column_stack((v, v + 1e-10 * w))
                for v, w in zip(
                    rng.normal(size=(4, 3)), rng.normal(size=(4, 3)), strict=True
                )
            ]
        ),
    ]
    for M in cases:
        got = optimized._profile(M, I, mass, tau)
        baseline = reference._profile(M, I, mass, tau)
        np.testing.assert_allclose(
            got.coefficients, baseline.coefficients, rtol=2e-8, atol=2e-8
        )
        np.testing.assert_array_equal(got.ranks, baseline.ranks)
        assert got.value == pytest.approx(baseline.value, rel=2e-12, abs=2e-11)


@pytest.mark.parametrize("tau", [0.0, 1e-9, 0.2])
def test_triangular_normal_solve_handles_multiple_rhs(tau: float) -> None:
    rng = np.random.default_rng(451)
    J, p = 8, 11
    M = rng.normal(size=(J, p, 2))
    M[:, :, 1] *= 0.03
    I = rng.normal(size=(J, p))
    state = optimized._profile(M, I, np.ones(J), tau)
    assert state.triangular is not None
    rhs = rng.normal(size=(J, 2, 5))
    got = state.normal_solve(rhs)
    expected = np.stack(
        [np.linalg.solve(M[j].T @ M[j] + tau * np.eye(2), rhs[j]) for j in range(J)]
    )
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    "m,p,tau", [(1, 6, 0.0), (2, 7, 0.2), (4, 8, 0.0), (4, 2, 0.2)]
)
def test_solver_matches_reference_for_rank_ridge_workspace_and_layout(
    m: int, p: int, tau: float
) -> None:
    rng = np.random.default_rng(700 + m + p)
    d, J = 11, 12
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0]
    B = Y.T
    U_base = rng.normal(size=(J, p, d * 2))
    U = U_base[:, :, ::2]
    I = rng.normal(size=(J, p))
    mass = np.geomspace(1e-2, 1e2, J)
    assert not U.flags.c_contiguous
    options = {"mass": mass, "max_steps": 4, "local_ridge": tau}

    expected = reference.solve(B, U, I, **options)
    got = optimized.solve(B, U, I, **options)
    np.testing.assert_allclose(
        got.index.T @ got.index,
        expected.index.T @ expected.index,
        rtol=2e-8,
        atol=2e-8,
    )
    assert got.diagnostics["profile_objective"] == pytest.approx(
        expected.diagnostics["profile_objective"], rel=2e-9, abs=2e-9
    )
    assert np.isfinite(got.diagnostics["profile_objective"])
    np.testing.assert_allclose(got.index @ got.index.T, np.eye(m), atol=2e-12)

    fallback = optimized.solve(B, U, I, **options, workspace_bytes=1)
    fallback_ref = reference.solve(B, U, I, **options, workspace_bytes=1)
    np.testing.assert_allclose(
        fallback.index.T @ fallback.index,
        fallback_ref.index.T @ fallback_ref.index,
        rtol=2e-8,
        atol=2e-8,
    )
    if m == 2 and p >= m and tau:
        assert fallback.diagnostics["workspace_fallbacks"] > 0


def test_batched_core_jacobian_matches_reference_and_finite_difference() -> None:
    rng = np.random.default_rng(981)
    J, p, d, m, q = 5, 8, 10, 2, 3
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0]
    V = np.linalg.qr(rng.normal(size=(d, q)))[0]
    V -= Y @ (Y.T @ V)
    V = np.linalg.qr(V)[0]
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.1, 10.0, J)
    for tau in (0.0, 0.2):
        state = optimized._profile(U @ Y, I, mass, tau)
        W = U @ V
        jac, residual = optimized._core_jacobian(state, W, mass, tau)
        ref_jac, ref_residual = reference._core_jacobian(state, W, mass, tau)
        np.testing.assert_allclose(jac, ref_jac, rtol=3e-10, atol=3e-10)
        np.testing.assert_allclose(residual, ref_residual, rtol=3e-12, atol=3e-12)

        K = rng.normal(size=(q, m))
        K /= np.linalg.norm(K)
        eps = 1e-5
        packed = []
        for sign in (-1.0, 1.0):
            _, M_new = optimized._polar(Y, state.M, V, W, sign * eps * K)
            trial = optimized._profile(M_new, I, mass, tau)
            rows = (
                trial.residual
                if tau == 0
                else np.concatenate(
                    (trial.residual, -np.sqrt(tau) * trial.coefficients), axis=1
                )
            )
            packed.append((np.sqrt(mass)[:, None] * rows).ravel())
        fd = (packed[1] - packed[0]) / (2 * eps)
        np.testing.assert_allclose(jac @ K.ravel(), fd, rtol=2e-6, atol=2e-6)


def test_gradient_multiple_chunks_matches_flat_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(829)
    J, p, d, m = 37, 13, 17, 2
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0]
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(1e-3, 1e3, J)
    state = optimized._profile(U @ Y, I, mass, 0.0)
    flat = reference._gradient(Y, U, state, mass)

    monkeypatch.setattr(optimized, "_GRADIENT_WORKSPACE_BYTES", 128)
    chunk = max(1, optimized._GRADIENT_WORKSPACE_BYTES // (8 * (d + m)))
    assert chunk < J
    got = optimized._gradient(Y, U, state, mass)
    np.testing.assert_allclose(got, flat, rtol=3e-12, atol=3e-10)
