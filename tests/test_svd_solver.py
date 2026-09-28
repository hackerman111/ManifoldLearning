from __future__ import annotations

import numpy as np
import pytest

import ADP.solver.SVD as svd_module
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver.SVD import (
    _FlatU,
    _v_certificate,
    _v_operator,
    _v_step,
    _v_step_direct,
    solve,
    solve_fixed_coefficients,
)


@pytest.mark.parametrize(
    ("ridge", "collinear"), [(0.0, False), (0.2, False), (0.2, True)]
)
def test_fixed_rank_step_matches_dense_objective_and_v_reference(
    ridge: float, collinear: bool
) -> None:
    rng = np.random.default_rng(2026)
    J, p, d, m = 8, 6, 5, 3
    U = rng.normal(size=(J, p, d))
    if collinear:
        U[..., -1] = U[..., 0] + 1e-10 * U[..., -1]
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = np.geomspace(1e-3, 1e3, J) if collinear else np.geomspace(0.2, 3.0, J)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    alpha = g @ a
    scale = np.repeat(np.sqrt(mass) * alpha, p)
    flat_U = _FlatU(U)
    operator = _v_operator(flat_U, scale)
    probe = rng.normal(size=d)
    dual = rng.normal(size=J * p)
    np.testing.assert_allclose(
        np.dot(operator @ probe, dual),
        np.dot(probe, operator.rmatvec(dual)),
        rtol=1e-12,
        atol=1e-12,
    )
    design = scale[:, None] * U.reshape(J * p, d)
    target = (np.sqrt(mass)[:, None] * I).ravel()
    z = P.T @ a
    expected_v = np.linalg.lstsq(
        np.vstack((design, np.sqrt(ridge) * np.eye(d))),
        np.concatenate((target, np.sqrt(ridge) * z)),
        rcond=None,
    )[0]
    v, raw_v, estimate, iterations, stop, certificate_args = _v_step(
        flat_U, target, P, alpha, a, np.sqrt(mass), ridge, 1e-8, None
    )
    certificate = _v_certificate(flat_U, raw_v, *certificate_args, ridge=ridge)
    np.testing.assert_allclose(v, expected_v / np.linalg.norm(expected_v), atol=1e-9)
    assert certificate < 1e-9
    assert np.isfinite(estimate)
    assert iterations > 0
    assert stop in range(8)
    warm = rng.normal(size=d)
    warm_v, warm_raw, _, _, _, warm_certificate_args = _v_step(
        flat_U,
        target,
        P,
        alpha,
        a,
        np.sqrt(mass),
        ridge,
        1e-8,
        None,
        x0=warm,
    )
    warm_certificate = _v_certificate(
        flat_U, warm_raw, *warm_certificate_args, ridge=ridge
    )
    np.testing.assert_allclose(
        warm_v, expected_v / np.linalg.norm(expected_v), atol=1e-9
    )
    assert warm_certificate < 1e-9

    if ridge > 0:
        direct_v, direct_raw, direct_certificate_args = _v_step_direct(
            flat_U, target, P, alpha, a, np.sqrt(mass), ridge
        )
        direct_certificate = _v_certificate(
            flat_U, direct_raw, *direct_certificate_args, ridge=ridge
        )
        np.testing.assert_allclose(
            direct_v, expected_v / np.linalg.norm(expected_v), atol=1e-9
        )
        assert direct_certificate < 1e-9

    B, diagnostics = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=2,
        lambda_penalty=ridge,
        inner_tol=1e-8,
        rank_tol=0,
        direct_max_dimension=None,
    )

    def dense_objective(matrix: np.ndarray) -> float:
        fitted = np.array([U[j] @ (matrix.T @ g[j]) for j in range(J)])
        return float(
            np.sum(mass[:, None] * (I - fitted) ** 2)
            + ridge * np.sum((matrix - P) ** 2)
        )

    assert np.linalg.matrix_rank(B) <= 2
    assert diagnostics["effective_rank"] <= 2
    assert diagnostics["u_vector_passes"] > 0
    assert not diagnostics["u_flat_copy"]
    assert diagnostics["lsmr_iterations_total"] > 0
    assert np.all(np.diff(diagnostics["rank_objective_history"]) <= 1e-10)
    np.testing.assert_allclose(
        diagnostics["rank_objective_history"][-1], dense_objective(B), atol=1e-9
    )
    assert dense_objective(B) < dense_objective(np.zeros_like(B))
    left, singular, right = np.linalg.svd(B, full_matrices=False)
    for k in range(diagnostics["effective_rank"]):
        component = np.outer(left[:, k], right[k])
        plus = dense_objective(B + 1e-5 * component)
        minus = dense_objective(B - 1e-5 * component)
        assert abs((plus - minus) / 2e-5) < 1e-6
        assert singular[k] > 0

    if ridge > 0:
        direct_B, direct_diagnostics = solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=2,
            lambda_penalty=ridge,
            inner_tol=1e-8,
            rank_tol=0,
            direct_max_dimension=d,
        )
        assert direct_diagnostics["direct_solves"] > 0
        assert direct_diagnostics["direct_fallbacks"] == 0
        assert direct_diagnostics["lsmr_iterations_total"] == 0
        np.testing.assert_allclose(
            direct_diagnostics["rank_objective_history"][-1],
            dense_objective(direct_B),
            atol=1e-9,
        )
        np.testing.assert_allclose(direct_B, B, rtol=2e-7, atol=2e-8)


def test_factor_projection_falls_back_to_direct_u_product(monkeypatch) -> None:
    rng = np.random.default_rng(83)
    J, p, d, m = 9, 4, 7, 4
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.3, 2.0, size=J)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    monkeypatch.setattr(svd_module, "_QR_CACHE_CONDITION_LIMIT", 0.0)

    B, diagnostics = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=3,
        lambda_penalty=0.4,
        inner_tol=1e-8,
        inner_maxiter=1,
        rank_tol=0,
        warm_start=False,
    )
    fitted = np.array([U[j] @ (B.T @ g[j]) for j in range(J)])
    dense_objective = float(
        np.sum(mass[:, None] * (I - fitted) ** 2) + 0.4 * np.sum((B - P) ** 2)
    )

    assert diagnostics["factor_cache_fallbacks"] == diagnostics["effective_rank"]
    assert diagnostics["inner_converged"] == (False,) * diagnostics["effective_rank"]
    np.testing.assert_allclose(
        diagnostics["rank_objective_history"][-1], dense_objective, atol=1e-9
    )


def test_public_multi_fit_accepts_explicit_svd_solver() -> None:
    X, Y, _ = _synthetic_data(90, 5, 2, 0.05, 17)
    config = ADP_Config(N_J=18, N_phi=6, N_loc=20, outer_steps=2, h_min=1e-6)
    model = ADP_multi_index(2, config, ADP_solver(solve, rank=1, inner_maxiter=12)).fit(
        X, Y
    )
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-10)
    assert len(model.trace_) == 2
    assert all(
        row["solver"]["linear_solver"] == "truncated_svd" for row in model.trace_
    )
    assert all(row["solver"]["effective_rank"] == 1 for row in model.trace_)
    assert np.all(np.isfinite(model.transform(X)))
