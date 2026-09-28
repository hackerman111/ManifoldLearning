from __future__ import annotations

import numpy as np
import pytest

import ADP.solver.SVD as svd_module
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver._multi_operator import forward
from ADP.solver.SVD import (
    _FlatU,
    _objective_from_factors,
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


@pytest.mark.parametrize("low_rank_target", ["matrix", "correction"])
def test_public_multi_fit_accepts_explicit_svd_solver(low_rank_target: str) -> None:
    X, Y, _ = _synthetic_data(90, 5, 2, 0.05, 17)
    config = ADP_Config(N_J=18, N_phi=6, N_loc=20, outer_steps=2, h_min=1e-6)
    model = ADP_multi_index(
        2,
        config,
        ADP_solver(solve, rank=1, inner_maxiter=12, low_rank_target=low_rank_target),
    ).fit(X, Y)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-10)
    assert len(model.trace_) == 2
    assert all(
        row["solver"]["linear_solver"] == "truncated_svd" for row in model.trace_
    )
    assert all(row["solver"]["effective_rank"] == 1 for row in model.trace_)
    assert all(
        row["solver"]["low_rank_target"] == low_rank_target for row in model.trace_
    )
    assert np.all(np.isfinite(model.transform(X)))


@pytest.mark.parametrize("ridge", [0.0, 0.4])
def test_correction_factors_match_dense_objective_and_descent(ridge: float) -> None:
    rng = np.random.default_rng(409)
    J, p, d, m = 9, 5, 7, 3
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.3, 2.0, size=J)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    base_residual = I - forward(U, P, g)

    def objective(B: np.ndarray) -> float:
        residual = I - forward(U, B, g)
        return float(np.sum(mass[:, None] * residual**2) + ridge * np.sum((B - P) ** 2))

    A = np.linalg.qr(rng.normal(size=(m, 2)))[0]
    V = np.linalg.qr(rng.normal(size=(d, 2)))[0]
    s = np.array([0.8, -0.3])
    W = U @ V
    factor_objective = _objective_from_factors(
        A, s, V, W, np.zeros_like(P), base_residual, g, mass, ridge, 0.0
    )
    np.testing.assert_allclose(factor_objective, objective(P + (A * s) @ V.T))

    zero, diagnostics_zero = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=0,
        lambda_penalty=ridge,
        low_rank_target="correction",
    )
    np.testing.assert_allclose(zero, P, rtol=0, atol=0)
    assert diagnostics_zero["correction_rank"] == 0
    assert diagnostics_zero["rank_objective_history"] == (objective(P),)
    stopped, stopped_diagnostics = solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=2,
        lambda_penalty=ridge,
        rank_tol=1e99,
        low_rank_target="correction",
    )
    np.testing.assert_allclose(stopped, P, rtol=0, atol=0)
    assert stopped_diagnostics["rank_stop_reason"] == "rank_tolerance"

    results = []
    for rank in (1, 2):
        B, diagnostics = solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=rank,
            lambda_penalty=ridge,
            rank_tol=0,
            inner_tol=1e-8,
            direct_max_dimension=None,
            low_rank_target="correction",
        )
        Delta = B - P
        results.append(objective(B))
        assert np.linalg.matrix_rank(Delta) <= rank
        assert diagnostics["correction_rank"] <= rank
        assert diagnostics["low_rank_target"] == "correction"
        assert diagnostics["lsmr_iterations_total"] > 0
        assert diagnostics["v_normal_residual_max"] < 1e-7
        assert len(diagnostics["rank_gain_history"]) == diagnostics["correction_rank"]
        assert np.all(np.diff(diagnostics["rank_objective_history"]) <= 1e-10)
        np.testing.assert_allclose(
            diagnostics["rank_objective_history"][-1], objective(B), atol=1e-9
        )
        np.testing.assert_allclose(np.sum(Delta**2), np.sum((B - P) ** 2))
        np.testing.assert_allclose(
            diagnostics["correction_frobenius_norm"],
            np.linalg.norm(Delta),
            atol=1e-10,
        )
        pulled = (U.swapaxes(1, 2) @ (I - forward(U, B, g))[..., None]).squeeze(-1)
        data_Q = g.T @ (mass[:, None] * pulled)
        np.testing.assert_allclose(
            data_Q - ridge * Delta, data_Q + ridge * (P - B), atol=1e-12
        )
        left, singular, right = np.linalg.svd(Delta, full_matrices=False)
        np.testing.assert_allclose(
            diagnostics["correction_singular_values"], singular[:rank], atol=1e-10
        )
        # Joint scale refit makes each accepted component stationary in scale.
        for k in range(diagnostics["correction_rank"]):
            direction = np.outer(left[:, k], right[k])
            h = 1e-5
            derivative = (
                objective(B + h * direction) - objective(B - h * direction)
            ) / (2 * h)
            assert abs(derivative) < 1e-6
        if rank == 1:
            a, v = left[:, 0], right[0]
            alpha = g @ a
            Uv = U @ v
            R = np.sum(mass[:, None] * base_residual * (alpha[:, None] * Uv))
            D = np.sum(mass[:, None] * (alpha[:, None] * Uv) ** 2) + ridge
            np.testing.assert_allclose(singular[0], R / D, atol=1e-9)
            np.testing.assert_allclose(
                objective(P) - objective(B), R * R / D, atol=1e-8
            )
    assert results[1] <= results[0] + 1e-10


def test_correction_direct_reference_and_public_basis() -> None:
    rng = np.random.default_rng(410)
    J, p, d, m = 10, 5, 6, 3
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = rng.uniform(0.5, 1.5, size=J)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    ridge = 0.3
    target = (np.sqrt(mass)[:, None] * (I - forward(U, P, g))).ravel()
    design = np.vstack([np.kron(g[j], U[j]) * np.sqrt(mass[j]) for j in range(J)])
    full_delta = np.linalg.lstsq(
        np.vstack((design, np.sqrt(ridge) * np.eye(m * d))),
        np.concatenate((target, np.zeros(m * d))),
        rcond=None,
    )[0].reshape(m, d)

    def objective(B: np.ndarray) -> float:
        return float(
            np.sum(mass[:, None] * (I - forward(U, B, g)) ** 2)
            + ridge * np.sum((B - P) ** 2)
        )

    full_objective = objective(P + full_delta)
    values = np.linalg.svd(full_delta, compute_uv=False)
    energy = np.cumsum(values**2) / np.sum(values**2)
    assert 0 < energy[0] <= energy[1] <= 1
    objectives = []
    for rank in (1, 2):
        B, diagnostics = solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=rank,
            lambda_penalty=ridge,
            inner_tol=1e-8,
            rank_tol=0,
            direct_max_dimension=d,
            low_rank_target="correction",
        )
        objectives.append(objective(B))
        assert diagnostics["direct_solves"] > 0
        assert diagnostics["direct_fallbacks"] == 0
        assert diagnostics["v_normal_residual_max"] < 1e-7
        assert objective(B) >= full_objective - 1e-9
    assert objectives[1] <= objectives[0] + 1e-10

    result = solve(
        P,
        U,
        I,
        mass=mass,
        rank=2,
        lambda_prox=ridge,
        low_rank_target="correction",
        inner_tol=1e-8,
    )
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=1e-10)
    assert result.diagnostics["completion"] == "updated_basis_qr"
    assert result.diagnostics["correction_rank"] <= 2
    zero = solve(
        P, U, I, mass=mass, rank=0, lambda_prox=ridge, low_rank_target="correction"
    )
    np.testing.assert_allclose(zero.index @ zero.index.T, np.eye(m), atol=1e-10)
    np.testing.assert_allclose(zero.index.T @ zero.index, P.T @ P, atol=1e-10)
