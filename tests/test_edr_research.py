from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver.grassman_research import solve
from ADP.solver.statistic_operator import DenseOperator, FactorizedOperator


def _number(diagnostics, name):
    return float(cast(float, diagnostics[name]))


def _count(diagnostics, name):
    return cast(int, diagnostics[name])


def _flag(diagnostics, name):
    return cast(bool, diagnostics[name])


def _history(diagnostics, name):
    return cast(tuple[float, ...], diagnostics[name])


def _problem(seed=761, *, n=42, d=9, J=12, p=7, m=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    C = rng.normal(size=(J * p, n))
    U = (C @ X).reshape(J, p, d) / np.sqrt(n)
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.3, 2.0, J)
    Y = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    return rng, U, C, X, I, mass, Y


def _direct_profile(U, I, mass, index, tau=0.0):
    value = 0.0
    m = index.shape[0]
    for j in range(len(I)):
        design = U[j] @ index.T
        aug = np.vstack((design, np.sqrt(tau) * np.eye(m)))
        target = np.r_[I[j], np.zeros(m)]
        coefficient = np.linalg.lstsq(aug, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        value += mass[j] * (residual @ residual + tau * (coefficient @ coefficient))
    return value


def _direct_full_gradient(U, I, mass, index, tau=0.0):
    Y = index.T
    ambient = np.zeros((U.shape[2], Y.shape[1]))
    for j in range(len(I)):
        design = U[j] @ Y
        aug = np.vstack((design, np.sqrt(tau) * np.eye(Y.shape[1])))
        target = np.r_[I[j], np.zeros(Y.shape[1])]
        coefficient = np.linalg.lstsq(aug, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        ambient -= 2 * mass[j] * U[j].T @ np.outer(residual, coefficient)
    return ambient - Y @ (Y.T @ ambient)


@pytest.mark.parametrize("method", ["schur_gn", "schur_newton", "metric"])
@pytest.mark.parametrize("tau", [0.0, 0.1])
def test_research_solver_true_profile_is_monotone_and_certified(method, tau):
    _, U, _, _, I, mass, Y = _problem()
    result = solve(
        Y,
        U,
        I,
        mass=mass,
        lambda_prox=0.04,
        max_steps=4,
        tol=1e-9,
        local_ridge=tau,
        method=method,
        core_steps=2,
        workspace_bytes=2**20,
    )
    diag = result.diagnostics
    assert np.all(np.isfinite(result.index))
    assert np.all(np.diff(_history(diag, "profile_history")) <= 2e-12)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=2e-12)
    assert _number(diag, "orthogonality_error") < 2e-12
    assert _number(diag, "profile_objective") == pytest.approx(
        _history(diag, "profile_history")[-1], rel=1e-12, abs=1e-12
    )
    assert _number(diag, "profile_objective") == pytest.approx(
        _direct_profile(U, I, mass, result.index, tau), rel=3e-11, abs=3e-11
    )
    assert _count(diag, "full_gradient_evaluations") >= 1
    if _flag(diag, "converged"):
        assert _number(diag, "gradient_norm") <= 1e-9


@pytest.mark.parametrize("representation", ["dense_operator", "factorized"])
def test_dense_and_operator_paths_match_for_same_exact_operator(representation):
    _, U, C, X, I, mass, Y = _problem(seed=762)
    dense = solve(Y, U, I, mass=mass, max_steps=3, tol=1e-10, lambda_prox=0.03)
    if representation == "dense_operator":
        operator = DenseOperator(U)
    else:
        operator = FactorizedOperator(
            csr_matrix(C / np.sqrt(len(X))),
            X,
            n_centers=U.shape[0],
            n_directions=U.shape[1],
        )
    actual = solve(Y, operator, I, mass=mass, max_steps=3, tol=1e-10, lambda_prox=0.03)
    np.testing.assert_allclose(
        actual.index.T @ actual.index,
        dense.index.T @ dense.index,
        rtol=3e-8,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        _history(actual.diagnostics, "profile_history"),
        _history(dense.diagnostics, "profile_history"),
        rtol=3e-9,
        atol=3e-10,
    )


@pytest.mark.parametrize("recycle_steps", [0, 1])
def test_recycling_never_substitutes_for_live_full_gradient_certificate(recycle_steps):
    _, U, _, _, I, mass, Y = _problem(seed=763)
    result = solve(
        Y,
        U,
        I,
        mass=mass,
        max_steps=4,
        tol=1e-14,
        recycle_steps=recycle_steps,
        core_steps=1,
        lambda_prox=0.02,
    )
    diag = result.diagnostics
    assert _count(diag, "full_gradient_evaluations") >= 1
    if recycle_steps:
        assert _count(diag, "recycled_steps") > 0
    else:
        assert _count(diag, "recycled_steps") == 0
    assert np.isfinite(_number(diag, "riemannian_gradient"))
    full_gradient = _direct_full_gradient(U, I, mass, result.index)
    expected_norm = np.linalg.norm(full_gradient)
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    assert _number(diag, "gradient_norm") == pytest.approx(
        expected_norm, rel=2e-10, abs=2e-10
    )
    assert _number(diag, "riemannian_gradient") == pytest.approx(
        expected_norm / scale, rel=2e-10, abs=2e-10
    )
    assert _flag(diag, "converged") == (
        expected_norm <= 1e-14 * scale and _flag(diag, "stationarity_applicable")
    )


def test_mass_none_matches_explicit_unit_mass():
    _, U, _, _, I, _, Y = _problem(seed=766)
    implicit, explicit = (
        solve(Y, U, I, mass=m, max_steps=2, method="metric")
        for m in (None, np.ones(len(I)))
    )
    np.testing.assert_allclose(
        implicit.index.T @ implicit.index, explicit.index.T @ explicit.index, atol=2e-12
    )
    np.testing.assert_allclose(
        _history(implicit.diagnostics, "profile_history"),
        _history(explicit.diagnostics, "profile_history"),
        atol=2e-12,
    )


@pytest.mark.parametrize("method", ["metric", "schur_newton"])
def test_public_multi_index_accepts_research_solver_callback(method):
    X, y, _ = _synthetic_data(72, 5, 2, 0.05, 767)
    config = ADP_Config(N_J=14, N_phi=5, N_loc=18, outer_steps=2, h_min=1e-6)
    model = ADP_multi_index(
        2,
        config,
        ADP_solver(partial(solve, method=method), max_steps=1, tol=1e-5),
    ).fit(X, y)
    basis = model.basis_
    assert basis is not None
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=2e-10)
    assert len(model.trace_) == 2
    assert all(
        row["solver"]["linear_solver"] == "grassman_research" for row in model.trace_
    )


def test_rank_boundary_workspace_and_full_space_are_finite():
    _, U, _, _, I, mass, Y = _problem(seed=764)
    singular = np.zeros_like(U)
    guarded = solve(Y, singular, I, mass=mass, local_ridge=0.0, max_steps=2)
    assert np.isfinite(_number(guarded.diagnostics, "profile_objective"))
    assert _count(guarded.diagnostics, "rank_guard_fallbacks") > 0
    tiny_workspace = solve(Y, U, I, mass=mass, max_steps=2, workspace_bytes=1)
    assert np.all(
        np.diff(_history(tiny_workspace.diagnostics, "profile_history")) <= 2e-12
    )
    assert _count(tiny_workspace.diagnostics, "workspace_fallbacks") > 0
    full = solve(np.eye(U.shape[-1]), U, I, mass=mass, max_steps=2)
    assert _number(full.diagnostics, "orthogonality_error") < 2e-12


@pytest.mark.parametrize("method", ["schur_gn", "schur_newton"])
def test_schur_derivative_guard_clears_core_before_metric_line_search(method):
    U = np.diag([1.0, 1e-5, 1e-6])[None, :, :]
    I = np.array([[1.0, 1.0, 1.0]])
    Y = np.eye(3)[:2]
    initial = _direct_profile(U, I, np.ones(1), Y)
    result = solve(Y, U, I, max_steps=1, tol=1e-10, method=method)
    assert _count(result.diagnostics, "rank_guard_fallbacks") > 0
    assert _count(result.diagnostics, "accepted_steps") >= 1
    assert _number(result.diagnostics, "profile_objective") < initial


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lambda_prox", -1.0),
        ("max_steps", -1),
        ("tol", np.nan),
        ("rank", 0),
        ("max_angle", 4.0),
        ("local_ridge", -0.2),
        ("method", "unknown"),
        ("workspace_bytes", 0),
        ("recycle_steps", -1),
    ],
)
def test_research_solver_rejects_invalid_settings(name, value):
    _, U, _, _, I, mass, Y = _problem(seed=765)
    with pytest.raises((ValueError, TypeError)):
        solve(Y, U, I, mass=mass, **{name: value})
