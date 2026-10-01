from __future__ import annotations

import numpy as np
import pytest

import ADP.solver.SVD as svd
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data


@pytest.mark.parametrize("target", ["matrix", "correction"])
@pytest.mark.parametrize("ridge", [0.0, 0.7, 1e4])
def test_gradient_step_matches_dense_derivative_and_exact_gain(target, ridge):
    rng = np.random.default_rng(941)
    J, p, m, d = 7, 4, 3, 6
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    U = rng.normal(size=(J, p, d))
    U[..., -1] = U[..., 0] + 1e-10 * U[..., -1]
    U[0] = 0
    g, I = rng.normal(size=(J, m)), rng.normal(size=(J, p))
    mass = np.geomspace(1e-3, 1e3, J)
    # Independent row-vectorized dense design, never used in production.
    design = np.concatenate(
        [np.sqrt(mass[j]) * np.kron(g[j][None, :], U[j]) for j in range(J)]
    )
    data = (np.sqrt(mass)[:, None] * I).ravel()
    initial = P.copy() if target == "correction" else np.zeros_like(P)

    def objective(B):
        return np.sum((data - design @ B.ravel()) ** 2) + ridge * np.sum((B - P) ** 2)

    Q = (
        design.T @ (data - design @ initial.ravel()) + ridge * (P - initial).ravel()
    ).reshape(m, d)
    direction = rng.normal(size=P.shape)
    step = 1e-5
    finite_difference = (
        objective(initial + step * direction) - objective(initial - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        finite_difference, -2 * np.sum(Q * direction), rtol=1e-8, atol=1e-7
    )
    left, _, right = np.linalg.svd(Q, full_matrices=False)
    candidates = []
    for a, v in zip(left.T, right, strict=True):
        atom = np.outer(a, v)
        numerator = np.sum(Q * atom)
        curvature = np.sum((design @ atom.ravel()) ** 2) + ridge
        gain = numerator**2 / curvature
        candidates.append((gain, initial + numerator / curvature * atom))
    gain, expected = max(candidates, key=lambda pair: pair[0])
    B, diagnostics = svd.solve_fixed_coefficients(
        P,
        U,
        I,
        g,
        mass,
        rank=1,
        lambda_penalty=ridge,
        low_rank_target=target,
        rank_one_search="gradient",
        rank_tol=0,
    )
    np.testing.assert_allclose(B, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(diagnostics["rank_gain_history"], [gain], rtol=1e-12)
    np.testing.assert_allclose(
        objective(initial) - objective(B), gain, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        diagnostics["rank_objective_history"][-1], objective(B), rtol=1e-12
    )
    assert diagnostics["lsmr_iterations_total"] == diagnostics["direct_solves"] == 0
    assert diagnostics["rank_one_search"] == "gradient"
    assert not diagnostics["v_normal_residual_applicable"]
    assert diagnostics["inner_iterations"] == ()
    assert diagnostics["inner_converged"] == ()


def test_gradient_prefers_smaller_singular_value_with_larger_gain():
    # Q=diag(10,2), but curvatures are 100 and 1. Gains: 1 and 4.
    P = np.eye(2)
    U = np.array([[[10.0, 0.0]], [[0.0, 1.0]]])
    I = np.array([[1.0], [2.0]])
    B, diagnostics = svd.solve_fixed_coefficients(
        P,
        U,
        I,
        np.eye(2),
        np.ones(2),
        rank=1,
        lambda_penalty=0,
        rank_one_search="gradient",
    )
    np.testing.assert_allclose(B, [[0, 0], [0, 2]], atol=1e-14)
    np.testing.assert_allclose(diagnostics["rank_gain_history"], [4])


def test_gradient_gain_tolerance_and_no_rank_growth_are_heuristic_stops():
    P = np.eye(3)
    U = np.array([[[1.0, 0, 0]], [[0, 2.0, 0]], [[0, 0, 3.0]]])
    I = np.ones((3, 1))
    g = np.tile([1.0, 0, 0], (3, 1))
    args = (P, U, I, g, np.ones(3))
    settings = {
        "rank": 2,
        "lambda_penalty": 0,
        "rank_one_search": "gradient",
        "rank_tol": 0,
    }
    first, _ = svd.solve_fixed_coefficients(*args, **{**settings, "rank": 1})
    B, diag = svd.solve_fixed_coefficients(*args, **settings)
    # Q retains one left direction: the next useful step cannot grow rank.
    # The current greedy compression rejects it, without claiming convergence.
    np.testing.assert_array_equal(B, first)
    assert diag["rank_stop_reason"] == "no_rank_growth"
    assert diag["effective_rank"] == 1
    stopped, diag = svd.solve_fixed_coefficients(*args, **{**settings, "rank_tol": 2.0})
    np.testing.assert_array_equal(stopped, np.zeros_like(P))
    assert diag["rank_stop_reason"] == "rank_tolerance"


@pytest.mark.parametrize("target", ["matrix", "correction"])
def test_gradient_multirank_and_public_fit_skip_direction_solves(target, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("gradient search must not call a conditional v solver")

    monkeypatch.setattr(svd, "_v_step", forbidden)
    monkeypatch.setattr(svd, "_v_step_direct", forbidden)
    rng = np.random.default_rng(142)
    P = np.linalg.qr(rng.normal(size=(8, 4)))[0].T
    U, I = rng.normal(size=(10, 5, 8)), rng.normal(size=(10, 5))
    g, mass = rng.normal(size=(10, 4)), np.geomspace(0.1, 10, 10)
    settings = {
        "rank": 3,
        "lambda_penalty": 0.2,
        "low_rank_target": target,
        "rank_one_search": "gradient",
        "rank_tol": 0,
    }
    B, diagnostics = svd.solve_fixed_coefficients(P, U, I, g, mass, **settings)
    B2, diagnostics2 = svd.solve_fixed_coefficients(P, U, I, g, mass, **settings)
    np.testing.assert_array_equal(B, B2)
    assert diagnostics == diagnostics2
    assert np.linalg.matrix_rank(B - P if target == "correction" else B) <= 3
    assert np.all(np.diff(diagnostics["rank_objective_history"]) <= 1e-10)
    assert np.all(np.isfinite(B))

    X, y, _ = _synthetic_data(90, 5, 2, 0.05, 17)
    config = ADP_Config(N_J=18, N_phi=6, N_loc=20, outer_steps=2, h_min=1e-6)
    model = ADP_multi_index(
        2,
        config,
        ADP_solver(
            svd.solve, rank=1, low_rank_target=target, rank_one_search="gradient"
        ),
    ).fit(X, y)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-10)
    assert all(row["solver"]["rank_one_search"] == "gradient" for row in model.trace_)
    assert np.all(np.isfinite(model.transform(X)))


def test_gradient_zero_rank_zero_gradient_and_invalid_search():
    P, U = np.eye(2), np.zeros((3, 1, 2))
    I, g, mass = np.ones((3, 1)), np.zeros((3, 2)), np.ones(3)
    for rank in (0, 1):
        B, diagnostics = svd.solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=rank,
            lambda_penalty=0,
            low_rank_target="correction",
            rank_one_search="gradient",
        )
        np.testing.assert_array_equal(B, P)
        assert diagnostics["effective_rank"] == 0
        assert diagnostics["rank_stop_reason"] == (
            "rank_limit" if rank == 0 else "zero_gradient"
        )
    for search in ("unknown", None, True):
        with pytest.raises(ValueError, match="rank_one_search"):
            svd.solve_fixed_coefficients(
                P,
                U,
                I,
                g,
                mass,
                rank=1,
                lambda_penalty=0,
                rank_one_search=search,
            )
    with pytest.raises(RuntimeError, match="must be finite"):
        svd._gradient_pair(svd._FlatU(U), np.full((2, 2), np.nan), g, mass, 0)
    with pytest.raises(RuntimeError, match="positive curvature"):
        svd._gradient_pair(svd._FlatU(U), P, g, mass, 0)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(RuntimeError, match="nonfinite"),
    ):
        svd._gradient_pair(svd._FlatU(np.full_like(U, 1e200)), P, g, mass, 0)
