from __future__ import annotations

import numpy as np
import pytest

from ADP.cli.main import _run, build_parser
from ADP.solver import HYBRID


def _workspace(*, budget: int = 64 * 1024**2) -> HYBRID.RidgeWorkspace:
    rng = np.random.default_rng(524)
    U = rng.normal(size=(19, 8, 3)) @ rng.normal(size=(3, 80))
    # Масштаб нормированных статистик: lambda=0.01 не теряется на фоне A* A.
    U *= 1e-3
    U[..., -1] = U[..., 0] + 1e-12 * U[..., -1]
    I = rng.normal(size=U.shape[:2])
    index = np.linalg.qr(rng.normal(size=(80, 2)))[0].T
    coefficients = rng.normal(size=(len(U), 2))
    mass = np.geomspace(1e-3, 1e3, len(U))
    return HYBRID.RidgeWorkspace(
        U, I, index, coefficients, mass, max_unknowns=0, max_bytes=budget
    )


@pytest.mark.parametrize("budget", [0, 48 * 160 * 3, 64 * 1024**2])
def test_recycled_shifts_preserve_ridge_center_and_minimum_norm(budget: int) -> None:
    workspace = _workspace(budget=budget)
    A = HYBRID.design_matrix(workspace.U, workspace.coefficients, workspace.root)
    for ridge in (0.01, 2.0, 1e3, 0.0):
        assert not workspace.rejects_trust(ridge, 1e6, 1000)
        # Малый ridge усиливает ошибку в слабых направлениях: reference-проверка
        # использует более строгий atol, не ослабляя сравнение координат.
        result = workspace.correction(ridge, 1e-12, 1000)
        augmented = np.vstack((A, np.sqrt(ridge) * np.eye(A.shape[1])))
        target = np.concatenate((workspace.residual, np.zeros(A.shape[1])))
        expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(result[0], expected, rtol=1e-6, atol=2e-8)
        normal = A.T @ (workspace.residual - A @ result[0]) - ridge * result[0]
        assert result[4] == pytest.approx(np.linalg.norm(normal), abs=1e-9)
        assert result[1] in (0, 1, 2, 4, 5)


def test_cached_directions_reduce_work_for_multiple_shifts() -> None:
    warm = _workspace()
    cold_iterations = warm_iterations = 0
    for ridge in (1.0, 10.0, 100.0):
        cold = _workspace()
        reference = cold.correction(ridge, 1e-8, 1000)
        assert not warm.rejects_trust(ridge, 1e6, 1000)
        result = warm.correction(ridge, 1e-8, 1000)
        np.testing.assert_allclose(result[0], reference[0], rtol=1e-6, atol=1e-8)
        cold_iterations += reference[2]
        warm_iterations += result[2]
    assert warm_iterations < cold_iterations
    assert warm.recycled_solves == 3
    assert warm.forward_calls > 0 and warm.adjoint_calls > 0


@pytest.mark.parametrize("requested", [1e-2, 1e-3, 1e-6])
def test_opt_in_certificate_bounds_actual_solution_error(requested: float) -> None:
    workspace = _workspace()
    ridge = 100.0
    workspace.rejects_trust(ridge, 1e6, 1000)
    result = workspace.correction(ridge, 1e-8, 1000, relative_tol=requested)
    A = HYBRID.design_matrix(workspace.U, workspace.coefficients, workspace.root)
    augmented = np.vstack((A, np.sqrt(ridge) * np.eye(A.shape[1])))
    target = np.concatenate((workspace.residual, np.zeros(A.shape[1])))
    expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
    assert result[3] <= requested
    assert np.linalg.norm(result[0] - expected) <= (
        requested * np.linalg.norm(result[0]) + 1e-12
    )
    assert workspace.projected_solves == 1 and result[2] == 0


def test_failed_coarse_solve_refines_without_exceeding_iteration_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(budget=0)
    original = HYBRID._lsmr_method
    settings_seen = []

    def inaccurate_first(operator, rhs, **settings):
        settings_seen.append(settings)
        if len(settings_seen) == 1:
            return np.zeros(operator.shape[1]), 2, 1
        return original(operator, rhs, **settings)

    monkeypatch.setattr(HYBRID, "_lsmr_method", inaccurate_first)
    result = workspace.correction(1e5, 1e-8, 300, relative_tol=1e-3)
    assert len(settings_seen) == 2
    assert settings_seen[1]["atol"] < settings_seen[0]["atol"]
    assert settings_seen[1]["maxiter"] == 299
    assert workspace.refinements == 1
    assert result[3] <= 1e-3 and result[2] <= 300


def test_iteration_failure_is_not_hidden_by_optional_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(budget=0)

    def failed(operator, rhs, **settings):
        return np.ones(operator.shape[1]), 7, settings["maxiter"]

    monkeypatch.setattr(HYBRID, "_lsmr_method", failed)
    result = workspace.correction(100, 1e-8, 1, relative_tol=1e-3)
    assert result[1] == 7 and result[2] == 1 and np.isinf(result[3])
    assert workspace.refinements == 0


def test_optional_accuracy_preserves_descent_and_is_reported() -> None:
    workspace = _workspace()
    result = HYBRID.solve(
        workspace.index,
        workspace.U,
        workspace.residual.reshape(workspace.U.shape[:2]) / workspace.root[:, None],
        mass=workspace.root**2,
        lambda_prox=100,
        max_steps=4,
        dense_max_unknowns=0,
        hybrid_inner_rtol=1e-3,
    )
    diagnostics = result.diagnostics
    assert diagnostics["hybrid_inner_rtol"] == 1e-3
    assert diagnostics["normal_residual_ratio"] <= 1e-3
    loss = np.asarray(diagnostics["loss_history"])
    assert np.all(np.diff(loss) <= 64 * np.finfo(float).eps * max(1, loss[0]))
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-13)


@pytest.mark.parametrize("value", [0.0, -1.0, 0.2, np.nan, np.inf])
def test_invalid_optional_accuracy_fails_at_boundary(value: float) -> None:
    workspace = _workspace()
    with pytest.raises(ValueError, match="hybrid_inner_rtol"):
        HYBRID.solve(
            workspace.index,
            workspace.U,
            np.ones(workspace.U.shape[:2]),
            hybrid_inner_rtol=value,
        )


@pytest.mark.parametrize("mode,solver", [("multi", "cg"), ("manifold", "hybrid")])
def test_cli_rejects_ignored_optional_accuracy(mode: str, solver: str) -> None:
    args = build_parser().parse_args(
        ["--mode", mode, "--solver", solver, "--hybrid-inner-rtol", "0.001"]
    )
    with pytest.raises(ValueError, match="hybrid_inner_rtol"):
        _run(args)
