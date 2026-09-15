from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from ADP.solver import LSMR
from ADP.solver.HYBRID import RidgeWorkspace, design_matrix, solve
from ADP.solver.LSMR import _index_distance, _local_refit


def _problem(seed: int = 971) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(9, 6, 7))
    I = rng.normal(size=U.shape[:2])
    index = np.linalg.qr(rng.normal(size=(7, 2)))[0].T
    coefficients = rng.normal(size=(len(U), 2))
    mass = np.geomspace(0.01, 100, len(U))
    return U, I, index, coefficients, mass


@pytest.mark.parametrize("ridge", [0.0, 1e-8, 0.7, 1e5])
@pytest.mark.parametrize("degenerate", [False, True])
def test_trust_bound_never_rejects_feasible_solution(
    ridge: float, degenerate: bool
) -> None:
    U, I, index, coefficients, mass = _problem()
    if degenerate:
        U[..., -1] = U[..., 0] + 1e-12 * U[..., -1]
        coefficients[:, 1] = coefficients[:, 0]
    workspace = RidgeWorkspace(U, I, index, coefficients, mass, max_unknowns=0)
    A = design_matrix(U, coefficients, np.sqrt(mass))
    augmented = np.vstack((A, np.sqrt(ridge) * np.eye(index.size)))
    rhs = np.concatenate((workspace.residual, np.zeros(index.size)))
    correction = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    radius = float(np.linalg.norm(correction))
    assert not workspace.rejects_trust(ridge, radius, 1000)
    assert not workspace.rejects_trust(ridge, radius * 1.1, 1000)
    # Даже при недостаточном rank поиск направления даёт полезную границу.
    assert workspace.rejects_trust(ridge, min(1.0, radius) * 1e-3, 1000)
    assert workspace.calls == workspace.iterations == 0
    assert workspace.screened_trials == 1


def test_trust_under_determined_and_zero_rhs() -> None:
    U, I, index, coefficients, mass = _problem()
    U, I = U[:3, :2], I[:3, :2]
    coefficients, mass = coefficients[:3], mass[:3]
    workspace = RidgeWorkspace(U, I, index, coefficients, mass, max_unknowns=0)
    A = design_matrix(U, coefficients, np.sqrt(mass))
    correction = np.linalg.lstsq(A, workspace.residual, rcond=None)[0]
    assert not workspace.rejects_trust(0, float(np.linalg.norm(correction)), 1000)
    coefficients[:] = 0
    workspace = RidgeWorkspace(U, I, index, coefficients, mass, max_unknowns=0)
    assert not workspace.rejects_trust(1.0, 0.1, 1000)
    assert workspace._trust_basis is None
    assert workspace.correction(1.0, 1e-8, 1000)[3] == 0


@pytest.mark.parametrize("limit", [0, 1, 4])
def test_trust_memory_and_work_limits(limit: int) -> None:
    U, I, index, coefficients, mass = _problem()
    workspace = RidgeWorkspace(
        U,
        I,
        index,
        coefficients,
        mass,
        max_unknowns=0,
        max_bytes=48 * index.size * limit,
    )
    workspace.rejects_trust(1, 0.1, 2)
    if limit == 0:
        assert workspace._trust_basis is None
        assert workspace.screening_matvecs == 0
    else:
        assert workspace._trust_basis is not None
        assert workspace._trust_basis.shape[1] <= min(limit, 2)
        assert workspace.screening_matvecs <= min(limit, 2) + 1


def test_screening_preserves_hpao_path() -> None:
    U, I, index, _, mass = _problem()
    settings = {
        "mass": mass,
        "max_steps": 3,
        "lambda_prox": 0.01,
        "tol": 1e-8,
        "dense_max_unknowns": 0,
        "trust_radius": 0.1,
    }
    reference = solve(index, U, I, dense_max_bytes=0, **settings)
    result = solve(index, U, I, **settings)
    assert result.diagnostics["linear_screened_trials"] > 0
    assert (
        result.diagnostics["lambda_history"] == reference.diagnostics["lambda_history"]
    )
    np.testing.assert_allclose(result.index, reference.index, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        result.diagnostics["loss_history"],
        reference.diagnostics["loss_history"],
        rtol=1e-9,
        atol=1e-10,
    )
    assert result.diagnostics["normal_residual_ratio"] < 0.1
    assert (
        result.diagnostics["linear_iterations_total"]
        < reference.diagnostics["linear_iterations_total"]
    )


def test_workspace_actions_are_adjoint_and_do_not_alias_buffers() -> None:
    U, I, index, coefficients, mass = _problem()
    U = U[..., ::-1]
    workspace = RidgeWorkspace(U, I, index, coefficients, mass, max_unknowns=0)
    A = design_matrix(U, coefficients, np.sqrt(mass))
    rng = np.random.default_rng(918)
    x, y = rng.normal(size=index.size), rng.normal(size=I.size)
    forward = workspace.matvec(x)
    reverse = workspace.rmatvec(y)
    workspace.matvec(2 * x)
    workspace.rmatvec(3 * y)
    np.testing.assert_allclose(forward, A @ x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(reverse, A.T @ y, rtol=1e-13, atol=1e-13)
    assert forward @ y == pytest.approx(x @ reverse, rel=1e-13)


@pytest.mark.parametrize("perturbation", [0.0, 1e-10, 0.1])
def test_subspace_step_matches_dense_projectors(perturbation: float) -> None:
    rng = np.random.default_rng(314)
    prior = np.linalg.qr(rng.normal(size=(17, 3)))[0].T
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    index = np.linalg.qr(
        (rotation @ prior + perturbation * rng.normal(size=prior.shape)).T
    )[0].T
    reference = np.linalg.norm(index.T @ index - prior.T @ prior) / np.sqrt(2)
    assert _index_distance(index, prior) == pytest.approx(
        reference, rel=1e-7, abs=2e-15
    )


def test_subspace_step_memory_is_linear_in_d() -> None:
    rng = np.random.default_rng(17)
    prior = np.linalg.qr(rng.normal(size=(1000, 3)))[0].T
    index = np.linalg.qr(rng.normal(size=(1000, 3)))[0].T
    tracemalloc.start()
    try:
        _index_distance(index, prior)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Один (1000,1000) float64 массив потребовал бы уже 8 MB.
    assert peak < 256 * 1024


@pytest.mark.parametrize("shape", [(0, 3, 5), (3, 0, 5), (3, 5, 0)])
def test_empty_statistics_fail_at_boundary(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        solve(np.ones((2, shape[-1])), np.empty(shape), np.empty(shape[:2]))


@pytest.mark.parametrize("P,m", [(20, 2), (20, 10), (3, 7)])
@pytest.mark.parametrize("budget", [1, 8192])
def test_batched_local_refit_preserves_rank_and_minimum_norm(
    P: int, m: int, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LSMR, "_LOCAL_REFIT_BYTES", budget)
    rng = np.random.default_rng(291)
    J, d = 13, 17
    U = rng.normal(size=(J, P, d))
    U[0] = 0
    U[1, 1:] = U[1, :1]
    U[2, 1:] = U[2, :1] + 1e-12 * U[2, 1:]
    index = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    # Почти вырожденный центр получает согласованный rhs без огромного решения.
    I = (U @ rng.normal(size=d))[..., None].squeeze(-1)
    I[3:] += rng.normal(size=I[3:].shape)
    expected, expected_rank = [], []
    for j, projected in enumerate(U @ index.T):
        solution, _, rank, _ = np.linalg.lstsq(projected, I[j], rcond=None)
        expected.append(solution)
        expected_rank.append(rank)
    coefficients, ranks = _local_refit(I, U, index)
    np.testing.assert_array_equal(ranks, expected_rank)
    # Чувствительные центры должны пройти исходный lstsq fallback.
    fitted = (U @ index.T @ coefficients[..., None]).squeeze(-1)
    reference = (U @ index.T @ np.asarray(expected)[..., None]).squeeze(-1)
    np.testing.assert_allclose(fitted, reference, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(coefficients, expected, rtol=1e-12, atol=1e-12)
