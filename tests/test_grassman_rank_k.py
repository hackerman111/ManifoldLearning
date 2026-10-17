from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ADP.solver import grassman, grassman_exp


def _frame(rng: np.random.Generator, d: int, m: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(d, m)))[0]


def _problem(seed: int = 901, *, J: int = 12, p: int = 8, d: int = 11, m: int = 3):
    rng = np.random.default_rng(seed)
    truth, init = _frame(rng, d, m), _frame(rng, d, m)
    U = rng.normal(size=(J, p, d))
    I = np.einsum("jpm,jm->jp", U @ truth, rng.normal(size=(J, m)))
    I += 0.1 * rng.normal(size=I.shape)
    return init.T, U, I, np.geomspace(0.1, 8.0, J)


def _direct_refit(M: np.ndarray, I: np.ndarray, mass: np.ndarray, ridge: float):
    coeff = np.empty((len(M), M.shape[2]))
    value = 0.0
    for j in range(len(M)):
        if ridge:
            design = np.vstack((M[j], np.sqrt(ridge) * np.eye(M.shape[2])))
            target = np.r_[I[j], np.zeros(M.shape[2])]
        else:
            design, target = M[j], I[j]
        coeff[j] = np.linalg.lstsq(design, target, rcond=None)[0]
        resid = I[j] - M[j] @ coeff[j]
        value += mass[j] * (resid @ resid + ridge * (coeff[j] @ coeff[j]))
    return coeff, float(value)


def _direct_gradient(B: np.ndarray, U: np.ndarray, I: np.ndarray, mass: np.ndarray):
    Y = B.T
    M = U @ Y
    coeff, _ = _direct_refit(M, I, mass, 0.0)
    residual = I - np.einsum("jpm,jm->jp", M, coeff)
    adjoint = np.einsum("jpd,jp->jd", U, residual)
    ambient = -2.0 * np.einsum("jd,jm,j->dm", adjoint, coeff, mass)
    return ambient - Y @ (Y.T @ ambient)


def _block(*args: Any, **kwargs: Any):
    return grassman.block_rank_k(*args, **kwargs)


def test_block_rank_k_geometry_and_one_step_angle_cap() -> None:
    B, U, I, mass = _problem()
    result = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=2,
        rank_strategy="energy",
        max_steps=1,
        max_angle=0.12,
        energy_tol=0.0,
    )
    assert np.all(np.isfinite(result.index))
    np.testing.assert_allclose(
        result.index @ result.index.T, np.eye(B.shape[0]), atol=4e-12
    )
    angles = np.linalg.svd(B @ result.index.T, compute_uv=False)
    principal = np.arccos(np.clip(angles, -1.0, 1.0))
    assert np.max(principal) <= 0.12 + 2e-8
    assert max(cast(tuple[int, ...], result.diagnostics["step_ranks"]), default=0) <= 2


@pytest.mark.parametrize("ridge", [0.0, 0.35])
def test_block_profiled_result_uses_direct_refit_with_unequal_mass(
    ridge: float,
) -> None:
    B, U, I, mass = _problem(seed=903)
    result = _block(B, U, I, mass=mass, rank_k=2, local_ridge=ridge, max_steps=2)
    coeff, value = _direct_refit(U @ result.index.T, I, mass, ridge)
    np.testing.assert_allclose(result.coefficients, coeff, rtol=2e-9, atol=2e-10)
    assert result.diagnostics["profile_objective"] == pytest.approx(
        value, rel=2e-10, abs=2e-10
    )


def test_spectral_block_matches_exp_spectral_for_same_rank() -> None:
    B, U, I, mass = _problem(seed=907)
    opts = {
        "mass": mass,
        "rank_k": 2,
        "rank_strategy": "energy",
        "max_steps": 3,
        "energy_tol": 0.0,
    }
    got = _block(B, U, I, **opts, block_variant="spectral")
    expected = grassman_exp.solve(
        B, U, I, mass=mass, method="spectral", rank=2, max_steps=3, energy_tol=0.0
    )
    np.testing.assert_allclose(
        got.index.T @ got.index, expected.index.T @ expected.index, rtol=2e-8, atol=2e-8
    )
    assert got.diagnostics["profile_objective"] == pytest.approx(
        expected.diagnostics["profile_objective"], rel=2e-9, abs=2e-9
    )


def test_optimized_block_matches_flat_angle_budget() -> None:
    B, U, I, mass = _problem(seed=911)
    got = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=2,
        block_variant="optimized",
        block_max_steps=3,
        max_steps=2,
        energy_tol=0.0,
    )
    expected = grassman_exp.solve(
        B,
        U,
        I,
        mass=mass,
        method="flat_gn",
        rank=2,
        flat_max_steps=3,
        max_steps=2,
        energy_tol=0.0,
    )
    np.testing.assert_allclose(
        got.index.T @ got.index, expected.index.T @ expected.index, rtol=2e-8, atol=2e-8
    )


def test_core_default_signature_and_result_remain_unchanged() -> None:
    B, U, I, mass = _problem(seed=913)
    baseline = grassman.solve(B, U, I, mass=mass, max_steps=2)
    again = grassman.solve(B, U, I, mass=mass, max_steps=2)
    np.testing.assert_array_equal(baseline.index, again.index)
    assert baseline.diagnostics == again.diagnostics
    assert "method" not in __import__("inspect").signature(grassman.solve).parameters
    assert (
        "block_rank_k" not in __import__("inspect").signature(grassman.solve).parameters
    )


def test_energy_rank_selection_tracks_gradient_spectrum() -> None:
    B, U, I, mass = _problem(seed=919)
    energy_tol = 0.2
    result = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=None,
        rank_strategy="energy",
        max_steps=1,
        energy_tol=energy_tol,
    )
    ranks = cast(tuple[int, ...], result.diagnostics["step_ranks"])
    singular = np.linalg.svd(_direct_gradient(B, U, I, mass), compute_uv=False)
    available = min(
        B.shape[0],
        B.shape[1] - B.shape[0],
        int(
            np.count_nonzero(
                singular > np.finfo(float).eps * max(B.shape) * singular[0]
            )
        ),
    )
    expected = min(
        available,
        int(
            np.searchsorted(
                np.cumsum(singular**2),
                (1 - energy_tol) * np.sum(singular**2),
            )
        )
        + 1,
    )
    assert ranks == (expected,)
    assert result.diagnostics["rank_strategy"] == "energy"


def test_cost_probes_are_bounded_and_profile_history_is_monotone() -> None:
    B, U, I, mass = _problem(seed=923)
    result = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=None,
        rank_strategy="cost",
        block_variant="spectral",
        max_steps=4,
        energy_tol=0.0,
    )
    diag = cast(dict[str, Any], result.diagnostics)
    assert int(diag["cost_probe_evaluations"]) == 2
    assert [rank for rank, _ in diag["cost_scores"]] == [1, 3]
    history = np.asarray(cast(tuple[float, ...], diag["profile_history"]))
    assert np.all(np.diff(history) <= 1e-10)
    assert int(diag["profile_evaluations"]) >= int(diag["cost_probe_evaluations"])


def test_full_gradient_certificate_survives_rank_one_steps() -> None:
    B, U, I, mass = _problem(seed=929, d=12, m=4)
    result = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=1,
        rank_strategy="energy",
        max_steps=1,
        tol=1e-12,
        energy_tol=0.0,
    )
    full_gradient = _direct_gradient(result.index, U, I, mass)
    raw_norm = np.linalg.norm(full_gradient)
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    diag = cast(dict[str, Any], result.diagnostics)
    assert cast(tuple[int, ...], diag["step_ranks"]) == (1,)
    assert cast(int, diag["accepted_steps"]) == 1
    assert cast(float, diag["gradient_norm"]) == pytest.approx(
        raw_norm, rel=2e-8, abs=2e-9
    )
    assert cast(float, diag["riemannian_gradient"]) == pytest.approx(
        raw_norm / scale, rel=2e-8, abs=2e-10
    )
    assert raw_norm > 1e-12 * scale
    assert not cast(bool, diag["converged"])


@pytest.mark.parametrize("kind", ["rank_deficient", "near_collinear"])
def test_rank_stress_falls_back_to_finite_orthonormal_result(kind: str) -> None:
    B, U, I, mass = _problem(seed=937)
    if kind == "rank_deficient":
        U[0] = np.outer(np.arange(1, U.shape[1] + 1), np.arange(1, U.shape[2] + 1))
    else:
        U[:, :, -1] = U[:, :, 0] + 1e-11 * U[:, :, -1]
    result = _block(B, U, I, mass=mass, rank_k=2, max_steps=2)
    assert np.allclose(result.index @ result.index.T, np.eye(B.shape[0]), atol=5e-12)
    if kind == "rank_deficient":
        assert int(cast(int, result.diagnostics["rank_guard_fallbacks"])) > 0


def test_workspace_fallback_is_reported_and_valid() -> None:
    B, U, I, mass = _problem(seed=941)
    result = _block(B, U, I, mass=mass, rank_k=2, workspace_bytes=1, max_steps=2)
    assert int(cast(int, result.diagnostics["workspace_fallbacks"])) > 0
    assert result.diagnostics["block_variant"] == "optimized"
    assert np.allclose(result.index @ result.index.T, np.eye(B.shape[0]), atol=5e-12)


def test_block_rank_k_validation_and_full_space() -> None:
    rng = np.random.default_rng(947)
    U, I = rng.normal(size=(5, 7, 4)), rng.normal(size=(5, 7))
    full = _block(np.eye(4), U, I, rank_k=None)
    np.testing.assert_allclose(full.index.T @ full.index, np.eye(4), atol=2e-12)
    with pytest.raises(ValueError, match="rank_k"):
        _block(np.eye(4), U, I, rank_k=1)
    for invalid in (0, -1, 4):
        with pytest.raises(ValueError, match="rank_k"):
            _block(np.eye(2, 6), rng.normal(size=(5, 7, 6)), I, rank_k=invalid)
    with pytest.raises(ValueError, match="rank_strategy"):
        _block(np.eye(2, 6), rng.normal(size=(5, 7, 6)), I, rank_strategy="unknown")
    with pytest.raises(ValueError, match="block_variant"):
        _block(np.eye(2, 6), rng.normal(size=(5, 7, 6)), I, block_variant="unknown")


def test_failed_optimized_block_falls_back_to_spectral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    B, U, I, mass = _problem(seed=949)

    def fail_flat(state: Any, *args: Any, **kwargs: Any):
        return np.zeros(1), state, 0, 0, 0

    monkeypatch.setattr(grassman_exp, "_flat_optimize", fail_flat)
    result = _block(
        B,
        U,
        I,
        mass=mass,
        rank_k=2,
        block_variant="optimized",
        max_steps=1,
        energy_tol=0.0,
    )
    diag = cast(dict[str, Any], result.diagnostics)
    assert int(diag["flat_fallbacks"]) == 1
    assert cast(int, diag["accepted_steps"]) == 1
    assert (
        cast(tuple[float, ...], diag["profile_history"])[-1]
        <= cast(tuple[float, ...], diag["profile_history"])[0]
    )
    np.testing.assert_allclose(
        result.index @ result.index.T, np.eye(B.shape[0]), atol=5e-12
    )


def test_adp_solver_fit_current_accepts_block_rank_k_callback() -> None:
    from ADP import ADP_Config, ADP_multi_index, ADP_solver

    rng = np.random.default_rng(953)
    X = rng.normal(size=(72, 5))
    Y = X[:, :2] @ np.array([0.8, -0.4]) + 0.1 * rng.normal(size=72)
    callback = lambda index, U, I, **kw: _block(  # noqa: E731
        index, U, I, **kw, rank_k=1
    )
    model = ADP_multi_index(
        2, ADP_Config(N_J=12, N_phi=4, outer_steps=2), ADP_solver(callback, max_steps=2)
    ).fit(X, Y)
    basis = cast(np.ndarray, model.basis_)
    assert basis.shape == (5, 2) and np.allclose(basis.T @ basis, np.eye(2), atol=2e-11)
