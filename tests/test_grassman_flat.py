from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ADP.solver import grassman_exp as solver


def _orthogonal_frame(
    rng: np.random.Generator, d: int, m: int, q: int
) -> tuple[np.ndarray, np.ndarray]:
    frame = np.linalg.qr(rng.normal(size=(d, m + q)))[0]
    return frame[:, :m], frame[:, m:]


def _ambient_exp(
    Y: np.ndarray, V: np.ndarray, At: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Direct product of q principal-plane rotations, independent of cache code."""
    YA = Y @ At.T
    return Y + (YA * (np.cos(theta) - 1.0) + V * np.sin(theta)) @ At


def _refit(
    M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float
) -> tuple[float, np.ndarray, np.ndarray]:
    J, _, m = M.shape
    coefficients = np.empty((J, m))
    residual = np.empty_like(I)
    value = 0.0
    for j in range(J):
        if tau:
            design = np.vstack((M[j], np.sqrt(tau) * np.eye(m)))
            target = np.r_[I[j], np.zeros(m)]
        else:
            design, target = M[j], I[j]
        coefficients[j] = np.linalg.lstsq(design, target, rcond=None)[0]
        residual[j] = I[j] - M[j] @ coefficients[j]
        value += mass[j] * (
            residual[j] @ residual[j] + tau * (coefficients[j] @ coefficients[j])
        )
    return float(value), coefficients, residual


def _profile_reference(
    M: np.ndarray, I: np.ndarray, mass: np.ndarray, tau: float
) -> np.ndarray:
    return _refit(M, I, mass, tau)[1]


def test_cached_design_matches_independent_ambient_exp_and_refit() -> None:
    rng = np.random.default_rng(55101)
    J, p, d, m, q = 7, 9, 13, 3, 2
    Y, V = _orthogonal_frame(rng, d, m, q)
    At = np.linalg.qr(rng.normal(size=(m, q)))[0].T
    U = rng.normal(size=(J, p, d))
    M, W = U @ Y, U @ V
    theta = np.array([0.21, -0.34])
    rotated_M = M @ At.T

    got = solver._flat_design(M, rotated_M, W, At, theta)
    expected_basis = _ambient_exp(Y, V, At, theta)
    expected_design = U @ expected_basis
    np.testing.assert_allclose(got, expected_design, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(expected_basis.T @ expected_basis, np.eye(m), atol=3e-14)

    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.2, 5.0, J)
    for tau in (0.0, 0.3):
        expected_value, expected_g, expected_r = _refit(got, I, mass, tau)
        state = solver._profile(got, I, mass, tau)
        assert state.value == pytest.approx(expected_value, rel=3e-12, abs=3e-12)
        np.testing.assert_allclose(
            state.coefficients, expected_g, rtol=3e-11, atol=3e-12
        )
        np.testing.assert_allclose(state.residual, expected_r, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("tau", [0.0, 0.25])
def test_full_q_angle_ridge_jacobian_matches_finite_difference(tau: float) -> None:
    rng = np.random.default_rng(55800 + int(100 * tau))
    J, p, d, m, q = 6, 10, 15, 3, 2
    Y, V = _orthogonal_frame(rng, d, m, q)
    At = np.linalg.qr(rng.normal(size=(m, q)))[0].T
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = np.geomspace(0.05, 8.0, J)
    M, W = U @ Y, U @ V
    theta = np.array([0.17, -0.23])
    rotated_M = M @ At.T
    Mtheta = solver._flat_design(M, rotated_M, W, At, theta)
    state = solver._profile(Mtheta, I, mass, tau)
    D = -rotated_M * np.sin(theta)[None, None, :] + W * np.cos(theta)[None, None, :]
    jac, packed = solver._flat_jacobian(state, D, At, mass, tau)

    expected_residual = []
    for j in range(J):
        block = state.residual[j]
        if tau:
            block = np.r_[block, -np.sqrt(tau) * state.coefficients[j]]
        expected_residual.extend(np.sqrt(mass[j]) * block)
    np.testing.assert_allclose(packed, expected_residual, rtol=2e-13, atol=2e-13)

    epsilon = 2e-6
    fd = np.empty_like(jac)
    for ell in range(q):
        direction = np.zeros(q)
        direction[ell] = epsilon
        packed_sides = []
        for sign in (-1.0, 1.0):
            Mside = solver._flat_design(M, rotated_M, W, At, theta + sign * direction)
            _, _, residual = _refit(Mside, I, mass, tau)
            coefficients = _profile_reference(Mside, I, mass, tau)
            blocks = []
            for j in range(J):
                block = residual[j]
                if tau:
                    block = np.r_[block, -np.sqrt(tau) * coefficients[j]]
                blocks.extend(np.sqrt(mass[j]) * block)
            packed_sides.append(np.asarray(blocks))
        fd[:, ell] = (packed_sides[1] - packed_sides[0]) / (2 * epsilon)
    np.testing.assert_allclose(jac, fd, rtol=2e-6, atol=2e-7)


def test_angle_pi_shift_preserves_projector_and_profile() -> None:
    rng = np.random.default_rng(55117)
    J, p, d, m, q = 5, 8, 12, 3, 2
    Y, V = _orthogonal_frame(rng, d, m, q)
    At = np.linalg.qr(rng.normal(size=(m, q)))[0].T
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    mass = np.geomspace(0.1, 3.0, J)
    M, W = U @ Y, U @ V
    rotated_M = M @ At.T
    theta = np.array([0.31, -0.14])
    shifted = theta.copy()
    shifted[1] += np.pi
    left = _ambient_exp(Y, V, At, theta)
    right = _ambient_exp(Y, V, At, shifted)
    np.testing.assert_allclose(left @ left.T, right @ right.T, atol=2e-13)
    for tau in (0.0, 0.4):
        profile_left = solver._profile(
            solver._flat_design(M, rotated_M, W, At, theta), I, mass, tau
        )
        profile_right = solver._profile(
            solver._flat_design(M, rotated_M, W, At, shifted), I, mass, tau
        )
        assert profile_left.value == pytest.approx(profile_right.value, abs=2e-11)


def test_profile_is_mass_scale_and_basis_gauge_invariant_with_ridge() -> None:
    rng = np.random.default_rng(55123)
    J, p, m = 9, 11, 3
    M, I = rng.normal(size=(J, p, m)), rng.normal(size=(J, p))
    mass = np.geomspace(1e-2, 1e2, J)
    tau = 0.2
    state = solver._profile(M, I, mass, tau)
    scaled = solver._profile(M, I, 7.5 * mass, tau)
    assert scaled.value == pytest.approx(7.5 * state.value, rel=2e-13)

    Q = np.linalg.qr(rng.normal(size=(m, m)))[0]
    gauged = solver._profile(M @ Q, I, mass, tau)
    assert gauged.value == pytest.approx(state.value, rel=3e-12, abs=3e-12)
    np.testing.assert_allclose(gauged.coefficients, state.coefficients @ Q, atol=2e-11)


def test_flat_gn_solver_is_covariant_to_initial_row_basis_gauge() -> None:
    rng = np.random.default_rng(55127)
    J, p, d, m = 24, 9, 14, 3
    truth, _ = _orthogonal_frame(rng, d, m, d - m)
    init, _ = _orthogonal_frame(rng, d, m, d - m)
    U = rng.normal(size=(J, p, d))
    I = np.einsum("jpm,jm->jp", U @ truth, rng.normal(size=(J, m)))
    I += 0.1 * rng.normal(size=(J, p))
    mass = np.geomspace(0.2, 4.0, J)
    Q = np.linalg.qr(rng.normal(size=(m, m)))[0]
    options = {
        "mass": mass,
        "method": "flat_gn",
        "local_ridge": 0.15,
        "max_steps": 3,
        "energy_tol": 0.0,
    }
    left = solver.solve(init.T, U, I, **options)
    right = solver.solve(Q @ init.T, U, I, **options)
    left_projector = left.index.T @ left.index
    right_projector = right.index.T @ right.index
    np.testing.assert_allclose(left_projector, right_projector, rtol=2e-8, atol=2e-8)
    assert left.diagnostics["profile_objective"] == pytest.approx(
        right.diagnostics["profile_objective"], rel=2e-9, abs=2e-9
    )


def test_flat_gn_updates_multiple_angles_and_keeps_monotone_orthonormal_result() -> (
    None
):
    rng = np.random.default_rng(55129)
    J, p, d, m = 18, 8, 12, 2
    truth, _ = _orthogonal_frame(rng, d, m, d - m)
    init, _ = _orthogonal_frame(rng, d, m, d - m)
    U = rng.normal(size=(J, p, d))
    coefficients = rng.normal(size=(J, m))
    I = np.einsum("jpm,jm->jp", U @ truth, coefficients) + 0.15 * rng.normal(
        size=(J, p)
    )
    result = solver.solve(
        init.T,
        U,
        I,
        mass=np.geomspace(0.5, 2.0, J),
        method="flat_gn",
        local_ridge=0.2,
        max_steps=3,
        flat_max_steps=8,
        energy_tol=0.0,
    )
    diagnostics = result.diagnostics
    assert int(cast(Any, diagnostics["flat_inner_iterations"])) > 0
    assert max(cast(tuple[int, ...], diagnostics["step_ranks"])) > 1
    angles = cast(tuple[tuple[float, ...], ...], diagnostics["flat_angles"])
    assert any(np.count_nonzero(np.abs(row) > 1e-8) > 1 for row in angles)
    history = np.asarray(diagnostics["profile_history"])
    assert np.all(np.diff(history) <= 1e-11)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=3e-12)
    assert np.all(np.isfinite(result.index))


def test_flat_gn_workspace_and_rank_fallbacks_preserve_valid_results() -> None:
    rng = np.random.default_rng(55131)
    J, p, d, m = 8, 7, 9, 2
    B = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    constrained = solver.solve(
        B, U, I, method="flat_gn", workspace_bytes=1, max_steps=2
    )
    assert int(cast(Any, constrained.diagnostics["workspace_fallbacks"])) > 0
    np.testing.assert_allclose(
        constrained.index @ constrained.index.T, np.eye(m), atol=3e-12
    )

    rank_deficient = rng.normal(size=U.shape)
    rank_deficient[0] = rng.normal(size=(1, 1, d)) * rng.normal(size=(1, p, 1))
    rank_result = solver.solve(B, rank_deficient, I, method="flat_gn", max_steps=1)
    assert int(cast(Any, rank_result.diagnostics["rank_guard_fallbacks"])) > 0
    np.testing.assert_allclose(
        rank_result.index @ rank_result.index.T, np.eye(m), atol=3e-12
    )


def test_flat_gn_full_space_and_option_validation() -> None:
    rng = np.random.default_rng(55137)
    d, J, p = 4, 6, 5
    B = np.eye(d)
    U, I = rng.normal(size=(J, p, d)), rng.normal(size=(J, p))
    full = solver.solve(B, U, I, method="flat_gn", flat_max_steps=3)
    np.testing.assert_allclose(full.index.T @ full.index, np.eye(d), atol=2e-12)
    assert full.diagnostics["converged"]

    with pytest.raises(ValueError, match="method"):
        solver.solve(B[:2], U, I, method="unknown")
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="flat_max_steps"):
            solver.solve(B[:2], U, I, method="flat_gn", flat_max_steps=invalid)
    with pytest.raises(TypeError, match="flat_max_steps"):
        solver.solve(B[:2], U, I, method="flat_gn", flat_max_steps=cast(Any, 1.5))


def test_ambient_projection_count_does_not_scale_with_flat_inner_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(55141)
    J, p, d, m = 12, 8, 10, 2
    truth, _ = _orthogonal_frame(rng, d, m, d - m)
    init, _ = _orthogonal_frame(rng, d, m, d - m)
    U = rng.normal(size=(J, p, d))
    I = np.einsum("jpm,jm->jp", U @ truth, rng.normal(size=(J, m)))
    project = solver._project
    count = 0

    def counted_project(operator: np.ndarray, basis: np.ndarray) -> np.ndarray:
        nonlocal count
        count += 1
        return project(operator, basis)

    monkeypatch.setattr(solver, "_project", counted_project)
    counts = []
    results = []
    for flat_steps in (1, 8):
        count = 0
        result = solver.solve(
            init.T,
            U,
            I,
            method="flat_gn",
            local_ridge=0.1,
            max_steps=3,
            flat_max_steps=flat_steps,
        )
        counts.append(count)
        results.append(result)
        assert count <= int(cast(Any, result.diagnostics["accepted_steps"])) + 2
    assert counts[0] == counts[1]
    assert cast(tuple[float, ...], results[0].diagnostics["profile_history"])[-1] >= 0
