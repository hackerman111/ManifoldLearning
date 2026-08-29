from types import SimpleNamespace

import numpy as np
import pytest

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_Solver
from ADP.solver import CG
from ADP.solver.LSMR import _gauge_fix, _local_refit


def test_cg_step_matches_dense_reference() -> None:
    rng = np.random.default_rng(17)
    J, p, d = 11, 5, 6
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = rng.uniform(0.5, 2.0, size=J)
    prior = rng.normal(size=d)
    prior /= np.linalg.norm(prior)
    lambda_prox = 0.3

    projected = U @ prior
    slopes = np.einsum("jp,jp->j", I, projected) / np.einsum(
        "jp,jp->j", projected, projected
    )
    weighted_slopes2 = mass * np.square(slopes)
    dense = np.einsum("j,jpd,jpe->de", weighted_slopes2, U, U, optimize=True)
    dense += lambda_prox * np.eye(d)
    rhs = np.einsum("j,jpd,jp->d", mass * slopes, U, I, optimize=True)
    rhs += lambda_prox * prior
    expected = np.linalg.solve(dense, rhs)
    expected /= np.linalg.norm(expected)
    if np.dot(expected, prior) < 0:
        expected = -expected

    operator, _ = CG._normal_operator(U, slopes, mass, lambda_prox)
    left = rng.normal(size=d)
    right = rng.normal(size=d)
    np.testing.assert_allclose(
        np.dot(operator @ left, right),
        np.dot(left, operator.rmatvec(right)),
        rtol=1e-13,
        atol=1e-13,
    )

    result = CG.solve(
        prior,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=1,
        tol=1e-10,
        cg_maxiter=100,
    )

    np.testing.assert_allclose(result.index, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(np.linalg.norm(result.index), 1.0, atol=1e-13)
    assert result.diagnostics["relative_linear_residual"] <= 1e-9
    assert (
        result.diagnostics["loss_history"][-1] <= result.diagnostics["loss_history"][0]
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        CG.solve(
            prior,
            U,
            I,
            mass=mass,
            lambda_prox=lambda_prox,
            max_steps=1,
            tol=1e-12,
            cg_maxiter=1,
        )


def test_public_solver_accepts_cg_beta_step() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    beta = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    data = SimpleNamespace(
        X=X,
        Y=np.sin(X @ beta),
        beta_init=np.array([1.0, 0.0, 0.0]),
        x_j=X[:8],
    )
    config = ADP_Config(
        N_loc=6,
        N_phi=3,
        outer_steps=1,
        h_min=1e6,
        batch_size=4,
    )

    estimate = ADP_Solver(config, data, solver=CG.cg).fit()

    assert estimate.shape == (3,)
    assert np.all(np.isfinite(estimate))
    np.testing.assert_allclose(np.linalg.norm(estimate), 1.0, atol=1e-12)


def test_multi_cg_step_matches_joint_dense_reference() -> None:
    rng = np.random.default_rng(23)
    J, p, d, m = 9, 5, 6, 2
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = rng.uniform(0.5, 2.0, size=J)
    prior, _ = np.linalg.qr(rng.normal(size=(d, m)), mode="reduced")
    prior = prior.T
    lambda_prox = 0.4
    coefficients, _ = _local_refit(I, U, prior)

    design = np.einsum("jm,jpd->jpmd", coefficients, U, optimize=True).reshape(
        J * p, m * d
    )
    row_mass = np.repeat(mass, p)
    dense = design.T @ (row_mass[:, None] * design)
    dense += lambda_prox * np.eye(m * d)
    rhs = design.T @ (row_mass * I.ravel())
    rhs += lambda_prox * prior.ravel()
    expected_raw = np.linalg.solve(dense, rhs).reshape(m, d)

    operator, preconditioner = CG._normal_operator(U, coefficients, mass, lambda_prox)
    left = rng.normal(size=m * d)
    right = rng.normal(size=m * d)
    np.testing.assert_allclose(
        np.dot(operator @ left, right),
        np.dot(left, operator.rmatvec(right)),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        preconditioner @ left,
        left / np.diag(dense),
        rtol=1e-13,
        atol=1e-13,
    )

    raw, info, _, residual = CG._global_step(
        I,
        U,
        coefficients,
        mass,
        prior,
        lambda_prox,
        1e-10,
        200,
    )
    np.testing.assert_allclose(raw, expected_raw, rtol=1e-10, atol=1e-11)
    assert info == 0
    assert residual <= 1e-9

    result = CG.solve(
        prior,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=1,
        tol=1e-10,
        cg_maxiter=200,
    )
    expected, _, rank = _gauge_fix(expected_raw, coefficients, prior)
    assert rank == m
    np.testing.assert_allclose(result.index, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=1e-12)
    assert result.diagnostics["relative_linear_residual"] <= 1e-9
    assert (
        result.diagnostics["loss_history"][-1] <= result.diagnostics["loss_history"][0]
    )

    rotation, _ = np.linalg.qr(rng.normal(size=(m, m)))
    rotated_raw, _, _, _ = CG._global_step(
        I,
        U,
        coefficients @ rotation.T,
        mass,
        rotation @ prior,
        lambda_prox,
        1e-10,
        200,
    )
    np.testing.assert_allclose(rotated_raw, rotation @ raw, rtol=1e-10, atol=1e-11)
    rotated_index = CG.cg(
        rotation @ prior,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=1,
        tol=1e-10,
        cg_maxiter=200,
    )
    np.testing.assert_allclose(
        rotated_index.T @ rotated_index,
        result.index.T @ result.index,
        rtol=1e-10,
        atol=1e-11,
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        CG.solve(
            prior,
            U,
            I,
            mass=mass,
            lambda_prox=lambda_prox,
            max_steps=1,
            tol=1e-12,
            cg_maxiter=1,
        )

    with pytest.raises(ValueError, match="lambda_prox must be finite and positive"):
        CG.solve(prior, U, I, mass=mass, lambda_prox=0)

    rank_deficient = np.repeat(prior[:1], m, axis=0)
    with pytest.raises(ValueError, match="full row rank"):
        CG.solve(rank_deficient, U, I, mass=mass, lambda_prox=lambda_prox)
