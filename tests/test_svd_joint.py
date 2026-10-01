from __future__ import annotations

import numpy as np
import pytest

from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.cli.main import _synthetic_data
from ADP.solver.SVD import _FlatU
from experiments.svd_joint_2026_09_30 import prototype as joint


def _problem(seed=955, d=7, m=3):
    rng = np.random.default_rng(seed)
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    U = rng.normal(size=(8, 5, d))
    U[0] = 0
    return (
        rng,
        P,
        U,
        rng.normal(size=(8, 5)),
        rng.normal(size=(8, m)),
        np.geomspace(0.01, 100, 8),
    )


def _dense(U, g, mass):
    return np.concatenate(
        [np.sqrt(mass[j]) * np.kron(g[j][None, :], U[j]) for j in range(len(U))]
    )


@pytest.mark.parametrize("ridge", [0.0, 0.7, 1e4])
@pytest.mark.parametrize("chunked", [False, True])
def test_full_core_matches_independent_augmented_lstsq(ridge, chunked, monkeypatch):
    rng, P, U, data, g, mass = _problem()
    U[..., -1] = U[..., 0] + 1e-12 * U[..., -1]
    A = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    V = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    W = U @ V
    if chunked:
        monkeypatch.setattr(joint, "_WORKSPACE_BYTES", 160)
    D = _dense(U, g, mass)
    # Build each matrix atom independently, including off-diagonal pairs.
    atoms = np.column_stack(
        [np.outer(A[:, i], V[:, k]).ravel() for i in range(2) for k in range(2)]
    )
    design = D @ atoms
    rhs = (np.sqrt(mass[:, None]) * data).ravel()
    expected = np.linalg.lstsq(
        np.vstack((design, np.sqrt(ridge) * atoms)),
        np.concatenate((rhs, np.sqrt(ridge) * P.ravel())),
        rcond=None,
    )[0].reshape(2, 2)
    actual, cert = joint._core(A, V, W, P, data, g, mass, ridge)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert cert < 1e-10
    # Nonzero off-diagonal core and cross-component Hessian.
    assert np.linalg.norm(actual - np.diag(np.diag(actual))) > 1e-3
    assert abs(design[:, 0] @ design[:, 3]) > 1e-3


def test_horizontal_gradients_and_variable_projection_finite_difference():
    rng, P, U, data, g, mass = _problem()
    A = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    V = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    M, _ = joint._core(A, V, U @ V, P, data, g, mass, 0.7)
    value, GA, GV, G = joint._evaluate(A, M, V, U @ V, P, data, g, mass, 0.7, _FlatU(U))
    D = _dense(U, g, mass)
    X = A @ M @ V.T
    rhs = (np.sqrt(mass[:, None]) * data).ravel()
    expected_G = (2 * D.T @ (D @ X.ravel() - rhs)).reshape(P.shape) + 1.4 * (X - P)
    np.testing.assert_allclose(G, expected_G, atol=1e-10)
    np.testing.assert_allclose(
        value, np.sum((D @ X.ravel() - rhs) ** 2) + 0.7 * np.sum((X - P) ** 2)
    )
    np.testing.assert_allclose(A.T @ GA, 0, atol=1e-10)
    np.testing.assert_allclose(V.T @ GV, 0, atol=1e-10)
    np.testing.assert_allclose(A.T @ G @ V, 0, atol=1e-10)
    dA, dV = rng.normal(size=A.shape), rng.normal(size=V.shape)
    dA -= A @ (A.T @ dA)
    dV -= V @ (V.T @ dV)

    def reduced(t):
        At = np.linalg.qr(A + t * dA)[0]
        Vt = np.linalg.qr(V + t * dV)[0]
        Mt, _ = joint._core(At, Vt, U @ Vt, P, data, g, mass, 0.7)
        return joint._evaluate(At, Mt, Vt, U @ Vt, P, data, g, mass, 0.7)

    eps = 1e-5
    fd = (reduced(eps) - reduced(-eps)) / (2 * eps)
    np.testing.assert_allclose(
        fd, np.sum(GA * dA) + np.sum(GV * dV), rtol=1e-7, atol=1e-7
    )
    R, S = (
        np.linalg.qr(rng.normal(size=(2, 2)))[0],
        np.linalg.qr(rng.normal(size=(2, 2)))[0],
    )
    rotated = joint._evaluate(
        A @ R, R.T @ M @ S, V @ S, (U @ V) @ S, P, data, g, mass, 0.7
    )
    np.testing.assert_allclose(rotated, value, rtol=1e-13)


@pytest.mark.parametrize("target", ["matrix", "correction"])
@pytest.mark.parametrize("ridge", [0.0, 0.7])
def test_joint_rank_descent_and_rotation_covariance(target, ridge):
    rng, P, U, I, g, mass = _problem()
    settings = {
        "rank": 2,
        "lambda_penalty": ridge,
        "maxiter": 30,
        "low_rank_target": target,
    }
    B, diag = joint.solve_fixed_coefficients(P, U, I, g, mass, **settings)
    D = _dense(U, g, mass)
    raw = np.sum(
        (D @ B.ravel() - (np.sqrt(mass[:, None]) * I).ravel()) ** 2
    ) + ridge * np.sum((B - P) ** 2)
    np.testing.assert_allclose(raw, diag["rank_objective_history"][-1], rtol=1e-12)
    assert np.all(np.diff(diag["rank_objective_history"]) <= 1e-10)
    assert np.linalg.matrix_rank(B - P if target == "correction" else B) <= 2
    O = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotated, _ = joint.solve_fixed_coefficients(O @ P, U, I, g @ O.T, mass, **settings)
    np.testing.assert_allclose(rotated, O @ B, rtol=1e-8, atol=1e-8)
    replay, replay_diag = joint.solve_fixed_coefficients(P, U, I, g, mass, **settings)
    np.testing.assert_array_equal(replay, B)
    assert replay_diag == diag


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_isotropic_objective_matches_global_truncated_svd(rank):
    rng = np.random.default_rng(333)
    m, d = 3, 6
    P = np.eye(d)[:m]
    U = np.tile(np.eye(d), (m, 1, 1))
    truth, g = rng.normal(size=(m, d)), np.eye(m)
    I = truth.copy()
    B, diag = joint.solve_fixed_coefficients(
        P, U, I, g, np.ones(m), rank=rank, lambda_penalty=0.7
    )
    left, values, right = np.linalg.svd((truth + 0.7 * P) / 1.7, full_matrices=False)
    expected = (left[:, :rank] * values[:rank]) @ right[:rank]
    np.testing.assert_allclose(B, expected, atol=1e-12)
    assert diag["joint_converged"]


@pytest.mark.parametrize("target", ["matrix", "correction"])
def test_joint_complete_adp_fit(target):
    X, y, _ = _synthetic_data(90, 5, 2, 0.05, 17)
    config = ADP_Config(N_J=18, N_phi=6, N_loc=20, outer_steps=2, h_min=1e-6)
    model = ADP_multi_index(
        2, config, ADP_solver(joint.solve, rank=1, maxiter=20, low_rank_target=target)
    ).fit(X, y)
    np.testing.assert_allclose(model.basis_.T @ model.basis_, np.eye(2), atol=1e-10)
    assert all(row["solver"]["optimizer"] == "joint_rank_r" for row in model.trace_)
    assert np.all(np.isfinite(model.transform(X)))


def test_joint_degenerate_and_validation():
    _, P, U, I, g, mass = _problem()
    for target in ("matrix", "correction"):
        B, diag = joint.solve_fixed_coefficients(
            P, U, I, g, mass, rank=0, lambda_penalty=0, low_rank_target=target
        )
        np.testing.assert_array_equal(
            B, P if target == "correction" else np.zeros_like(P)
        )
        assert diag["joint_stop_reason"] == "rank_zero"
    B, diag = joint.solve_fixed_coefficients(
        P, U, I, np.zeros_like(g), mass, rank=2, lambda_penalty=0
    )
    np.testing.assert_array_equal(B, np.zeros_like(P))
    assert diag["joint_stop_reason"] == "zero_gradient"
    for kwargs, exc in [
        ({"rank": True}, TypeError),
        ({"rank": 4}, ValueError),
        ({"maxiter": 0}, ValueError),
        ({"tol": np.nan}, ValueError),
        ({"lambda_penalty": -1}, ValueError),
        ({"low_rank_target": "bad"}, ValueError),
    ]:
        with pytest.raises(exc):
            joint.solve_fixed_coefficients(
                P, U, I, g, mass, **{"rank": 2, "lambda_penalty": 0.7, **kwargs}
            )
    with pytest.raises(ValueError, match="finite"):
        joint.solve_fixed_coefficients(
            P, U, I, g * np.nan, mass, rank=2, lambda_penalty=0.7
        )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises((ValueError, RuntimeError), match="finite"),
    ):
        joint.solve_fixed_coefficients(
            P, U, I * 1e200, g, mass, rank=2, lambda_penalty=0.7
        )
