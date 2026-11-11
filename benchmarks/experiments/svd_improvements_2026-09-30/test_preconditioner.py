from __future__ import annotations

import numpy as np
import pytest

import ADP.solver.SVD as svd


def _fixture(ridge):
    rng = np.random.default_rng(774)
    J, p, d, m = 9, 4, 12, 3
    U = rng.normal(size=(J, p, d))
    if ridge:
        U *= np.geomspace(0.01, 100, d)
        U[..., 1] = U[..., 0] + 1e-7 * U[..., 1]
    U[..., -1] = 0
    P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    I = rng.normal(size=(J, p))
    g = rng.normal(size=(J, m))
    mass = np.geomspace(1e-3, 1e3, J)
    a = rng.normal(size=m)
    a /= np.linalg.norm(a)
    return rng, P, U, I, g, mass, a


@pytest.mark.parametrize("ridge", [0.0, 1e-4, 0.3])
@pytest.mark.parametrize("warm", [False, True])
def test_preconditioned_v_matches_dense_shifted_ridge(ridge, warm):
    rng, P, U, I, g, mass, a = _fixture(ridge)
    rows = np.repeat(np.sqrt(mass) * (g @ a), U.shape[1])
    design = rows[:, None] * U.reshape(-1, U.shape[2])
    target = (np.sqrt(mass)[:, None] * I).ravel()
    z = P.T @ a if ridge else np.zeros(U.shape[2])
    expected = np.linalg.lstsq(
        np.vstack((design, np.sqrt(ridge) * np.eye(U.shape[2]))),
        np.r_[target, np.sqrt(ridge) * z],
        rcond=None,
    )[0]
    flat = svd._FlatU(U)
    _, raw, _, _, _, args = svd._v_step(
        flat,
        target,
        P,
        g @ a,
        a,
        np.sqrt(mass),
        ridge,
        1e-8,
        None,
        x0=rng.normal(size=U.shape[2]) if warm else None,
        precondition=True,
        krylov_tol=1e-13,
    )
    np.testing.assert_allclose(raw, expected, rtol=1e-7, atol=1e-8)
    assert svd._v_certificate(flat, raw, *args, ridge=ridge) < 1e-7
    assert (flat.column_energy is None) == (ridge == 0)


def test_preconditioner_diagonal_stream_cache_and_augmented_adjoint(monkeypatch):
    rng, _, U, _, g, mass, a = _fixture(0.3)
    rows = np.repeat(np.sqrt(mass) * (g @ a), U.shape[1])
    flat = svd._FlatU(U)
    expected = np.sum((rows[:, None] * flat.flat) ** 2, axis=0)
    np.testing.assert_allclose(flat.normal_diagonal(rows), expected, rtol=1e-13)
    cached_passes = flat.passes
    np.testing.assert_allclose(flat.normal_diagonal(2 * rows), 4 * expected, rtol=1e-13)
    assert flat.passes == cached_passes
    monkeypatch.setattr(svd, "_DIRECT_WORKSPACE_BYTES", U.shape[2] * U.itemsize)
    streamed = svd._FlatU(U)
    np.testing.assert_allclose(streamed.normal_diagonal(rows), expected, rtol=1e-13)
    assert streamed.column_energy is None
    scale = 1 / np.sqrt(expected + 0.3)
    operator = svd._v_augmented_operator(flat, rows, np.sqrt(0.3), scale)
    dense = (
        np.vstack((rows[:, None] * flat.flat, np.sqrt(0.3) * np.eye(U.shape[2])))
        * scale
    )
    x = rng.normal(size=operator.shape[1])
    y = rng.normal(size=operator.shape[0])
    np.testing.assert_allclose(operator @ x, dense @ x, rtol=1e-12)
    np.testing.assert_allclose(operator.rmatvec(y), dense.T @ y, rtol=1e-12)
    np.testing.assert_allclose(x @ operator.rmatvec(y), (operator @ x) @ y, rtol=1e-12)


def test_preconditioner_keeps_original_certificate_and_iteration_limit():
    _, P, U, I, g, mass, _ = _fixture(0.3)
    with pytest.raises(RuntimeError, match="normal-residual check"):
        svd.solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=2,
            lambda_penalty=0.3,
            inner_maxiter=1,
            inner_tol=1e-8,
            lsmr_maxiter=1,
            direct_max_dimension=None,
            warm_start=False,
            precondition_v=True,
        )
    with pytest.raises(TypeError, match="precondition_v must be a boolean"):
        svd.solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=2,
            lambda_penalty=0.3,
            precondition_v=1,
        )


def test_nonfinite_original_certificate_is_rejected(monkeypatch):
    _, P, U, I, g, mass, _ = _fixture(0.3)
    monkeypatch.setattr(svd, "_v_certificate", lambda *args, **kwargs: float("nan"))
    with pytest.raises(RuntimeError, match="normal-residual check"):
        svd.solve_fixed_coefficients(
            P,
            U,
            I,
            g,
            mass,
            rank=2,
            lambda_penalty=0.3,
            precondition_v=True,
        )
