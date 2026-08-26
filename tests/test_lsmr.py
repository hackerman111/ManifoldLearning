from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "ADP" / "solver" / "LSMR.py"
SPEC = importlib.util.spec_from_file_location("adp_lsmr_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LSMR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LSMR
SPEC.loader.exec_module(LSMR)


def test_dense_hpao_contract() -> None:
    rng = np.random.default_rng(7)
    J, p, d, m = 9, 5, 4, 2
    U = rng.normal(size=(J, p, d))
    I = rng.normal(size=(J, p))
    mass = rng.uniform(0.5, 2.0, size=J)

    initial, _ = np.linalg.qr(rng.normal(size=(d, m)), mode="reduced")
    P = initial.T  # (m, d)
    coefficients, _ = LSMR._local_refit(I, U, P)
    operator = LSMR._linear_operator(U, coefficients, np.sqrt(mass), P.shape)

    delta = rng.normal(size=P.size)
    z = rng.normal(size=I.size)
    np.testing.assert_allclose(
        np.dot(operator @ delta, z),
        np.dot(delta, operator.rmatvec(z)),
        rtol=1e-12,
        atol=1e-12,
    )

    residual = np.sqrt(mass)[:, None] * (I - LSMR._predict(U, P, coefficients))
    dense_operator = np.column_stack([operator @ column for column in np.eye(P.size)])
    lambda_prox = 0.2
    expected = np.linalg.solve(
        dense_operator.T @ dense_operator + lambda_prox * np.eye(P.size),
        dense_operator.T @ residual.ravel(),
    )
    correction, *_ = LSMR._global_correction(
        I, U, P, coefficients, mass, lambda_prox, 1e-8, None
    )
    np.testing.assert_allclose(correction, expected, rtol=1e-9, atol=1e-10)

    raw = P + delta.reshape(P.shape) / 10.0
    P_gauge, coefficients_gauge, _ = LSMR._gauge_fix(raw, coefficients, P)
    fitted_raw = np.einsum("jpd,md,jm->jp", U, raw, coefficients, optimize=True)
    fitted_gauge = np.einsum(
        "jpd,md,jm->jp", U, P_gauge, coefficients_gauge, optimize=True
    )
    np.testing.assert_allclose(fitted_gauge, fitted_raw, rtol=1e-12, atol=1e-12)

    initial_loss = LSMR._loss(I, U, P, coefficients, mass)
    result = LSMR.solve(
        P,
        U,
        I,
        mass=mass,
        lambda_prox=0.2,
        max_steps=4,
        tol=1e-8,
    )
    final_loss = LSMR._loss(I, U, result.index, result.coefficients, mass)

    assert final_loss <= initial_loss + 1e-12
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=1e-12)
    assert result.diagnostics["accepted_steps"] >= 1
    assert result.diagnostics["normal_residual_ratio"] <= 0.1

    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)
    slopes, _ = LSMR._local_refit(I, U, beta)
    single_loss = LSMR._loss(I, U, beta, slopes, mass)
    beta_result = LSMR.solve(
        beta,
        U,
        I,
        mass=mass,
        lambda_prox=lambda_prox,
        max_steps=4,
        tol=1e-8,
    )
    assert (
        LSMR._loss(I, U, beta_result.index, beta_result.coefficients, mass)
        <= single_loss + 1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(beta_result.index), 1.0, atol=1e-12)
    assert LSMR.lsmr(beta, U, I, mass=mass, max_steps=1).shape == (d,)

    with pytest.raises(ValueError, match="nonzero finite norm"):
        LSMR.solve(np.zeros(d), U, I)
