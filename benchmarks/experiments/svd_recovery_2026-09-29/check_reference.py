"""Small augmented dense reference, adjoint and degeneracy checks."""

import json

import numpy as np
from prototype import fixed, solve

from ADP.solver.LSMR import _linear_operator

rng = np.random.default_rng(72999)
records = []
for ridge in (0.0, 1e-4, 0.5):
    for degenerate in (False, True):
        J, p, m, d = 13, 5, 3, 8
        P = np.linalg.qr(rng.normal(size=(d, m)))[0].T
        U = rng.normal(size=(J, p, d))
        if degenerate:
            U[..., -1] = U[..., 0]
        g = rng.normal(size=(J, m))
        mass = np.exp(rng.uniform(-4, 4, J))
        mass[0] = 0
        I = rng.normal(size=(J, p))
        operator = _linear_operator(U, g, np.sqrt(mass), P.shape)
        dense = np.einsum("j,jpD,jM->jpMD", np.sqrt(mass), U, g).reshape(J * p, m * d)
        x, y = rng.normal(size=m * d), rng.normal(size=J * p)
        np.testing.assert_allclose(operator @ x, dense @ x, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvec(y), dense.T @ y, atol=1e-12)
        rhs = (np.sqrt(mass)[:, None] * I).ravel() - dense @ P.ravel()
        reference = (
            np.linalg.lstsq(
                np.vstack((dense, np.sqrt(ridge) * np.eye(m * d))),
                np.concatenate((rhs, np.zeros(m * d))),
                rcond=None,
            )[0].reshape(m, d)
            + P
        )
        B, diagnostics = fixed(P, U, I, g, mass, ridge)
        np.testing.assert_allclose(B, reference, rtol=2e-7, atol=2e-8)
        result = solve(P, U, I, mass=np.maximum(mass, 1e-8), lambda_prox=ridge)
        np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=1e-12)
        records.append(
            {
                "ridge": ridge,
                "degenerate": degenerate,
                "error": float(np.linalg.norm(B - reference)),
                **diagnostics,
            }
        )
print(json.dumps(records, indent=2))
