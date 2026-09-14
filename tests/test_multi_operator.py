from __future__ import annotations

import numpy as np
import pytest

from ADP.solver._multi_operator import adjoint, forward
from ADP.solver.CG import _normal_operator
from ADP.solver.LSMR import _linear_operator


@pytest.mark.parametrize("ridge", [0.0, 1e-10, 1e3])
@pytest.mark.parametrize("strided", [False, True])
def test_multi_actions_match_explicit_design(ridge: float, strided: bool) -> None:
    """Joint cross-terms и adjoint сохраняются при разных массах и strides."""
    rng = np.random.default_rng(781)
    J, P, d, m = 7, 5, 6, 2
    U = rng.normal(size=(J, P, d * 2))
    U = U[..., ::2] if strided else U[..., :d].copy()
    U[..., -1] = U[..., 0] + 1e-10 * U[..., -1]
    coefficients = rng.normal(size=(J, m))
    coefficients[0] *= 1e-10
    mass = np.geomspace(1e-6, 1e6, J)
    matrix = rng.normal(size=(m, d))
    data = rng.normal(size=(J, P))
    # Малый прямой эталон: столбцы соответствуют row-major vec(B).
    design = np.array(
        [
            [coefficients[j, a] * U[j, p, r] for a in range(m) for r in range(d)]
            for j in range(J)
            for p in range(P)
        ]
    )
    np.testing.assert_allclose(
        forward(U, matrix, coefficients).ravel(),
        design @ matrix.ravel(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint(U, data, coefficients).ravel(),
        design.T @ data.ravel(),
        rtol=1e-12,
        atol=1e-12,
    )
    operator = _linear_operator(U, coefficients, np.sqrt(mass), matrix.shape)
    weighted = np.repeat(np.sqrt(mass), P)[:, None] * design
    np.testing.assert_allclose(operator @ matrix.ravel(), weighted @ matrix.ravel())
    np.testing.assert_allclose(
        operator.rmatvec(data.ravel()), weighted.T @ data.ravel()
    )
    assert np.vdot(operator @ matrix.ravel(), data.ravel()) == pytest.approx(
        np.vdot(matrix.ravel(), operator.rmatvec(data.ravel())), rel=1e-12, abs=1e-12
    )
    normal, _ = _normal_operator(U, coefficients, mass, ridge)
    actual = normal @ matrix.ravel()
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(
        actual,
        weighted.T @ (weighted @ matrix.ravel()) + ridge * matrix.ravel(),
        rtol=1e-12,
        atol=1e-8,
    )
