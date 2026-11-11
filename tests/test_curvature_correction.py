"""Moment consistency and basis invariance of the isolated curvature correction."""

import numpy as np
import pytest

from ADP.engine.common.statistic import calculate_statistics
from ADP.solver import grassman
from experiments.local_polynomial import neighborhoods


def fixture(quadratic=True):
    rng = np.random.default_rng(72)
    x = rng.normal(size=(200, 6))
    basis = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    z = x @ basis
    hessian = rng.normal(size=(3, 3))
    hessian += hessian.T
    y = 0.4 + z @ rng.normal(size=3)
    if quadratic:
        y += 0.5 * np.einsum("ni,ij,nj->n", z, hessian, z)
    weights = rng.uniform(0.1, 1.0, size=(4, len(x)))
    phi = rng.normal(size=(4, 5, 6))
    stats = calculate_statistics(x, y, weights, phi, normalized=True)
    return basis, stats, neighborhoods(x, y, weights), phi


def test_removes_quadratic_moment_residual_at_truth():
    from experiments.curvature_correction import correct

    basis, stats, data, phi = fixture()
    before = grassman._profile(
        grassman._project(stats.U, basis), stats.I, stats.mass, 0.0
    )
    assert before.value > 1
    corrected = correct(basis, stats, data, phi, ridge=0.0)
    after = grassman._profile(
        grassman._project(corrected.U, basis), corrected.I, corrected.mass, 0.0
    )
    assert after.value < 1e-20
    assert (
        np.linalg.norm(grassman._gradient(basis, corrected.U, after, corrected.mass))
        < 1e-9
    )
    np.testing.assert_array_equal(corrected.U, stats.U)
    np.testing.assert_array_equal(corrected.mass, stats.mass)
    np.testing.assert_array_equal(corrected.S, stats.S)


@pytest.mark.parametrize("quadratic", [False, True])
def test_basis_invariance_and_linear_response_identity(quadratic):
    from experiments.curvature_correction import correct

    basis, stats, data, phi = fixture(quadratic)
    rotation = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))[0]
    corrected = correct(basis, stats, data, phi, ridge=0.0)
    rotated = correct(basis @ rotation, stats, data, phi, ridge=0.0)
    np.testing.assert_allclose(rotated.I, corrected.I, atol=1e-11)
    if not quadratic:
        np.testing.assert_allclose(corrected.I, stats.I, atol=1e-11)
