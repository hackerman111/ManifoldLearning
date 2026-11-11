"""Batched preparation parity, including the numerical-nullspace guard."""

from dataclasses import replace

import numpy as np
import pytest

from ADP.core.ADP_Statistic import ADP_Statistics
from experiments import coupled_penalties as reference
from experiments import full_ridge_batched as direct


@pytest.mark.parametrize("kind", ["full", "singular", "weak", "nullspace", "zero"])
def test_batched_profile_matches_reference(kind):
    rng = np.random.default_rng(0)
    j, d, m = 7, 6, 2
    q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    eigenvalues = {
        "full": [3, 2, 1, 0.7, 0.3, 0.1],
        "singular": [2, 1, 0, 0, 0, 0],
        "weak": [2, 1, 0.1, 0.01, 1e-11, 1e-14],
        "nullspace": [2, 1, 0, 0, 0, 0],
        "zero": [0] * 6,
    }[kind]
    covariance = np.broadcast_to((q * eigenvalues) @ q.T, (j, d, d)).copy()
    cross = np.broadcast_to(q[:, 0] if kind != "zero" else np.zeros(d), (j, d)).copy()
    incoming = (
        q[:, 2:4] if kind == "nullspace" else np.linalg.qr(rng.normal(size=(d, m)))[0]
    )
    stats = ADP_Statistics(
        cross,
        covariance,
        np.ones(j),
        rng.normal(size=(j, d)),
        np.full(j, 20),
        np.empty((j, 0)),
        rng.normal(size=j),
    )
    white, _ = reference.full_moments(stats, covariance, cross)
    problem = reference.build(white, incoming, "full")
    candidate = direct.build(stats, incoming)
    assert candidate.tau == pytest.approx(problem.tau, rel=1e-13, abs=0)
    assert candidate.scale == pytest.approx(problem.scale, rel=1e-13)
    for basis in (incoming, np.linalg.qr(rng.normal(size=(d, m)))[0]):
        actual, expected = (
            direct.profile(basis, candidate),
            reference.profile(basis, problem),
        )
        assert actual.value == pytest.approx(expected.value, rel=1e-10, abs=1e-10)
        np.testing.assert_allclose(
            actual.coefficients, expected.coefficients, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            actual.gradient, expected.gradient, rtol=1e-9, atol=1e-9
        )
    if kind == "nullspace":
        assert candidate.info["whitened_fallback"]
    if kind == "full":
        assert not candidate.info["whitened_fallback"]


def test_batched_full_solve_matches_reference():
    rng = np.random.default_rng(975)
    covariance = rng.normal(size=(7, 6, 6))
    covariance = covariance @ covariance.swapaxes(1, 2) + np.eye(6)
    stats = ADP_Statistics(
        rng.normal(size=(7, 6)),
        covariance,
        np.ones(7),
        rng.normal(size=(7, 6)),
        np.full(7, 20),
        np.empty((7, 0)),
        rng.normal(size=7),
    )
    basis = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    actual = direct.solve(basis.T, stats, max_steps=5)
    expected = reference.solve(
        basis.T, stats, "full", moments=(stats.U, stats.I), max_steps=5
    )
    np.testing.assert_allclose(
        actual.index.T @ actual.index, expected.index.T @ expected.index, atol=1e-9
    )
    with pytest.raises(ValueError, match="semidefinite"):
        direct.build(replace(stats, U=-covariance), basis)
