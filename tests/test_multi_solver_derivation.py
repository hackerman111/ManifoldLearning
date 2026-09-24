from __future__ import annotations

import numpy as np
import pytest

from experiments.multi_solver_certificate import evaluate
from experiments.multi_solver_derivation import evaluate_reduced


def _retract(B: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    raw = B + tangent
    eigenvalues, eigenvectors = np.linalg.eigh(raw @ raw.T)
    return ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T) @ raw


@pytest.mark.parametrize("m", [2, 5])
@pytest.mark.parametrize("structural_rank", [False, True])
def test_truncated_gradient_matches_tangent_finite_difference(
    m: int, structural_rank: bool
) -> None:
    rng = np.random.default_rng(93 + m)
    J, P, d = 4, 7, 8
    U = rng.normal(size=(J, P, d))
    if structural_rank:
        U[0] = np.outer(rng.normal(size=P), rng.normal(size=d))
        U[1] = 0
    I = rng.normal(size=(J, P))
    mass = np.geomspace(0.2, 3.0, J)
    B = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    tangent = rng.normal(size=B.shape)
    tangent -= (tangent @ B.T) @ B
    tangent /= np.linalg.norm(tangent)
    base = evaluate_reduced(B, U, I, mass)
    assert base.objective == pytest.approx(evaluate(B, U, I, mass).objective, abs=1e-11)
    derivative = float(np.sum(base.gradient * tangent))
    errors = []
    for step in (1e-3, 1e-4, 1e-5, 1e-6):
        plus = evaluate_reduced(_retract(B, step * tangent), U, I, mass).objective
        minus = evaluate_reduced(_retract(B, -step * tangent), U, I, mass).objective
        errors.append(abs((plus - minus) / (2 * step) - derivative))
    assert errors[1] < 1e-6 * max(1.0, abs(derivative))
    assert errors[1] < errors[0]

    rotation = np.linalg.qr(rng.normal(size=(m, m)))[0]
    rotated = evaluate_reduced(rotation @ B, U, I, mass)
    assert rotated.objective == pytest.approx(base.objective, abs=1e-11)
    np.testing.assert_allclose(rotated.gradient, rotation @ base.gradient, atol=1e-8)


def test_positive_discarded_singular_can_change_exact_minimum() -> None:
    B = np.eye(2)
    U = np.array([[[1.0, 0.0], [0.0, 1e-17]]])
    I = np.array([[0.0, 1.0]])
    result = evaluate_reduced(B, U, I, np.ones(1))
    assert result.objective == pytest.approx(0.5)
    assert result.value_defect == pytest.approx(0.5)
    assert result.value_defect > 1e-10 * max(1.0, result.objective)
    with pytest.raises(ValueError, match="mass must be positive"):
        evaluate_reduced(B, U, I, np.zeros(1))
