from __future__ import annotations

import numpy as np
import pytest

from ADP.solver import LSMR
from experiments.multi_solver_certificate import evaluate


def test_reference_certificate_matches_hpao_and_basis_rotation() -> None:
    rng = np.random.default_rng(39)
    U = rng.normal(size=(8, 7, 5))
    I = rng.normal(size=(8, 7))
    mass = np.geomspace(0.1, 10.0, 8)
    B = np.linalg.qr(rng.normal(size=(5, 2)))[0].T
    reference = evaluate(B, U, I, mass)
    coefficients, ranks = LSMR._local_refit(I, U, B)
    loss = LSMR._loss(I, U, B, coefficients, mass)
    gradient, local, orthogonality = LSMR._stationarity(
        I, U, B, coefficients, mass, loss
    )
    assert reference.objective == pytest.approx(loss, abs=1e-12)
    assert reference.riemannian_gradient == pytest.approx(gradient, abs=1e-12)
    assert reference.local_gradient == pytest.approx(local, abs=1e-12)
    assert reference.orthogonality == pytest.approx(orthogonality, abs=1e-12)
    assert reference.rank_loss == np.count_nonzero(ranks < 2)

    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = evaluate(rotation @ B, U, I, mass)
    assert rotated.objective == pytest.approx(reference.objective, abs=1e-12)
    assert rotated.horizontal_gradient == pytest.approx(
        reference.horizontal_gradient, abs=1e-12
    )
    with pytest.raises(ValueError, match="mass must be positive"):
        evaluate(B, U, I, np.zeros(8))
