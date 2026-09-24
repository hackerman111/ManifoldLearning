from __future__ import annotations

import numpy as np
import pytest

from ADP.solver.LSMR import solve as solve_lsmr
from experiments.multi_solver_certificate import evaluate
from experiments.warm_reduced import solve


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(921)
    U = rng.normal(size=(8, 7, 6))
    I = rng.normal(size=(8, 7))
    mass = np.geomspace(0.2, 5.0, 8)
    B = np.linalg.qr(rng.normal(size=(6, 2)))[0].T
    return B, U, I, mass


def test_warm_prefix_matches_hpao_and_final_certificate() -> None:
    B, U, I, mass = _problem()
    prefix = solve_lsmr(B, U, I, mass=mass, lambda_prox=0.05, max_steps=1)
    result = solve(B, U, I, mass=mass, lambda_prox=0.05, max_steps=8)
    diagnostic = result.diagnostics
    assert diagnostic["warm_steps"] == 1
    assert diagnostic["warm_loss_final"] == pytest.approx(
        prefix.diagnostics["loss"], abs=1e-12
    )
    assert diagnostic["warm_normal_residual_ratio"] <= 0.1
    assert diagnostic["loss"] <= diagnostic["warm_loss_final"] + 1e-12
    assert diagnostic["total_accepted_steps"] == 1 + diagnostic["accepted_steps"]
    reference = evaluate(result.index, U, I, mass)
    assert reference.objective == pytest.approx(diagnostic["loss"], abs=1e-10)
    assert reference.riemannian_gradient == pytest.approx(
        diagnostic["riemannian_gradient"], abs=1e-10
    )


def test_warm_step_does_not_count_as_reduced_certificate() -> None:
    B, U, I, mass = _problem()
    result = solve(B, U, I, mass=mass, max_steps=1, tol=100.0)
    assert result.diagnostics["warm_steps"] == 1
    assert result.diagnostics["accepted_steps"] == 1
    assert not result.diagnostics["converged"]
    assert result.diagnostics["certificate_failures"] == ("consecutive_steps",)


def test_invalid_mass_and_unproved_zero_prox_fail() -> None:
    B, U, I, mass = _problem()
    with pytest.raises(ValueError, match="positive lambda_prox"):
        solve(B, U, I, mass=mass, lambda_prox=0)
    with pytest.raises(ValueError, match="mass must contain only finite positive"):
        solve(B, U, I, mass=np.zeros_like(mass))
