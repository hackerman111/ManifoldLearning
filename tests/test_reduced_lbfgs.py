from __future__ import annotations

import numpy as np
import pytest

from experiments import reduced_lbfgs
from experiments.multi_solver_certificate import evaluate


@pytest.mark.parametrize("m", [2, 5])
@pytest.mark.parametrize("structural_rank", [False, True])
def test_guarded_solver_decreases_reference_objective(
    m: int, structural_rank: bool
) -> None:
    rng = np.random.default_rng(511 + m)
    J, P, d = 7, 8, 9
    U = rng.normal(size=(J, P, d))
    if structural_rank:
        U[0] = 0
        U[1] = np.outer(rng.normal(size=P), rng.normal(size=d))
    I = rng.normal(size=(J, P))
    if structural_rank:
        I[1] = U[1] @ rng.normal(size=d)
    mass = np.geomspace(0.1, 10.0, J)
    B = np.linalg.qr(rng.normal(size=(d, m)))[0].T
    initial = evaluate(B, U, I, mass)
    result = reduced_lbfgs.solve(B, U, I, mass=mass, max_steps=8)
    final = evaluate(result.index, U, I, mass)
    assert result.diagnostics["accepted_steps"] >= 1
    assert final.objective <= initial.objective + 1e-12
    assert final.objective == pytest.approx(result.diagnostics["loss"], abs=1e-10)
    assert final.riemannian_gradient == pytest.approx(
        result.diagnostics["riemannian_gradient"], abs=1e-10
    )
    assert final.rank_loss == initial.rank_loss
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(m), atol=1e-12)
    assert np.all(np.diff(result.diagnostics["loss_history"]) <= 1e-12)
    assert np.all(np.isfinite(result.coefficients))


def test_one_accepted_step_cannot_fake_two_step_certificate() -> None:
    rng = np.random.default_rng(515)
    U = rng.normal(size=(6, 7, 5))
    I = rng.normal(size=(6, 7))
    B = np.linalg.qr(rng.normal(size=(5, 2)))[0].T
    result = reduced_lbfgs.solve(B, U, I, max_steps=1, tol=100.0)
    assert result.diagnostics["accepted_steps"] == 1
    assert not result.diagnostics["converged"]
    assert result.diagnostics["consecutive_certified_steps"] == 1
    assert result.diagnostics["certificate_failures"] == ("consecutive_steps",)


def test_solver_path_is_invariant_to_basis_rotation() -> None:
    rng = np.random.default_rng(518)
    U = rng.normal(size=(8, 7, 6))
    I = rng.normal(size=(8, 7))
    B = np.linalg.qr(rng.normal(size=(6, 2)))[0].T
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    first = reduced_lbfgs.solve(B, U, I, max_steps=6)
    second = reduced_lbfgs.solve(rotation @ B, U, I, max_steps=6)
    np.testing.assert_allclose(
        first.index.T @ first.index,
        second.index.T @ second.index,
        atol=1e-9,
    )
    assert first.diagnostics["loss"] == pytest.approx(
        second.diagnostics["loss"], abs=1e-10
    )


def test_guards_reject_invalid_mass_ill_conditioning_and_line_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(516)
    U = rng.normal(size=(6, 7, 5))
    I = rng.normal(size=(6, 7))
    B = np.linalg.qr(rng.normal(size=(5, 2)))[0].T
    with pytest.raises(ValueError, match="mass must contain only finite positive"):
        reduced_lbfgs.solve(B, U, I, mass=np.zeros(6))
    with pytest.raises(RuntimeError, match="absolute guard"):
        reduced_lbfgs.solve(B, U * 1e-30, I)
    monkeypatch.setattr(reduced_lbfgs, "_retract", lambda B, _: B)
    with pytest.raises(RuntimeError, match="line search exhausted"):
        reduced_lbfgs.solve(B, U, I, max_backtracks=2)
