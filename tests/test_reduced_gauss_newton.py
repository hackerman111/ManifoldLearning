from __future__ import annotations

import numpy as np
import pytest

from experiments import reduced_gauss_newton as gn
from experiments.multi_solver_certificate import evaluate
from experiments.multi_solver_derivation import _retract, evaluate_reduced


def _problem(m=2, deficient=False):
    rng = np.random.default_rng(617 + m)
    U = rng.normal(size=(6, 7, 8))
    I = rng.normal(size=(6, 7))
    B = np.linalg.qr(rng.normal(size=(8, m)))[0].T
    if deficient:
        U[0] = 0
        U[1] = np.outer(rng.normal(size=7), rng.normal(size=8))
        I[1] = U[1] @ rng.normal(size=8)
    mass = np.geomspace(0.01, 100, 6)
    return B, U, I, mass


@pytest.mark.parametrize("m", [2, 5])
@pytest.mark.parametrize("deficient", [False, True])
def test_residual_jacobian_adjoint_gradient_and_ridge_reference(m, deficient):
    B, U, I, mass = _problem(m, deficient)
    norms = np.sum(U**2, axis=(1, 2))
    state = gn._ReducedState(B, U, I, mass, norms)
    reference = evaluate_reduced(B, U, I, mass)
    assert state.objective == pytest.approx(reference.objective, abs=1e-10)
    np.testing.assert_allclose(state.coefficients, reference.coefficients, atol=1e-11)
    np.testing.assert_allclose(
        state.gradient, gn._horizontal(B, reference.gradient), rtol=1e-10, atol=1e-10
    )
    rng = np.random.default_rng(813)
    vector, dual = rng.normal(size=B.size), rng.normal(size=I.size)
    J = state.operator
    assert (J @ vector) @ dual == pytest.approx(vector @ J.rmatvec(dual), abs=1e-10)
    tangent = gn._horizontal(B, vector.reshape(B.shape))
    tangent /= np.linalg.norm(tangent)
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        plus = gn._ReducedState(_retract(B, step * tangent), U, I, mass, norms)
        minus = gn._ReducedState(_retract(B, -step * tangent), U, I, mass, norms)
        fd = (plus.residual - minus.residual).ravel() / (2 * step)
        errors.append(np.linalg.norm(fd - J @ tangent.ravel()))
    assert errors[-1] < 1e-5 and errors[-1] < errors[0] / 100
    dense = np.column_stack([J @ e for e in np.eye(B.size)])
    ridge = 0.2
    expected = np.linalg.lstsq(
        np.vstack((dense, np.sqrt(ridge) * np.eye(B.size))),
        np.concatenate((-state.residual.ravel(), np.zeros(B.size))),
        rcond=None,
    )[0]
    direction, ratio, _ = gn._direction(state, ridge)
    np.testing.assert_allclose(direction.ravel(), expected, rtol=1e-7, atol=1e-9)
    assert ratio <= 0.1


def test_solver_certificate_basis_invariance_and_failure_boundaries(monkeypatch):
    B, U, I, mass = _problem()
    result = gn.solve(B, U, I, mass=mass, max_steps=6)
    final = evaluate(result.index, U, I, mass)
    assert final.objective == pytest.approx(result.diagnostics["loss"], abs=1e-10)
    assert final.riemannian_gradient == pytest.approx(
        result.diagnostics["riemannian_gradient"], abs=1e-10
    )
    assert np.all(np.diff(result.diagnostics["loss_history"]) <= 1e-10)
    rotated = gn.solve(B[::-1], U, I, mass=mass, max_steps=6)
    np.testing.assert_allclose(
        result.index.T @ result.index, rotated.index.T @ rotated.index, atol=1e-9
    )
    single = gn.solve(B, U, I, mass=mass, max_steps=1, tol=100)
    assert not single.diagnostics["converged"]
    assert single.diagnostics["consecutive_certified_steps"] == 1
    with pytest.raises(ValueError, match="positive"):
        gn.solve(B, U, I, mass=np.zeros(6))
    with pytest.raises(RuntimeError, match="absolute guard"):
        gn.solve(B, U * 1e-30, I)
    with pytest.raises(ValueError, match="positive lambda"):
        gn.solve(B, U, I, lambda_prox=0)
    monkeypatch.setattr(gn, "_retract", lambda B, _: B)
    with pytest.raises(RuntimeError, match="line search exhausted"):
        gn.solve(B, U, I)


def test_positive_discarded_singular_value_and_stationary_initial():
    B = np.eye(2, 4)
    U = np.array([[[1.0, 0, 0.2, 0.1], [0, 1e-17, 0, 0], [0, 0, 1e-23, 0]]])
    I, mass = np.array([[2.0, 0, 1.0]]), np.ones(1)
    norms = np.sum(U**2, axis=(1, 2))
    state = gn._ReducedState(B, U, I, mass, norms)
    assert state.ranks.tolist() == [1] and state.singular[0, 1] > 0
    reference = evaluate_reduced(B, U, I, mass)
    np.testing.assert_allclose(
        state.gradient, gn._horizontal(B, reference.gradient), atol=1e-20
    )
    rng = np.random.default_rng(442)
    x, y = rng.normal(size=B.size), rng.normal(size=I.size)
    assert (state.operator @ x) @ y == pytest.approx(
        x @ state.operator.rmatvec(y), abs=1e-25
    )
    zero = gn.solve(B, np.zeros_like(U), I)
    assert zero.diagnostics["stop_reason"] == "stationary_initial"
    assert not zero.diagnostics["converged"]
    assert zero.diagnostics["accepted_steps"] == 0


def test_failed_linear_certificate_has_a_finite_budget(monkeypatch):
    calls = []

    def inaccurate(operator, rhs, **settings):
        calls.append(settings["maxiter"])
        return np.ones(operator.shape[1]), 7, settings["maxiter"]

    monkeypatch.setattr(gn, "_lsmr_method", inaccurate)
    B, U, I, mass = _problem()
    with pytest.raises(RuntimeError, match="linear certificate failed"):
        gn.solve(B, U, I, mass=mass)
    assert calls == [5 * B.size] * 12
