"""Независимые проверки относительного демпфирования intrinsic GN."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ADP.solver import grassman
from ADP.solver.grassman import _gradient, _profile
from experiments.grassman_accuracy_2026_10_18 import backtracking, relative_gn


def fixture(m: int = 2):
    rng = np.random.default_rng(319 + m)
    U = rng.normal(size=(7, m + 3, m + 3))
    U[0] = 0.0
    U[0, : m - 1, : m - 1] = np.eye(m - 1)
    I = rng.normal(size=U.shape[:2])
    mass = np.geomspace(0.3, 2.0, len(I))
    return np.eye(m + 3)[:m], U, I, mass


def dense_fit(M, I, mass):
    coefficients = np.array(
        [np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(M, I, strict=True)]
    )
    residual = I - (M @ coefficients[..., None]).squeeze(-1)
    return coefficients, residual, float(np.sum(mass[:, None] * residual**2))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_first_step_matches_finite_difference_augmented_reference(m):
    index, U, I, mass = fixture(m)
    Y, q = index.T, 1
    state = _profile(U @ Y, I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    V = np.linalg.svd(G, full_matrices=False)[0][:, :q]
    V, _ = np.linalg.qr(V - Y @ (Y.T @ V), mode="reduced")
    W = U @ V
    regular = np.arange(1, len(I))
    M, target, weights = (U @ Y)[regular], I[regular], mass[regular]
    root = np.sqrt(weights)[:, None]
    residual = (root * dense_fit(M, target, weights)[1]).ravel()
    columns = []
    for column in range(q * m):
        K = np.zeros((q, m))
        K.flat[column] = 1e-6
        plus = dense_fit(M + W[regular] @ K, target, weights)[1]
        minus = dense_fit(M - W[regular] @ K, target, weights)[1]
        columns.append((root * (plus - minus) / 2e-6).ravel())
    jac = np.column_stack(columns)
    damping = 0.05 * float(np.sum(jac**2)) / (q * m)
    design = np.vstack((jac, np.sqrt(damping) * np.eye(q * m)))
    rhs = np.concatenate((-residual, np.zeros(q * m)))
    K = np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(q, m)
    K *= min(1.0, np.tan(0.5) / np.linalg.norm(K, ord=2))
    slope = float(np.sum((V.T @ G) * K))
    assert slope < 0
    for backtrack in range(16):
        step = K * 0.5**backtrack
        eig, rotation = np.linalg.eigh(np.eye(m) + step.T @ step)
        Yn = (Y + V @ step) @ ((rotation / np.sqrt(eig)) @ rotation.T)
        value = dense_fit(U @ Yn, I, mass)[2]
        if value <= state.value + 1e-4 * slope * 0.5**backtrack:
            break
    else:
        pytest.fail("reference line search did not accept a proposal")

    result = relative_gn.solve(index, U, I, mass=mass, max_steps=1, rank=q, tol=1e-12)
    coefficients, _, value = dense_fit(U @ result.index.T, I, mass)
    assert result.index.T @ result.index == pytest.approx(Yn @ Yn.T, abs=3e-8)
    assert result.coefficients == pytest.approx(coefficients, abs=2e-11)
    assert result.diagnostics["profile_objective"] == pytest.approx(value, abs=2e-11)
    assert result.diagnostics["normalized_gn_proposals"] == 1
    assert result.diagnostics["method"] == "core_gn_relative"
    assert not result.diagnostics["stationarity_applicable"]
    assert result.diagnostics["local_rank_loss"] == 1


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize(
    ("response", "weight"), [(0.1, 0.25), (0.1, 4), (10, 0.25), (10, 4)]
)
def test_first_step_is_invariant_to_positive_response_and_mass_scaling(
    m, response, weight
):
    index, U, I, mass = fixture(m)
    first = relative_gn.solve(index, U, I, mass=mass, max_steps=1, tol=1e-13)
    scaled = relative_gn.solve(
        index, U, response * I, mass=weight * mass, max_steps=1, tol=1e-13
    )
    assert (
        first.diagnostics["accepted_steps"] == scaled.diagnostics["accepted_steps"] == 1
    )
    assert scaled.index.T @ scaled.index == pytest.approx(
        first.index.T @ first.index, abs=2e-10
    )
    assert scaled.diagnostics["profile_objective"] == pytest.approx(
        first.diagnostics["profile_objective"] * weight * response**2, rel=2e-11
    )
    assert scaled.coefficients == pytest.approx(
        response * first.coefficients, abs=2e-10
    )


@pytest.mark.parametrize("ridge", [0.0, 0.2])
def test_smooth_path_and_ridge_retain_original_absolute_damping(ridge):
    index, U, I, mass = fixture()
    if not ridge:
        U[0] = np.random.default_rng(33).normal(size=U[0].shape)
    original = backtracking.solve(
        index, U, I, mass=mass, max_steps=10, intrinsic_gn=True, local_ridge=ridge
    )
    result = relative_gn.solve(index, U, I, mass=mass, max_steps=10, local_ridge=ridge)
    np.testing.assert_array_equal(result.index, original.index)
    np.testing.assert_array_equal(result.coefficients, original.coefficients)
    assert result.diagnostics["normalized_gn_proposals"] == 0
    assert result.diagnostics["effective_max_steps"] == 5


@pytest.mark.parametrize("bad", [0.0, np.inf, np.nan])
@pytest.mark.parametrize("public", [False, True])
def test_degenerate_damping_skips_gn_without_false_stationarity(
    monkeypatch, bad, public
):
    index, U, I, mass = fixture()
    module = grassman if public else backtracking
    original = module._core_jacobian

    def invalid_jacobian(*args, **kwargs):
        jac, residual = original(*args, **kwargs)
        return np.full_like(jac, bad), residual

    monkeypatch.setattr(module, "_core_jacobian", invalid_jacobian)
    method = grassman.solve if public else relative_gn.solve
    options: dict[str, Any] = (
        {
            "relative_gn": True,
            "quadratic_trials": 0,
            "constant_centers": False,
            "smooth_max_steps": 5,
        }
        if public
        else {}
    )
    result = method(index, U, I, mass=mass, max_steps=1, tol=1e-12, **options)
    assert result.diagnostics["damping_scale_fallbacks"] == 1
    assert result.diagnostics["gn_solves"] == 0
    assert result.diagnostics["accepted_steps"] == 1
    assert not result.diagnostics["converged"]
    assert np.all(np.isfinite(result.index))
    assert result.diagnostics["profile_objective"] == pytest.approx(
        dense_fit(U @ result.index.T, I, mass)[2], abs=2e-11
    )


def test_workspace_guard_precedes_jacobian_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Jacobian must not be allocated beyond workspace budget")

    monkeypatch.setattr(backtracking, "_core_jacobian", forbidden)
    index, U, I, mass = fixture()
    result = relative_gn.solve(index, U, I, mass=mass, max_steps=1, workspace_bytes=1)
    assert result.diagnostics["workspace_fallbacks"] == 1
    assert result.diagnostics["normalized_gn_proposals"] == 0
    assert result.diagnostics["accepted_steps"] == 1


def test_flag_requires_boolean_and_intrinsic_variant():
    index, U, I, mass = fixture()
    with pytest.raises(TypeError, match="normalized_damping"):
        backtracking.solve(index, U, I, mass=mass, normalized_damping=cast(Any, 1))
    with pytest.raises(ValueError, match="requires intrinsic_gn"):
        backtracking.solve(index, U, I, mass=mass, normalized_damping=True)


@pytest.mark.parametrize(
    ("m", "scenario"),
    [
        (1, "intrinsic"),
        (2, "intrinsic"),
        (3, "intrinsic"),
        (2, "projected"),
        (2, "sensitive"),
        (2, "smooth"),
        (2, "ridge"),
        (2, "workspace"),
        (2, "unequal_mass"),
        (2, "rank_one"),
        (2, "constant"),
    ],
)
def test_public_relative_port_exactly_matches_frozen_algorithm(m, scenario):
    index, U, I, mass = fixture(m)
    settings: dict[str, Any] = {}
    if scenario == "projected":
        U[0, m - 1, m] = 1.0  # rank(U)>rank(UY): direct rank-one fallback
    elif scenario == "sensitive":
        U[1] = 0.0
        U[1, 0, 0], U[1, 1, 1] = 1.0, 1e-8
    elif scenario == "smooth":
        U[0] = np.random.default_rng(33).normal(size=U[0].shape)
    elif scenario == "ridge":
        settings["local_ridge"] = 0.2
    elif scenario == "workspace":
        settings["workspace_bytes"] = 1
    elif scenario == "unequal_mass":
        mass = np.geomspace(1e-4, 1e4, len(mass))
    elif scenario == "rank_one":
        settings["rank"] = 1
    elif scenario == "constant":
        U[:] = 0.0
    expected = relative_gn.solve(index, U, I, mass=mass, max_steps=10, **settings)
    actual = grassman.solve(
        index,
        U,
        I,
        mass=mass,
        max_steps=10,
        relative_gn=True,
        quadratic_trials=0,
        constant_centers=False,
        smooth_max_steps=5,
        **settings,
    )
    np.testing.assert_array_equal(actual.index, expected.index)
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
    assert actual.diagnostics == expected.diagnostics


def test_public_relative_option_is_enabled_and_validates_boolean():
    index, U, I, mass = fixture()
    original = grassman.solve(index, U, I, mass=mass)
    explicit = grassman.solve(index, U, I, mass=mass, relative_gn=True)
    np.testing.assert_array_equal(original.index, explicit.index)
    assert original.diagnostics == explicit.diagnostics
    with pytest.raises(TypeError, match="relative_gn"):
        grassman.solve(index, U, I, mass=mass, relative_gn=cast(Any, 1))
