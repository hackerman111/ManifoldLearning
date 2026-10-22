"""Reference checks for exact zero-operator center exclusion."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from experiments.grassman_accuracy_2026_10_18.candidate import solve as local_solve
from experiments.grassman_accuracy_2026_10_18.constant_centers import solve


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1890)
    U = rng.normal(size=(4, 5, 4))
    U[[1, 3]] = 0
    I = rng.normal(size=(4, 5))
    I[[1, 3]] *= 30.0
    mass = np.array([0.3, 1.7, 2.1, 0.8])
    index = np.eye(2, 4)
    return index, U, I, mass


@pytest.mark.parametrize("ridge", [0.0, 0.2])
def test_zero_center_objective_gradient_and_coefficients(ridge: float) -> None:
    index, U, I, mass = fixture()
    active = np.any(U != 0, axis=(1, 2))
    original_scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    active_scale = max(1.0, float(np.sum(mass[active, None] * I[active] ** 2)))
    constant = float(np.sum(mass[~active, None] * I[~active] ** 2))
    kwargs = {"max_steps": 2, "local_ridge": ridge}
    got = solve(index, U, I, mass=mass, tol=1e-7, **kwargs)
    active_result = local_solve(
        index,
        U[active],
        I[active],
        mass=mass[active],
        tol=1e-7 * original_scale / active_scale,
        **kwargs,
    )
    diag = got.diagnostics
    active_diag = active_result.diagnostics
    assert got.index == pytest.approx(active_result.index, abs=1e-13)
    assert got.coefficients.shape == (len(I), index.shape[0])
    assert got.coefficients[active] == pytest.approx(active_result.coefficients)
    assert np.array_equal(got.coefficients[~active], np.zeros((2, index.shape[0])))
    assert diag["profile_objective"] == pytest.approx(
        active_diag["profile_objective"] + constant, abs=2e-11
    )
    assert diag["profile_history"] == pytest.approx(
        tuple(x + constant for x in active_diag["profile_history"]), abs=2e-11
    )
    assert diag["gradient_norm"] == pytest.approx(active_diag["gradient_norm"])
    assert cast(float, diag["riemannian_gradient"]) * original_scale == pytest.approx(
        cast(float, diag["gradient_norm"]), rel=2e-13, abs=2e-13
    )
    designs = U @ got.index.T
    residual = I - np.einsum("jpm,jm->jp", designs, got.coefficients)
    reference = np.sum(
        mass
        * (np.sum(residual**2, axis=1) + ridge * np.sum(got.coefficients**2, axis=1))
    )
    assert diag["profile_objective"] == pytest.approx(reference, abs=2e-10)


@pytest.mark.parametrize("all_zero", [False, True])
def test_nozero_and_allzero_inputs_match_direct_delegate(all_zero: bool) -> None:
    index, U, I, mass = fixture()
    if all_zero:
        U[:] = 0
    else:
        U[[1, 3]] = U[[0, 2]]
    kwargs = {"mass": mass, "tol": 2e-6, "max_steps": 2, "local_ridge": 0.1}
    got, direct = solve(index, U, I, **kwargs), local_solve(index, U, I, **kwargs)
    assert got.index == pytest.approx(direct.index, abs=1e-13)
    assert got.coefficients == pytest.approx(direct.coefficients, abs=1e-13)
    assert got.diagnostics == direct.diagnostics


def test_zero_mass_is_rejected() -> None:
    index, U, I, mass = fixture()
    mass[1] = 0
    with pytest.raises(ValueError, match="mass"):
        solve(index, U, I, mass=mass)


@pytest.mark.parametrize("tol", [-1.0, 0.0, np.nan, np.inf])
def test_invalid_tolerances_are_rejected(tol: float) -> None:
    index, U, I, mass = fixture()
    with pytest.raises(ValueError, match="tol"):
        solve(index, U, I, mass=mass, tol=tol, max_steps=1)
