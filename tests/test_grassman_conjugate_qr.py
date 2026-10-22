"""Checks for the opt-in per-center profile on conjugate spectral steps."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, cast

import numpy as np
import pytest

from experiments.grassman_accuracy_2026_10_18 import backtracking
from experiments.grassman_accuracy_2026_10_18.conjugate_guard import (
    solve as conjugate_solve,
)


def _case(m: int, *, smooth: bool = False):
    rng = np.random.default_rng(730 + m + smooth)
    p, d, centers = m + 3, m + 2, 8
    U = rng.normal(size=(centers, p, d))
    if not smooth:
        U[0] = 0.0
    I = rng.normal(size=(centers, p))
    mass = np.geomspace(0.2, 2.0, centers)
    return np.eye(d)[:m], U, I, mass


def _direct_fit(U, Y, I, mass, ridge=0.0):
    coefficients, ranks, value = [], [], 0.0
    for matrix, target, weight in zip(U @ Y, I, mass, strict=True):
        design = np.vstack((matrix, np.sqrt(ridge) * np.eye(Y.shape[1])))
        augmented = np.r_[target, np.zeros(Y.shape[1])]
        coef, _, rank, _ = np.linalg.lstsq(design, augmented, rcond=None)
        residual = target - matrix @ coef
        coefficients.append(coef)
        ranks.append(rank)
        value += weight * (residual @ residual + ridge * (coef @ coef))
    return np.asarray(coefficients), np.asarray(ranks), float(value)


def _qr_solve():
    return importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.conjugate_qr"
    ).solve


def test_conjugate_qr_wrapper_and_per_center_option_exist():
    assert "per_center_profile" in inspect.signature(backtracking.solve).parameters
    assert _qr_solve().__kwdefaults__["max_steps"] == 10


@pytest.mark.parametrize("m", [1, 2, 3])
def test_per_center_conjugate_matches_full_refit_on_rank_guard_case(m):
    index, U, I, mass = _case(m)
    qr = _qr_solve()(index, U, I, mass=mass, max_steps=2, tol=1e-12)
    old = conjugate_solve(index, U, I, mass=mass, max_steps=2, tol=1e-12)
    coef, ranks, value = _direct_fit(U, qr.index.T, I, mass)
    assert qr.index.T @ qr.index == pytest.approx(old.index.T @ old.index, abs=2e-8)
    assert qr.coefficients == pytest.approx(coef, abs=2e-8)
    assert qr.coefficients == pytest.approx(old.coefficients, abs=2e-8)
    assert qr.diagnostics["profile_objective"] == pytest.approx(value, abs=2e-8)
    assert qr.diagnostics["local_rank_loss"] == np.count_nonzero(ranks < m)
    assert np.all(np.diff(qr.diagnostics["profile_history"]) <= 1e-10)


def test_per_center_conjugate_supports_ridge_and_smooth_profile_budget():
    index, U, I, mass = _case(2)
    qr = _qr_solve()(index, U, I, mass=mass, local_ridge=0.2, max_steps=2)
    coef, _ranks, value = _direct_fit(U, qr.index.T, I, mass, ridge=0.2)
    assert qr.coefficients == pytest.approx(coef, abs=2e-9)
    assert qr.diagnostics["profile_objective"] == pytest.approx(value, abs=2e-9)

    index, U, I, mass = _case(2, smooth=True)
    qr = _qr_solve()(index, U, I, mass=mass, max_steps=10, tol=1e-12)
    old = conjugate_solve(index, U, I, mass=mass, max_steps=10, tol=1e-12)
    assert qr.diagnostics["effective_max_steps"] == 5
    assert qr.index == pytest.approx(old.index, abs=2e-12)
    assert qr.coefficients == pytest.approx(old.coefficients, abs=2e-12)


def test_per_center_flag_defaults_off_and_rejects_invalid_combinations():
    index, U, I, mass = _case(2)
    default = backtracking.solve(
        index, U, I, mass=mass, spectral_guard=True, conjugate_guard=True
    )
    explicit = backtracking.solve(
        index,
        U,
        I,
        mass=mass,
        spectral_guard=True,
        conjugate_guard=True,
        per_center_profile=False,
    )
    assert default.index == pytest.approx(explicit.index, abs=0)
    assert default.coefficients == pytest.approx(explicit.coefficients, abs=0)
    with pytest.raises(TypeError, match="per_center_profile"):
        backtracking.solve(index, U, I, per_center_profile=cast(Any, 1))
    with pytest.raises(ValueError, match="intrinsic_gn"):
        backtracking.solve(index, U, I, per_center_profile=True, intrinsic_gn=True)
