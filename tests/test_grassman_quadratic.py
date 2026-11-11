"""Independent checks for true-profile quadratic angle proposals."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ADP.solver.grassman import _gradient, _profile, _project
from experiments.grassman_accuracy_2026_10_18.backtracking import solve
from experiments.grassman_accuracy_2026_10_18.quadratic import solve as quadratic_solve


def direct_value(U, Y, I, mass, ridge=0.0):
    designs = U @ Y
    value = 0.0
    for j, design in enumerate(designs):
        if ridge:
            augmented = np.vstack((design, np.sqrt(ridge) * np.eye(Y.shape[1])))
            target = np.r_[I[j], np.zeros(Y.shape[1])]
        else:
            augmented, target = design, I[j]
        coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        value += mass[j] * (residual @ residual + ridge * (coefficient @ coefficient))
    return float(value)


def case():
    rng = np.random.default_rng(96800)
    U = rng.normal(size=(12, 6, 5))
    U[0] = 0.0
    I, mass = rng.normal(size=(12, 6)), np.geomspace(0.2, 2.0, 12)
    index = np.linalg.qr(rng.normal(size=(5, 2)))[0].T
    return index, U, I, mass


def direct_proposals(index, U, I, mass, max_angle=0.5, count=3):
    Y = index.T
    state = _profile(_project(U, Y), I, mass, 0.0)
    G = _gradient(Y, U, state, mass)
    left, _, right_t = np.linalg.svd(G, full_matrices=False)
    V = left[:, :1] - Y @ (Y.T @ left[:, :1])
    V, _ = np.linalg.qr(V, mode="reduced")
    a = right_t[0]
    sign = -1.0 if float((V[:, 0] @ G) @ a) > 0 else 1.0
    v = sign * V[:, 0]
    slope = float(v @ G @ a)
    initial = direct_value(U, Y, I, mass)
    angle, accepted, tried = max_angle, [], []
    for _ in range(count):
        Yn = Y + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
        value = direct_value(U, Yn, I, mass)
        tried.append((angle, Yn, value))
        if value <= initial + 1e-4 * angle * slope:
            accepted.append((Yn, value))
        curvature = 2 * (value - initial - angle * slope) / angle**2
        if not np.isfinite(curvature) or curvature <= 0:
            break
        proposed = float(np.clip(-slope / curvature, max_angle * 0.5**19, max_angle))
        if abs(proposed - angle) <= 1e-6 * angle:
            break
        angle = proposed
    if not accepted:
        for k in range(20):
            angle = max_angle * 0.5**k
            Yn = Y + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
            value = direct_value(U, Yn, I, mass)
            tried.append((angle, Yn, value))
            if value <= initial + 1e-4 * angle * slope:
                accepted.append((Yn, value))
                break
    return min(accepted, key=lambda pair: pair[1]), tried, len(accepted) > 0


def test_quadratic_proposals_match_direct_geometry_and_best_armijo_trial() -> None:
    index, U, I, mass = case()
    (expected_Y, expected_value), tried, accepted = direct_proposals(index, U, I, mass)
    result = solve(index, U, I, mass=mass, max_steps=1, quadratic_trials=3)
    diag = result.diagnostics
    assert accepted and len(tried) >= 1
    assert diag["rank_guard_fallbacks"] == 1
    assert diag["profile_evaluations"] == len(tried)
    assert result.index.T @ result.index == pytest.approx(
        expected_Y @ expected_Y.T, abs=3e-11
    )
    assert diag["profile_objective"] == pytest.approx(expected_value, abs=3e-10)
    assert diag["profile_objective"] == pytest.approx(
        direct_value(U, result.index.T, I, mass), abs=3e-10
    )
    assert np.all(np.diff(diag["profile_history"]) <= 1e-10)
    assert diag["orthogonality_error"] < 1e-12


def test_default_zero_trials_matches_direct_original_armijo() -> None:
    index, U, I, mass = case()
    expected, _, _ = direct_proposals(index, U, I, mass, count=0)
    result = solve(index, U, I, mass=mass, max_steps=1)
    assert result.index.T @ result.index == pytest.approx(
        expected[0] @ expected[0].T, abs=3e-11
    )
    assert result.diagnostics["profile_objective"] == pytest.approx(
        expected[1], abs=3e-10
    )


def test_zero_trials_matches_frozen_armijo_source_exactly() -> None:
    root = Path(__file__).parents[1]
    snapshot = root / "benchmarks/results/outputs/r_sdr"
    folder = snapshot / "grassman-armijo-dev-2026-10-18-repaired/source"
    path = folder / "benchmarks/experiments/grassman_accuracy_2026_10_18/backtracking.py"
    inputs = root / "benchmarks/experiments/grassman_accuracy_2026_10_18/frozen"
    names = ("001.npz", "014.npz")
    if not path.is_file() or not all((inputs / name).is_file() for name in names):
        pytest.skip("optional frozen Armijo source or inputs are unavailable")
    spec = importlib.util.spec_from_file_location("_frozen_armijo", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    for name in names:
        with np.load(inputs / name) as data:
            args = (data["index_init"], data["U"], data["I"])
            current = solve(*args, mass=data["mass"], max_steps=10)
            frozen = old.solve(*args, mass=data["mass"], max_steps=10)
            np.testing.assert_array_equal(current.index, frozen.index)
            np.testing.assert_array_equal(current.coefficients, frozen.coefficients)
            assert current.diagnostics == frozen.diagnostics


def test_sign_gauge_stopping_rank_guard_and_wrapper_budgets() -> None:
    index, U, I, mass = case()
    base = solve(index, U, I, mass=mass, max_steps=2, quadratic_trials=3)
    rotated = np.linalg.qr(np.array([[1.0, 0.3], [-0.2, 1.0]]))[0]
    other = solve(rotated.T @ index, U, I, mass=mass, max_steps=2, quadratic_trials=3)
    assert base.index.T @ base.index == pytest.approx(
        other.index.T @ other.index, abs=2e-10
    )
    assert base.diagnostics["profile_objective"] == pytest.approx(
        other.diagnostics["profile_objective"], abs=2e-9
    )
    smooth = U.copy()
    smooth[0] = np.random.default_rng(7).normal(size=smooth[0].shape)
    q = quadratic_solve
    budgets = [
        q(index, U, I, mass=mass, max_steps=10).diagnostics["effective_max_steps"],
        q(index, smooth, I, mass=mass, max_steps=10).diagnostics["effective_max_steps"],
        solve(index, smooth, I, mass=mass, max_steps=10, quadratic_trials=3,
              smooth_max_steps=None).diagnostics["effective_max_steps"],
    ]
    assert budgets == [10, 5, 10]
    U[:] = 0
    stopped = solve(index, U, I, mass=mass, max_steps=10, quadratic_trials=3)
    assert stopped.diagnostics["stop_reason"] == "rank_boundary_stationary"


@pytest.mark.parametrize("trials", [True, -1, 4, 1.5])
def test_invalid_quadratic_trial_counts_rejected(trials) -> None:
    index, U, I, mass = case()
    with pytest.raises(ValueError, match="quadratic_trials"):
        solve(index, U, I, mass=mass, max_steps=1, quadratic_trials=trials)
