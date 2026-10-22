"""Independent checks for rank-boundary direct-profile Armijo steps."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ADP.solver.grassman import _gradient, _profile, _project
from experiments.grassman_accuracy_2026_10_18.backtracking import solve


def direct_value(U: np.ndarray, Y: np.ndarray, I: np.ndarray,
                 mass: np.ndarray, ridge: float = 0.0) -> float:
    designs = U @ Y
    value = 0.0
    for j, design in enumerate(designs):
        augmented = np.vstack((design, np.sqrt(ridge) * np.eye(Y.shape[1])))
        target = np.r_[I[j], np.zeros(Y.shape[1])]
        coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        value += mass[j] * (residual @ residual + ridge * (coefficient @ coefficient))
    return float(value)


def rank_guard_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(95800)
    U = rng.normal(size=(9, 6, 5))
    U[0] = 0.0
    I = rng.normal(size=(9, 6))
    mass = np.geomspace(0.2, 3.0, len(I))
    Y = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    return Y.T, U, I, mass


def first_armijo_step(index: np.ndarray, U: np.ndarray, I: np.ndarray,
                      mass: np.ndarray, max_angle: float) -> tuple[np.ndarray, float]:
    Y = index.T
    state = _profile(_project(U, Y), I, mass, 0.0)
    gradient = _gradient(Y, U, state, mass)
    left, _, right_t = np.linalg.svd(gradient, full_matrices=False)
    V = left[:, :1] - Y @ (Y.T @ left[:, :1])
    V, _ = np.linalg.qr(V, mode="reduced")
    a = right_t[:1].T[:, 0]
    sign = -1.0 if float((V[:, 0] @ gradient) @ a) > 0 else 1.0
    v = sign * V[:, 0]
    slope = float(v @ gradient @ a)
    initial = direct_value(U, Y, I, mass)
    for halvings in range(20):
        angle = max_angle * 0.5**halvings
        Yn = Y + (Y @ a * (np.cos(angle) - 1) + v * np.sin(angle))[:, None] * a
        value = direct_value(U, Yn, I, mass)
        if value <= initial + 1e-4 * angle * slope:
            return Yn, value
    raise AssertionError("independent reference found no Armijo step")


def test_rank_guard_one_step_matches_direct_geometric_reference() -> None:
    index, U, I, mass = rank_guard_case()
    expected_Y, expected_value = first_armijo_step(index, U, I, mass, 0.5)
    result = solve(index, U, I, mass=mass, max_steps=1, max_angle=0.5)
    diag = result.diagnostics
    assert diag["rank_guard_fallbacks"] == 1
    assert diag["accepted_steps"] == 1
    assert cast(int, diag["local_rank_loss"]) > 0
    assert result.index.T @ result.index == pytest.approx(
        expected_Y @ expected_Y.T, abs=2e-12
    )
    assert diag["profile_objective"] == pytest.approx(expected_value, abs=2e-11)
    assert diag["profile_objective"] == pytest.approx(
        direct_value(U, result.index.T, I, mass), abs=2e-11
    )
    assert np.max(np.diff(diag["profile_history"])) <= 1e-12
    assert cast(float, diag["orthogonality_error"]) < 1e-12


def test_rank_guard_line_search_is_sign_and_gauge_invariant() -> None:
    index, U, I, mass = rank_guard_case()
    base = solve(index, U, I, mass=mass, max_steps=2)
    signs = np.diag([-1.0, 1.0])
    flipped = solve(signs @ index, U, I, mass=mass, max_steps=2)
    rotation = np.linalg.qr(np.array([[1.0, 0.4], [-0.2, 1.0]]))[0]
    gauged = solve(rotation.T @ index, U, I, mass=mass, max_steps=2)
    for other in (flipped, gauged):
        assert base.index.T @ base.index == pytest.approx(
            other.index.T @ other.index, abs=2e-11
        )
        assert base.diagnostics["profile_objective"] == pytest.approx(
            other.diagnostics["profile_objective"], abs=2e-10
        )


def test_step_budget_caps_smooth_start_but_keeps_rank_guard_budget() -> None:
    index, U, I, mass = rank_guard_case()
    nonsmooth = solve(index, U, I, mass=mass, max_steps=9)
    smooth_U = U.copy()
    smooth_U[0] = np.random.default_rng(11).normal(size=smooth_U[0].shape)
    smooth = solve(index, smooth_U, I, mass=mass, max_steps=9)
    assert nonsmooth.diagnostics["effective_max_steps"] == 9
    assert smooth.diagnostics["effective_max_steps"] == 5
    assert cast(int, nonsmooth.diagnostics["rank_guard_fallbacks"]) >= 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"tol": 0.0}, "tol"), ({"max_angle": np.pi / 2}, "max_angle"),
     ({"rank": 3}, "rank"), ({"mass": np.zeros(9)}, "mass")],
)
def test_invalid_inputs_are_rejected(kwargs: dict, message: str) -> None:
    index, U, I, mass = rank_guard_case()
    options = {"mass": mass, "max_steps": 1}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        solve(index, U, I, **options)
