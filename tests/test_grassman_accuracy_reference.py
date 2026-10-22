"""Independent augmented-lstsq checks for the experimental angle profile."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.grassman_accuracy_2026_10_18.candidate import AngleProfile


def direct_value(
    M: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    tau: float,
    t: float,
) -> float:
    """Evaluate the defining augmented least-squares problem center by center."""
    u = M @ a
    designs = M + (u * (np.cos(t) - 1) + w * np.sin(t))[:, :, None] * a
    total = 0.0
    for j, design in enumerate(designs):
        augmented = np.vstack((design, np.sqrt(tau) * np.eye(M.shape[2])))
        target = np.r_[I[j], np.zeros(M.shape[2])]
        coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
        residual = I[j] - design @ coefficient
        total += mass[j] * (residual @ residual + tau * (coefficient @ coefficient))
    return float(total)


def check_profile(
    M: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
    I: np.ndarray,
    mass: np.ndarray,
    tau: float,
    t: float,
) -> AngleProfile:
    curve = AngleProfile(M, w, a, I, mass, tau)
    assert curve.value(t) == pytest.approx(
        direct_value(M, w, a, I, mass, tau, t), rel=3e-12, abs=3e-12
    )
    return curve


def test_constant_columns_near_full_design_cutoff_refit() -> None:
    M = np.zeros((1, 5, 2))
    M[0, 0, 0], M[0, 1, 1] = 1e-15, 1.0
    w = np.zeros((1, 5))
    w[0, 2] = 0.7
    I = np.zeros((1, 5))
    I[0, 0] = 1.0
    curve = check_profile(M, w, np.array([0.0, 1.0]), I, np.array([2.3]), 0.0, 0.2)
    assert curve.refit_centers == 1


def test_tau_zero_with_fewer_rows_than_constant_columns_refits() -> None:
    M = np.array([[[1.0, 0.2, 0.0, 0.0], [0.0, 0.3, 0.4, 0.0]]])
    w = np.array([[0.3, -0.2]])
    I = np.array([[0.7, -1.1]])
    curve = check_profile(
        M, w, np.array([0.0, 0.0, 0.0, 1.0]), I, np.array([0.4]), 0.0, 0.31
    )
    assert not curve.safe[0]
    assert curve.refit_centers == 1


def test_mixed_safe_unsafe_centers_keep_mass_weighted_profile() -> None:
    M = np.zeros((2, 5, 2))
    M[0, 0, 0], M[0, 1, 1] = 1.0, 0.8
    w = np.zeros((2, 5))
    w[:, 2] = (0.4, -0.6)
    I = np.array([[0.2, -0.3, 0.8, 1.1, -0.4], [0.7, 0.1, -0.2, 0.5, 0.9]])
    mass = np.array([3.7, 0.17])
    curve = check_profile(M, w, np.array([0.0, 1.0]), I, mass, 0.0, 0.27)
    assert curve.safe.tolist() == [True, False]
    assert curve.refit_centers == 1


def test_small_positive_ridge_uses_direct_augmented_fallback() -> None:
    M = np.zeros((1, 5, 2))
    M[0, 1, 1] = 1.0
    w = np.zeros((1, 5))
    w[0, 2] = 0.5
    I = np.array([[0.4, 0.2, -0.7, 0.3, 0.8]])
    curve = check_profile(M, w, np.array([0.0, 1.0]), I, np.array([1.2]), 1e-16, 0.2)
    assert curve.refit_centers == 1


def test_raw_direction_cancellation_near_pi_over_four_refits() -> None:
    M = np.zeros((1, 4, 2))
    M[0, 1, 0], M[0, 0, 1] = 1.0, 1.0
    w = np.zeros((1, 4))
    w[0, 0] = -1.0
    I = np.array([[1.0, 0.2, -0.3, 0.4]])
    curve = check_profile(M, w, np.array([0.0, 1.0]), I, np.ones(1), 0.0, np.pi / 4)
    assert curve.refit_centers == 1


def test_random_orthogonal_coefficient_gauge_preserves_profile() -> None:
    rng = np.random.default_rng(1042)
    M = rng.normal(size=(3, 6, 3))
    w = rng.normal(size=(3, 6))
    I = rng.normal(size=(3, 6))
    mass = np.array([0.2, 1.3, 4.1])
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    gauged_M, gauged_a = M @ rotation, rotation.T @ a
    first = check_profile(M, w, a, I, mass, 0.03, -0.19)
    second = check_profile(gauged_M, w, gauged_a, I, mass, 0.03, -0.19)
    assert first.value(-0.19) == pytest.approx(second.value(-0.19), abs=3e-12)
