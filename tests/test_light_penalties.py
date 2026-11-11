"""Direct mathematical references for cheap penalties on projected moments."""

import numpy as np
import pytest

from ADP.core.ADP_Statistic import ADP_Statistics
from ADP.solver import grassman
from experiments.light_penalties import prepare, solve


def fixture():
    rng = np.random.default_rng(543)
    j, p, d, m = 7, 4, 6, 2
    stats = ADP_Statistics(
        I=rng.normal(size=(j, p)),
        U=rng.normal(size=(j, p, d)),
        mass=rng.uniform(1, 3, size=j),
        mean=np.zeros((j, d)),
        n_eff=np.full(j, 10),
        eta=np.zeros((j, p)),
    )
    start = np.linalg.qr(rng.normal(size=(d, m)))[0]
    anchor = np.linalg.qr(rng.normal(size=(d, m)))[0]
    trial = np.linalg.qr(rng.normal(size=(d, m)))[0]
    return stats, start, anchor, trial


def direct_value(stats, b, tau=0):
    result = 0.0
    for u, target, mass in zip(stats.U, stats.I, stats.mass, strict=True):
        design = np.vstack((u @ b, np.sqrt(tau) * np.eye(b.shape[1])))
        rhs = np.r_[target, np.zeros(b.shape[1])]
        coefficients = np.linalg.lstsq(design, rhs, rcond=None)[0]
        result += mass * np.linalg.norm(design @ coefficients - rhs) ** 2
    return result


@pytest.mark.parametrize("arm,beta", [("proj0.01", 0.01), ("proj0.1", 0.1)])
def test_virtual_centers_equal_exact_projector_penalty(arm, beta):
    stats, start, anchor, trial = fixture()
    u, target, mass, tau, info = prepare(stats, start.T, anchor, arm)
    actual = grassman._profile(u @ trial, target, mass, tau).value
    energy = np.sum(stats.mass[:, None] * stats.I**2)
    expected = direct_value(stats, trial) + beta * energy * (
        1 - np.linalg.norm(anchor.T @ trial) ** 2 / trial.shape[1]
    )
    assert actual == pytest.approx(expected, rel=2e-12)
    assert info["virtual_centers"] == trial.shape[1]
    # Rotating the preliminary basis must preserve its projector penalty.
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    u2, t2, m2, tau2, _ = prepare(stats, start.T, anchor @ rotation, arm)
    assert grassman._profile(u2 @ trial, t2, m2, tau2).value == pytest.approx(actual)


@pytest.mark.parametrize("arm,alpha", [("ridge0.001", 0.001), ("ridge0.01", 0.01)])
def test_ridge_uses_frozen_scale_and_matches_augmented_regression(arm, alpha):
    stats, start, anchor, trial = fixture()
    u, target, mass, tau, info = prepare(stats, start.T, anchor, arm)
    expected_tau = alpha * np.median(np.sum((stats.U @ start) ** 2, axis=(1, 2))) / 2
    assert tau == pytest.approx(expected_tau)
    actual = grassman._profile(u @ trial, target, mass, tau).value
    assert actual == pytest.approx(direct_value(stats, trial, tau), rel=2e-12)
    assert info["virtual_centers"] == 0


@pytest.mark.parametrize("arm", ["base", "pad"])
def test_zero_penalty_preserves_original_profile(arm):
    stats, start, anchor, trial = fixture()
    original_u, original_i = stats.U.copy(), stats.I.copy()
    u, target, mass, tau, _ = prepare(stats, start.T, anchor, arm)
    actual = grassman._profile(u @ trial, target, mass, tau).value
    assert actual == pytest.approx(direct_value(stats, trial), rel=2e-12)
    np.testing.assert_array_equal(stats.U, original_u)
    np.testing.assert_array_equal(stats.I, original_i)


def test_outer_loop_receives_coefficients_for_real_centers_only():
    stats, start, anchor, _ = fixture()
    result = solve(start.T, stats, anchor, "proj0.1", max_steps=2)
    assert result.coefficients.shape == (len(stats.mass), start.shape[1])
    reference = np.array(
        [
            np.linalg.lstsq(u @ result.index.T, target, rcond=None)[0]
            for u, target in zip(stats.U, stats.I, strict=True)
        ]
    )
    np.testing.assert_allclose(result.coefficients, reference, rtol=1e-10, atol=1e-10)


def test_zero_signal_does_not_add_zero_mass_centers():
    stats, start, anchor, _ = fixture()
    stats.I[:] = 0
    u, target, mass, tau, info = prepare(stats, start.T, anchor, "proj0.1")
    assert info["virtual_centers"] == 0
    assert np.all(mass > 0)
    assert grassman._profile(u @ start, target, mass, tau).value == 0


def test_unknown_arm_is_rejected():
    stats, start, anchor, _ = fixture()
    with pytest.raises(ValueError, match="arm"):
        prepare(stats, start.T, anchor, "unknown")
