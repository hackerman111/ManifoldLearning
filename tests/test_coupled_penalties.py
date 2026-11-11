"""Direct references for jointly profiled graph penalties."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from ADP.core.ADP_Statistic import ADP_Statistics
from experiments.coupled_penalties import build, full_moments, profile, solve


def fixture():
    rng = np.random.default_rng(370)
    j, p, d, m = 7, 4, 6, 2
    stats = ADP_Statistics(
        I=rng.normal(size=(j, p)),
        U=rng.normal(size=(j, p, d)),
        mass=rng.uniform(1, 3, j),
        mean=rng.normal(size=(j, d)),
        n_eff=np.full(j, 20),
        eta=np.zeros((j, p)),
        S=rng.normal(size=j),
    )
    b = np.linalg.qr(rng.normal(size=(d, m)))[0]
    return rng, stats, b


def direct(stats, b, problem):
    j, m = len(stats.mass), b.shape[1]
    rows, targets = [], []
    for k in range(j):
        block = np.zeros((stats.I.shape[1] + m, j * m))
        block[:, k * m : (k + 1) * m] = np.sqrt(problem.pi[k]) * np.vstack(
            (stats.U[k] @ b, np.sqrt(problem.tau) * np.eye(m))
        )
        rows.append(block)
        targets.append(np.r_[np.sqrt(problem.pi[k]) * stats.I[k], np.zeros(m)])
    for (a, c), w, delta, dy in zip(
        problem.edges, problem.weights, problem.delta, problem.dy, strict=True
    ):
        block = np.zeros((1 + m, j * m))
        block[0, a * m : (a + 1) * m] = np.sqrt(problem.lmean * w) * delta @ b / 2
        block[0, c * m : (c + 1) * m] = np.sqrt(problem.lmean * w) * delta @ b / 2
        block[1:, a * m : (a + 1) * m] = -np.sqrt(problem.lsmooth * w) * np.eye(m)
        block[1:, c * m : (c + 1) * m] = np.sqrt(problem.lsmooth * w) * np.eye(m)
        rows.append(block)
        targets.append(np.r_[np.sqrt(problem.lmean * w) * dy, np.zeros(m)])
    design, target = np.vstack(rows), np.concatenate(targets)
    g = np.linalg.lstsq(design, target, rcond=None)[0]
    return g.reshape(j, m), np.linalg.norm(design @ g - target) ** 2


@pytest.mark.parametrize(
    "arm", ["chart", "mean0.01", "mean0.1", "smooth0.1", "joint0.1"]
)
def test_joint_profile_equals_direct_augmented_regression(arm):
    _, stats, b = fixture()
    problem = build(stats, b, arm)
    state = profile(b, problem)
    g, value = direct(stats, b, problem)
    np.testing.assert_allclose(state.coefficients, g, atol=2e-9, rtol=2e-9)
    assert state.value == pytest.approx(value, rel=2e-12)
    assert state.normal_residual < 1e-8


def test_envelope_and_chart_gradient_and_basis_equivariance():
    rng, stats, b = fixture()
    problem = build(stats, b, "joint0.1")
    state = profile(b, problem)
    direction = rng.normal(size=b.shape)
    h = 1e-6
    fd = (
        profile(b + h * direction, problem).value
        - profile(b - h * direction, problem).value
    ) / (2 * h)
    assert fd == pytest.approx(np.sum(state.gradient * direction), abs=2e-7, rel=2e-7)
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = profile(b @ rotation, problem)
    assert rotated.value == pytest.approx(state.value, rel=1e-12)
    np.testing.assert_allclose(
        rotated.coefficients, state.coefficients @ rotation, atol=2e-9
    )
    complement = np.linalg.qr(b, mode="complete")[0][:, 2:]
    k = rng.normal(scale=0.1, size=(4, 2))
    direction = rng.normal(size=k.shape)
    q, r = np.linalg.qr(b + complement @ k, mode="reduced")
    chart_state = profile(q, problem)
    horizontal = chart_state.gradient - q @ (q.T @ chart_state.gradient)
    gradient = complement.T @ np.linalg.solve(r, horizontal.T).T

    def value(t):
        return profile(
            np.linalg.qr(b + complement @ (k + t * direction))[0], problem
        ).value

    assert (value(h) - value(-h)) / (2 * h) == pytest.approx(
        np.sum(gradient * direction), abs=2e-7
    )


def test_full_moment_loss_equals_singular_weighted_regression():
    rng, stats, b = fixture()
    covariances, cross, references = [], [], []
    for _ in stats.mass:
        z = rng.normal(size=(5, 6))
        z -= z.mean(axis=0)
        y = rng.normal(size=5)
        y -= y.mean()
        covariances.append(z.T @ z / 5)
        cross.append(z.T @ y / 5)
        references.append((z, y))
    converted, info = full_moments(stats, np.array(covariances), np.array(cross))
    for j, (z, y) in enumerate(references):
        g = rng.normal(size=2)
        actual = np.linalg.norm(converted.I[j] - converted.U[j] @ b @ g) ** 2
        constant = y @ y / 5 - cross[j] @ np.linalg.pinv(covariances[j]) @ cross[j]
        assert actual + constant == pytest.approx(
            np.mean((y - z @ b @ g) ** 2), abs=1e-10
        )
    assert info["rank_max"] <= 4


def test_zero_signal_and_duplicate_centers_are_finite():
    _, stats, b = fixture()
    stats = replace(
        stats,
        I=np.zeros_like(stats.I),
        S=np.zeros_like(stats.S),
        mean=np.zeros_like(stats.mean),
    )
    problem = build(stats, b, "joint0.1")
    state = profile(b, problem)
    assert state.value == 0
    np.testing.assert_array_equal(state.coefficients, 0)
    assert np.isfinite(state.gradient).all()


@pytest.mark.parametrize("zero", [False, True])
def test_full_moments_share_truncated_mask_and_match_retained_predictors(zero):
    rng, stats, basis = fixture()
    values = np.zeros(6) if zero else np.array([2.0, 1.0, 1e-14, 0.0, 0.0, 0.0])
    z = np.vstack((np.diag(np.sqrt(6 * values)), -np.diag(np.sqrt(6 * values))))
    y = rng.normal(size=len(z))
    y -= y.mean()
    covariance, cross = z.T @ z / len(z), z.T @ y / len(z)
    converted, info = full_moments(
        stats, np.repeat(covariance[None], 7, axis=0), np.repeat(cross[None], 7, axis=0)
    )
    retained = np.zeros((6, 6)) if zero else np.diag([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        converted.U[0] @ converted.U[0], retained @ covariance @ retained, atol=1e-15
    )
    assert np.linalg.norm(converted.U[0][:, 2:]) == 0
    assert converted.I[0][2] == 0
    coefficient = rng.normal(size=2)
    projected_cross = retained @ cross
    constant = (
        y @ y / len(y)
        - projected_cross
        @ np.linalg.pinv(retained @ covariance @ retained)
        @ projected_cross
    )
    assert np.linalg.norm(
        converted.I[0] - converted.U[0] @ basis @ coefficient
    ) ** 2 + constant == pytest.approx(
        np.mean((y - z @ retained @ basis @ coefficient) ** 2), abs=1e-12
    )
    assert info["rank_max"] == (0 if zero else 2)


def test_solver_preserves_shapes_orthogonality_and_descends():
    _, stats, b = fixture()
    result = solve(b.T, stats, "joint0.1", max_steps=3)
    assert result.coefficients.shape == (len(stats.mass), 2)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    history = result.diagnostics["profile_history"]
    assert all(y <= x + 1e-10 for x, y in pairwise(history))


def test_invalid_arm_rejected():
    _, stats, b = fixture()
    with pytest.raises(ValueError, match="arm"):
        build(stats, b, "typo")
