"""Independent raw-data and dense references for the curvature experiment."""

import numpy as np
import pytest

from experiments import curvature_penalties as cp
from experiments import local_polynomial as poly


def fixture(m=2, support_lengths=(7, 7, 7, 7, 7)):
    rng = np.random.default_rng(2037 + m)
    x, y = rng.normal(size=(27, 7)), rng.normal(size=27)
    weights = np.zeros((5, 27))
    for row, count in zip(weights, support_lengths, strict=True):
        support = rng.choice(27, count, replace=False)
        row[support] = rng.uniform(0.2, 1, len(support))
    data = poly.neighborhoods(x, y, weights)
    basis = np.linalg.qr(rng.normal(size=(7, m)))[0]
    directions = rng.normal(size=(5, 4, 7))
    return rng, data, basis, directions


def raw_rows(local, basis, probes, arm):
    z = local.delta @ basis
    m = basis.shape[1]
    rows = [*z.T, *(z**2 / np.sqrt(2)).T]
    rows += [z[:, a] * z[:, b] for a in range(m) for b in range(a + 1, m)]
    phi = np.array(rows).T
    phi -= local.weights @ phi
    if arm in ("m1quad", "rquad"):
        covariance = local.delta.T @ (local.weights[:, None] * local.delta)
        values, vectors = np.linalg.eigh(covariance)
        keep = values > 1e-12 * values[-1]
        inverse_root = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
        transform = inverse_root @ (local.delta.T * local.weights)
        if arm == "rquad":
            quadratic = (local.delta @ probes.T) ** 2
            quadratic -= local.weights @ quadratic
            var = local.weights @ quadratic**2
            den = np.maximum(var, 1e-12 * max(1.0, float(var.max())))
            extra = quadratic.T * local.weights / np.sqrt(len(probes) * den[:, None])
            transform = np.vstack((transform, extra))
    else:
        transform = np.diag(np.sqrt(local.weights))
    return transform @ phi, transform @ local.response


def dense_reference(data, basis, directions, problem):
    j, m = len(data.local), basis.shape[1]
    q = m + m * (m + 1) // 2
    rows, rhs = [], []
    for k, local in enumerate(data.local):
        design, target = raw_rows(local, basis, directions[k], problem.arm)
        design = np.vstack((design, np.sqrt(problem.ridge) * np.eye(q)))
        target = np.r_[target, np.zeros(q)]
        block = np.zeros((len(target), j * q))
        block[:, k * q : (k + 1) * q] = np.sqrt(problem.pi[k]) * design
        rows.append(block)
        rhs.append(np.sqrt(problem.pi[k]) * target)
    selector = np.c_[np.zeros((q - m, m)), np.eye(q - m)]
    if problem.arm == "hgraph":
        for (a, b), weight in zip(problem.edges, problem.edge_weights, strict=True):
            block = np.zeros((q - m, j * q))
            block[:, a * q : (a + 1) * q] = (
                np.sqrt(2 * problem.coupling * weight) * selector
            )
            block[:, b * q : (b + 1) * q] = (
                -np.sqrt(2 * problem.coupling * weight) * selector
            )
            rows.append(block)
            rhs.append(np.zeros(q - m))
    if problem.arm == "hcommon":
        for k in range(j):
            block = np.hstack([-weight * selector for weight in problem.pi])
            block[:, k * q : (k + 1) * q] += selector
            rows.append(np.sqrt(2 * problem.coupling * problem.pi[k]) * block)
            rhs.append(np.zeros(q - m))
    design, target = np.vstack(rows), np.concatenate(rhs)
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    return theta.reshape(j, q), np.linalg.norm(design @ theta - target) ** 2


@pytest.mark.parametrize("arm", cp.ARMS)
def test_joint_coefficients_and_objective_against_raw_dense_ls(arm):
    _, data, basis, directions = fixture()
    problem = cp.build(data, basis, directions, arm)
    state = cp.profile(basis, problem)
    theta, value = dense_reference(data, basis, directions, problem)
    np.testing.assert_allclose(state.theta, theta, rtol=2e-7, atol=3e-8)
    assert state.value == pytest.approx(value, rel=2e-10, abs=1e-12)
    assert state.normal_residual <= 1e-8


@pytest.mark.parametrize("arm", cp.ARMS)
def test_variable_support_buckets_match_raw_reference(arm):
    _, data, basis, directions = fixture(3, (1, 3, 6, 9, 17))
    problem = cp.build(data, basis, directions, arm)
    assert len(problem.groups) > 1
    state = cp.profile(basis, problem)
    theta, value = dense_reference(data, basis, directions, problem)
    np.testing.assert_allclose(state.theta, theta, atol=3e-8, rtol=2e-7)
    assert state.value == pytest.approx(value, abs=1e-11)


@pytest.mark.parametrize("arm", cp.ARMS)
def test_envelope_chart_gradient_and_rotation(arm):
    rng, data, basis, directions = fixture()
    problem = cp.build(data, basis, directions, arm)
    state = cp.profile(basis, problem)
    vector, epsilon = rng.normal(size=basis.shape), 2e-6
    finite = (
        cp.profile(basis + epsilon * vector, problem).value
        - cp.profile(basis - epsilon * vector, problem).value
    ) / (2 * epsilon)
    assert finite == pytest.approx(np.sum(state.gradient * vector), rel=1e-6, abs=1e-7)
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = cp.profile(basis @ rotation, problem)
    assert rotated.value == pytest.approx(state.value, abs=1e-11)
    np.testing.assert_allclose(
        rotated.theta[:, :2], state.theta[:, :2] @ rotation, atol=1e-8
    )
    expected_h = np.einsum("ab,jbc,cd->jad", rotation.T, state.hessians, rotation)
    np.testing.assert_allclose(rotated.hessians, expected_h, atol=1e-8)
    complement = np.linalg.qr(basis, mode="complete")[0][:, 2:]
    point, direction = rng.normal(scale=0.1, size=(5, 2)), rng.normal(size=(5, 2))
    b, r = np.linalg.qr(basis + complement @ point)
    at_point = cp.profile(b, problem)
    horizontal = at_point.gradient - b @ (b.T @ at_point.gradient)
    pullback = complement.T @ np.linalg.solve(r, horizontal.T).T

    def value(t):
        return cp.profile(
            np.linalg.qr(basis + complement @ (point + t * direction))[0], problem
        ).value

    assert (value(epsilon) - value(-epsilon)) / (2 * epsilon) == pytest.approx(
        np.sum(pullback * direction), rel=1e-6, abs=1e-7
    )


@pytest.mark.parametrize("m", (2, 3, 5))
def test_wls_matches_existing_profile_and_saturated_first_moment(m):
    _, data, basis, directions = fixture(m)
    actual = cp.profile(basis, cp.build(data, basis, directions, "wls2"))
    previous = poly.profile(basis, data, ridge=1e-4)
    total_mass = data.mass.sum()
    assert actual.value == pytest.approx(previous.value / total_mass, rel=1e-11)
    np.testing.assert_allclose(
        actual.gradient, previous.gradient / total_mass, atol=1e-10
    )
    first = cp.profile(basis, cp.build(data, basis, directions, "m1quad"))
    assert first.value == pytest.approx(actual.value, abs=1e-11)


def test_rquad_is_literal_centered_second_moment_residual():
    _, data, basis, directions = fixture(5)
    state = cp.profile(basis, cp.build(data, basis, directions, "rquad"))
    expected = 0.0
    for k, local in enumerate(data.local):
        projected = local.delta @ basis
        prediction = projected @ state.theta[k, :5] + 0.5 * np.einsum(
            "na,ab,nb->n", projected, state.hessians[k], projected
        )
        prediction -= local.weights @ prediction
        residual = local.response - prediction
        probes = (local.delta @ directions[k].T) ** 2
        probes -= local.weights @ probes
        variance = local.weights @ probes**2
        denominator = np.maximum(variance, 1e-12 * max(1.0, float(variance.max())))
        moments = (local.weights * residual) @ probes
        expected += data.mass[k] / data.mass.sum() * np.mean(moments**2 / denominator)
    assert state.components["second_moment"] == pytest.approx(expected, abs=1e-12)


def test_zero_signal_and_optimizer_output():
    _, data, basis, directions = fixture()
    for local in data.local:
        local.response[:] = 0
    for arm in cp.ARMS:
        state = cp.profile(basis, cp.build(data, basis, directions, arm))
        assert state.value == 0
        np.testing.assert_array_equal(state.gradient, 0)
    result = cp.solve(basis.T, data, directions, "hcommon", max_steps=2)
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    assert result.coefficients.shape == (5, 2)
