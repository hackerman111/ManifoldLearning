"""Reference and derivative checks for the isolated local polynomial estimator."""

import numpy as np
import pytest


def fixture(seed=11, n=90, d=6, m=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    basis = np.linalg.qr(rng.normal(size=(d, m)))[0]
    z = x @ basis
    hessian = rng.normal(size=(m, m))
    hessian = hessian + hessian.T
    y = 0.7 + z @ rng.normal(size=m) + 0.5 * np.einsum("ni,ij,nj->n", z, hessian, z)
    weights = rng.uniform(0.1, 1, size=(4, n))
    return x, y, basis, weights


def test_quadratic_represents_an_exact_quadratic_response():
    from experiments import local_polynomial as poly

    x, y, basis, weights = fixture()
    locals_ = poly.neighborhoods(x, y, weights)
    quadratic = poly.profile(basis, locals_, degree=2, ridge=0.0)
    linear = poly.profile(basis, locals_, degree=1, ridge=0.0)
    assert quadratic.value < 1e-20
    assert np.linalg.norm(quadratic.gradient) < 1e-8
    assert linear.value > 1


@pytest.mark.parametrize("degree", [1, 2])
def test_profile_matches_dense_ridge_and_is_basis_invariant(degree):
    from experiments import local_polynomial as poly

    x, y, basis, weights = fixture()
    ridge = 1e-3
    locals_ = poly.neighborhoods(x, y, weights)
    state = poly.profile(basis, locals_, degree=degree, ridge=ridge)
    reference = 0.0
    for w in weights:
        a = w / w.sum()
        z = (x - a @ x) @ basis
        features = [z]
        if degree == 2:
            features.extend(
                [
                    z**2 / np.sqrt(2),
                    np.column_stack(
                        [
                            z[:, i] * z[:, j]
                            for i in range(z.shape[1])
                            for j in range(i + 1, z.shape[1])
                        ]
                    ),
                ]
            )
        features = np.column_stack(features)
        design = np.column_stack((np.ones(len(x)), features))
        penalty = np.diag([0.0] + [np.sqrt(ridge)] * features.shape[1])
        augmented = np.vstack((np.sqrt(a)[:, None] * design, penalty))
        target = np.r_[np.sqrt(a) * y, np.zeros(design.shape[1])]
        coef = np.linalg.lstsq(augmented, target, rcond=None)[0]
        reference += w.sum() * (
            np.sum(a * (y - design @ coef) ** 2) + ridge * np.sum(coef[1:] ** 2)
        )
    assert state.value == pytest.approx(reference, rel=1e-11, abs=1e-11)
    rotation = np.linalg.qr(
        np.random.default_rng(4).normal(size=(basis.shape[1], basis.shape[1]))
    )[0]
    rotated = poly.profile(basis @ rotation, locals_, degree=degree, ridge=ridge)
    assert rotated.value == pytest.approx(state.value, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("degree", [1, 2])
def test_profile_ambient_gradient_matches_finite_differences(degree):
    from experiments import local_polynomial as poly

    x, y, basis, weights = fixture()
    basis = basis + 0.08 * np.random.default_rng(9).normal(size=basis.shape)
    locals_ = poly.neighborhoods(x, y, weights)
    state = poly.profile(basis, locals_, degree=degree, ridge=1e-3)
    direction = np.random.default_rng(6).normal(size=basis.shape)
    step = 1e-6
    difference = (
        poly.profile(basis + step * direction, locals_, degree=degree, ridge=1e-3).value
        - poly.profile(
            basis - step * direction, locals_, degree=degree, ridge=1e-3
        ).value
    ) / (2 * step)
    assert difference == pytest.approx(
        np.sum(state.gradient * direction), rel=1e-6, abs=1e-6
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_qr_chart_gradient_matches_finite_differences(degree):
    from experiments import local_polynomial as poly

    x, y, basis, weights = fixture()
    full = np.linalg.qr(basis, mode="complete")[0]
    reference, complement = full[:, :3], full[:, 3:]
    k = np.random.default_rng(8).normal(scale=0.2, size=(3, 3))
    direction = np.random.default_rng(2).normal(size=k.shape)
    locals_ = poly.neighborhoods(x, y, weights)
    args = (reference, complement, locals_)
    value, gradient = poly.chart_value_gradient(k, *args, degree=degree, ridge=1e-3)
    assert np.isfinite(value)
    step = 1e-6
    plus = poly.chart_value_gradient(
        k + step * direction, *args, degree=degree, ridge=1e-3
    )[0]
    minus = poly.chart_value_gradient(
        k - step * direction, *args, degree=degree, ridge=1e-3
    )[0]
    assert (plus - minus) / (2 * step) == pytest.approx(
        np.sum(gradient * direction), rel=1e-6, abs=1e-6
    )


def test_ridge_handles_small_and_single_point_neighborhoods():
    from experiments import local_polynomial as poly

    x, y, basis, _ = fixture()
    weights = np.zeros((2, len(x)))
    weights[0, 0] = 1
    weights[1, :3] = [1, 0.5, 0.2]
    locals_ = poly.neighborhoods(x, y, weights)
    state = poly.profile(basis, locals_, degree=2, ridge=1e-4)
    assert np.isfinite(state.value)
    assert np.isfinite(state.gradient).all()
    np.testing.assert_array_equal(state.coefficients[0], np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        poly.neighborhoods(x, y, np.zeros_like(weights))
    with pytest.raises(ValueError, match="nonnegative"):
        poly.neighborhoods(x, y, -np.ones_like(weights))


def test_optimizer_returns_orthonormal_basis_and_monotone_profile():
    from experiments import local_polynomial as poly

    x, y, truth, weights = fixture(n=100, d=5, m=2)
    initial = np.linalg.qr(
        truth + np.random.default_rng(5).normal(scale=0.2, size=truth.shape)
    )[0]
    result = poly.solve(
        initial.T, poly.neighborhoods(x, y, weights), degree=2, max_steps=20
    )
    np.testing.assert_allclose(result.index @ result.index.T, np.eye(2), atol=1e-12)
    history = np.array(result.diagnostics["profile_history"])
    assert np.all(np.diff(history) <= 1e-10 * max(1, history[0]))
    assert history[-1] < 0.1 * history[0]


def test_optimizer_does_not_return_an_unaccepted_trial(monkeypatch):
    from types import SimpleNamespace

    from experiments import local_polynomial as poly

    x, y, truth, weights = fixture(n=100, d=5, m=2)
    initial = np.linalg.qr(
        truth + np.random.default_rng(5).normal(scale=0.2, size=truth.shape)
    )[0]

    def aborted(fun, x0, **kwargs):
        value, gradient = fun(x0)
        candidate = x0 - 0.01 * gradient / np.linalg.norm(gradient)
        assert fun(candidate)[0] < value
        return SimpleNamespace(
            x=x0, nit=0, nfev=2, success=False, message="aborted test"
        )

    monkeypatch.setattr(poly, "minimize", aborted)
    result = poly.solve(initial.T, poly.neighborhoods(x, y, weights), degree=2)
    np.testing.assert_allclose(
        result.index.T @ result.index, initial @ initial.T, atol=1e-12
    )
    assert result.diagnostics["accepted_steps"] == 0
    assert len(result.diagnostics["profile_history"]) == 1


def test_identity_statistics_are_full_centered_moments_with_response_mean():
    from experiments.lls_pilot import identity_statistics

    x, y, _, weights = fixture()
    x = x + np.arange(x.shape[1]) * 100
    y = y + 200
    directions = np.ones((len(weights), x.shape[1], x.shape[1]))
    stats = identity_statistics(x, y, weights, directions, normalized=True)
    for j, w in enumerate(weights):
        a = w / w.sum()
        delta, residual = x - a @ x, y - a @ y
        np.testing.assert_allclose(
            stats.U[j], delta.T @ (a[:, None] * delta), atol=1e-10
        )
        np.testing.assert_allclose(stats.I[j], delta.T @ (a * residual), atol=1e-10)
        assert stats.S[j] == pytest.approx(a @ y)


@pytest.mark.parametrize("arm", ["base", "identity", "wls1", "wls2"])
def test_study_runs_streamed_neighborhoods_through_full_fit(arm):
    import json

    from experiments.moment_quadratic_study import fit

    rng = np.random.default_rng(3801)
    x = rng.normal(size=(320, 6))
    y = x[:, 0] ** 2 + 0.5 * x[:, 1] ** 2 + 0.1 * rng.normal(size=len(x))
    fitted, stages, elapsed, _ = fit(x, y, 2, 3811, arm)
    assert stages
    assert elapsed > 0
    np.testing.assert_allclose(fitted.index @ fitted.index.T, np.eye(2), atol=1e-10)
    assert json.loads(json.dumps(stages, allow_nan=False))
