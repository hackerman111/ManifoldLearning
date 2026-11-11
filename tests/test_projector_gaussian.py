"""The experimental loop must retain ADP semantics and Gaussian identities."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from ADP import ADP_Config
from ADP.core.ADP_Config import epanechnikov
from ADP.core.experimental_centers import AdaptiveCentersConfig
from ADP.engine.common.index_fit import fit_index
from experiments.anisotropic_metrics import calculate_statistics, solve
from experiments.two_sided_meanmave import TimeOnlyProfiler


@pytest.mark.parametrize("floor", [0.0, 1.0])
def test_compact_control_reproduces_current_engine(floor):
    from experiments.projector_gaussian import fit

    rng = np.random.default_rng(2840041)
    x = rng.normal(size=(180, 6))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.1 * rng.normal(size=180)
    config = ADP_Config(seed=2840042, N_J=24, N_lin=24, N_loc=10, N_phi=4)
    expected = fit_index(
        x,
        y,
        config=replace(
            config, experimental_centers=AdaptiveCentersConfig(lambda_floor=floor)
        ),
        mode="multi",
        index_dim=2,
        solve=partial(solve, arm="two_sided", max_steps=3),
        profiler=TimeOnlyProfiler(),
        statistics_function=calculate_statistics,
        trace_indices=True,
    )
    actual = fit(
        x, y, m=2, config=config, kernel=epanechnikov, floor=floor, max_steps=3
    )
    np.testing.assert_array_equal(actual.initial_index, expected.initial_index)
    np.testing.assert_allclose(actual.index, expected.index, atol=2e-11)
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=2e-11)
    assert (
        actual.metadata["selected_iteration"] == expected.metadata["selected_iteration"]
    )
    assert actual.metadata["stop_reason"] == expected.metadata["stop_reason"]
    for old, new in zip(
        expected.metadata["trace"], actual.metadata["trace"], strict=True
    ):
        for key in ("basis", "eigenvalues", "h", "factor", "err"):
            np.testing.assert_allclose(new[key], old[key], atol=2e-11)


def test_exact_common_covariance_ridge_contraction():
    # This constructs consistent moments; quadrature independently checks them below.
    from experiments.projector_gaussian import gaussian_kernel

    rng = np.random.default_rng(2840043)
    d, m = 7, 2
    start, truth = [np.linalg.qr(rng.normal(size=(d, m)))[0] for _ in range(2)]
    p = start @ start.T
    h, alpha = 0.8, 0.2
    tensor = p + alpha**2 * (np.eye(d) - p)
    covariance = np.linalg.inv(np.eye(d) + 4 * tensor / h**2)
    centers = rng.normal(size=(8, d))
    means = centers @ (4 * tensor / h**2) @ covariance
    z = means @ truth
    var = np.diag(truth.T @ covariance @ truth)
    gradients = np.column_stack(
        (np.ones(len(z)), 4 * (z[:, 1] ** 3 + 3 * z[:, 1] * var[1]))
    )
    cross = gradients @ truth.T @ covariance
    tau = 0.01 / (1 + 4 / h**2)
    slopes = np.linalg.solve(covariance + tau * np.eye(d), cross.T)
    estimated = np.linalg.svd(slopes, full_matrices=False)[0][:, :m]
    error = lambda b: np.linalg.norm((np.eye(d) - b @ b.T) @ truth)
    assert error(estimated) <= 0.01 * error(start) + 1e-12
    np.testing.assert_allclose(
        gaussian_kernel(np.array([0.0, 1.0, 2.0])), np.exp([0.0, -2.0, -4.0])
    )


def test_gaussian_tilt_identity_by_direct_quadrature():
    from experiments.projector_gaussian import gaussian_kernel

    nodes, weights = np.polynomial.hermite.hermgauss(60)
    x = np.array(np.meshgrid(nodes, nodes)).reshape(2, -1).T * np.sqrt(2)
    quadrature = np.outer(weights, weights).ravel() / np.pi
    center, b = np.array([0.3, -0.7]), np.array([0.6, 0.8])
    tensor, h = np.array([[1.0, 0.2], [0.2, 0.4]]), 1.5
    delta = x - center
    w = quadrature * gaussian_kernel(np.sum((delta @ tensor) * delta, axis=1) / h**2)
    w /= w.sum()
    covariance = np.linalg.inv(np.eye(2) + 4 * tensor / h**2)
    mean = covariance @ (4 * tensor / h**2) @ center
    z = x @ b
    response = z + 0.2 * z**4
    direct_cross = (x - w @ x).T @ (w * (response - w @ response))
    zmean, zvar = mean @ b, b @ covariance @ b
    expected = covariance @ b * (1 + 0.8 * (zmean**3 + 3 * zmean * zvar))
    np.testing.assert_allclose(w @ x, mean, atol=1e-10)
    np.testing.assert_allclose(
        (x - mean).T @ (w[:, None] * (x - mean)), covariance, atol=1e-10
    )
    np.testing.assert_allclose(direct_cross, expected, atol=1e-9)


def test_kernel_change_keeps_compact_initializer_and_finite_stages():
    from experiments.projector_gaussian import fit

    rng = np.random.default_rng(2840044)
    x = rng.normal(size=(150, 5))
    y = np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2
    cfg = ADP_Config(seed=25, N_J=16, N_lin=20, N_loc=10, N_phi=3, outer_steps=2)
    compact = fit(x, y, m=2, config=cfg, kernel=epanechnikov, floor=0, max_steps=2)
    gaussian = fit(x, y, m=2, config=cfg, max_steps=2)
    np.testing.assert_array_equal(compact.initial_index, gaussian.initial_index)
    np.testing.assert_allclose(gaussian.index @ gaussian.index.T, np.eye(2), atol=1e-12)
    assert len(gaussian.metadata["trace"]) == 2
    assert all(t["median_neff"] > 0 for t in gaussian.metadata["trace"])


def test_rejects_unsupported_config_without_silently_ignoring_it():
    from experiments.projector_gaussian import fit

    x = np.zeros((40, 4))
    with pytest.raises(ValueError, match="configuration"):
        fit(x, np.zeros(40), m=2, config=ADP_Config(training_set="exclude_centers"))
