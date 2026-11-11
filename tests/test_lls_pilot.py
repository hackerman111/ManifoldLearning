import importlib

import numpy as np
import pytest


@pytest.mark.parametrize(
    "n,d,dependent", [(30, 6, False), (4, 9, False), (20, 6, True)]
)
def test_lls_preserves_weighted_least_squares(n, d, dependent):
    module = importlib.import_module("experiments.lls_pilot")
    rng = np.random.default_rng(719)
    x, y = rng.normal(size=(n, d)), rng.normal(size=n)
    if dependent:
        x[:, -1] = x[:, 0]
    weights = rng.uniform(0.1, 1, size=(2, n))
    weights[0, -1] = 0
    directions = np.broadcast_to(np.eye(d), (2, d, d))
    stats = module.lls_statistics(
        x, y, iter([(0, weights)]), directions, normalized=True
    )
    legacy = module.lls_statistics(x, y, weights, directions, normalized=False)
    np.testing.assert_allclose(legacy.U, stats.U * stats.mass[:, None, None])
    np.testing.assert_allclose(legacy.I, stats.I * stats.mass[:, None])
    for j in range(2):
        a = weights[j] / weights[j].sum()
        xc, yc = x - a @ x, y - a @ y
        np.testing.assert_allclose(
            stats.U[j].T @ stats.U[j], xc.T @ (a[:, None] * xc), atol=1e-13
        )
        np.testing.assert_allclose(
            stats.U[j].T @ stats.I[j], xc.T @ (a * yc), atol=1e-13
        )
        offsets = []
        for _ in range(4):
            v = rng.normal(size=d)
            full = np.sum(a * (yc - xc @ v) ** 2)
            compressed = np.sum((stats.I[j] - stats.U[j] @ v) ** 2)
            offsets.append(full - compressed)
        np.testing.assert_allclose(offsets, np.repeat(offsets[0], 4), atol=1e-12)
        assert offsets[0] >= -1e-12


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_lls_rejects_invalid_weights(bad):
    module = importlib.import_module("experiments.lls_pilot")
    weights = np.ones((1, 8))
    weights[0, 0] = bad
    with pytest.raises(ValueError):
        module.lls_statistics(
            np.ones((8, 3)), np.ones(8), weights, np.eye(3)[None], normalized=True
        )


def test_identity_statistics_are_full_covariance_moments():
    module = importlib.import_module("experiments.lls_pilot")
    rng = np.random.default_rng(812)
    x, y = rng.normal(size=(12, 4)), rng.normal(size=12)
    w = rng.uniform(size=(1, 12))
    a = w[0] / w.sum()
    stats = module.identity_statistics(x, y, w, np.zeros((1, 4, 4)), normalized=True)
    xc, yc = x - a @ x, y - a @ y
    np.testing.assert_allclose(stats.U[0], xc.T @ (a[:, None] * xc), atol=1e-14)
    np.testing.assert_allclose(stats.I[0], xc.T @ (a * yc), atol=1e-14)


def test_pilot_base_replays_public_model_exactly():
    from ADP import ADP_Config, ADP_multi_index, ADP_solver
    from ADP.solver.grassman import solve
    from experiments.lls_pilot import SOLVER, run_fit
    from experiments.r_sdr.design import Case, generate
    from experiments.r_sdr.runner import adp_settings

    case, seed = Case("replay", 80, 5), 9451
    x, y, _ = generate(case, seed)
    model = ADP_multi_index(
        case.m,
        ADP_Config(seed=seed, **adp_settings(case, {})),
        ADP_solver(solve, **SOLVER),
    ).fit(x, y)
    base, _ = run_fit(x, y, case, seed, "base")
    phid, _ = run_fit(x, y, case, seed, "phid")
    np.testing.assert_array_equal(base.index.T, model.basis_)
    np.testing.assert_array_equal(base.initial_index.T, model.result_.beta_init)
    np.testing.assert_array_equal(phid.index, base.index)
    for actual, expected in zip(base.metadata["trace"], model.trace_, strict=True):
        np.testing.assert_array_equal(actual["basis"], expected["basis"])
