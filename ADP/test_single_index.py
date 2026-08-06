import numpy as np

from ADP import (
    ADP_Config,
    ADP_SolverResult,
    ADP_single_index,
    ADP_solver,
)


def test_single_index_recovers_direction_and_preserves_local_mass():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(240, 3))
    beta_true = np.array([1.0, -0.7, 0.4])
    beta_true /= np.linalg.norm(beta_true)
    Y = np.sin(X @ beta_true) + 0.05 * rng.normal(size=len(X))

    config = ADP_Config(
        seed=9,
        N_loc=10,
        N_lin=12,
        N_J=60,
        N_phi=3,
        lambda_penalty=100.0,
        h_min=0.15,
        batch_size=20,
    )
    model = ADP_single_index(config).fit(X, Y)

    assert model.score_direction(beta_true) > 0.99
    assert model.trace_
    assert all(step["mean_mass"] >= config.N_loc - 1e-10 for step in model.trace_)
    assert model.h_k_ / config.a < config.h_min
    np.testing.assert_allclose(model.projector_, np.outer(model.beta_, model.beta_))
    assert model.transform(X[:2]).shape == (2,)


def test_solver_is_configured_separately_from_model():
    seen = {}

    def identity_solver(
        statistics,
        initial_index,
        *,
        lambda_penalty,
        local_ridge,
        marker,
    ):
        seen.update(
            marker=marker,
            lambda_penalty=lambda_penalty,
            local_ridge=local_ridge,
            directions=statistics.I.shape,
        )
        return ADP_SolverResult(initial_index.copy(), None, {"status": "identity"})

    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3))
    Y = X[:, 0]
    solver = ADP_solver(identity_solver, marker=17)
    config = ADP_Config(
        seed=4,
        N_loc=5,
        N_J=20,
        N_phi=2,
        h_min=10.0,
        index_init="random",
    )

    model = ADP_single_index(config, solver=solver).fit(X, Y)

    assert seen == {
        "marker": 17,
        "lambda_penalty": config.lambda_penalty,
        "local_ridge": config.local_ridge,
        "directions": (20, 2),
    }
    assert len(model.trace_) == 1


def test_default_parameters_are_resolved_from_data():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(120, 3))
    beta_true = np.array([1.0, 0.5, -0.2])
    beta_true /= np.linalg.norm(beta_true)
    Y = np.sin(X @ beta_true) + 0.05 * rng.normal(size=len(X))

    model = ADP_single_index(
        ADP_Config(lambda_penalty=100.0),
    ).fit(X, Y)

    assert model.effective_parameters_["N_lin"] == 6
    assert model.effective_parameters_["N_J"] == 120
    assert model.effective_parameters_["N_phi"] == 3
    assert model.stop_reason_ in {"h_min", "local_mass_limit"}
    assert model.score_direction(beta_true) > 0.99
