import time

import numpy as np

from adp import ADP, ADPConfig
from adp.common.types import LocalStatistics


def _statistics() -> LocalStatistics:
    return LocalStatistics(
        variant="new",
        imav=np.array([[1.2, -0.4, 0.7], [0.3, 0.8, -1.1]]),
        centers=np.zeros((2, 3)),
        h=1.0,
        weights_mean=3.0,
        S=np.array([[0.7, -0.2, 0.4], [0.1, 0.6, -0.5]]),
        U=np.array(
            [
                [[0.9, 0.2, -0.1], [0.3, -0.7, 0.4], [0.2, 0.5, 0.8]],
                [[-0.4, 0.6, 0.1], [0.8, 0.2, -0.3], [0.5, -0.1, 0.7]],
            ]
        ),
    )


def test_objective_matches_the_regularized_local_and_beta_subproblems():
    ridge = 0.35
    lambda_penalty = 0.6
    model = ADP.create(
        "new",
        ADPConfig(ridge=ridge, lambda_penalty=lambda_penalty, show_progress=False),
    )
    stats = _statistics()
    beta = np.array([0.4, -0.3, 0.8])
    prior = np.array([0.7, 0.1, -0.2])
    intercepts = np.array([0.25, -0.45])
    slopes = np.array([1.1, -0.6])

    residual = (
        stats.imav
        - intercepts[:, None] * stats.S
        - slopes[:, None] * (stats.U @ beta)
    )
    expected = float(np.sum(residual**2))
    expected += ridge * float(
        np.sum(intercepts**2) + np.sum(slopes**2) + np.sum(beta**2)
    )
    expected += lambda_penalty * float(np.sum((beta - prior) ** 2))

    actual = model._objective(
        stats,
        beta,
        intercepts,
        slopes,
        prior,
        lambda_penalty,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_local_solver_lstsq_fallback_keeps_ridge(monkeypatch):
    import adp.variants.random_projection as random_projection

    ridge = 0.8
    model = ADP.create(
        "new",
        ADPConfig(ridge=ridge, show_progress=False),
    )
    stats = _statistics()
    beta = np.array([0.4, -0.3, 0.8])
    projected = stats.U @ beta
    design = np.stack((stats.S, projected), axis=-1)
    gram = np.swapaxes(design, 1, 2) @ design
    gram += ridge * np.eye(2)[None, :, :]
    rhs = np.einsum("jpk,jp->jk", design, stats.imav, optimize=True)
    expected = np.stack(
        [np.linalg.solve(gram[index], rhs[index]) for index in range(gram.shape[0])]
    )

    def fail_batched_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("forced fallback")

    monkeypatch.setattr(random_projection.np.linalg, "solve", fail_batched_solve)

    intercepts, slopes = model._solve_local_coefficients(stats, beta)

    np.testing.assert_allclose(intercepts, expected[:, 0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(slopes, expected[:, 1], rtol=1e-13, atol=1e-13)


def test_float32_fit_keeps_beta_in_the_configured_dtype():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            dtype="float32",
            random_state=17,
            show_progress=False,
        ),
    )
    data = model.generate_data(n=40, d=4, noise=0.01, link="linear")

    result = model.fit(
        data.X,
        data.y,
        centers=data.centers,
        beta0=data.beta.astype(np.float32),
        directions=data.directions,
    )

    assert result.statistics.U.dtype == np.dtype("float32")
    assert result.beta.dtype == np.dtype("float32")
    assert result.intercepts.dtype == np.dtype("float32")
    assert result.slopes.dtype == np.dtype("float32")
    assert all(beta.dtype == np.dtype("float32") for beta in result.beta_path)


def test_float32_beta_solver_does_not_promote_its_linear_system():
    model = ADP.create(
        "new",
        ADPConfig(dtype="float32", tol=1e-7, ridge=1e-4, show_progress=False),
    )
    stats = _statistics()
    stats.imav = stats.imav.astype(np.float32)
    stats.S = stats.S.astype(np.float32)
    stats.U = stats.U.astype(np.float32)
    prior = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    beta = model._solve_beta(
        stats,
        intercepts=np.array([0.2, -0.1], dtype=np.float32),
        slopes=np.array([0.8, -0.5], dtype=np.float32),
        prior=prior,
        lambda_penalty=0.6,
        x0=prior,
    )

    assert beta.dtype == np.dtype("float32")


class _RecordingBetaSolver:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve(
        self,
        statistics,
        intercepts,
        slopes,
        prior,
        lambda_penalty,
        x0=None,
    ):
        prior_values = np.asarray(prior, dtype=float).copy()
        start_values = np.asarray(x0, dtype=float).copy()
        self.calls.append((prior_values, start_values))
        return start_values + np.array([0.0, 0.2, -0.1])


class _RecordingNeverStop:
    def __init__(self) -> None:
        self.objective_deltas: list[float] = []

    def should_stop(self, phase, state, *, step=None, **metrics):
        if phase == "inner":
            self.objective_deltas.append(float(metrics["objective_delta"]))
        return False


def _recording_model(*, record_telemetry: bool):
    beta_solver = _RecordingBetaSolver()
    stop_rule = _RecordingNeverStop()
    model = ADP.create(
        "new",
        ADPConfig(
            inner_steps=3,
            objective_check_every=1,
            record_telemetry=record_telemetry,
            ridge=0.2,
            show_progress=False,
        ),
        stage_factories={
            "beta_solver": lambda context: beta_solver,
            "stop_rule": lambda context: stop_rule,
        },
    )
    return model, beta_solver, stop_rule


def test_each_inner_beta_update_uses_the_previous_inner_beta_as_prior():
    model, beta_solver, _ = _recording_model(record_telemetry=False)

    model._alternating_solve(
        _statistics(),
        beta_start=np.array([1.0, 0.0, 0.0]),
        lambda_penalty=0.6,
        outer=0,
        outer_started=time.perf_counter(),
    )

    assert len(beta_solver.calls) == 3
    for prior, start in beta_solver.calls:
        np.testing.assert_allclose(prior, start, rtol=0.0, atol=1e-15)
    assert not np.allclose(beta_solver.calls[0][0], beta_solver.calls[1][0])


def test_objective_stopping_delta_compares_values_within_the_same_inner_step():
    model, _, stop_rule = _recording_model(record_telemetry=True)

    _, _, _, history = model._alternating_solve(
        _statistics(),
        beta_start=np.array([1.0, 0.0, 0.0]),
        lambda_penalty=0.6,
        outer=0,
        outer_started=time.perf_counter(),
    )

    expected_deltas = [
        abs(float(step.objective_before) - float(step.objective_after))
        for step in history
    ]
    assert np.isinf(stop_rule.objective_deltas[0])
    np.testing.assert_allclose(
        stop_rule.objective_deltas[1:],
        expected_deltas[1:],
        rtol=1e-14,
        atol=1e-14,
    )


def test_outer_telemetry_reports_the_last_proximal_step_objective_pair():
    model, _, _ = _recording_model(record_telemetry=True)
    stats = _statistics()
    beta_start = np.array([1.0, 0.0, 0.0])

    beta, intercepts, slopes, history = model._alternating_solve(
        stats,
        beta_start=beta_start,
        lambda_penalty=0.6,
        outer=0,
        outer_started=time.perf_counter(),
    )
    outer_row, _ = model.algorithm._build_outer_telemetry(
        stats,
        beta_start,
        beta,
        intercepts,
        slopes,
        history,
        outer=0,
        n_observations=20,
        iteration_started=time.perf_counter(),
        bandwidth_update_time=0.0,
        optimization_time=0.0,
        stage_timings_before={},
        stage_calls_before={},
    )

    assert history[0].objective_before != history[-1].objective_before
    assert outer_row["objective_before"] == history[-1].objective_before
    assert outer_row["objective_after"] == history[-1].objective_after
    expected_relative_decrease = (
        float(history[-1].objective_before) - float(history[-1].objective_after)
    ) / abs(float(history[-1].objective_before))
    np.testing.assert_allclose(
        outer_row["relative_objective_decrease"],
        expected_relative_decrease,
        rtol=1e-14,
        atol=1e-14,
    )
