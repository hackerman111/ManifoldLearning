import numpy as np
import pytest
from pathlib import Path

from adp import (
    ADP,
    ADPAlgorithm,
    ADPConfig,
    ADPState,
    StageContext,
    StageExecutionError,
    StageFactory,
    StageRegistry,
)


def test_default_stage_registries_are_isolated():
    first = StageRegistry.with_defaults()
    second = StageRegistry.with_defaults()

    first.register("beta_solver", "experimental", lambda context: object())

    assert "experimental" in first.available("beta_solver")
    assert "experimental" not in second.available("beta_solver")
    assert "cg" in second.available("beta_solver")
    assert second.available("statistics_builder") == (
        "cpu_batched",
        "cpu_compact_factored",
        "random_projection",
    )


def test_model_resolves_cpu_batched_statistics_builder():
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"statistics_builder": "cpu_batched"},
    )

    assert model.algorithm.stage_names["statistics_builder"] == "cpu_batched"
    assert type(model.algorithm.components["statistics_builder"]).__name__ == (
        "CpuBatchedStatisticsBuilder"
    )


def test_model_resolves_cpu_compact_factored_statistics_builder():
    model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"statistics_builder": "cpu_compact_factored"},
    )

    assert model.algorithm.stage_names["statistics_builder"] == (
        "cpu_compact_factored"
    )
    assert type(model.algorithm.components["statistics_builder"]).__name__ == (
        "CpuCompactFactoredStatisticsBuilder"
    )


def test_factory_types_are_part_of_public_api():
    from adp.stages import (
        BandwidthSelector,
        BetaInitializer,
        BetaSolver,
        CenterSelector,
        DirectionSampler,
        LocalSolver,
        StatisticsBuilder,
        StopRule,
    )

    assert ADPAlgorithm.__name__ == "ADPAlgorithm"
    assert StageContext.__name__ == "StageContext"
    assert StageFactory is not None
    assert {
        BandwidthSelector,
        BetaInitializer,
        BetaSolver,
        CenterSelector,
        DirectionSampler,
        LocalSolver,
        StatisticsBuilder,
        StopRule,
    }


class PriorBetaSolver:
    def __init__(self):
        self.calls = 0

    def solve(
        self,
        statistics,
        intercepts,
        slopes,
        prior,
        lambda_penalty,
        x0=None,
    ):
        self.calls += 1
        return np.asarray(prior).copy()


def test_model_resolves_named_stage_from_custom_registry():
    registry = StageRegistry.with_defaults()
    registry.register("beta_solver", "prior", lambda context: PriorBetaSolver())

    model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"beta_solver": "prior"},
        registry=registry,
    )

    assert model.algorithm.stage_names["beta_solver"] == "prior"
    assert isinstance(model.algorithm.components["beta_solver"], PriorBetaSolver)


def test_direct_factory_has_priority_over_named_stage():
    registry = StageRegistry.with_defaults()
    registry.register("beta_solver", "prior", lambda context: object())

    model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"beta_solver": "prior"},
        stage_factories={"beta_solver": lambda context: PriorBetaSolver()},
        registry=registry,
    )

    assert model.algorithm.stage_names["beta_solver"] == "custom"
    assert isinstance(model.algorithm.components["beta_solver"], PriorBetaSolver)


def test_unknown_named_stage_lists_available_implementations():
    with pytest.raises(ValueError, match="missing.*cg"):
        ADP.create(
            "new",
            ADPConfig(show_progress=False),
            stages={"beta_solver": "missing"},
        )


def test_fit_uses_custom_beta_solver_and_records_stage_diagnostics():
    solver = PriorBetaSolver()
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=12,
            n_directions=4,
            min_neighbors=5,
            outer_steps=1,
            inner_steps=3,
            show_progress=False,
            random_state=9,
        ),
        stage_factories={"beta_solver": lambda context: solver},
    )
    data = model.generate_data(n=60, d=4, noise=0.01, link="linear")

    result = model.fit(
        data.X,
        data.y,
        centers=data.centers,
        beta0=data.beta,
        directions=data.directions,
    )

    assert solver.calls > 0
    assert result.stage_names["beta_solver"] == "custom"
    assert result.stage_calls["beta_solver"] == solver.calls
    assert result.stage_timings["beta_solver"] >= 0.0
    assert set(result.stage_names) == set(model.algorithm.components)

    solver.calls = 0
    delegated = model._solve_beta(
        result.statistics,
        result.intercepts,
        result.slopes,
        data.beta,
        model.config.resolved_lambda(),
        x0=result.beta,
    )
    assert solver.calls == 1
    assert np.allclose(delegated, data.beta)


def test_invalid_custom_beta_solver_output_reports_stage_and_iteration():
    class ZeroBetaSolver:
        def solve(self, *args, **kwargs):
            prior = np.asarray(args[3])
            return np.zeros_like(prior)

    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=1,
            show_progress=False,
            random_state=10,
        ),
        stage_factories={"beta_solver": lambda context: ZeroBetaSolver()},
    )
    data = model.generate_data(n=40, d=3, noise=0.01, link="linear")

    with pytest.raises(
        StageExecutionError,
        match=r"beta_solver.*custom.*outer=0.*inner=0",
    ):
        model.fit(
            data.X,
            data.y,
            centers=data.centers,
            beta0=data.beta,
            directions=data.directions,
        )


def test_invalid_local_solver_output_is_not_misattributed_to_beta_solver():
    class InvalidLocalSolver:
        def solve(self, statistics, beta):
            wrong_size = statistics.centers.shape[0] + 1
            return np.zeros(wrong_size), np.ones(wrong_size)

    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=1,
            show_progress=False,
            random_state=11,
        ),
        stage_factories={"local_solver": lambda context: InvalidLocalSolver()},
    )
    data = model.generate_data(n=40, d=3, noise=0.01, link="linear")

    with pytest.raises(
        StageExecutionError,
        match=r"local_solver.*custom.*outer=0.*inner=0",
    ):
        model.fit(
            data.X,
            data.y,
            centers=data.centers,
            beta0=data.beta,
            directions=data.directions,
        )


def test_stop_rule_receives_complete_adp_state():
    seen = []

    class InspectingStopRule:
        def should_stop(self, phase, state, *, step=None, **metrics):
            seen.append((phase, state, step, metrics))
            return phase == "inner"

    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            show_progress=False,
            random_state=14,
        ),
        stage_factories={"stop_rule": lambda context: InspectingStopRule()},
    )
    data = model.generate_data(n=40, d=3, noise=0.01, link="linear")

    model.fit(data.X, data.y)

    inner_state = next(state for phase, state, _, _ in seen if phase == "inner")
    assert isinstance(inner_state, ADPState)
    assert inner_state.X.shape == data.X.shape
    assert inner_state.y.shape == data.y.shape
    assert inner_state.centers is not None
    assert inner_state.beta is not None
    assert inner_state.statistics is not None


def test_fit_records_actual_inner_and_outer_stop_decisions():
    class LastAllowedStepStopRule:
        def should_stop(self, phase, state, *, step=None, **metrics):
            return phase == "inner" and metrics["inner"] == 1

    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            show_progress=False,
            random_state=141,
        ),
        stage_factories={
            "stop_rule": lambda context: LastAllowedStepStopRule()
        },
    )
    data = model.generate_data(n=40, d=3, noise=0.01, link="linear")

    result = model.fit(data.X, data.y)

    assert len(result.history) == 2
    assert result.history[-1].inner_stop_reason == "tolerance"
    assert result.stop_reason == "scheduled_completion"


def test_invalid_initializer_is_attributed_before_normalization():
    class ZeroInitializer:
        def initialize(self, X, y):
            return np.zeros(X.shape[1])

    model = ADP.create(
        "new",
        ADPConfig(n_centers=8, n_directions=3, show_progress=False, random_state=15),
        stage_factories={"beta_initializer": lambda context: ZeroInitializer()},
    )
    data = model.generate_data(n=40, d=3)

    with pytest.raises(StageExecutionError, match="beta_initializer.*custom"):
        model.fit(data.X, data.y, centers=data.centers, directions=data.directions)


def test_invalid_statistics_are_attributed_at_statistics_boundary():
    class InvalidStatisticsBuilder:
        def compute(self, *args, **kwargs):
            return object()

    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=1,
            show_progress=False,
            random_state=16,
        ),
        stage_factories={
            "statistics_builder": lambda context: InvalidStatisticsBuilder()
        },
    )
    data = model.generate_data(n=40, d=3)

    with pytest.raises(
        StageExecutionError,
        match=r"statistics_builder.*custom.*outer=0",
    ):
        model.fit(
            data.X,
            data.y,
            centers=data.centers,
            beta0=data.beta,
            directions=data.directions,
        )


def test_inner_training_loop_has_single_implementation_path():
    solver_source = Path("adp/engine/solver.py").read_text()

    assert "for inner in" not in solver_source
    assert "algorithm._alternating_solve" in solver_source
