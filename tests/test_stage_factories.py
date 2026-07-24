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
from adp.common.types import LocalStatistics


def test_default_stage_registries_are_isolated():
    first = StageRegistry.with_defaults()
    second = StageRegistry.with_defaults()

    first.register("beta_solver", "experimental", lambda context: object())

    assert "experimental" in first.available("beta_solver")
    assert "experimental" not in second.available("beta_solver")
    assert "cg" in second.available("beta_solver")
    assert second.available("statistics_builder") == ("random_projection",)


def test_zero_intercept_local_solver_is_available_without_changing_default():
    registry = StageRegistry.with_defaults()
    model = ADP.create("new", ADPConfig(show_progress=False))

    assert registry.available("local_solver") == (
        "least_squares",
        "zero_intercept",
    )
    assert model.algorithm.stage_names["local_solver"] == "least_squares"


def test_initial_direction_and_bandwidth_experiment_variants_are_registered():
    registry = StageRegistry.with_defaults()

    assert {
        "default",
        "e1",
        "pca",
        "ridge_0",
        "ridge_1e-4",
        "ridge_1e-5",
        "ridge_1e-6",
        "ridge_1e-7",
        "ridge_1e-2",
    } <= set(registry.available("beta_initializer"))
    assert {
        "adaptive_mass",
        "local_mass_mean",
        "local_mass_q0",
        "local_mass_q05",
        "local_mass_q10",
        "local_mass_q25",
        "knn_q90_k1",
        "knn_q90_k2",
        "knn_q90_k4",
    } <= set(registry.available("bandwidth_selector"))


def test_builtin_initial_direction_variants_follow_their_definitions():
    X = np.array(
        [
            [-0.2, -3.0, 1.0],
            [0.1, -1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.2, 3.0, 1.0],
        ]
    )
    y = np.array([-2.0, -1.0, 1.0, 2.0])

    e1_model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"beta_initializer": "e1"},
    )
    pca_model = ADP.create(
        "new",
        ADPConfig(show_progress=False),
        stages={"beta_initializer": "pca"},
    )
    ridge_models = {
        eta_scale: ADP.create(
            "new",
            ADPConfig(show_progress=False),
            stages={"beta_initializer": initializer},
        )
        for eta_scale, initializer in (
            (1e-2, "ridge_1e-2"),
            (1e-4, "ridge_1e-4"),
            (1e-5, "ridge_1e-5"),
            (1e-6, "ridge_1e-6"),
            (1e-7, "ridge_1e-7"),
        )
    }

    e1 = e1_model.algorithm.components["beta_initializer"].initialize(X, y)
    pca = pca_model.algorithm.components["beta_initializer"].initialize(X, y)

    np.testing.assert_array_equal(e1, np.array([1.0, 0.0, 0.0]))
    assert abs(pca[1]) == pytest.approx(1.0, abs=3e-3)
    gram = X.T @ X
    for eta_scale, ridge_model in ridge_models.items():
        ridge = ridge_model.algorithm.components["beta_initializer"].initialize(
            X,
            y,
        )
        eta = eta_scale * np.trace(gram) / X.shape[1]
        expected_ridge = np.linalg.solve(
            gram + eta * np.eye(X.shape[1]),
            X.T @ (y - y.mean()),
        )
        expected_ridge /= np.linalg.norm(expected_ridge)
        np.testing.assert_allclose(
            ridge,
            expected_ridge,
            rtol=1e-12,
            atol=1e-12,
        )


def test_fit_exposes_distinct_beta_reference_and_normalized_zero_step():
    model = ADP.create(
        "new",
        ADPConfig(
            n_centers=8,
            n_directions=3,
            min_neighbors=4,
            outer_steps=1,
            inner_steps=2,
            record_telemetry=True,
            show_progress=False,
            random_state=71,
        ),
        stages={"beta_initializer": "e1"},
    )
    data = model.generate_data(n=40, d=3, noise=0.01, link="linear")

    result = model.fit(
        data.X,
        data.y,
        centers=data.centers,
        directions=data.directions,
    )

    np.testing.assert_array_equal(result.beta_ref, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.linalg.norm(result.beta_hat0), 1.0)
    np.testing.assert_allclose(result.beta_hat0, result.beta_path[0])
    assert {
        "local_mass_min",
        "local_mass_q05",
        "local_mass_q10",
        "local_mass_q25",
    } <= set(result.outer_telemetry[0])


def test_knn_bandwidth_variant_uses_q90_of_requested_neighbor_distance():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    centers = np.array([[0.0], [10.0]])
    model = ADP.create(
        "new",
        ADPConfig(min_neighbors=2.0, show_progress=False),
        stages={"bandwidth_selector": "knn_q90_k1"},
    )

    h0 = model.algorithm.components["bandwidth_selector"].select_initial(
        X,
        centers,
        None,
    )

    # The second nearest observations are at distances 1 and 8.
    assert h0 == pytest.approx(np.quantile([1.0, 8.0], 0.9))


def test_knn_bandwidth_rounds_the_scaled_neighbor_count_once():
    X = np.arange(6.0).reshape(-1, 1)
    centers = np.array([[0.0]])
    model = ADP.create(
        "new",
        ADPConfig(min_neighbors=2.5, show_progress=False),
        stages={"bandwidth_selector": "knn_q90_k2"},
    )

    h0 = model.algorithm.components["bandwidth_selector"].select_initial(
        X,
        centers,
        None,
    )

    # ceil(2 * 2.5) = 5, and the fifth observation is at distance 4.
    assert h0 == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("name", "quantile"),
    [
        ("local_mass_q0", 0.0),
        ("local_mass_q05", 0.05),
        ("local_mass_q10", 0.1),
        ("local_mass_q25", 0.25),
    ],
)
def test_quantile_bandwidth_variants_reach_the_named_local_mass_quantile(
    name,
    quantile,
):
    X = np.array([[0.0], [0.2], [0.4], [2.0], [4.0]])
    centers = np.array([[0.0], [0.4], [4.0]])
    model = ADP.create(
        "new",
        ADPConfig(
            min_neighbors=1.5,
            scale_expand_steps=20,
            scale_search_steps=30,
            show_progress=False,
        ),
        stages={"bandwidth_selector": name},
    )

    h0 = model.algorithm.components["bandwidth_selector"].select_initial(
        X,
        centers,
        None,
    )
    q = model._cached_pairwise_norm2(X, centers) / (h0 * h0)
    masses = model.backend.kernel(q, model.config.kernel).sum(axis=1)

    assert np.quantile(masses, quantile) >= model.config.min_neighbors - 2e-3


def test_zero_intercept_local_solver_uses_scalar_adp_regression_and_dtype():
    dtype = np.float32
    beta = np.array([1.0, -0.5], dtype=dtype)
    U = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=dtype,
    )
    imav = np.array([[2.0, -1.0, 0.5], [4.0, 5.0, 6.0]], dtype=dtype)
    statistics = LocalStatistics(
        variant="new",
        imav=imav,
        centers=np.zeros((2, 2), dtype=dtype),
        h=1.0,
        weights_mean=1.0,
        S=np.ones((2, 3), dtype=dtype),
        U=U,
    )
    model = ADP.create(
        "new",
        ADPConfig(dtype="float32", show_progress=False),
        stages={"local_solver": "zero_intercept"},
    )

    intercepts, slopes = model.algorithm.components["local_solver"].solve(
        statistics,
        beta,
    )

    projected = U @ beta
    expected = np.sum(imav * projected, axis=1) / np.maximum(
        np.sum(projected * projected, axis=1),
        np.finfo(dtype).tiny,
    )
    np.testing.assert_array_equal(intercepts, np.zeros(2, dtype=dtype))
    np.testing.assert_allclose(slopes, expected)
    assert intercepts.dtype == dtype
    assert slopes.dtype == dtype


@pytest.mark.parametrize("name", ("cpu_batched", "cpu_compact_factored"))
def test_removed_statistics_builders_cannot_be_selected(name):
    with pytest.raises(ValueError, match="statistics_builder"):
        ADP.create(
            "new",
            ADPConfig(show_progress=False),
            stages={"statistics_builder": name},
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
