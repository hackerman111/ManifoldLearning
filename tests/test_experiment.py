from __future__ import annotations

import csv
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ADP import ADP_Config
from ADP.cli.experiment import (
    CATALOG,
    Build,
    Experiment,
    ExperimentPoint,
    _effective_config,
    _experiment_id,
    _generate_data,
    _has_numerical_failures,
    _link,
    _local_subspace_metrics,
    _make_seed_bundle,
    _outcome_fields,
    _selected_experiments,
    _subspace_metrics,
    custom_experiment,
    main as experiment_main,
)
from ADP.cli.experiment_plots import (
    _phase_plot,
    _phase_summaries,
    _wilson_interval,
    _write_phase_plots,
    build_report,
)
from ADP.engine.utils import _prepare_xy


def test_catalog_has_all_main_experiments() -> None:
    legacy = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7.1",
        "7.2",
        "8.1",
        "8.2",
        "8.3",
    )
    assert tuple(CATALOG)[: len(legacy)] == legacy
    assert tuple(CATALOG)[-1] == "custom"
    assert {
        selector: len(experiment.points("full"))
        for selector, experiment in CATALOG.items()
        if selector in legacy
    } == {
        "1": 8,
        "2": 44,
        "3": 66,
        "4": 48,
        "5": 42,
        "6": 36,
        "7.1": 12,
        "7.2": 12,
        "8.1": 8,
        "8.2": 28,
        "8.3": 28,
    }
    assert len(CATALOG["si-d"].full) == 10
    assert len(CATALOG["si-centers"].full) == 7
    assert len(CATALOG["mi-nphi"].full) == 7
    assert len(CATALOG["mi-tensor"].full) == 2
    assert len(CATALOG["si-breaking"].full) == 8960
    assert len(CATALOG["mi-breaking"].full) == 8960
    assert CATALOG["si-breaking"].full_runs == 100
    assert CATALOG["mi-breaking"].full_runs == 100
    assert CATALOG["2"].full_runs == 25
    assert CATALOG["2"].quality_threshold == 0.9
    assert CATALOG["2"].report_fields == ("d", "n_over_d")
    assert len(CATALOG["manifold"].full) == 3
    assert CATALOG["manifold"].quality_threshold == 0.2
    assert CATALOG["manifold"].smoke.mode == "manifold"
    manifold_points = {
        "manifold": 3,
        "manifold-n": 4,
        "manifold-d": 6,
        "manifold-noise": 6,
        "manifold-scale": 4,
        "manifold-corr": 5,
        "manifold-nlin": 5,
        "manifold-nloc": 5,
        "manifold-centers": 5,
        "manifold-nphi": 6,
        "manifold-neighbors": 5,
        "manifold-lambda": 5,
        "manifold-sync": 4,
    }
    assert {
        selector: len(CATALOG[selector].full) for selector in manifold_points
    } == manifold_points
    assert all(
        CATALOG[selector].full_runs == 5
        and CATALOG[selector].quality_threshold == 0.2
        and CATALOG[selector].hypothesis
        for selector in manifold_points
    )
    assert {
        selector: len(CATALOG[selector].full)
        for selector in ("mi-1", "mi-2", "mi-3", "mi-4", "mi-5")
    } == {"mi-1": 33, "mi-2": 66, "mi-3": 48, "mi-4": 42, "mi-5": 240}
    assert all(
        CATALOG[selector].full_runs == 25
        and CATALOG[selector].quality_threshold == 0.95
        for selector in ("mi-1", "mi-2", "mi-3", "mi-4", "mi-5")
    )
    assert {
        selector: (
            CATALOG[selector].condition_field,
            CATALOG[selector].common_random_fields,
            CATALOG[selector].condition_group_fields,
        )
        for selector in ("mi-1", "mi-2", "mi-3", "mi-4", "mi-5")
    } == {
        "mi-1": (None, (), ()),
        "mi-2": ("sigma_eps", ("sigma_eps",), ()),
        "mi-3": ("rho_corr", ("rho_corr",), ()),
        "mi-4": ("sigma_x", ("sigma_x",), ()),
        "mi-5": (
            "index_dim",
            ("index_dim", "link", "link_scale"),
            ("link", "link_scale"),
        ),
    }
    assert {point.d for point in CATALOG["mi-5"].full} == {25, 50}
    assert {point.d for point in CATALOG["mi-1"].full} == {5, 25, 50}
    assert all(
        {point.d for point in CATALOG[selector].full} == {25, 50}
        for selector in ("mi-2", "mi-3", "mi-4")
    )
    assert {point.index_dim for point in CATALOG["mi-5"].full} == {2, 3, 5, 7, 10}
    assert {
        selector: (
            CATALOG[selector].full_runs,
            CATALOG[selector].quality_threshold,
            CATALOG[selector].condition_field,
            CATALOG[selector].common_random_fields,
        )
        for selector in ("3", "4", "5")
    } == {
        "3": (25, 0.9, "sigma_eps", ("sigma_eps",)),
        "4": (25, 0.9, "rho_corr", ("rho_corr",)),
        "5": (25, 0.9, "sigma_x", ("sigma_x",)),
    }
    assert not any(
        method in selector.lower()
        for selector in CATALOG
        for method in ("mave", "sir", "ade")
    )


def test_tex_tuning_grids_are_detailed_paired_and_bounded() -> None:
    numeric = (
        "si-nlin",
        "si-centers",
        "si-displacement",
        "si-kmax",
        "si-nloc",
        "si-nphi",
        "si-lambda",
        "si-a",
        "si-hmin",
        "mi-nlin",
        "mi-centers",
        "mi-displacement",
        "mi-kmax",
        "mi-nloc",
        "mi-nphi",
        "mi-lambda",
        "mi-a",
        "mi-hmin",
    )
    assert all(len(CATALOG[selector].full) == 7 for selector in numeric)
    assert tuple(point.lambda_penalty for point in CATALOG["si-lambda"].full) == (
        0.0,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    )
    for selector in numeric:
        experiment = CATALOG[selector]
        assert experiment.common_random_fields == (experiment.condition_field,)
        assert (
            len(
                {
                    _make_seed_bundle(
                        selector,
                        point,
                        7,
                        common_random_fields=experiment.common_random_fields,
                    )
                    for point in experiment.full
                }
            )
            == 1
        )

    selected = _selected_experiments(
        "tex-tuning",
        custom=custom_experiment(),
    )
    assert len(selected) == 24
    assert sum(len(experiment.full) for experiment in selected) == 138
    assert all(
        point.outer_steps == 3 for experiment in selected for point in experiment.full
    )
    assert all(
        experiment.common_random_fields == (experiment.condition_field,)
        for experiment in selected
    )


def test_remaining_tex_grids_are_detailed_paired_and_bounded() -> None:
    numeric = (
        "si-n",
        "si-noise",
        "si-tau",
        "si-frequency-sin",
        "si-frequency-xsin",
        "si-scale",
        "mi-n",
        "mi-noise",
        "mi-tau",
        "mi-frequency-additive",
        "mi-frequency-multiplicative",
        "mi-scale",
    )
    assert all(len(CATALOG[selector].full) == 7 for selector in numeric)
    selected = _selected_experiments("tex-all", custom=custom_experiment())
    assert len(selected) == 40
    assert sum(len(experiment.full) for experiment in selected) == 246
    assert all(
        point.outer_steps == 3 for experiment in selected for point in experiment.full
    )
    assert all(
        experiment.common_random_fields == (experiment.condition_field,)
        for experiment in selected
    )


def test_single_function_hypotheses_are_paired_and_bounded() -> None:
    expected_points = {
        "si-function-classes": 12,
        "si-parity-frequency": 14,
        "si-gradient-support": 14,
        "si-init-by-function": 12,
    }
    selected = _selected_experiments(
        "si-hypotheses",
        custom=custom_experiment(),
    )

    assert tuple(experiment.selector for experiment in selected) == tuple(
        expected_points
    )
    assert {
        experiment.selector: len(experiment.full) for experiment in selected
    } == expected_points
    assert (
        sum(len(experiment.full) * experiment.full_runs for experiment in selected)
        == 156
    )
    assert all(experiment.full_runs == 3 for experiment in selected)
    assert all(experiment.hypothesis for experiment in selected)
    assert all(
        point.d == 30
        and point.n == 600
        and point.outer_steps == 5
        and point.solver_max_steps == 8
        for experiment in selected
        for point in experiment.full
    )
    for experiment in selected:
        assert (
            len(
                {
                    _make_seed_bundle(
                        experiment.selector,
                        point,
                        7,
                        common_random_fields=experiment.common_random_fields,
                    )
                    for point in experiment.full
                }
            )
            == 1
        )


def test_second_batch_has_focused_paired_multi_and_single_experiments() -> None:
    multi_points = {
        "mi-focus-d": 7,
        "mi-focus-frequency-additive": 7,
        "mi-focus-nphi": 7,
        "mi-focus-tensor": 2,
        "mi-focus-init": 2,
        "mi-focus-noise": 7,
    }
    single_selectors = (
        "si-function-classes",
        "si-parity-frequency",
        "si-gradient-support",
        "si-init-by-function",
    )
    selected = _selected_experiments("batch-2", custom=custom_experiment())
    focused_multi = selected[: len(multi_points)]

    assert tuple(experiment.selector for experiment in focused_multi) == tuple(
        multi_points
    )
    assert tuple(
        experiment.selector for experiment in selected[len(multi_points) :]
    ) == (single_selectors)
    assert {
        experiment.selector: len(experiment.full) for experiment in focused_multi
    } == multi_points
    assert (
        sum(len(experiment.full) * experiment.full_runs for experiment in focused_multi)
        == 96
    )
    assert (
        sum(len(experiment.full) * experiment.full_runs for experiment in selected)
        == 252
    )
    assert all(
        experiment.full_runs == 3 and experiment.hypothesis
        for experiment in focused_multi
    )
    assert tuple(point.d for point in CATALOG["mi-focus-d"].full) == (
        20,
        22,
        24,
        25,
        26,
        28,
        30,
    )
    assert tuple(
        point.link_scale for point in CATALOG["mi-focus-frequency-additive"].full
    ) == (2.0, 2.15, 2.3, 2.5, 2.7, 2.85, 3.0)
    assert tuple(point.N_phi for point in CATALOG["mi-focus-nphi"].full) == (
        20,
        24,
        28,
        30,
        32,
        36,
        40,
    )
    assert tuple(point.sigma_eps for point in CATALOG["mi-focus-noise"].full) == (
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.9,
        1.0,
    )
    assert all(
        experiment.quality_threshold == 0.95
        for experiment in focused_multi
        if experiment.selector not in {"mi-focus-tensor", "mi-focus-init"}
    )
    for experiment in focused_multi:
        assert (
            len(
                {
                    _make_seed_bundle(
                        experiment.selector,
                        point,
                        11,
                        common_random_fields=experiment.common_random_fields,
                    )
                    for point in experiment.full
                }
            )
            == 1
        )


def test_third_batch_repeats_single_boundaries_on_paired_seeds() -> None:
    expected_points = {
        "si-focus-d": 7,
        "si-focus-frequency": 14,
        "si-focus-tau": 7,
        "si-focus-displacement": 7,
    }
    selected = _selected_experiments("batch-3", custom=custom_experiment())

    assert tuple(experiment.selector for experiment in selected) == tuple(
        expected_points
    )
    assert {
        experiment.selector: len(experiment.full) for experiment in selected
    } == expected_points
    assert (
        sum(len(experiment.full) * experiment.full_runs for experiment in selected)
        == 175
    )
    assert all(
        experiment.full_runs == 5
        and experiment.quality_threshold == 0.9
        and experiment.hypothesis
        for experiment in selected
    )
    assert tuple(point.d for point in CATALOG["si-focus-d"].full) == (
        40,
        44,
        48,
        50,
        52,
        56,
        60,
    )
    assert tuple(
        point.link_scale for point in CATALOG["si-focus-frequency"].full[:7]
    ) == (2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0)
    assert tuple(point.tau for point in CATALOG["si-focus-tau"].full) == (
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
    )
    assert tuple(
        point.center_displacement for point in CATALOG["si-focus-displacement"].full
    ) == (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
    assert all(
        point.outer_steps == 3 and point.solver_max_steps == 3
        for experiment in selected
        for point in experiment.full
    )
    for experiment in selected:
        assert (
            len(
                {
                    _make_seed_bundle(
                        experiment.selector,
                        point,
                        13,
                        common_random_fields=experiment.common_random_fields,
                    )
                    for point in experiment.full
                }
            )
            == 1
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("cubic", np.array([-1.0, 0.0, 1.0])),
        ("quartic", np.array([1.0, 0.0, 1.0])),
        ("cos_scaled", np.cos(np.array([-2.0, 0.0, 2.0]))),
        ("tanh_scaled", np.tanh(np.array([-2.0, 0.0, 2.0]))),
        ("absolute", np.array([1.0, 0.0, 1.0])),
        ("relu", np.array([0.0, 0.0, 1.0])),
        ("gaussian_bump", np.exp(-0.5 * np.array([4.0, 0.0, 4.0]))),
    ),
)
def test_new_single_links_match_their_formulas(
    name: str,
    expected: np.ndarray,
) -> None:
    np.testing.assert_allclose(
        _link(np.array([-1.0, 0.0, 1.0]), name, scale=2.0),
        expected,
    )


def test_subspace_metrics_match_dense_projector_and_worst_angle() -> None:
    true_basis = np.eye(4)[:, :2]
    angle = np.deg2rad(60.0)
    estimate = np.column_stack(
        (
            np.eye(4)[:, 0],
            np.cos(angle) * np.eye(4)[:, 1] + np.sin(angle) * np.eye(4)[:, 2],
        )
    )

    trace_score, projector_distance, max_sine, max_angle = _subspace_metrics(
        true_basis,
        estimate,
    )
    dense_reference = (
        np.linalg.norm(
            estimate.T @ (np.eye(4) - true_basis @ true_basis.T),
            ord="fro",
        )
        ** 2
    )

    assert trace_score == pytest.approx(0.625, abs=1e-14)
    assert projector_distance == pytest.approx(dense_reference, abs=1e-14)
    assert max_sine == pytest.approx(np.sin(angle), abs=1e-14)
    assert max_angle == pytest.approx(60.0, abs=1e-12)
    assert _subspace_metrics(true_basis, true_basis) == pytest.approx(
        (1.0, 0.0, 0.0, 0.0),
        abs=1e-14,
    )
    with pytest.raises(RuntimeError, match="not orthonormal"):
        _subspace_metrics(true_basis, 2 * estimate)


@pytest.mark.parametrize("selector", ("6", "7.1", "7.2", "8.1", "8.3"))
def test_legacy_condition_levels_use_common_random_numbers(selector: str) -> None:
    experiment = CATALOG[selector]
    points = tuple(
        point for point in experiment.full if point.d == 25 and point.n_over_d == 2
    )
    assert len(points) > 1
    assert (
        len(
            {
                _make_seed_bundle(
                    selector,
                    point,
                    7,
                    common_random_fields=experiment.common_random_fields,
                )
                for point in points
            }
        )
        == 1
    )


def test_outlier_cases_use_common_random_numbers() -> None:
    experiment = CATALOG["8.2"]
    points = tuple(
        point for point in experiment.full if point.d == 25 and point.n_over_d == 2
    )
    assert len(points) == 7
    assert (
        len(
            {
                _make_seed_bundle(
                    experiment.selector,
                    point,
                    7,
                    common_random_fields=experiment.common_random_fields,
                )
                for point in points
            }
        )
        == 1
    )


def test_catalog_data_is_deterministic() -> None:
    experiment = CATALOG["8.3"]
    point = experiment.smoke
    seeds = _make_seed_bundle(experiment.selector, point, 7)

    first = _generate_data(experiment.selector, point, seeds, 7)
    second = _generate_data(experiment.selector, point, seeds, 7)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    np.testing.assert_array_equal(first.beta, second.beta)


@pytest.mark.parametrize(
    ("selector", "condition_field"),
    (
        ("3", "sigma_eps"),
        ("4", "rho_corr"),
        ("5", "sigma_x"),
        ("mi-2", "sigma_eps"),
        ("mi-3", "rho_corr"),
        ("mi-4", "sigma_x"),
    ),
)
def test_condition_levels_use_common_random_numbers(
    selector: str,
    condition_field: str,
) -> None:
    experiment = CATALOG[selector]
    points = [
        point for point in experiment.full if point.d == 25 and point.n_over_d == 2
    ]
    bundles = {
        _make_seed_bundle(
            selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        for point in points
    }
    other_dimension = 50 if selector.startswith("mi-") else 100
    other_point = next(
        point
        for point in experiment.full
        if point.d == other_dimension and point.n_over_d == 2
    )
    other_bundle = _make_seed_bundle(
        selector,
        other_point,
        7,
        common_random_fields=experiment.common_random_fields,
    )

    assert len({getattr(point, condition_field) for point in points}) > 1
    assert len(bundles) == 1
    assert other_bundle not in bundles


def test_noise_experiment_scales_one_noise_draw() -> None:
    experiment = CATALOG["3"]
    points = {
        point.sigma_eps: point
        for point in experiment.full
        if point.d == 25 and point.n_over_d == 2
    }

    def generate(level: float):
        point = points[level]
        seeds = _make_seed_bundle(
            experiment.selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        return _generate_data(experiment.selector, point, seeds, 7)

    zero, low, high = generate(0.0), generate(0.2), generate(0.8)

    np.testing.assert_array_equal(zero.X, low.X)
    np.testing.assert_array_equal(zero.beta, high.beta)
    np.testing.assert_allclose(
        (low.Y - zero.Y) / 0.2,
        (high.Y - zero.Y) / 0.8,
        rtol=1e-14,
        atol=1e-14,
    )


def test_scale_experiment_changes_only_feature_scale() -> None:
    experiment = CATALOG["5"]
    points = {
        point.sigma_x: point
        for point in experiment.full
        if point.d == 25 and point.n_over_d == 2
    }

    def generate(level: float):
        point = points[level]
        seeds = _make_seed_bundle(
            experiment.selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        return _generate_data(experiment.selector, point, seeds, 7)

    low, high = generate(0.125), generate(8.0)

    np.testing.assert_array_equal(high.X, 64 * low.X)
    np.testing.assert_array_equal(low.beta, high.beta)
    np.testing.assert_allclose(low.Y, high.Y, rtol=1e-14, atol=1e-14)


def test_multi_scale_experiment_changes_only_feature_scale() -> None:
    experiment = CATALOG["mi-4"]
    points = {
        point.sigma_x: point
        for point in experiment.full
        if point.d == 25 and point.n_over_d == 2
    }

    def generate(level: float):
        point = points[level]
        seeds = _make_seed_bundle(
            experiment.selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        return _generate_data(experiment.selector, point, seeds, 7)

    low, high = generate(0.125), generate(8.0)

    np.testing.assert_array_equal(high.X, 64 * low.X)
    np.testing.assert_array_equal(low.beta, high.beta)
    np.testing.assert_allclose(low.Y, high.Y, rtol=1e-14, atol=1e-14)


def test_multi_index_dimension_uses_one_nested_basis() -> None:
    experiment = CATALOG["mi-5"]
    paired_points = [
        point for point in experiment.full if point.d == 25 and point.n_over_d == 2
    ]
    points = [
        point
        for point in paired_points
        if point.link == "multi_additive" and point.link_scale == 1
    ]
    generated = {}
    bundles = {
        _make_seed_bundle(
            experiment.selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        for point in paired_points
    }
    for point in points:
        seeds = _make_seed_bundle(
            experiment.selector,
            point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        generated[point.index_dim] = _generate_data(
            experiment.selector,
            point,
            seeds,
            7,
        )

    reference = generated[10]
    assert len(bundles) == 1
    other_point = next(
        point for point in experiment.full if point.d == 50 and point.n_over_d == 2
    )
    assert (
        _make_seed_bundle(
            experiment.selector,
            other_point,
            7,
            common_random_fields=experiment.common_random_fields,
        )
        not in bundles
    )
    for index_dim, data in generated.items():
        np.testing.assert_array_equal(data.X, reference.X)
        np.testing.assert_array_equal(data.beta, reference.beta[:, :index_dim])
        np.testing.assert_allclose(
            data.beta.T @ data.beta,
            np.eye(index_dim),
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"condition_field": ""},
        {"common_random_fields": ("missing",)},
        {
            "condition_field": "sigma_eps",
            "condition_group_fields": ("link", "link"),
        },
    ),
)
def test_experiment_condition_fields_must_name_point_fields(
    kwargs: dict[str, object],
) -> None:
    point = ExperimentPoint(4, 5)

    with pytest.raises(ValueError, match="ExperimentPoint field"):
        Experiment("test", "test", point, (point,), **kwargs)


@pytest.mark.parametrize("hypothesis", ("", "two\nlines"))
def test_experiment_hypothesis_must_be_nonempty_and_single_line(
    hypothesis: str,
) -> None:
    point = ExperimentPoint(4, 5)

    with pytest.raises(ValueError, match="hypothesis"):
        Experiment("test", "test", point, (point,), hypothesis=hypothesis)


@pytest.mark.parametrize("basis_pool_dim", (1, 5))
def test_basis_pool_dimension_must_contain_target_subspace(
    basis_pool_dim: int,
) -> None:
    with pytest.raises(ValueError, match="basis_pool_dim"):
        ExperimentPoint(
            4,
            5,
            mode="multi",
            index_dim=2,
            link="multi_additive",
            basis_pool_dim=basis_pool_dim,
        )


def test_multi_index_report_data_is_deterministic_and_orthonormal() -> None:
    experiment = CATALOG["mi-link"]
    point = experiment.full[0]
    seeds = _make_seed_bundle(experiment.selector, point, 7)

    first = _generate_data(experiment.selector, point, seeds, 7)
    second = _generate_data(experiment.selector, point, seeds, 7)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    np.testing.assert_array_equal(first.beta, second.beta)
    assert first.beta.shape == (point.d, point.index_dim)
    np.testing.assert_allclose(
        first.beta.T @ first.beta,
        np.eye(point.index_dim),
        atol=1e-12,
    )


def test_manifold_data_and_local_metrics_are_deterministic() -> None:
    experiment = CATALOG["manifold"]
    point = experiment.smoke
    seeds = _make_seed_bundle(experiment.selector, point, 7)

    first = _generate_data(experiment.selector, point, seeds, 7)
    second = _generate_data(experiment.selector, point, seeds, 7)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    np.testing.assert_array_equal(first.beta, second.beta)
    assert first.beta.shape == (point.n, point.d, 1)
    np.testing.assert_allclose(
        np.linalg.norm(first.beta[:, :, 0], axis=1),
        1.0,
        atol=1e-14,
    )
    true = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    estimate = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    quality, worst_sine, worst_angle = _local_subspace_metrics(true, estimate)
    assert quality == pytest.approx(1 / np.sqrt(2))
    assert worst_sine == pytest.approx(1.0)
    assert worst_angle == pytest.approx(90.0)


def test_manifold_basic_catalog_is_paired_and_selectable() -> None:
    selected = _selected_experiments("manifold-basic", custom=custom_experiment())

    assert selected[0].selector == "manifold"
    assert len(selected) == 13
    for experiment in selected[1:]:
        assert experiment.condition_field is not None
        assert (
            len(
                {
                    _make_seed_bundle(
                        experiment.selector,
                        point,
                        7,
                        common_random_fields=experiment.common_random_fields,
                    )
                    for point in experiment.full
                }
            )
            == 1
        )
        assert all(point.mode == "manifold" for point in experiment.full)
    for selector in ("manifold-n", "manifold-d"):
        assert all(
            point.n_over_d == pytest.approx(point.n / point.d)
            for point in CATALOG[selector].full
        )

    algorithm = CATALOG["manifold-lambda"]
    first, last = algorithm.full[0], algorithm.full[-1]
    first_data = _generate_data(
        algorithm.selector,
        first,
        _make_seed_bundle(
            algorithm.selector,
            first,
            7,
            common_random_fields=algorithm.common_random_fields,
        ),
        7,
    )
    last_data = _generate_data(
        algorithm.selector,
        last,
        _make_seed_bundle(
            algorithm.selector,
            last,
            7,
            common_random_fields=algorithm.common_random_fields,
        ),
        7,
    )
    np.testing.assert_array_equal(first_data.X, last_data.X)
    np.testing.assert_array_equal(first_data.Y, last_data.Y)
    np.testing.assert_array_equal(first_data.beta, last_data.beta)


def test_effective_config_fits_small_catalog_point() -> None:
    point = CATALOG["2"].smoke

    effective = _effective_config(ADP_Config(), point, model_seed=7)

    assert (effective.N_loc, effective.N_lin, effective.N_J, effective.N_phi) == (
        8,
        8,
        8,
        4,
    )


def test_phase_outcomes_have_strict_precedence() -> None:
    assert (
        _outcome_fields("numerical_failure", False, None, 0.9, "higher")["failure_mode"]
        == "numerical_failure"
    )
    assert (
        _outcome_fields("nonconverged", False, 0.95, 0.9, "higher")["failure_mode"]
        == "nonconverged"
    )
    assert (
        _outcome_fields("success", True, 0.85, 0.9, "higher")["failure_mode"]
        == "converged_bad_quality"
    )
    assert _outcome_fields("success", True, 0.95, 0.9, "higher") == {
        "convergence_pass": True,
        "quality_pass": True,
        "recovered": True,
        "failure_mode": "recovered",
    }


def test_phase_summary_keeps_all_scaling_cells_separate() -> None:
    rows = [
        {
            "experiment": "2",
            "title": "Масштабирование",
            "d": str(point.d),
            "n_over_d": str(point.n_over_d),
            "build": "ADP",
            "status": "success",
            "quality": "0.95",
            "cosine_abs": "0.95",
            "solver_diagnostics": '{"converged": true}',
        }
        for point in CATALOG["2"].full
    ]
    recovery = {"metric": "cosine_abs", "direction": "higher", "threshold": 0.9}

    summary = _phase_summaries(rows, recovery)

    assert len(summary) == 44
    assert {(row["d"], row["n_over_d"]) for row in summary} == {
        (str(point.d), str(point.n_over_d)) for point in CATALOG["2"].full
    }
    assert all(row["recovery_rate"] == 1 for row in summary)


def test_conditioned_phase_summary_keeps_levels_separate_and_plots(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "experiment": "3",
            "title": "Устойчивость к шуму",
            "d": "25",
            "n_over_d": ratio,
            "sigma_eps": level,
            "build": "ADP",
            "status": "success",
            "quality": quality,
            "cosine_abs": quality,
            "solver_diagnostics": '{"converged": true}',
        }
        for ratio, level, quality in (
            ("2.0", "0.2", "0.95"),
            ("2.0", "0.8", "0.50"),
            ("5.0", "0.2", "0.96"),
            ("5.0", "0.8", "0.60"),
        )
    ]
    recovery = {"metric": "cosine_abs", "direction": "higher", "threshold": 0.9}

    summary = _phase_summaries(rows, recovery, "sigma_eps")
    _phase_plot(
        summary,
        (("recovery_rate", "P(recovered)"),),
        tmp_path / "phase.png",
        "Граница работоспособности",
        "sigma_eps",
    )

    assert len(summary) == 4
    assert {
        (row["n_over_d"], row["condition_field"], row["condition_level"])
        for row in summary
    } == {
        ("2.0", "sigma_eps", "0.2"),
        ("2.0", "sigma_eps", "0.8"),
        ("5.0", "sigma_eps", "0.2"),
        ("5.0", "sigma_eps", "0.8"),
    }
    assert (tmp_path / "phase.png").is_file()


def test_conditioned_phase_summary_rejects_nonnumeric_level() -> None:
    rows = [
        {
            "experiment": "3",
            "title": "Устойчивость к шуму",
            "d": "25",
            "n_over_d": "2.0",
            "sigma_eps": "bad",
            "build": "ADP",
            "status": "success",
            "quality": "0.95",
            "cosine_abs": "0.95",
            "solver_diagnostics": '{"converged": true}',
        }
    ]
    recovery = {"metric": "cosine_abs", "direction": "higher", "threshold": 0.9}

    with pytest.raises(ValueError, match="condition level"):
        _phase_summaries(rows, recovery, "sigma_eps")


def test_grouped_multi_phase_summary_does_not_mix_links(tmp_path: Path) -> None:
    rows = [
        {
            "experiment": "mi-5",
            "title": "Multi-index: размер подпространства и link",
            "d": "25",
            "n_over_d": "2.0",
            "index_dim": str(index_dim),
            "link": link,
            "link_scale": str(link_scale),
            "build": "ADP",
            "status": "success",
            "quality": "0.05",
            "projector_distance": "0.05",
            "solver_diagnostics": '{"converged": true}',
        }
        for index_dim, link, link_scale in product(
            (2, 10),
            ("multi_additive", "multi_multiplicative"),
            (1, 2, 3, 4),
        )
    ]
    recovery = {
        "metric": "projector_distance",
        "direction": "lower",
        "threshold": 0.1,
    }

    summary = _phase_summaries(
        rows,
        recovery,
        "index_dim",
        ("link", "link_scale"),
    )
    _write_phase_plots(
        summary,
        tmp_path,
        "index_dim",
        ("link", "link_scale"),
    )

    assert len(summary) == 16
    assert {(row["link"], row["link_scale"]) for row in summary} == {
        (link, str(link_scale))
        for link, link_scale in product(
            ("multi_additive", "multi_multiplicative"),
            (1, 2, 3, 4),
        )
    }
    assert len(tuple(tmp_path.glob("phase_diagram-*.png"))) == 8
    assert len(tuple(tmp_path.glob("failure_modes-*.png"))) == 8


def test_phase_report_writes_boundaries_and_plots(tmp_path) -> None:
    series = tmp_path / "series"
    series.mkdir()
    manifest = {
        "experiment": "2",
        "title": "Масштабирование",
        "report_fields": ["d", "n_over_d"],
        "recovery": {
            "metric": "cosine_abs",
            "direction": "higher",
            "threshold": 0.9,
        },
    }
    (series / "series.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = (
        ("5", "2.0", "success", "0.95", True),
        ("5", "2.0", "nonconverged", "0.95", False),
        ("25", "2.0", "success", "0.50", True),
        ("25", "2.0", "success", "0.60", True),
    )
    fieldnames = (
        "experiment",
        "title",
        "point",
        "run",
        "seed",
        "build",
        "d",
        "n_over_d",
        "status",
        "quality_metric",
        "quality_direction",
        "quality",
        "cosine_abs",
        "initial_quality",
        "last_quality",
        "fit_time_sec",
        "max_stage_traced_peak_mib",
        "trace",
        "solver_diagnostics",
    )
    with (series / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run, (d, ratio, status, quality, converged) in enumerate(rows):
            writer.writerow(
                {
                    "experiment": "2",
                    "title": "Масштабирование",
                    "point": 0 if d == "5" else 1,
                    "run": run,
                    "seed": run,
                    "build": "ADP",
                    "d": d,
                    "n_over_d": ratio,
                    "status": status,
                    "quality_metric": "cosine_abs",
                    "quality_direction": "higher",
                    "quality": quality,
                    "cosine_abs": quality,
                    "initial_quality": quality,
                    "last_quality": quality,
                    "fit_time_sec": 1,
                    "max_stage_traced_peak_mib": 1,
                    "trace": "[]",
                    "solver_diagnostics": json.dumps({"converged": converged}),
                }
            )

    build_report(series)

    with (series / "phase_summary.csv").open(encoding="utf-8") as stream:
        phase = list(csv.DictReader(stream))
    with (series / "boundary.csv").open(encoding="utf-8") as stream:
        boundary = list(csv.DictReader(stream))
    assert len(phase) == 2
    assert phase[0]["recovery_rate"] == "0.5"
    assert [row["boundary_kind"] for row in boundary] == [
        "transition,numerical",
        "estimator",
    ]
    assert (series / "plots/2/phase_diagram.png").is_file()
    assert (series / "plots/2/failure_modes.png").is_file()


def test_wilson_interval_for_one_failure() -> None:
    low, high = _wilson_interval(0, 1)

    assert low == pytest.approx(0.0, abs=1e-15)
    assert high == pytest.approx(0.7934506856)


@pytest.mark.parametrize("value", ("", ".", "..", "nested/id"))
def test_experiment_id_must_be_path_safe(value: str) -> None:
    with pytest.raises(ValueError, match="path-safe"):
        _experiment_id(value)


def test_high_dimensional_data_requires_random_initialization() -> None:
    X = np.ones((3, 4))
    Y = np.ones(3)

    with pytest.raises(ValueError, match="n must exceed"):
        _prepare_xy(X, Y)
    prepared, _ = _prepare_xy(X, Y, require_overdetermined=False)

    assert prepared.shape == (3, 4)


def test_paired_ab_writes_log_and_plots(tmp_path) -> None:
    config = ADP_Config(
        N_loc=6,
        N_lin=8,
        N_J=8,
        N_phi=3,
        outer_steps=1,
        h_min=1_000_000,
        index_init="random",
        batch_size=4,
    )
    experiment = custom_experiment(n=20, d=3, noise=0.01)
    experiment = replace(
        experiment,
        full=(experiment.smoke, experiment.smoke),
        hypothesis="Проверяемая гипотеза",
    )

    series = experiment.run(
        Build("A", config, solver_max_steps=2),
        Build("B", replace(config, batch_size=5), solver_max_steps=2),
        runs=1,
        seed=11,
        profile="full",
        output_dir=tmp_path,
    )

    with (series / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["build"] for row in rows] == ["A", "B", "B", "A"]
    assert len({row["seed_bundle"] for row in rows}) == 1
    assert {row["model_seed"] for row in rows} == {"11"}
    assert {row["status"] for row in rows} <= {"success", "nonconverged"}
    assert all(json.loads(row["trace"]) for row in rows)
    assert {row["selected_iteration"] for row in rows} == {"0"}
    manifest = json.loads((series / "series.json").read_text())
    assert series == tmp_path / manifest["experiment_id"] / "custom"
    assert manifest["schema_version"] == 9
    assert manifest["subspace_metrics"]["trace_score"] == (
        "trace(P_hat @ P_true) / m = mean(cos(theta_j)^2)"
    )
    assert manifest["subspace_metrics"]["projector_distance"] == ("sum(sin(theta_j)^2)")
    assert manifest["hypothesis"] == "Проверяемая гипотеза"
    assert manifest["subspace_metrics"]["max_principal_sine"] == ("max(sin(theta_j))")
    assert manifest["seed"] == 11
    assert (series / "plots/custom/quality.png").is_file()
    assert (series / "plots/custom/runtime.png").is_file()
    assert (series / "plots/custom/memory.png").is_file()
    assert (series / "plots/custom/failures.png").is_file()
    assert (series / "plots/custom/stages.png").is_file()
    assert (series / "plots/custom/trajectory.png").is_file()
    assert (series / "plots/custom/paired_delta.png").is_file()
    assert (series / "summary.csv").is_file()
    assert (series / "summary.md").is_file()
    assert (series / "trace_summary.csv").is_file()
    assert (series / "failures.csv").is_file()
    assert not _has_numerical_failures(series)

    with (series / "runs.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writerow({**rows[0], "status": "numerical_failure"})
    assert _has_numerical_failures(series)


def test_manifold_experiment_writes_local_quality_and_prediction(tmp_path) -> None:
    series = CATALOG["manifold"].run(
        Build("ADP", ADP_Config()),
        seed=3,
        output_dir=tmp_path,
        plots=False,
    )

    with (series / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "success"
    assert row["quality_metric"] == "local_projector_distance"
    assert row["quality_direction"] == "lower"
    assert row["recovered"] == "False"
    assert row["failure_mode"] == "converged_bad_quality"
    assert np.isfinite(float(row["prediction_rmse"]))
    assert json.loads(row["effective_config"])["solver"] == "cg"
    manifest = json.loads((series / "series.json").read_text())
    assert manifest["schema_version"] == 9
    assert manifest["manifold_link"] == "0.5 * (x_1^2 + x_2^2)"


def test_single_build_writes_tables_without_plots(tmp_path, monkeypatch) -> None:
    progress_bar = MagicMock()
    progress_bar.__enter__.return_value = progress_bar
    progress_factory = MagicMock(return_value=progress_bar)
    monkeypatch.setattr("experiments.runner.tqdm", progress_factory)
    config = ADP_Config(
        N_loc=6,
        N_lin=8,
        N_J=8,
        N_phi=3,
        outer_steps=1,
        h_min=1_000_000,
        index_init="random",
        batch_size=4,
    )
    experiment = custom_experiment(n=20, d=3, noise=0.01)

    series = experiment.run(
        Build(
            "ADP",
            config,
            solver_max_steps=2,
            solver="cg",
            cg_maxiter=100,
        ),
        seed=11,
        output_dir=tmp_path,
        plots=False,
        progress=True,
    )

    with (series / "runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["build"] for row in rows] == ["ADP"]
    assert json.loads(rows[0]["requested_config"])["solver"] == "cg"
    assert json.loads(rows[0]["effective_config"])["solver"] == "cg"
    manifest = json.loads((series / "series.json").read_text())
    assert manifest["builds"][0]["solver"] == "cg"
    assert (series / "summary.csv").is_file()
    assert (series / "summary.md").is_file()
    assert (series / "trace_summary.csv").is_file()
    assert (series / "failures.csv").is_file()
    assert not (series / "plots").exists()
    progress_factory.assert_called_once_with(
        total=1,
        desc="custom",
        unit="fit",
        disable=False,
    )
    progress_bar.update.assert_called_once_with()


def test_cli_returns_nonzero_for_numerical_failure(tmp_path, monkeypatch) -> None:
    series = tmp_path / "series"
    series.mkdir()
    (series / "runs.csv").write_text("status\nnumerical_failure\n", encoding="utf-8")
    monkeypatch.setattr(Experiment, "run", lambda *args, **kwargs: series)

    assert experiment_main(["--experiment", "custom", "--no-plots"]) == 1


def test_cli_groups_experiments_under_one_id(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, bool]] = []
    builds: list[Build] = []

    def fake_run(self: Experiment, *args: object, **kwargs: object) -> Path:
        experiment_id = kwargs["experiment_id"]
        progress = kwargs["progress"]
        assert isinstance(experiment_id, str)
        assert isinstance(progress, bool)
        assert isinstance(args[0], Build)
        builds.append(args[0])
        calls.append((self.selector, experiment_id, progress))
        series = tmp_path / experiment_id / self.selector.replace(".", "_")
        series.mkdir(parents=True)
        (series / "runs.csv").write_text("status\nsuccess\n", encoding="utf-8")
        return series

    monkeypatch.setattr(Experiment, "run", fake_run)

    assert (
        experiment_main(
            [
                "--experiment",
                "1,2",
                "--output-dir",
                str(tmp_path),
                "--no-plots",
                "--solver",
                "cg",
                "--solver-max-steps",
                "10",
                "--cg-maxiter",
                "17",
            ]
        )
        == 0
    )
    assert [selector for selector, _, _ in calls] == ["1", "2"]
    assert len({experiment_id for _, experiment_id, _ in calls}) == 1
    assert all(progress for _, _, progress in calls)
    assert {
        (build.solver, build.solver_max_steps, build.cg_maxiter) for build in builds
    } == {("cg", 10, 17)}
