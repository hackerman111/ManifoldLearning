import math
from collections import Counter
from itertools import product

import pytest

from adp.evaluation.single_index.scenarios import (
    EXPERIMENT_COUNTS,
    EXPERIMENT_SELECTORS,
    full_parameter_grid,
    parse_experiment_selectors,
    parse_seed_selection,
    smoke_parameter_grid,
)
from adp.evaluation.single_index.types import (
    ExperimentParameters,
    SingleIndexSeriesConfig,
)


EXPECTED_COUNTS = {
    "1": 800,
    "2": 2_000,
    "3": 4_200,
    "4": 3_600,
    "5": 3_000,
    "6": 3_600,
    "7.1": 1_200,
    "7.2": 1_200,
    "8.1": 800,
    "8.2": 2_000,
    "8.3": 1_600,
}


def test_full_grids_have_exact_independent_configuration_counts():
    configuration_counts = {
        selector: len(full_parameter_grid(selector))
        for selector in EXPERIMENT_SELECTORS
    }

    assert configuration_counts == {
        selector: count // 100 for selector, count in EXPECTED_COUNTS.items()
    }
    assert EXPERIMENT_COUNTS == EXPECTED_COUNTS
    assert sum(EXPERIMENT_COUNTS.values()) == 24_000


def test_each_experiment_changes_only_its_declared_parameters():
    defaults = ExperimentParameters(d=25, n_over_d=5)
    allowed = {
        "1": {"d", "n_over_d", "link", "sigma_eps"},
        "2": {"d", "n_over_d"},
        "3": {"d", "n_over_d", "sigma_eps"},
        "4": {"d", "n_over_d", "rho_corr"},
        "5": {"d", "n_over_d", "sigma_x"},
        "6": {"d", "n_over_d", "link"},
        "7.1": {"d", "n_over_d", "x_distribution"},
        "7.2": {"d", "n_over_d", "noise_distribution"},
        "8.1": {"d", "n_over_d", "heteroscedastic"},
        "8.2": {"d", "n_over_d", "outlier_fraction", "outlier_scale"},
        "8.3": {"d", "n_over_d", "delta"},
    }

    for selector in EXPERIMENT_SELECTORS:
        for parameters in full_parameter_grid(selector):
            changed = {
                name
                for name in parameters.__dataclass_fields__
                if getattr(parameters, name) != getattr(defaults, name)
            }
            assert changed <= allowed[selector]


def test_full_grids_use_the_literal_cartesian_products():
    expected = {
        "1": {
            (d, n_over_d, link, 0.0)
            for d, n_over_d, link in product(
                (5, 25),
                (5.0, 10.0),
                ("linear", "quadratic"),
            )
        },
        "2": set(
            product(
                (5, 25, 50, 100),
                (1.0, 1.15, 2.0, 5.0, 10.0),
            )
        ),
        "3": set(
            product(
                (25, 100),
                (2.0, 5.0, 10.0),
                (0.0, 0.316, 0.5, 0.707, 1.0, 1.414, 2.0),
            )
        ),
        "4": set(
            product(
                (25, 100),
                (2.0, 5.0, 10.0),
                (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
            )
        ),
        "5": set(
            product(
                (25, 100),
                (2.0, 5.0, 10.0),
                (0.25, 0.5, 1.0, 2.0, 4.0),
            )
        ),
        "6": set(
            product(
                (25, 100),
                (2.0, 5.0, 10.0),
                ("linear", "quadratic", "square", "sin", "tanh", "oscillating"),
            )
        ),
        "7.1": set(
            product(
                (25, 100),
                (2.0, 5.0),
                ("gaussian", "uniform", "student_t5"),
            )
        ),
        "7.2": set(
            product(
                (25, 100),
                (2.0, 5.0),
                ("gaussian", "student_t5", "student_t3"),
            )
        ),
        "8.1": set(product((25, 100), (2.0, 5.0), (False, True))),
        "8.3": set(product((25, 100), (2.0, 5.0), (0.0, 0.1, 0.25, 0.5))),
    }
    fields = {
        "1": ("d", "n_over_d", "link", "sigma_eps"),
        "2": ("d", "n_over_d"),
        "3": ("d", "n_over_d", "sigma_eps"),
        "4": ("d", "n_over_d", "rho_corr"),
        "5": ("d", "n_over_d", "sigma_x"),
        "6": ("d", "n_over_d", "link"),
        "7.1": ("d", "n_over_d", "x_distribution"),
        "7.2": ("d", "n_over_d", "noise_distribution"),
        "8.1": ("d", "n_over_d", "heteroscedastic"),
        "8.3": ("d", "n_over_d", "delta"),
    }

    for selector, names in fields.items():
        observed = {
            tuple(getattr(parameters, name) for name in names)
            for parameters in full_parameter_grid(selector)
        }
        assert observed == expected[selector]

    outlier_configurations = {
        (0.0, 1.0),
        (0.01, 5.0),
        (0.01, 10.0),
        (0.05, 5.0),
        (0.05, 10.0),
    }
    expected_outliers = {
        (d, n_over_d, fraction, scale)
        for d, n_over_d, (fraction, scale) in product(
            (25, 100),
            (2.0, 5.0),
            outlier_configurations,
        )
    }
    observed_outliers = {
        (
            parameters.d,
            parameters.n_over_d,
            parameters.outlier_fraction,
            parameters.outlier_scale,
        )
        for parameters in full_parameter_grid("8.2")
    }
    assert observed_outliers == expected_outliers


def test_selector_and_seed_parsers_are_canonical_and_strict():
    assert parse_experiment_selectors("all") == EXPERIMENT_SELECTORS
    assert parse_experiment_selectors("8.3,1,1") == ("1", "8.3")
    assert parse_seed_selection("0:3") == (0, 1, 2, 3)
    assert parse_seed_selection("5,2,5") == (2, 5)

    with pytest.raises(ValueError, match="unknown experiment selector"):
        parse_experiment_selectors("9.1")
    with pytest.raises(ValueError, match="seed range"):
        parse_seed_selection("3:1")
    with pytest.raises(ValueError, match="nonnegative"):
        parse_seed_selection("-1")


def test_parameter_sizes_and_series_validation():
    parameters = ExperimentParameters(d=25, n_over_d=1.15, center_fraction=0.25)
    assert parameters.n == math.ceil(25 * 1.15)
    assert parameters.n_centers == math.ceil(parameters.n * 0.25)

    with pytest.raises(ValueError, match="jobs"):
        SingleIndexSeriesConfig(jobs=0)
    with pytest.raises(ValueError, match="center_fraction"):
        SingleIndexSeriesConfig(center_fraction=0.0)
    with pytest.raises(ValueError, match="diagnostic_seeds"):
        SingleIndexSeriesConfig(diagnostic_seeds=(-1,))


def test_local_solver_selection_is_ordered_unique_and_strict():
    config = SingleIndexSeriesConfig(
        local_solvers=("zero_intercept", "least_squares", "zero_intercept")
    )

    assert config.local_solvers == ("zero_intercept", "least_squares")
    with pytest.raises(ValueError, match="local_solvers"):
        SingleIndexSeriesConfig(local_solvers=())
    with pytest.raises(ValueError, match="unknown local solver"):
        SingleIndexSeriesConfig(local_solvers=("missing",))


def test_full_grid_values_are_finite_and_valid():
    distribution_counts = Counter()
    for selector in EXPERIMENT_SELECTORS:
        for parameters in full_parameter_grid(selector):
            assert parameters.d > 0
            assert parameters.n > 0
            assert parameters.n_centers == parameters.n
            assert 0.0 <= parameters.rho_corr < 1.0
            assert parameters.sigma_x > 0.0
            assert parameters.sigma_eps >= 0.0
            distribution_counts[parameters.x_distribution] += 1
    assert {"gaussian", "uniform", "student_t5"} <= set(distribution_counts)


def test_experiment_two_smoke_uses_original_single_configuration():
    grid = smoke_parameter_grid("2")

    assert grid == (ExperimentParameters(d=4, n_over_d=2.0),)
