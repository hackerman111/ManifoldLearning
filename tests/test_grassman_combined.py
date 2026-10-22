"""Прямые проверки объединённых profile-поисков и нулевых центров."""

from __future__ import annotations

import csv
import json
from typing import Any, cast

import numpy as np
import pytest
from test_grassman_quadratic import case, direct_proposals, direct_value

from ADP.solver import grassman
from experiments.grassman_accuracy_2026_10_18 import backtracking


def test_public_defaults_enable_combined_twenty_steps_without_smooth_cap():
    import inspect

    signature = inspect.signature(grassman.solve).parameters
    assert signature["max_steps"].default == 20
    assert signature["relative_gn"].default is True
    assert signature["quadratic_trials"].default == 3
    assert signature["constant_centers"].default is True
    assert signature["smooth_max_steps"].default is None
    index, U, I, mass = case()
    default = grassman.solve(index, U, I, mass=mass)
    explicit = grassman.solve(
        index,
        U,
        I,
        mass=mass,
        max_steps=20,
        relative_gn=True,
        quadratic_trials=3,
        constant_centers=True,
        smooth_max_steps=None,
    )
    np.testing.assert_array_equal(default.index, explicit.index)
    assert default.diagnostics == explicit.diagnostics
    assert default.diagnostics["effective_max_steps"] == 20


def test_large_constant_does_not_hide_active_profile_search():
    index, U, I, mass = case()
    I[0] *= 1e12
    active = np.any(U != 0, axis=(1, 2))
    scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    reduced_scale = max(1.0, float(np.sum(mass[active, None] * I[active] ** 2)))
    settings: dict[str, Any] = {
        "max_steps": 2,
        "relative_gn": True,
        "quadratic_trials": 3,
        "smooth_max_steps": None,
    }
    reduced = grassman.solve(
        index,
        U[active],
        I[active],
        mass=mass[active],
        constant_centers=False,
        tol=1e-26 * scale / reduced_scale,
        **settings,
    )
    got = grassman.solve(
        index, U, I, mass=mass, constant_centers=True, tol=1e-26, **settings
    )
    np.testing.assert_array_equal(got.index, reduced.index)
    assert got.diagnostics["accepted_steps"] == 2
    history = cast(tuple[float, ...], reduced.diagnostics["profile_history"])
    assert history[-1] < history[0]


@pytest.mark.parametrize("all_zero", [False, True])
def test_no_partial_zero_reduction_matches_direct_path(all_zero):
    index, U, I, mass = case()
    U[:] = 0 if all_zero else U[1]
    settings = {
        "mass": mass,
        "max_steps": 2,
        "relative_gn": True,
        "quadratic_trials": 3,
        "smooth_max_steps": None,
    }
    got = grassman.solve(index, U, I, constant_centers=True, **settings)
    direct = grassman.solve(index, U, I, constant_centers=False, **settings)
    np.testing.assert_array_equal(got.index, direct.index)
    np.testing.assert_array_equal(got.coefficients, direct.coefficients)
    assert got.diagnostics["constant_centers"] == (len(I) if all_zero else 0)
    assert {
        k: v for k, v in got.diagnostics.items() if k != "constant_centers"
    } == direct.diagnostics


def test_public_quadratic_matches_direct_rank_one_proposals():
    index, U, I, mass = case()
    expected, tried, _ = direct_proposals(index, U, I, mass)
    got = grassman.solve(
        index,
        U,
        I,
        mass=mass,
        max_steps=1,
        relative_gn=False,
        quadratic_trials=3,
        constant_centers=False,
    )
    assert got.index.T @ got.index == pytest.approx(
        expected[0] @ expected[0].T, abs=3e-11
    )
    assert got.diagnostics["profile_objective"] == pytest.approx(expected[1], abs=3e-10)
    assert got.diagnostics["profile_evaluations"] == len(tried)


@pytest.mark.parametrize("ridge", [0.0, 0.2])
def test_zero_center_reduction_restores_full_objective_and_stopping_scale(ridge):
    index, U, I, mass = case()
    U[3] = 0
    I[[0, 3]] *= 100
    active = np.any(U != 0, axis=(1, 2))
    total_scale = max(1.0, float(np.sum(mass[:, None] * I**2)))
    reduced_scale = max(1.0, float(np.sum(mass[active, None] * I[active] ** 2)))
    options = {
        "max_steps": 7,
        "relative_gn": True,
        "quadratic_trials": 3,
        "smooth_max_steps": None,
        "local_ridge": ridge,
    }
    got = grassman.solve(
        index, U, I, mass=mass, tol=1e-9, constant_centers=True, **options
    )
    ref = backtracking.solve(
        index,
        U[active],
        I[active],
        mass=mass[active],
        tol=1e-9 * total_scale / reduced_scale,
        intrinsic_gn=True,
        normalized_damping=True,
        **{k: v for k, v in options.items() if k != "relative_gn"},
    )
    np.testing.assert_array_equal(got.index, ref.index)
    np.testing.assert_array_equal(got.coefficients[active], ref.coefficients)
    np.testing.assert_array_equal(got.coefficients[~active], np.zeros((2, 2)))
    assert got.diagnostics["constant_centers"] == 2
    assert got.diagnostics["effective_max_steps"] == 7
    assert got.diagnostics["profile_objective"] == pytest.approx(
        direct_value(U, got.index.T, I, mass, ridge), rel=1e-13
    )
    assert cast(
        float, got.diagnostics["riemannian_gradient"]
    ) * total_scale == pytest.approx(got.diagnostics["gradient_norm"], rel=1e-13)
    assert got.diagnostics["local_rank_loss"] == (0 if ridge else 2)
    assert np.all(np.diff(got.diagnostics["profile_history"]) <= 1e-9)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_combined_intrinsic_path_matches_audited_prototype(m):
    rng = np.random.default_rng(88 + m)
    U = rng.normal(size=(12, m + 3, m + 4))
    U[0] = 0
    U[1] = 0
    U[1, : m - 1, : m - 1] = np.eye(m - 1)
    I = rng.normal(size=U.shape[:2])
    mass = np.geomspace(0.2, 2, len(I))
    index = np.eye(m + 4)[:m]
    options = {
        "mass": mass,
        "max_steps": 6,
        "quadratic_trials": 3,
        "smooth_max_steps": None,
        "tol": 1e-10,
    }
    got = grassman.solve(
        index, U, I, relative_gn=True, constant_centers=False, **options
    )
    ref = backtracking.solve(
        index, U, I, intrinsic_gn=True, normalized_damping=True, **options
    )
    np.testing.assert_array_equal(got.index, ref.index)
    np.testing.assert_array_equal(got.coefficients, ref.coefficients)
    assert got.diagnostics["profile_history"] == ref.diagnostics["profile_history"]
    assert (
        got.diagnostics["stationarity_applicable"]
        == ref.diagnostics["stationarity_applicable"]
    )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"quadratic_trials": True}, "quadratic_trials"),
        ({"quadratic_trials": -1}, "quadratic_trials"),
        ({"quadratic_trials": 4}, "quadratic_trials"),
        ({"constant_centers": 1}, "constant_centers"),
        ({"smooth_max_steps": 0}, "smooth_max_steps"),
        ({"smooth_max_steps": True}, "smooth_max_steps"),
    ],
)
def test_invalid_combination_options_rejected(options, match):
    index, U, I, mass = case()
    with pytest.raises((TypeError, ValueError), match=match):
        grassman.solve(index, U, I, mass=mass, **options)


def test_multi_candidate_analysis_keeps_shared_pairs_and_explicit_baselines(tmp_path):
    from experiments.grassman_accuracy_2026_10_18.analyze import analyze

    methods = ["ADP-Grassmann", "ADP-Grassmann-relative", "ADP-Grassmann-combined"]
    protocol = {
        "methods": methods,
        "runs": 2,
        "seed": 1,
        "cases": [{"name": "a"}],
        "planned_fits": 6,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": protocol}))
    rows = [
        {
            "case": "a",
            "seed": seed,
            "method": method,
            "status": "ok",
            "fit_seconds": edr,
            "projector_distance": edr,
            "peak_rss_mib": 100,
            "orthogonality_error": 1e-15,
            "data_sha256": str(seed),
            "x_csv_sha256": str(seed),
            "y_csv_sha256": str(seed),
            "method_seed": seed,
        }
        for seed in (1, 2)
        for method, edr in zip(methods, (0.2, 0.4, 0.1), strict=True)
    ]
    with (tmp_path / "runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    got = analyze(tmp_path, candidate=methods[2], baseline=methods[0], write=False)
    incremental = analyze(
        tmp_path, candidate=methods[2], baseline=methods[1], write=False
    )
    assert got["overall"]["edr_ratio_geomean"] == pytest.approx(0.5)
    assert incremental["overall"]["edr_ratio_geomean"] == pytest.approx(0.25)
    assert got["paired_hash_audit"]["bad_blocks"] == []
    assert got["overall"]["joint_success_pairs"] == 2
