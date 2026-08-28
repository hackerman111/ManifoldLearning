from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from ADP import ADP_Config
from ADP.engine.utils import _prepare_xy
from ADP.experiment import (
    CATALOG,
    Build,
    Experiment,
    _effective_config,
    _generate_data,
    _has_numerical_failures,
    _make_seed_bundle,
    custom_experiment,
    main as experiment_main,
)


def test_catalog_has_all_main_experiments() -> None:
    assert tuple(CATALOG) == (
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
        "custom",
    )
    assert {
        selector: len(experiment.points("full"))
        for selector, experiment in CATALOG.items()
    } == {
        "1": 8,
        "2": 20,
        "3": 42,
        "4": 36,
        "5": 30,
        "6": 36,
        "7.1": 12,
        "7.2": 12,
        "8.1": 8,
        "8.2": 20,
        "8.3": 16,
        "custom": 1,
    }


def test_catalog_data_is_deterministic() -> None:
    experiment = CATALOG["8.3"]
    point = experiment.smoke
    seeds = _make_seed_bundle(experiment.selector, point, 7)

    first = _generate_data(experiment.selector, point, seeds, 7)
    second = _generate_data(experiment.selector, point, seeds, 7)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.Y, second.Y)
    np.testing.assert_array_equal(first.beta, second.beta)


def test_effective_config_fits_small_catalog_point() -> None:
    point = CATALOG["2"].smoke

    effective = _effective_config(ADP_Config(), point, model_seed=7)

    assert (effective.N_loc, effective.N_lin, effective.N_J, effective.N_phi) == (
        8,
        8,
        8,
        4,
    )


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
    experiment = replace(experiment, full=(experiment.smoke, experiment.smoke))

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
    assert json.loads((series / "series.json").read_text())["seed"] == 11
    assert (series / "plots/custom/quality.png").is_file()
    assert (series / "plots/custom/runtime.png").is_file()
    assert (series / "plots/custom/paired_delta.png").is_file()
    assert not _has_numerical_failures(series)

    with (series / "runs.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writerow({**rows[0], "status": "numerical_failure"})
    assert _has_numerical_failures(series)


def test_cli_returns_nonzero_for_numerical_failure(tmp_path, monkeypatch) -> None:
    series = tmp_path / "series"
    series.mkdir()
    (series / "runs.csv").write_text("status\nnumerical_failure\n", encoding="utf-8")
    monkeypatch.setattr(Experiment, "run", lambda *args, **kwargs: series)

    assert experiment_main(["--experiment", "custom", "--no-plots"]) == 1
