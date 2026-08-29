from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ADP import ADP_Config
from ADP.cli.experiment import (
    CATALOG,
    Build,
    Experiment,
    _effective_config,
    _experiment_id,
    _generate_data,
    _has_numerical_failures,
    _make_seed_bundle,
    custom_experiment,
    main as experiment_main,
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
    }
    assert len(CATALOG["si-d"].full) == 10
    assert len(CATALOG["si-centers"].full) == 20
    assert len(CATALOG["mi-nphi"].full) == 4
    assert len(CATALOG["mi-tensor"].full) == 2
    assert len(CATALOG["si-breaking"].full) == 8960
    assert len(CATALOG["mi-breaking"].full) == 8960
    assert CATALOG["si-breaking"].full_runs == 100
    assert CATALOG["mi-breaking"].full_runs == 100
    assert not any(
        method in selector.lower()
        for selector in CATALOG
        for method in ("mave", "sir", "ade")
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


def test_effective_config_fits_small_catalog_point() -> None:
    point = CATALOG["2"].smoke

    effective = _effective_config(ADP_Config(), point, model_seed=7)

    assert (effective.N_loc, effective.N_lin, effective.N_J, effective.N_phi) == (
        8,
        8,
        8,
        4,
    )


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
    manifest = json.loads((series / "series.json").read_text())
    assert series == tmp_path / manifest["experiment_id"] / "custom"
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


def test_single_build_writes_tables_without_plots(tmp_path, monkeypatch) -> None:
    progress_bar = MagicMock()
    progress_bar.__enter__.return_value = progress_bar
    progress_factory = MagicMock(return_value=progress_bar)
    monkeypatch.setattr("ADP.cli.experiment.tqdm", progress_factory)
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
                "--cg-maxiter",
                "17",
            ]
        )
        == 0
    )
    assert [selector for selector, _, _ in calls] == ["1", "2"]
    assert len({experiment_id for _, experiment_id, _ in calls}) == 1
    assert all(progress for _, _, progress in calls)
    assert {(build.solver, build.cg_maxiter) for build in builds} == {("cg", 17)}
