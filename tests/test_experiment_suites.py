from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ADP import ADP_Config
from ADP.cli.experiment_plots import build_report
from experiments import manifold, multi, single
from experiments.data import _make_seed_bundle
from experiments.models import Build, ModelMode
from experiments.registry import CATALOG
from experiments.suite import main, select_suite


def test_single_hybrid_is_rejected_before_writing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main("single", ["--solver", "hybrid", "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="use lsmr or cg"):
        CATALOG["1"].run(
            Build("ADP", ADP_Config(), solver="hybrid"), output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_all_failed_builds_produce_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("intentional numerical failure")

    monkeypatch.setattr("experiments.runner._fit", fail)
    path = CATALOG["1"].run(
        Build("A", ADP_Config()),
        Build("B", ADP_Config()),
        output_dir=tmp_path,
        plots=True,
    )
    assert (path / "plots" / "1" / "stages.png").is_file()
    with (path / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["status"] == "numerical_failure" for row in rows)


@pytest.mark.parametrize(
    "module,mode", [(single, "single"), (multi, "multi"), (manifold, "manifold")]
)
def test_family_catalogs_are_disjoint(module, mode) -> None:
    items = module.catalog()
    assert len({item.selector for item in items}) == len(items)
    assert all(point.mode == mode for item in items for point in item.full)
    assert set(module.DEFAULT_SELECTORS) <= {item.selector for item in items}
    assert not any(name.endswith("-breaking") for name in module.DEFAULT_SELECTORS)


@pytest.mark.parametrize("mode", ["single", "multi", "manifold"])
def test_overview_retains_original_points_and_random_streams(mode: ModelMode) -> None:
    for item in select_suite(mode, None):
        original = CATALOG[item.selector]
        points = item.points("overview")
        assert points
        assert set(points) <= set(original.full)
        assert item.quality_threshold is not None
        assert item.common_random_fields == original.common_random_fields
        for point in points:
            assert _make_seed_bundle(
                item.selector, point, 7, common_random_fields=item.common_random_fields
            ) == _make_seed_bundle(
                original.selector,
                point,
                7,
                common_random_fields=original.common_random_fields,
            )


def test_overview_preserves_endpoints_and_categories() -> None:
    experiment = CATALOG["si-parity-frequency"]
    points = experiment.points("overview")
    assert {point.link for point in points} == {"sin_scaled", "cos_scaled"}
    assert {point.link_scale for point in points} == {0.5, 2.0, 3.5}
    assert len(points) == 6
    assert experiment.points("full") == experiment.full
    assert experiment.points("smoke") == (experiment.smoke,)


@pytest.mark.parametrize("mode", ["single", "multi", "manifold"])
def test_standalone_script_dry_run_outside_repository(
    tmp_path: Path, mode: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "experiments" / f"{mode}.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "profile=overview" in result.stdout
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "arguments", [["--runs", "0"], ["--threads", "0"], ["--experiment", "mi-n"]]
)
def test_suite_rejects_invalid_requests_before_writing(
    tmp_path: Path, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main("single", [*arguments, "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mode,selector", [("single", "si-n"), ("multi", "mi-n"), ("manifold", "manifold")]
)
def test_suite_smoke_writes_reproducible_overview(
    tmp_path: Path, mode: ModelMode, selector: str
) -> None:
    code = main(
        mode,
        [
            "--profile",
            "smoke",
            "--experiment",
            selector,
            "--no-plots",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert code == 0
    (root,) = tmp_path.iterdir()
    suite = json.loads((root / "suite.json").read_text())
    assert suite["completed"] == [selector]
    assert suite["status"] == "completed"
    with (root / "overview.csv").open() as stream:
        (row,) = list(csv.DictReader(stream))
    assert row["n_total"] == "1"
    assert row["n_numerical_failure"] == "0"
    series = json.loads((root / selector / "series.json").read_text())
    assert len(series["points"]) == 1
    assert series["profile"] == "smoke"
    assert series["recovery"]["threshold"] is not None
    assert (root / selector / "phase_summary.csv").is_file()
    assert (root / "overview.md").is_file()


def test_failure_is_counted_and_next_series_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("intentional solver failure")

    monkeypatch.setattr("experiments.runner._fit", fail)
    assert (
        main(
            "single",
            [
                "--profile",
                "smoke",
                "--experiment",
                "si-n,si-d",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ],
        )
        == 1
    )
    (root,) = tmp_path.iterdir()
    suite = json.loads((root / "suite.json").read_text())
    assert suite["status"] == "completed_with_failures"
    assert suite["completed"] == ["si-n", "si-d"]
    with (root / "overview.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(
        row["n_numerical_failure"] == "1" and row["recovery_rate"] == "0.0"
        for row in rows
    )
    assert all("intentional solver failure" in row["errors"] for row in rows)


def test_overview_does_not_change_source_recovery_protocol() -> None:
    original = CATALOG["si-n"]
    assert original.quality_threshold is None
    (item,) = select_suite("single", "si-n")
    assert replace(item, quality_threshold=None) == original


@pytest.mark.parametrize("mode", ["single", "multi", "manifold"])
def test_complete_smoke_suite(tmp_path: Path, mode: ModelMode) -> None:
    assert (
        main(mode, ["--profile", "smoke", "--no-plots", "--output-dir", str(tmp_path)])
        == 0
    )
    (root,) = tmp_path.iterdir()
    manifest = json.loads((root / "suite.json").read_text())
    assert len(manifest["completed"]) == len(select_suite(mode, None))


def test_categorical_recovery_report_preserves_levels_and_plots(tmp_path: Path) -> None:
    assert (
        main(
            "multi",
            [
                "--profile",
                "smoke",
                "--experiment",
                "mi-link",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ],
        )
        == 0
    )
    (root,) = tmp_path.iterdir()
    series = root / "mi-link"
    with (series / "runs.csv").open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        (row,) = list(reader)
    assert columns is not None
    second = {
        **row,
        "link": "multi_multiplicative",
        "status": "numerical_failure",
        "quality": "",
        "trace_score": "",
        "convergence_pass": "False",
    }
    with (series / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([row, second])
    build_report(series, plots=True)
    with (series / "phase_summary.csv").open() as stream:
        phases = list(csv.DictReader(stream))
    assert {phase["condition_level"] for phase in phases} == {
        "multi_additive",
        "multi_multiplicative",
    }
    assert all(phase["n_total"] == "1" for phase in phases)
    assert (series / "plots" / "mi-link" / "phase_diagram.png").is_file()
