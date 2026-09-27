from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ADP.core.ADP_Config import ADP_Config
from experiments.data import _make_seed_bundle
from experiments.multiv2_quality import (
    SELECTION_SEED,
    VALIDATION_SEED,
    analyze,
    main,
    make_case,
)
from experiments.runner import _effective_config


def _args(candidate_file: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        inner_steps=None, outer_steps=None, factor=None, candidate_file=candidate_file
    )


def test_inner_points_share_seed_bundles_and_validation_uses_new_seeds(
    tmp_path: Path, capsys
) -> None:
    experiment, labels = make_case("n1000", "inner", _args())
    assert labels == ("baseline", "solver_max_steps=50", "solver_max_steps=80")
    selection = {
        _make_seed_bundle(
            experiment.selector,
            point,
            SELECTION_SEED,
            common_random_fields=experiment.common_random_fields,
        )
        for point in experiment.full
    }
    assert len(selection) == 1
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"d10": {"solver_max_steps": 80}, "n1000": {"solver_max_steps": 80}})
    )
    validation, _ = make_case("n1000", "validation", _args(candidate))
    held_out = _make_seed_bundle(
        validation.selector,
        validation.full[0],
        VALIDATION_SEED,
        common_random_fields=validation.common_random_fields,
    )
    assert held_out not in selection
    assert main(["--stage", "inner", "--dry-run", "--output-dir", str(tmp_path)]) == 0
    assert "60 fits" in capsys.readouterr().out
    assert sorted(tmp_path.iterdir()) == [candidate]
    assert (
        main(
            [
                "--stage",
                "validation",
                "--candidate-file",
                str(candidate),
                "--dry-run",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "80 fits" in capsys.readouterr().out


def test_analysis_requires_complete_paired_rows_and_validation_does_not_select(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"d10": {"solver_max_steps": 80}, "n1000": {"solver_max_steps": 80}})
    )
    experiment, labels = make_case("d10", "validation", _args(candidate))
    series = tmp_path / "series"
    series.mkdir()
    (series / "series.json").write_text(
        json.dumps({"points": [asdict(p) for p in experiment.full]})
    )
    rows = []
    for index, point in enumerate(experiment.full):
        for run in range(20):
            seed = VALIDATION_SEED + run
            bundle = _make_seed_bundle(
                experiment.selector,
                point,
                seed,
                common_random_fields=experiment.common_random_fields,
            )
            effective = _effective_config(ADP_Config(), point, bundle.init)
            config = {
                field: getattr(effective, field)
                for field in experiment.common_random_fields
                if field != "solver_max_steps"
            }
            config["solver_max_steps"] = point.solver_max_steps
            rows.append(
                {
                    "point": str(index),
                    "run": str(run),
                    "seed": str(seed),
                    "seed_bundle": json.dumps(asdict(bundle), sort_keys=True),
                    "model_seed": str(bundle.init),
                    "effective_config": json.dumps(config),
                    "requested_config": "{}",
                    "status": "nonconverged",
                    "failure_mode": "quality_not_recovered",
                    "stop_reason": "outer_steps",
                    "quality": "0.8",
                    "initial_quality": "0.7",
                    "fit_time_sec": "1",
                    "max_stage_traced_peak_mib": "10",
                    "outer_iterations": "3",
                    "convergence_pass": "False",
                    "recovered": "False",
                    "error": "",
                    "trace": "[]",
                }
            )
    with (series / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="incomplete series"):
        analyze(
            experiment,
            labels,
            series,
            runs=20,
            seed=VALIDATION_SEED,
            stage="validation",
        )
    with (series / "runs.csv").open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(rows[0])).writerow(rows[-1])
    result = analyze(
        experiment, labels, series, runs=20, seed=VALIDATION_SEED, stage="validation"
    )
    assert "selected_candidate" not in result
    assert all(row["effective_config"] for row in result["summaries"])
