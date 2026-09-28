"""Focused regression for the manifold grid boundary and recovery protocol."""

import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from experiments.manifold_grid import Cell, _run_case, main


def test_grid_boundary_and_recovery_protocol(tmp_path: Path) -> None:
    output = tmp_path / "grid"
    with threadpool_limits(limits=1):
        assert (
            main(["--series", "dimension", "--runs", "1", "--output-dir", str(output)])
            == 0
        )
        strict = _run_case(Cell(200, 3, 1, 0.35, 0.1), 81000, "raise")
        curved = _run_case(Cell(400, 10, 2, 0.35, 0.1), 81000, "stop")

    rows = [
        json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    first = rows[0]

    assert manifest["schema_version"] == 2
    assert manifest["estimator"]["scale_boundary"] == "stop"
    assert len(rows) == 4 and all(row["error"] is None for row in rows)
    assert first["stop_reason"] == "function_mass_boundary"
    assert first["identifiable_target"] is True and first["recovered"] is False
    assert first["chart_estimation_max"] > 0.2
    assert first["query_oracle_max"] > 0
    assert summary["series"]["dimension"]["identifiable_finished"] == 4
    assert summary["series"]["dimension"]["function_boundary_stops"] == 4
    assert strict["error"] == (
        "RuntimeError: function mass target is infeasible even with alpha=0"
    )
    assert curved["identifiable_target"] is False and curved["recovered"] is None
