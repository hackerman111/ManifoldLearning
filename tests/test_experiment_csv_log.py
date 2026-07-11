import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adp.common.experiment_log as experiment_log
from adp.common.experiment_log import (
    CSVTable,
    configuration_fingerprint,
    flatten_mapping,
    merge_csv_shards,
    replace_single_row_csv,
    stable_run_id,
)


def test_csv_table_appends_rows_with_stable_header(tmp_path):
    table = CSVTable(
        tmp_path / "runs.csv",
        ("schema_version", "run_id", "failed"),
    )

    table.append({"run_id": "r1", "failed": False})
    table.append({"run_id": "r2", "failed": True})

    frame = pd.read_csv(table.path)
    assert list(frame["run_id"]) == ["r1", "r2"]
    assert set(frame["schema_version"]) == {1}


def test_csv_table_rejects_columns_outside_schema(tmp_path):
    table = CSVTable(tmp_path / "runs.csv", ("schema_version", "run_id"))

    with pytest.raises(ValueError, match="unexpected CSV columns: extra"):
        table.append({"run_id": "r1", "extra": 1})


def test_flatten_mapping_never_writes_json_cells():
    flat = flatten_mapping(
        {
            "config": {"outer_steps": 2},
            "methods": ("full_adp", "fixed_h"),
            "output": None,
        }
    )

    assert flat == {
        "config_outer_steps": 2,
        "methods": "full_adp|fixed_h",
        "output": None,
    }


def test_merge_csv_shards_streams_rows_and_removes_sources(tmp_path):
    fieldnames = ("schema_version", "run_id", "value")
    shards = [tmp_path / "part-1.csv", tmp_path / "part-2.csv"]
    for index, shard in enumerate(shards, start=1):
        with shard.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {"schema_version": 1, "run_id": f"r{index}", "value": index}
            )

    destination = tmp_path / "merged.csv"
    merge_csv_shards(shards, destination, fieldnames)

    assert list(pd.read_csv(destination)["run_id"]) == ["r1", "r2"]
    assert not any(path.exists() for path in shards)


def test_stable_run_id_is_deterministic_and_sensitive_to_seed():
    first = stable_run_id("4", "scenario", "full_adp", 10)
    second = stable_run_id("4", "scenario", "full_adp", 10)
    different = stable_run_id("4", "scenario", "full_adp", 11)

    assert first == second
    assert first != different


def test_configuration_fingerprint_is_order_independent_and_value_sensitive():
    first = {
        "solver": {"steps": np.int64(8), "tolerance": 1e-6},
        "methods": ("full_adp", "save"),
        "output": Path("benchmark_outputs"),
    }
    reordered = {
        "output": Path("benchmark_outputs"),
        "methods": ("full_adp", "save"),
        "solver": {"tolerance": 1e-6, "steps": 8},
    }
    changed = {
        **reordered,
        "solver": {"tolerance": 1e-5, "steps": 8},
    }

    assert configuration_fingerprint(first) == configuration_fingerprint(reordered)
    assert configuration_fingerprint(first) != configuration_fingerprint(changed)


def test_stable_run_id_remains_compatible_and_accepts_config_fingerprint():
    legacy = stable_run_id("4", "scenario", "full_adp", 10)

    assert legacy == "run-f2d684012acb080a"
    assert stable_run_id(
        "4",
        "scenario",
        "full_adp",
        10,
        config_fingerprint="config-a",
    ) != legacy


def test_replace_single_row_csv_preserves_old_file_when_publish_fails(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "series.csv"
    path.write_text("schema_version,status\n1,complete\n", encoding="utf-8")
    original = path.read_text(encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("publish failed")

    monkeypatch.setattr(experiment_log.os, "replace", fail_replace)

    with pytest.raises(OSError, match="publish failed"):
        replace_single_row_csv(path, {"schema_version": 1, "status": "running"})

    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "series.csv.tmp").exists()


def test_replace_single_row_csv_atomically_writes_flat_row(tmp_path):
    path = replace_single_row_csv(
        tmp_path / "series.csv",
        {"schema_version": 1, "config": {"profile": "smoke"}},
    )

    assert pd.read_csv(path).to_dict("records") == [
        {"schema_version": 1, "config_profile": "smoke"}
    ]
