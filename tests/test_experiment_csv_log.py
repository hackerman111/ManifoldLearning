import csv

import pandas as pd
import pytest

from adp.common.experiment_log import (
    CSVTable,
    flatten_mapping,
    merge_csv_shards,
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
