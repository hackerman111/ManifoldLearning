from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
Scalar = str | int | float | bool | None


def flatten_mapping(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Scalar]:
    """Flatten nested mappings into scalar columns without JSON cells."""

    flattened: dict[str, Scalar] = {}
    for raw_key, value in values.items():
        key = f"{prefix}_{raw_key}" if prefix else str(raw_key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=key))
        elif isinstance(value, Path):
            flattened[key] = str(value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            flattened[key] = "|".join(str(item) for item in value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif hasattr(value, "item"):
            flattened[key] = value.item()
        else:
            raise TypeError(f"CSV value for {key!r} is not scalar: {type(value).__name__}")
    return flattened


def stable_run_id(
    experiment: str,
    scenario_id: str,
    method: str,
    seed: int,
) -> str:
    """Build a compact deterministic identifier for one experiment job."""

    payload = "\x1f".join((str(experiment), scenario_id, method, str(seed)))
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
    return f"run-{digest}"


class CSVTable:
    """Append rows to a CSV file while enforcing one stable schema."""

    def __init__(
        self,
        path: str | Path,
        fieldnames: Iterable[str],
        *,
        schema_version: int = SCHEMA_VERSION,
    ) -> None:
        self.path = Path(path)
        self.fieldnames = tuple(fieldnames)
        self.schema_version = int(schema_version)
        if not self.fieldnames:
            raise ValueError("CSV fieldnames must not be empty")
        if len(set(self.fieldnames)) != len(self.fieldnames):
            raise ValueError("CSV fieldnames must be unique")

    def append(self, row: Mapping[str, Any]) -> None:
        self.append_many((row,))

    def append_many(self, rows: Iterable[Mapping[str, Any]]) -> int:
        iterator = iter(rows)
        try:
            first = self._prepare_row(next(iterator))
        except StopIteration:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            self._validate_existing_header()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(first)
            count = 1
            for row in iterator:
                writer.writerow(self._prepare_row(row))
                count += 1
        return count

    def write_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            self._validate_existing_header()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def _prepare_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        prepared = dict(row)
        if "schema_version" in self.fieldnames:
            prepared.setdefault("schema_version", self.schema_version)
        unexpected = sorted(set(prepared) - set(self.fieldnames))
        if unexpected:
            raise ValueError("unexpected CSV columns: " + ", ".join(unexpected))
        return {name: prepared.get(name) for name in self.fieldnames}

    def _validate_existing_header(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != self.fieldnames:
            raise ValueError(
                f"CSV header mismatch for {self.path}: "
                f"expected {self.fieldnames}, got {tuple(header)}"
            )


def merge_csv_shards(
    shards: Iterable[str | Path],
    destination: str | Path,
    fieldnames: Iterable[str],
) -> Path:
    """Merge CSV shards using bounded memory and atomically publish output."""

    source_paths = [Path(path) for path in shards]
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(fieldnames)
    temporary = destination_path.with_name(destination_path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for source in source_paths:
                with source.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames or ()) != columns:
                        raise ValueError(
                            f"CSV shard header mismatch for {source}: "
                            f"expected {columns}, got {tuple(reader.fieldnames or ())}"
                        )
                    writer.writerows(reader)
        os.replace(temporary, destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    for source in source_paths:
        source.unlink()
    return destination_path


def write_single_row_csv(
    path: str | Path,
    row: Mapping[str, Any],
) -> Path:
    flattened = flatten_mapping(row)
    fieldnames = tuple(flattened)
    table = CSVTable(path, fieldnames)
    table.append(flattened)
    return table.path


def write_artifacts_csv(
    path: str | Path,
    artifacts: Mapping[str, str | Path],
) -> Path:
    table = CSVTable(
        path,
        ("schema_version", "artifact_type", "name", "path"),
    )
    rows = [
        {
            "artifact_type": Path(value).suffix.lstrip(".") or "directory",
            "name": name,
            "path": str(value),
        }
        for name, value in artifacts.items()
    ]
    if rows:
        table.append_many(rows)
    else:
        table.write_header()
    return table.path
