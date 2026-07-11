from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np

from ...common.experiment_log import (
    SCHEMA_VERSION,
    CSVTable,
    Scalar,
    configuration_fingerprint,
    flatten_mapping,
    replace_single_row_csv,
)
from .schema import (
    ARTIFACT_COLUMNS,
    FAILURE_COLUMNS,
    INITIAL_PARAMETER_COLUMNS,
    ITERATION_COLUMNS,
    RUN_COLUMNS,
    SERIES_COLUMNS,
    SOLVER_ITERATION_COLUMNS,
)
from .types import SingleIndexJob, SingleIndexSeriesConfig


_TABLE_COLUMNS = {
    "runs": RUN_COLUMNS,
    "iterations": ITERATION_COLUMNS,
    "solver_iterations": SOLVER_ITERATION_COLUMNS,
    "failures": FAILURE_COLUMNS,
}
_TABLE_KEYS = {
    "runs": ("run_id",),
    "iterations": ("run_id", "outer_k"),
    "solver_iterations": ("run_id", "outer_k", "inner_k", "cg_k"),
    "failures": ("run_id",),
}


class SingleIndexSeriesStore:
    """Crash-safe normalized CSV storage for one benchmark series."""

    def __init__(
        self,
        series_dir: Path,
        config: SingleIndexSeriesConfig,
        *,
        series_id: str,
        config_fingerprint: str,
        requested_jobs: int,
    ) -> None:
        self.series_dir = series_dir
        self.config = config
        self.series_id = series_id
        self.config_fingerprint = config_fingerprint
        self.requested_jobs = requested_jobs
        self.shard_dir = series_dir / f".{series_id}_shards"

    @classmethod
    def create(
        cls,
        root: Path,
        config: SingleIndexSeriesConfig,
        jobs: Sequence[SingleIndexJob],
    ) -> Self:
        fingerprint = _config_fingerprint(config)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        series_id = f"{stamp}-{fingerprint.removeprefix('cfg-')}"
        series_dir = Path(root) / series_id
        series_dir.mkdir(parents=True, exist_ok=False)
        store = cls(
            series_dir,
            config,
            series_id=series_id,
            config_fingerprint=fingerprint,
            requested_jobs=len(jobs),
        )
        store.shard_dir.mkdir()
        store._write_series(status="running")
        store._write_initial_parameters(jobs)
        return store

    @classmethod
    def resume(
        cls,
        series_dir: Path,
        config: SingleIndexSeriesConfig,
    ) -> Self:
        series_path = Path(series_dir) / "single_index_series.csv"
        row = _read_single_row(series_path)
        schema_version = int(row.get("schema_version", -1))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema version mismatch: expected {SCHEMA_VERSION}, got {schema_version}"
            )
        expected = _config_fingerprint(config)
        actual = str(row.get("config_fingerprint", ""))
        if expected != actual:
            raise ValueError(
                f"configuration fingerprint mismatch: expected {actual}, got {expected}"
            )
        store = cls(
            Path(series_dir),
            config,
            series_id=str(row["series_id"]),
            config_fingerprint=actual,
            requested_jobs=int(row.get("requested_jobs", 0)),
        )
        store.shard_dir.mkdir(exist_ok=True)
        store._discard_orphan_shard_rows()
        return store

    def pending_jobs(
        self,
        jobs: Sequence[SingleIndexJob],
    ) -> Iterator[SingleIndexJob]:
        statuses = self._committed_statuses()
        for job in jobs:
            status = statuses.get(job.run_id)
            if status is None:
                yield job
            elif status == "failed" and self.config.retry_failed:
                yield job

    def append_worker_rows(
        self,
        table: str,
        rows: Iterable[Mapping[str, Scalar]],
    ) -> int:
        if table not in _TABLE_COLUMNS:
            raise ValueError(f"unknown single-index table: {table}")
        shard = self.shard_dir / f"{table}-{os.getpid()}.csv"
        prepared = (
            {
                **row,
                "schema_version": SCHEMA_VERSION,
                "series_id": self.series_id,
            }
            for row in rows
        )
        return CSVTable(shard, _TABLE_COLUMNS[table]).append_many(prepared)

    def finalize(self, *, status: str) -> Mapping[str, Path]:
        if status not in {"complete", "partial", "failed"}:
            raise ValueError("series status must be complete, partial or failed")

        runs_path = self._merge_table("runs")
        committed_ids = set(self._committed_statuses(final_only=True))
        iterations_path = self._merge_table(
            "iterations",
            allowed_run_ids=committed_ids,
        )
        solver_path = self._merge_table(
            "solver_iterations",
            allowed_run_ids=committed_ids,
        )
        failures_path = self._merge_table(
            "failures",
            allowed_run_ids=committed_ids,
        )

        if self.shard_dir.exists() and not any(self.shard_dir.iterdir()):
            self.shard_dir.rmdir()

        counts = _status_counts(runs_path)
        self._write_series(
            status=status,
            completed_jobs=counts.get("success", 0),
            failed_jobs=counts.get("failed", 0),
            unavailable_jobs=counts.get("unavailable", 0),
            finished_at_utc=_utc_now(),
        )
        series_path = self.series_dir / "single_index_series.csv"
        parameters_path = self.series_dir / "single_index_initial_parameters.csv"
        artifacts_path = self.series_dir / "single_index_artifacts.csv"
        primary = {
            "series": series_path,
            "initial_parameters": parameters_path,
            "runs": runs_path,
            "iterations": iterations_path,
            "solver_iterations": solver_path,
            "failures": failures_path,
        }
        self._write_artifacts(artifacts_path, primary)
        return {**primary, "artifacts": artifacts_path}

    def _write_series(self, *, status: str, **updates: Scalar) -> None:
        path = self.series_dir / "single_index_series.csv"
        existing = _read_single_row(path) if path.exists() else {}
        git_commit, git_branch, git_dirty = _git_metadata()
        defaults: dict[str, Scalar] = {
            "schema_version": SCHEMA_VERSION,
            "series_id": self.series_id,
            "config_fingerprint": self.config_fingerprint,
            "status": status,
            "profile": self.config.profile,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "requested_jobs": self.requested_jobs,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "unavailable_jobs": 0,
            "process_jobs": self.config.jobs,
            "statistics_workers": self.config.statistics_workers,
            "base_seed": self.config.base_seed,
        }
        values = {**defaults, **existing, **updates, "status": status}
        replace_single_row_csv(path, _complete_row(SERIES_COLUMNS, values))

    def _write_initial_parameters(self, jobs: Sequence[SingleIndexJob]) -> None:
        path = self.series_dir / "single_index_initial_parameters.csv"
        table = CSVTable(path, INITIAL_PARAMETER_COLUMNS)
        rows = []
        for job in jobs:
            flat = flatten_mapping(
                {
                    "data": job.scenario.data,
                    "algorithm": job.scenario.algorithm,
                    "solver": job.scenario.solver,
                }
            )
            values: dict[str, Scalar] = {
                "schema_version": SCHEMA_VERSION,
                "series_id": self.series_id,
                "run_id": job.run_id,
                "scenario_id": job.scenario.scenario_id,
                "family": job.scenario.family,
                "executor": job.scenario.executor,
                "method": job.method,
                "repeat": job.repeat,
                "data_seed": job.seeds.data,
                "beta_seed": job.seeds.beta,
                "centers_seed": job.seeds.centers,
                "directions_seed": job.seeds.directions,
                "init_seed": job.seeds.init,
                "hypothesis": job.scenario.hypothesis,
                **flat,
                "algorithm_statistics_workers": self.config.statistics_workers,
            }
            rows.append(_complete_row(INITIAL_PARAMETER_COLUMNS, values))
        if rows:
            table.append_many(rows)
        else:
            table.write_header()

    def _committed_statuses(self, *, final_only: bool = False) -> dict[str, str]:
        paths = [self.series_dir / "single_index_runs.csv"]
        if not final_only and self.shard_dir.exists():
            paths.extend(sorted(self.shard_dir.glob("runs-*.csv")))
        statuses: dict[str, str] = {}
        for path in paths:
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    statuses[str(row["run_id"])] = str(row["status"])
        return statuses

    def _discard_orphan_shard_rows(self) -> None:
        committed_ids = set(self._committed_statuses())
        for table in ("iterations", "solver_iterations", "failures"):
            columns = _TABLE_COLUMNS[table]
            for shard in sorted(self.shard_dir.glob(f"{table}-*.csv")):
                temporary = shard.with_name(shard.name + ".tmp")
                try:
                    with shard.open(newline="", encoding="utf-8") as source:
                        reader = csv.DictReader(source)
                        if tuple(reader.fieldnames or ()) != columns:
                            raise ValueError(
                                f"CSV shard header mismatch for {shard}: "
                                f"expected {columns}, got {tuple(reader.fieldnames or ())}"
                            )
                        with temporary.open("w", newline="", encoding="utf-8") as output:
                            writer = csv.DictWriter(output, fieldnames=columns)
                            writer.writeheader()
                            for row in reader:
                                if str(row.get("run_id", "")) in committed_ids:
                                    writer.writerow(row)
                    os.replace(temporary, shard)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise

    def _merge_table(
        self,
        table: str,
        *,
        allowed_run_ids: set[str] | None = None,
    ) -> Path:
        columns = _TABLE_COLUMNS[table]
        destination = self.series_dir / f"single_index_{table}.csv"
        shards = sorted(self.shard_dir.glob(f"{table}-*.csv"))
        sources = ([destination] if destination.exists() else []) + shards
        temporary = destination.with_name(destination.name + ".tmp")
        seen: set[tuple[str, ...]] = set()
        try:
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for source in sources:
                    with source.open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        if tuple(reader.fieldnames or ()) != columns:
                            raise ValueError(
                                f"CSV shard header mismatch for {source}: "
                                f"expected {columns}, got {tuple(reader.fieldnames or ())}"
                            )
                        for row in reader:
                            run_id = str(row.get("run_id", ""))
                            if allowed_run_ids is not None and run_id not in allowed_run_ids:
                                continue
                            key = tuple(str(row.get(name, "")) for name in _TABLE_KEYS[table])
                            if key in seen:
                                raise ValueError(f"duplicate {table} key: {key}")
                            seen.add(key)
                            writer.writerow(row)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        for shard in shards:
            shard.unlink()
        return destination

    def _write_artifacts(
        self,
        path: Path,
        artifacts: Mapping[str, Path],
    ) -> None:
        path.unlink(missing_ok=True)
        rows = []
        for name, artifact_path in artifacts.items():
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "series_id": self.series_id,
                    "artifact_type": artifact_path.suffix.lstrip(".") or "directory",
                    "name": name,
                    "path": str(artifact_path.relative_to(self.series_dir)),
                    "size_bytes": artifact_path.stat().st_size,
                    "status": "created",
                    "error": "",
                }
            )
        CSVTable(path, ARTIFACT_COLUMNS).append_many(rows)


def _config_fingerprint(config: SingleIndexSeriesConfig) -> str:
    values = asdict(config)
    values.pop("retry_failed", None)
    values.pop("allow_download", None)
    return configuration_fingerprint(values)


def _complete_row(
    columns: Sequence[str],
    values: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    return {column: values.get(column) for column in columns}


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    return dict(rows[0])


def _status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status", ""))
            counts[status] = counts.get(status, 0) + 1
    return counts


def _git_metadata() -> tuple[str, str, bool]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ("git", *args),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty = bool(run("status", "--porcelain"))
    return commit, branch, dirty


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
