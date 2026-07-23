from __future__ import annotations

import csv
import os
import platform
import subprocess
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np

from ...common.experiment_log import Scalar, configuration_fingerprint
from .schema import (
    ARTIFACT_COLUMNS,
    INNER_ITERATION_COLUMNS,
    LOCAL_DIAGNOSTIC_COLUMNS,
    OUTER_ITERATION_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SCHEMA_VERSION,
    SERIES_COLUMNS,
    SOLVER_ITERATION_COLUMNS,
)
from .types import RunOutcome, SingleIndexJob, SingleIndexSeriesConfig


_SHARD_COLUMNS = {
    "run_summary": RUN_SUMMARY_COLUMNS,
    "outer_iterations": OUTER_ITERATION_COLUMNS,
    "inner_iterations": INNER_ITERATION_COLUMNS,
    "local_diagnostics": LOCAL_DIAGNOSTIC_COLUMNS,
    "solver_iterations": SOLVER_ITERATION_COLUMNS,
}
_OUTCOME_ATTRIBUTES = {
    "run_summary": "run_row",
    "outer_iterations": "outer_rows",
    "inner_iterations": "inner_rows",
    "local_diagnostics": "local_rows",
    "solver_iterations": "solver_rows",
}
_TABLE_KEYS = {
    "run_summary": ("run_id",),
    "outer_iterations": ("run_id", "outer_k"),
    "inner_iterations": ("run_id", "outer_k", "inner_k"),
    "local_diagnostics": ("run_id", "outer_k", "center_j"),
    "solver_iterations": ("run_id", "outer_k", "inner_k", "solver_k"),
}
_DETAIL_TABLES = (
    "outer_iterations",
    "inner_iterations",
    "local_diagnostics",
    "solver_iterations",
)


class SingleIndexSeriesStore:
    """Atomic per-run shards and bounded public CSV aggregation."""

    def __init__(
        self,
        series_dir: Path,
        config: SingleIndexSeriesConfig,
        *,
        series_id: str,
        config_fingerprint: str,
        requested_jobs: int,
        planned_run_ids: Sequence[str] = (),
    ) -> None:
        self.series_dir = Path(series_dir)
        self.config = config
        self.series_id = series_id
        self.config_fingerprint = config_fingerprint
        self.requested_jobs = int(requested_jobs)
        self.shard_dir = self.series_dir / ".shards"
        self._planned_run_ids = tuple(planned_run_ids)

    @classmethod
    def create(
        cls,
        root: Path,
        config: SingleIndexSeriesConfig,
        jobs: Sequence[SingleIndexJob],
    ) -> Self:
        run_ids = tuple(job.run_id for job in jobs)
        _require_unique_run_ids(run_ids)
        fingerprint = _config_fingerprint(config)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        series_id = f"{stamp}-{fingerprint.removeprefix('cfg-')}"
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        series_dir = root / series_id
        series_dir.mkdir(parents=False, exist_ok=False)
        store = cls(
            series_dir,
            config,
            series_id=series_id,
            config_fingerprint=fingerprint,
            requested_jobs=len(jobs),
            planned_run_ids=run_ids,
        )
        store.shard_dir.mkdir()
        store._write_series(status="running")
        return store

    @classmethod
    def resume(
        cls,
        series_dir: Path,
        config: SingleIndexSeriesConfig,
    ) -> Self:
        series_dir = Path(series_dir)
        row = _read_single_row(series_dir / "series.csv", SERIES_COLUMNS)
        actual_version = int(float(str(row.get("schema_version", -1))))
        if actual_version != SCHEMA_VERSION:
            raise ValueError(
                "schema version mismatch: "
                f"expected {SCHEMA_VERSION}, got {actual_version}"
            )
        expected = _config_fingerprint(config)
        actual = str(row.get("config_fingerprint", ""))
        if expected != actual:
            raise ValueError(
                "configuration fingerprint mismatch: "
                f"expected {actual}, got {expected}"
            )
        store = cls(
            series_dir,
            config,
            series_id=str(row["series_id"]),
            config_fingerprint=actual,
            requested_jobs=int(float(str(row.get("requested_jobs", 0)))),
        )
        store.shard_dir.mkdir(exist_ok=True)
        return store

    def completed_run_ids(self) -> set[str]:
        return set(self._committed_statuses())

    def pending_jobs(
        self,
        jobs: Sequence[SingleIndexJob],
    ) -> Iterator[SingleIndexJob]:
        run_ids = tuple(job.run_id for job in jobs)
        _require_unique_run_ids(run_ids)
        self._planned_run_ids = run_ids
        statuses = self._committed_statuses()
        for job in jobs:
            status = statuses.get(job.run_id)
            if status is None:
                yield job
            elif status == "numerical_failure" and self.config.retry_failed:
                yield job

    def commit(self, outcome: RunOutcome) -> None:
        run_id = _outcome_run_id(outcome)
        _validate_run_id(run_id)
        prepared = self._prepare_outcome_rows(outcome, run_id)
        target = self.shard_dir / run_id
        marker = target / "run_summary.csv"
        if marker.exists():
            self._commit_replacement(target, prepared)
            return

        target.mkdir(parents=False, exist_ok=True)
        for table in _DETAIL_TABLES:
            self._atomic_write_rows(
                target / f"{table}.csv",
                _SHARD_COLUMNS[table],
                prepared[table],
            )
        self._atomic_write_rows(
            marker,
            RUN_SUMMARY_COLUMNS,
            prepared["run_summary"],
        )

    def finalize(self, *, status: str) -> Mapping[str, Path]:
        if status not in {"complete", "partial", "failed"}:
            raise ValueError("series status must be complete, partial or failed")
        committed = self._committed_statuses()
        order = self._merge_order(committed)
        saved = {
            table: self._merge_table(table, order)
            for table in _SHARD_COLUMNS
        }
        counts = _status_counts(saved["run_summary"])
        self._write_series(
            status=status,
            committed_jobs=sum(counts.values()),
            success_jobs=counts.get("success", 0),
            nonconverged_jobs=counts.get("nonconverged", 0),
            numerical_failure_jobs=counts.get("numerical_failure", 0),
            finished_at_utc=_utc_now(),
        )
        saved["series"] = self.series_dir / "series.csv"
        saved["artifacts"] = self.series_dir / "artifacts.csv"
        self._write_artifacts(saved)
        return saved

    def _prepare_outcome_rows(
        self,
        outcome: RunOutcome,
        run_id: str,
    ) -> dict[str, tuple[Mapping[str, Scalar], ...]]:
        prepared: dict[str, tuple[Mapping[str, Scalar], ...]] = {}
        for table, attribute in _OUTCOME_ATTRIBUTES.items():
            value = getattr(outcome, attribute)
            rows = (value,) if table == "run_summary" else tuple(value)
            for row in rows:
                if str(row.get("run_id", "")) != run_id:
                    raise ValueError(
                        f"{table} row belongs to {row.get('run_id')!r}, "
                        f"expected {run_id!r}"
                    )
            _require_unique_rows(table, rows)
            prepared[table] = rows
        return prepared

    def _commit_replacement(
        self,
        target: Path,
        prepared: Mapping[str, tuple[Mapping[str, Scalar], ...]],
    ) -> None:
        replacement = self.shard_dir / f"pending-{target.name}-{uuid.uuid4().hex}"
        replacement.mkdir(parents=False, exist_ok=False)
        try:
            for table in _DETAIL_TABLES:
                self._atomic_write_rows(
                    replacement / f"{table}.csv",
                    _SHARD_COLUMNS[table],
                    prepared[table],
                )
            self._atomic_write_rows(
                replacement / "run_summary.csv",
                RUN_SUMMARY_COLUMNS,
                prepared["run_summary"],
            )

            (target / "run_summary.csv").unlink()
            for table in _DETAIL_TABLES:
                os.replace(
                    replacement / f"{table}.csv",
                    target / f"{table}.csv",
                )
            os.replace(
                replacement / "run_summary.csv",
                target / "run_summary.csv",
            )
        finally:
            for path in replacement.glob("*") if replacement.exists() else ():
                path.unlink(missing_ok=True)
            if replacement.exists():
                replacement.rmdir()

    def _atomic_write_rows(
        self,
        path: Path,
        columns: Sequence[str],
        rows: Iterable[Mapping[str, Scalar]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f"pending-{path.name}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    values = {
                        **row,
                        "schema_version": SCHEMA_VERSION,
                        "series_id": self.series_id,
                    }
                    writer.writerow(_complete_row(columns, values))
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _committed_statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        if not self.shard_dir.exists():
            return statuses
        for run_dir in sorted(self.shard_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("pending-"):
                continue
            marker = run_dir / "run_summary.csv"
            if not marker.exists():
                continue
            row = _read_single_row(marker, RUN_SUMMARY_COLUMNS)
            run_id = str(row.get("run_id", ""))
            if run_id != run_dir.name:
                raise ValueError(
                    f"run marker identity mismatch: {run_id!r} != {run_dir.name!r}"
                )
            statuses[run_id] = str(row.get("status", ""))
        return statuses

    def _merge_order(self, committed: Mapping[str, str]) -> tuple[str, ...]:
        if self._planned_run_ids:
            planned = tuple(
                run_id for run_id in self._planned_run_ids if run_id in committed
            )
            extras = tuple(sorted(set(committed) - set(planned)))
            return planned + extras
        return tuple(sorted(committed))

    def _merge_table(self, table: str, run_ids: Sequence[str]) -> Path:
        columns = _SHARD_COLUMNS[table]
        destination = self.series_dir / f"{table}.csv"
        temporary = destination.with_name(f"pending-{destination.name}")
        seen: set[tuple[str, ...]] = set()
        try:
            with temporary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for run_id in run_ids:
                    source = self.shard_dir / run_id / f"{table}.csv"
                    if not source.exists():
                        if table == "run_summary":
                            raise ValueError(f"missing commit marker for {run_id}")
                        continue
                    with source.open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        if tuple(reader.fieldnames or ()) != tuple(columns):
                            raise ValueError(
                                f"CSV shard header mismatch for {source}: "
                                f"expected {tuple(columns)}, "
                                f"got {tuple(reader.fieldnames or ())}"
                            )
                        for row in reader:
                            if str(row.get("run_id", "")) != run_id:
                                raise ValueError(
                                    f"row identity mismatch in {source}: "
                                    f"{row.get('run_id')!r}"
                                )
                            key = tuple(str(row.get(name, "")) for name in _TABLE_KEYS[table])
                            if key in seen:
                                raise ValueError(f"duplicate {table} key: {key}")
                            seen.add(key)
                            writer.writerow(_complete_row(columns, row))
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def _write_series(self, *, status: str, **updates: Scalar) -> None:
        path = self.series_dir / "series.csv"
        existing: dict[str, str] = {}
        if path.exists():
            existing = _read_single_row(path, SERIES_COLUMNS)
        git_commit, git_branch, git_dirty = _git_metadata()
        defaults: dict[str, Scalar] = {
            "schema_version": SCHEMA_VERSION,
            "series_id": self.series_id,
            "config_fingerprint": self.config_fingerprint,
            "status": status,
            "profile": self.config.profile,
            "experiments": "|".join(self.config.experiments),
            "seeds": _encode_seeds(self.config.seeds),
            "diagnostic_seeds": _encode_seeds(self.config.diagnostic_seeds),
            "local_solvers": "|".join(self.config.local_solvers),
            "center_fraction": self.config.center_fraction,
            "process_jobs": self.config.jobs,
            "statistics_workers": 1,
            "requested_jobs": self.requested_jobs,
            "committed_jobs": 0,
            "success_jobs": 0,
            "nonconverged_jobs": 0,
            "numerical_failure_jobs": 0,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        }
        values = {**defaults, **existing, **updates, "status": status}
        self._atomic_write_rows(path, SERIES_COLUMNS, (values,))

    def _write_artifacts(self, saved: Mapping[str, Path]) -> None:
        path = self.series_dir / "artifacts.csv"
        artifact_size = 0
        for _ in range(4):
            rows = []
            for name, artifact_path in saved.items():
                size = (
                    artifact_size
                    if artifact_path == path
                    else artifact_path.stat().st_size
                )
                rows.append(
                    {
                        "artifact_type": "csv",
                        "name": name,
                        "path": str(artifact_path.relative_to(self.series_dir)),
                        "size_bytes": size,
                        "status": "created",
                        "error": "",
                    }
                )
            self._atomic_write_rows(path, ARTIFACT_COLUMNS, rows)
            new_size = path.stat().st_size
            if new_size == artifact_size:
                break
            artifact_size = new_size

    def _discard_pending_files(self) -> None:
        if not self.shard_dir.exists():
            return
        for run_dir in self.shard_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("pending-"):
                continue
            for pending in run_dir.glob("pending-*"):
                if pending.is_file():
                    pending.unlink()


def _config_fingerprint(config: SingleIndexSeriesConfig) -> str:
    return configuration_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "profile": config.profile,
            "experiments": config.experiments,
            "seeds": config.seeds,
            "diagnostic_seeds": config.diagnostic_seeds,
            "local_solvers": config.local_solvers,
            "center_fraction": config.center_fraction,
        }
    )


def _outcome_run_id(outcome: RunOutcome) -> str:
    run_id = str(outcome.run_row.get("run_id", ""))
    if not run_id:
        raise ValueError("run outcome requires a nonempty run_id")
    return run_id


def _validate_run_id(run_id: str) -> None:
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"unsafe run_id: {run_id!r}")


def _require_unique_run_ids(run_ids: Sequence[str]) -> None:
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("planned jobs contain duplicate run_id values")
    for run_id in run_ids:
        _validate_run_id(run_id)


def _require_unique_rows(
    table: str,
    rows: Sequence[Mapping[str, Scalar]],
) -> None:
    keys: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in _TABLE_KEYS[table])
        if key in keys:
            raise ValueError(f"duplicate {table} key: {key}")
        keys.add(key)


def _complete_row(
    columns: Sequence[str],
    values: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    return {column: values.get(column) for column in columns}


def _read_single_row(path: Path, columns: Sequence[str]) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(
                f"CSV header mismatch for {path}: expected {tuple(columns)}, "
                f"got {tuple(reader.fieldnames or ())}"
            )
        rows = list(reader)
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


def _encode_seeds(seeds: tuple[int, ...] | None) -> str:
    if seeds is None:
        return "default"
    return "|".join(str(seed) for seed in seeds)


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


__all__ = ["SingleIndexSeriesStore"]
