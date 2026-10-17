"""Serial launcher for the frozen metric-on-Spokoiny experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .workers import SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
FULL_ARMS = ("core", "metric2", "metric1")
FROZEN_ARMS = (*FULL_ARMS, "isotropic2", "isotropic2_matched")


def _manifest() -> dict[str, Any]:
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
        if (ROOT / name).is_file()
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    return {
        "git_commit": git,
        "git_dirty": bool(dirty),
        "git_dirty_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (HERE / "PROTOCOL.md").read_bytes()
        ).hexdigest(),
        "source_sha256": sources,
    }


def _prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    current = _manifest()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if any(
            saved.get(key) != current[key]
            for key in ("protocol_sha256", "source_sha256")
        ):
            raise RuntimeError("existing outputs use different protocol/source hashes")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(current, stream, indent=2, sort_keys=True)
            stream.write("\n")


def _snapshots(output: Path) -> list[dict[str, Any]]:
    from .analysis import read_rows

    snapshots = _snapshot_records(read_rows(output / "capture.jsonl"), output)
    keys = {(row["seed"], row["phase"]) for row in snapshots}
    if len(snapshots) != 9 or keys != {
        (seed, phase) for seed in range(3) for phase in ("early", "middle", "late")
    }:
        raise RuntimeError("expected exactly three phases for each development seed")
    return snapshots


def _snapshot_records(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    snapshots = []
    for row in rows:
        if row.get("status") != "ok" or row.get("arm") != "core":
            continue
        for curve in row.get("capture_curves", []):
            if curve.get("capture_kind") != "independent_snapshot_input":
                continue
            index = int(curve["call_index"])
            phase = {0: "early", 80: "middle", 160: "late"}.get(index, "late")
            path = output / "captures" / f"core_seed{row['seed']}_call{index}.npz"
            if path.is_file():
                snapshots.append(
                    {
                        "seed": int(row["seed"]),
                        "phase": phase,
                        "call_index": index,
                        "path": path,
                    }
                )
    keys = [(row["seed"], row["call_index"]) for row in snapshots]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate captured (seed, call_index) snapshot")
    return sorted(snapshots, key=lambda item: (item["seed"], item["call_index"]))


def _rotate(arms: tuple[str, ...], offset: int) -> tuple[str, ...]:
    offset %= len(arms)
    return arms[offset:] + arms[:offset]


def _jobs(stage: str, output: Path) -> tuple[list[dict[str, Any]], int]:
    if stage == "fullfit5":
        return (
            [
                {"mode": "fullfit", "seed": s, "arm": a, "budget": 5}
                for s in range(3)
                for a in _rotate(FULL_ARMS, s)
            ],
            180,
        )
    if stage == "capture":
        jobs = [
            {"mode": "capture", "seed": s, "arm": "core", "budget": 5} for s in range(3)
        ]
        jobs.append({"mode": "capture", "seed": 0, "arm": "metric2", "budget": 5})
        return jobs, 180
    if stage == "frozen":
        snapshots = _snapshots(output)
        expected = {
            (seed, phase) for seed in range(3) for phase in ("early", "middle", "late")
        }
        if (
            len(snapshots) != 9
            or {(row["seed"], row["phase"]) for row in snapshots} != expected
        ):
            raise RuntimeError(
                "frozen stage requires all nine unique seed/phase inputs"
            )
        jobs = []
        for snapshot in snapshots:
            for budget in (5, 100):
                for arm in _rotate(FROZEN_ARMS, snapshot["seed"] + budget):
                    jobs.append(
                        {
                            "mode": "frozen",
                            **snapshot,
                            "snapshot": snapshot["path"],
                            "arm": arm,
                            "budget": budget,
                        }
                    )
        if len(jobs) != 90:
            raise RuntimeError(f"expected 90 frozen solves, got {len(jobs)}")
        return jobs, 60
    if stage == "budget20":
        return (
            [
                {"mode": "fullfit", "seed": s, "arm": a, "budget": 20}
                for s in range(3)
                for a in _rotate(("core", "metric2"), s)
            ],
            180,
        )
    if stage == "validation":
        from .analysis import development_gate, read_rows

        gate = development_gate(read_rows(output / "fullfit5.jsonl"))
        if not gate["passed"]:
            return ([{"mode": "skipped", "gate": gate}], 180)
        return (
            [
                {"mode": "fullfit", "seed": s, "arm": a, "budget": 5}
                for s in range(3, 8)
                for a in _rotate(("core", "metric2"), s)
            ],
            180,
        )
    raise ValueError(stage)


def _run_job(job: dict[str, Any], output: Path, timeout: int) -> dict[str, Any]:
    record = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in job.items()
    }
    if job["mode"] == "skipped":
        return {**record, "status": "skipped", "reason": "development gate failed"}
    command = [sys.executable, "-m", "experiments.metric_spokoiny_2026_10_15.workers"]
    for key in ("mode", "seed", "arm", "budget", "snapshot"):
        if key in job:
            command.extend((f"--{key}", str(job[key])))
    command.extend(("--output", str(output)))
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout
        )
        if process.returncode:
            return {
                **record,
                "status": "process_error",
                "returncode": process.returncode,
                "stderr": process.stderr[-5000:],
            }
        row = json.loads(process.stdout.strip().splitlines()[-1])
        row.update(
            {key: record[key] for key in ("phase", "call_index") if key in record}
        )
        return row
    except subprocess.TimeoutExpired as error:
        return {
            **record,
            "status": "timeout",
            "timeout_sec": timeout,
            "stderr": str(error.stderr)[-5000:],
        }
    except (IndexError, json.JSONDecodeError) as error:
        return {**record, "status": "worker_output_error", "error": repr(error)}


def run(stage: str, output: Path) -> Path:
    _prepare(output)
    path = output / f"{stage}.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    jobs, timeout = _jobs(stage, output)
    with path.open("x", encoding="utf-8") as stream:
        for position, job in enumerate(jobs):
            row = _run_job(job, output, timeout)
            row.update(run_stage=stage, order_position=position)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("fullfit5", "capture", "frozen", "budget20", "validation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.stage, args.output.resolve())
    print(result.relative_to(ROOT))


if __name__ == "__main__":
    main()
