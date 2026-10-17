"""Run serial subprocess workers and audit smoke/development execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import (
    ARMS,
    CASES,
    DEV_SEEDS,
    FIT_SETTINGS,
    OUT,
    ROOT,
    THREAD_ENV,
    git_state,
    write_json,
)
from .run import assert_frozen


def run_child(case: str, seed: int, arm: str, output: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(dict.fromkeys(THREAD_ENV, "1"))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "experiments.adaptive_centers.worker",
        "--case",
        case,
        "--seed",
        str(seed),
        "--arm",
        arm,
        "--output",
        str(output),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(FIT_SETTINGS["timeout_sec"]),
            check=False,
        )
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "timeout_sec": FIT_SETTINGS["timeout_sec"],
            "stdout": str(exc.stdout)[-1000:],
            "stderr": str(exc.stderr)[-4000:],
        }


def read_worker(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            fallback = {**fallback, "worker_read_error": f"{type(exc).__name__}: {exc}"}
    return fallback


def audit_smoke_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_arms = {arm.name for arm in ARMS}
    by_arm = {str(row.get("arm")): row for row in rows}
    valid = [row for row in rows if row.get("status") == "ok"]
    reference = next((row for row in valid if row.get("arm") == "baseline"), {})
    expected_hashes = reference.get("input_hashes", {})
    hash_match = bool(expected_hashes) and all(
        all(
            row.get("input_hashes", {}).get(key) == expected_hashes.get(key)
            for key in ("X", "Y", "truth", "initial_basis")
        )
        for row in valid
    )
    thread_ok = all(
        bool(row.get("environment", {}).get("threadpools", []))
        and all(
            pool.get("num_threads") == 1
            for pool in row.get("environment", {}).get("threadpools", [])
        )
        and set(row.get("environment", {}).get("thread_env", {})) == set(THREAD_ENV)
        and all(
            value == "1"
            for value in row.get("environment", {}).get("thread_env", {}).values()
        )
        for row in valid
    )
    finite_ok = all(
        math.isfinite(float(row.get("edr_error", float("nan"))))
        and math.isfinite(float(row.get("fit_wall_sec", float("nan"))))
        for row in valid
    )
    coefficient_ok = all(
        bool(row.get("coefficient_mapping", {}).get("row_count_matches_selection"))
        for row in valid
    )
    result = {
        "expected_arms": sorted(expected_arms),
        "observed_arms": sorted(by_arm),
        "all_arms_returned": set(by_arm) == expected_arms,
        "all_fits_ok": len(valid) == len(expected_arms),
        "input_and_initialization_hashes_match": hash_match,
        "one_thread_verified": thread_ok,
        "edr_and_wall_finite": finite_ok,
        "coefficient_mapping_valid": coefficient_ok,
    }
    result["pass"] = all(
        result[key]
        for key in (
            "all_arms_returned",
            "all_fits_ok",
            "input_and_initialization_hashes_match",
            "one_thread_verified",
            "edr_and_wall_finite",
            "coefficient_mapping_valid",
        )
    )
    return result


def _append_worker(
    stream, worker_dir: Path, case: str, seed: int, arm: str, order: int
) -> None:
    worker = worker_dir / f"fit_{case}_{seed}_{arm}.json"
    if worker.exists():
        raise FileExistsError(worker)
    status = run_child(case, seed, arm, worker)
    row = read_worker(
        worker,
        {
            "status": status["status"],
            "case": case,
            "seed": seed,
            "arm": arm,
            "error": status,
        },
    )
    row["order_index"] = order
    row["worker_status"] = status
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def run_development() -> None:
    assert_frozen()
    output = OUT / "results" / "fullfit_dev.jsonl"
    if output.exists():
        raise FileExistsError(output)
    worker_dir = OUT / "results" / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for case_index, case in enumerate(CASES):
            for seed in DEV_SEEDS:
                offset = (seed + case_index) % len(ARMS)
                ordered_arms = ARMS[offset:] + ARMS[:offset]
                for order_index, arm in enumerate(ordered_arms):
                    _append_worker(
                        stream, worker_dir, case.name, seed, arm.name, order_index
                    )
    frozen = assert_frozen()
    write_json(
        OUT / "results" / "verification_dev.json",
        {
            "source_hashes_match_freeze": True,
            "protocol_hash_matches_freeze": True,
            "manifest_sha256": hashlib.sha256(
                (OUT / "results" / "freeze_manifest.json").read_bytes()
            ).hexdigest(),
            "source_count": len(frozen["source_sha256"]),
            "git_after_run": git_state(),
        },
    )


def run_smoke(case_name: str) -> None:
    assert_frozen()
    if case_name not in {case.name for case in CASES}:
        raise ValueError(f"unknown case: {case_name}")
    seed = 92799
    output = OUT / "results" / f"smoke_{case_name}_{seed}.jsonl"
    if output.exists():
        raise FileExistsError(output)
    worker_dir = OUT / "results" / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for order_index, arm in enumerate(ARMS):
            worker = worker_dir / f"smoke_{case_name}_{seed}_{arm.name}.json"
            if worker.exists():
                raise FileExistsError(worker)
            status = run_child(case_name, seed, arm.name, worker)
            row = read_worker(
                worker,
                {
                    "status": status["status"],
                    "case": case_name,
                    "seed": seed,
                    "arm": arm.name,
                    "error": status,
                },
            )
            row["order_index"] = order_index
            row["worker_status"] = status
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
    smoke_rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    smoke_audit = audit_smoke_rows(smoke_rows)
    write_json(OUT / "results" / f"smoke_audit_{case_name}_{seed}.json", smoke_audit)
    if not smoke_audit["pass"]:
        raise RuntimeError("smoke audit failed; development runs are not authorized")
    frozen = assert_frozen()
    write_json(
        OUT / "results" / f"verification_smoke_{case_name}_{seed}.json",
        {
            "source_hashes_match_freeze": True,
            "protocol_hash_matches_freeze": True,
            "manifest_sha256": hashlib.sha256(
                (OUT / "results" / "freeze_manifest.json").read_bytes()
            ).hexdigest(),
            "source_count": len(frozen["source_sha256"]),
            "git_after_run": git_state(),
            "excluded_from_development_and_heldout": True,
        },
    )
