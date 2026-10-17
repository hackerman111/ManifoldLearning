"""Frozen serial launcher; no experiment selection is performed automatically."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

from .common import (
    ARMS,
    DEV_SEEDS,
    HOLDOUT_SEEDS,
    OUT,
    SCENARIOS,
    SETTINGS,
    THREAD_ENV,
    _assert_frozen,
    _fresh_worker,
    _read_row,
    _run_child,
    array_hash,
    environment,
    git_state,
    jsonl_append,
    scenario_metadata,
    source_hashes,
)
from .inner import make_frozen


def _manifest() -> dict[str, Any]:
    return {
        "task": "grassman-rank-k-2026-10-16",
        "status": "frozen-before-runs",
        "development_seeds": DEV_SEEDS,
        "heldout_seeds": HOLDOUT_SEEDS,
        "scenarios": {scenario: scenario_metadata(scenario) for scenario in SCENARIOS},
        "arms": ARMS,
        "settings": SETTINGS,
        "inner": {
            "m": (2, 3, 5, 8),
            "d": (20, 100, 300),
            "J": 64,
            "cost_J": (16, 256),
            "seeds": (81600, 81700),
            "noise_sd": 0.12,
        },
        "execution": {
            "processes": 1,
            "threads": dict.fromkeys(THREAD_ENV, 1),
            "fullfit_timeout_sec": 180,
            "inner_timeout_sec": 60,
            "rss": "resource.ru_maxrss (KiB on Linux)",
            "wall": "time.perf_counter; fullfit excludes first-stat hash overhead",
            "fullfit_timing_boundary": (
                "data/config generation excluded; fit includes wrapper except "
                "measured first-stat hashing, which is subtracted"
            ),
            "inner_timing_boundary": (
                "one solver call includes scalar gradient/profile traces; "
                "kernel time separately accumulated"
            ),
            "selection": "manual, after lead audit",
        },
        "source_sha256": source_hashes(),
        "git": git_state(),
        "environment_at_freeze": environment(),
    }


def freeze() -> None:
    path = OUT / "freeze_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    (OUT / "captures").mkdir(parents=True, exist_ok=True)
    workloads = []
    for m in (2, 3, 5, 8):
        for d in (20, 100, 300):
            for seed in (81600, 81700):
                snap = make_frozen(m, d, seed)
                file = OUT / "captures" / f"inner_m{m}_d{d}_seed{seed}_J64.npz"
                if file.exists():
                    raise FileExistsError(file)
                np.savez_compressed(
                    file,
                    index_init=snap["index_init"],
                    U=snap["U"],
                    I=snap["I"],
                    mass=snap["mass"],
                    truth=snap["truth"],
                )
                workloads.append(
                    {
                        "m": m,
                        "d": d,
                        "seed": seed,
                        "J": 64,
                        "path": str(file.relative_to(OUT)),
                        "hashes": {key: array_hash(snap[key]) for key in snap},
                    }
                )
        if m == 5:
            for J in (16, 256):
                for seed in (81600, 81700):
                    snap = make_frozen(m, 100, seed, J=J)
                    file = OUT / "captures" / f"inner_m5_d100_seed{seed}_J{J}.npz"
                    if file.exists():
                        raise FileExistsError(file)
                    if file.exists():
                        raise FileExistsError(file)
                    np.savez_compressed(
                        file,
                        index_init=snap["index_init"],
                        U=snap["U"],
                        I=snap["I"],
                        mass=snap["mass"],
                        truth=snap["truth"],
                    )
                    workloads.append(
                        {
                            "m": m,
                            "d": 100,
                            "seed": seed,
                            "J": J,
                            "path": str(file.relative_to(OUT)),
                            "hashes": {key: array_hash(snap[key]) for key in snap},
                        }
                    )
    manifest = _manifest()
    manifest["inner_workloads"] = workloads
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fullfit_dev() -> None:
    _assert_frozen()
    output = OUT / "results" / "fullfit_dev.jsonl"
    capture_output = OUT / "results" / "capture.jsonl"
    if output.exists() or capture_output.exists():
        raise FileExistsError(output if output.exists() else capture_output)
    capture_hashes: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for seed in DEV_SEEDS:
            worker = OUT / "results" / "workers" / f"capture_{scenario}_{seed}.json"
            worker.parent.mkdir(parents=True, exist_ok=True)
            _fresh_worker(worker)
            status = _run_child(
                "experiments.grassman_rank_k.fullfit",
                [
                    "--scenario",
                    scenario,
                    "--seed",
                    str(seed),
                    "--capture",
                    "--output",
                    str(worker),
                ],
                180,
            )
            row = (
                _read_row(worker)
                if worker.exists()
                else {
                    "status": status["status"],
                    "scenario": scenario,
                    "seed": seed,
                    "arm": "capture",
                    "error": status,
                }
            )
            row.update(
                {
                    "mode": "capture",
                    "scenario": scenario,
                    "seed": seed,
                    **{k: v for k, v in status.items() if k != "status"},
                }
            )
            jsonl_append(capture_output, row)
            if row.get("status") == "captured":
                capture_hashes[(scenario, seed)] = row["input_hashes"]
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in DEV_SEEDS:
            offset = (seed + scenario_index) % len(ARMS)
            order = ARMS[offset:] + ARMS[:offset]
            for order_index, arm in enumerate(order):
                worker = (
                    OUT / "results" / "workers" / f"fit_{scenario}_{seed}_{arm}.json"
                )
                _fresh_worker(worker)
                status = _run_child(
                    "experiments.grassman_rank_k.fullfit",
                    [
                        "--scenario",
                        scenario,
                        "--seed",
                        str(seed),
                        "--arm",
                        arm,
                        "--output",
                        str(worker),
                    ],
                    180,
                )
                row = (
                    _read_row(worker)
                    if worker.exists()
                    else {
                        "status": status["status"],
                        "scenario": scenario,
                        "seed": seed,
                        "arm": arm,
                        "error": status,
                    }
                )
                expected = capture_hashes.get((scenario, seed))
                expected_first = (
                    None
                    if expected is None
                    else {
                        "index_init": expected.get("index_init"),
                        "U": expected.get("U"),
                        "I": expected.get("I"),
                        "mass": expected.get("mass"),
                    }
                )
                row["first_statistics_hashes_expected"] = expected_first
                row["pair_hash_match"] = bool(
                    expected_first
                    and expected_first == row.get("first_statistics_hashes")
                )
                row["data_hash_match"] = bool(
                    expected
                    and all(
                        row.get("input_hashes", {}).get(k) == expected.get(k)
                        for k in ("X", "y", "truth")
                    )
                )
                row.update(
                    {
                        "order_index": order_index,
                        "worker_status": status,
                        **{k: v for k, v in status.items() if k != "status"},
                    }
                )
                jsonl_append(output, row)


def fullfit_heldout() -> None:
    _assert_frozen()
    selection_path = OUT / "results" / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "write reviewed results/selection.json after development audit"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected_arm")
    if selected not in ARMS:
        raise ValueError("selection.json selected_arm must name a declared arm")
    output = OUT / "results" / "fullfit_heldout.jsonl"
    if output.exists():
        raise FileExistsError(output)
    capture_output = OUT / "results" / "capture_heldout.jsonl"
    if capture_output.exists():
        raise FileExistsError(capture_output)
    development = OUT / "results" / "fullfit_dev.jsonl"
    expected_development_hash = hashlib.sha256(development.read_bytes()).hexdigest()
    if selection.get("development_runs_sha256") != expected_development_hash:
        raise ValueError(
            "selection.json must identify the audited fullfit_dev.jsonl SHA-256"
        )
    arms = tuple(dict.fromkeys((selected, "core", "spectral")))
    capture_hashes: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for seed in HOLDOUT_SEEDS:
            worker = OUT / "results" / "workers" / f"capture_{scenario}_{seed}.json"
            worker.parent.mkdir(parents=True, exist_ok=True)
            _fresh_worker(worker)
            status = _run_child(
                "experiments.grassman_rank_k.fullfit",
                [
                    "--scenario",
                    scenario,
                    "--seed",
                    str(seed),
                    "--capture",
                    "--output",
                    str(worker),
                ],
                180,
            )
            row = (
                _read_row(worker)
                if worker.exists()
                else {
                    "status": status["status"],
                    "scenario": scenario,
                    "seed": seed,
                    "arm": "capture",
                    "error": status,
                }
            )
            row.update(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "arm": "capture",
                    "worker_status": status,
                }
            )
            jsonl_append(capture_output, row)
            if row.get("status") == "captured":
                capture_hashes[(scenario, seed)] = row["input_hashes"]
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in HOLDOUT_SEEDS:
            offset0 = (seed + scenario_index) % len(arms)
            order = arms[offset0:] + arms[:offset0]
            for order_index, arm in enumerate(order):
                worker = (
                    OUT / "results" / "workers" / f"fit_{scenario}_{seed}_{arm}.json"
                )
                _fresh_worker(worker)
                status = _run_child(
                    "experiments.grassman_rank_k.fullfit",
                    [
                        "--scenario",
                        scenario,
                        "--seed",
                        str(seed),
                        "--arm",
                        arm,
                        "--output",
                        str(worker),
                    ],
                    180,
                )
                row = (
                    _read_row(worker)
                    if worker.exists()
                    else {
                        "status": status["status"],
                        "scenario": scenario,
                        "seed": seed,
                        "arm": arm,
                        "error": status,
                    }
                )
                expected = capture_hashes.get((scenario, seed), {})
                expected_first = {
                    key: expected.get(key) for key in ("index_init", "U", "I", "mass")
                }
                row["first_statistics_hashes_expected"] = expected_first
                row["pair_hash_match"] = bool(
                    expected_first.get("U")
                    and expected_first == row.get("first_statistics_hashes")
                )
                row["data_hash_match"] = bool(
                    expected
                    and all(
                        row.get("input_hashes", {}).get(key) == expected.get(key)
                        for key in ("X", "y", "truth")
                    )
                )
                row.update(
                    {
                        "selection": selection,
                        "selection_sha256": hashlib.sha256(
                            selection_path.read_bytes()
                        ).hexdigest(),
                        "development_runs_sha256": expected_development_hash,
                        "order_index": order_index,
                        "scenario_index": scenario_index,
                        "worker_status": status,
                    }
                )
                jsonl_append(output, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("freeze", "fullfit-dev", "fullfit-heldout", "inner"),
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "fullfit-dev":
        fullfit_dev()
    elif args.stage == "fullfit-heldout":
        fullfit_heldout()
    else:
        from .run_inner import inner_runs

        inner_runs()


if __name__ == "__main__":
    main()
