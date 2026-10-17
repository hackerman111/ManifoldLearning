"""Run the bounded independent polish supplement serially after inner runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import (
    OUT,
    _assert_frozen,
    _fresh_worker,
    _read_row,
    _run_child,
    jsonl_append,
)


def main() -> None:
    _assert_frozen()
    results = OUT / "results"
    source_path = OUT / "reference_polish.py"
    runner_path = Path(__file__)
    test_path = OUT.parents[1] / "tests" / "test_grassman_rank_polish.py"
    frozen_path = OUT / "freeze_manifest.json"
    inner_path = results / "inner.jsonl"
    manifest_path = results / "reference_polish_manifest.json"
    output_path = results / "reference_polish.jsonl"
    if manifest_path.exists() or output_path.exists():
        raise FileExistsError(manifest_path if manifest_path.exists() else output_path)

    rows = [
        json.loads(line) for line in inner_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != 792:
        raise RuntimeError(f"inner.jsonl must contain 792 rows, found {len(rows)}")
    cases = [
        row
        for row in rows
        if row.get("arm") == "reference" and row.get("converged") is not True
    ]
    if len(cases) > 24:
        raise RuntimeError(f"polish case cap is 24, found {len(cases)}")
    worker_paths = [
        results
        / "workers"
        / f"refpolish_{row.get('m')}_{row.get('d')}_{row.get('seed')}.json"
        for row in cases
    ]
    if len(set(worker_paths)) != len(worker_paths):
        raise RuntimeError("polish cases do not have unique worker output paths")
    existing_workers = [path for path in worker_paths if path.exists()]
    if existing_workers:
        raise FileExistsError(existing_workers[0])

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source_path, runner_path, test_path, frozen_path)
    }
    manifest = {
        "inner_jsonl_sha256": hashlib.sha256(inner_path.read_bytes()).hexdigest(),
        "source_hashes": hashes,
        "selected_case_count": len(cases),
        "selected_case_keys": [
            {key: row.get(key) for key in ("scenario", "m", "d", "J", "seed")}
            for row in cases
        ],
        "settings": {
            "lbfgs_max_steps_per_start": 5000,
            "cg_max_steps_per_start": 1000,
            "starts_per_case": 3,
            "tolerance": 1e-9,
            "timeout_sec": 180,
        },
        "execution": "serial, pinned thread counts through common._run_child",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    output_path.write_text("")

    for case in cases:
        m, d, seed = (case[key] for key in ("m", "d", "seed"))
        snapshot_value = case.get("snapshot")
        snapshot = (
            Path(snapshot_value)
            if isinstance(snapshot_value, str)
            else (OUT / "captures" / f"inner_m{m}_d{d}_seed{seed}_J64.npz")
        )
        if not snapshot.is_absolute():
            snapshot = OUT / snapshot
        worker = results / "workers" / f"refpolish_{m}_{d}_{seed}.json"
        _fresh_worker(worker)
        worker_status = _run_child(
            "experiments.grassman_rank_k.reference_polish",
            ["--snapshot", str(snapshot), "--output", str(worker)],
            180,
        )
        polish_result = _read_row(worker) if worker.exists() else None
        diagnostics = case.get("diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        output = {
            "scenario": case.get("scenario"),
            "m": m,
            "d": d,
            "J": case.get("J"),
            "seed": seed,
            "arm": "reference_polish",
            "worker_status": worker_status,
            "error": None
            if polish_result is not None and worker_status.get("status") == "ok"
            else worker_status,
            "original_reference": {
                "objective": case.get("objective"),
                "normalized_gradient": case.get(
                    "normalized_gradient", diagnostics.get("normalized_gradient")
                ),
                "converged": case.get("converged"),
                "status": case.get("status", case.get("worker_status")),
            },
            "polish_result": polish_result,
        }
        jsonl_append(output_path, output)


if __name__ == "__main__":
    main()
