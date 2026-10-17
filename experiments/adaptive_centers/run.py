"""Freeze the experiment and dispatch authorized execution stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from typing import Any

from .common import (
    ARMS,
    CASES,
    DEV_SEEDS,
    FIT_SETTINGS,
    HELDOUT_SEEDS,
    OUT,
    ROOT,
    THREAD_ENV,
    environment,
    git_state,
    provenance_hashes,
    source_hashes,
    write_json,
)


def manifest() -> dict[str, Any]:
    return {
        "task": "adaptive-centers-fullfit-2026-10",
        "status": "frozen-before-runs",
        "protocol_sha256": hashlib.sha256(
            (OUT / "PROTOCOL.md").read_bytes()
        ).hexdigest(),
        "development_seeds": DEV_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "cases": [asdict(case) for case in CASES],
        "arms": [asdict(arm) for arm in ARMS],
        "fit_settings": FIT_SETTINGS,
        "planned_development_fits": len(CASES) * len(DEV_SEEDS) * len(ARMS),
        "execution": {
            "processes": 1,
            "threads": dict.fromkeys(THREAD_ENV, 1),
            "timeout_sec": FIT_SETTINGS["timeout_sec"],
            "rss": "resource.ru_maxrss (KiB on Linux)",
            "wall": "time.perf_counter around ADP_multi_index.fit",
            "data_generation": "outside fit wall; separately hashed, deterministic",
        },
        "source_sha256": source_hashes(),
        "provenance_sha256": provenance_hashes(),
        "git": git_state(),
        "environment_at_freeze": environment(),
    }


def freeze() -> None:
    path = OUT / "results" / "freeze_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = manifest()
    snapshot = OUT / "results" / "source_snapshot.zip"
    snapshot_names = (
        *frozen["source_sha256"],
        *frozen["provenance_sha256"],
        "experiments/adaptive_centers/PROTOCOL.md",
    )
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in snapshot_names:
            archive.write(ROOT / relative, relative)
    frozen["source_snapshot"] = str(snapshot.relative_to(ROOT))
    frozen["source_snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    write_json(path, frozen)


def assert_frozen() -> dict[str, Any]:
    path = OUT / "results" / "freeze_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("run --stage freeze first")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if source_hashes() != frozen.get("source_sha256"):
        raise RuntimeError("numerical or harness sources changed after freeze")
    if hashlib.sha256((OUT / "PROTOCOL.md").read_bytes()).hexdigest() != frozen.get(
        "protocol_sha256"
    ):
        raise RuntimeError("PROTOCOL.md changed after freeze")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "smoke", "dev"), required=True)
    parser.add_argument("--case", default=CASES[0].name)
    parser.add_argument(
        "--start-authorized",
        action="store_true",
        help="required to start numerical fits after lead authorization",
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
        return
    if not args.start_authorized:
        parser.error("numerical runs require --start-authorized after lead approval")
    from .execute import run_development, run_smoke

    if args.stage == "smoke":
        run_smoke(args.case)
    else:
        run_development()


if __name__ == "__main__":
    main()
