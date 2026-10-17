"""Freeze and serially run the isolated P1 localization microbenchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .worker import CASES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SEEDS = (93000, 93001, 93002)
NUMERICAL_SOURCES = (
    "ADP/core/ADP_Config.py",
    "ADP/engine/common/calculus.py",
    "ADP/engine/common/experimental_geometry.py",
    "ADP/engine/common/utils.py",
    "ADP/engine/common/weights.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> dict[str, str]:
    paths = [*HERE.glob("*.py"), *(ROOT / path for path in NUMERICAL_SOURCES)]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def freeze() -> None:
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    parent_manifest = OUT.parents[1] / "results" / "freeze_manifest.json"
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    parent_sha = sha(parent_manifest)
    manifest = {
        "task": "adaptive-centers-p1-localization-microbenchmark",
        "parent_freeze_manifest_sha256": parent_sha,
        "parent_source_snapshot_sha256": parent["source_snapshot_sha256"],
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "source_sha256": source_files(),
        "cases": CASES,
        "seeds": SEEDS,
        "modes": ("weights", "alpha_weights"),
        "arms": ("baseline", "projected"),
        "thread_environment": dict.fromkeys(THREAD_KEYS, "1"),
        "runs": len(CASES) * len(SEEDS) * 2 * 2,
        "git_commit": parent["git"].get("commit"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def assert_frozen() -> dict:
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_files() != manifest["source_sha256"]:
        raise RuntimeError("supplement or numerical source changed after freeze")
    if sha(HERE / "PROTOCOL.md") != manifest["protocol_sha256"]:
        raise RuntimeError("supplement protocol changed after freeze")
    parent_manifest = OUT.parents[1] / "results" / "freeze_manifest.json"
    if sha(parent_manifest) != manifest["parent_freeze_manifest_sha256"]:
        raise RuntimeError("parent full-fit freeze manifest changed")
    return manifest


def run_all() -> None:
    assert_frozen()
    stream_path = OUT / "runs.jsonl"
    if stream_path.exists():
        raise FileExistsError(stream_path)
    env = os.environ.copy()
    env.update(dict.fromkeys(THREAD_KEYS, "1"))
    done = 0
    with stream_path.open("x", encoding="utf-8") as stream:
        for case in CASES:
            for seed in SEEDS:
                for mode in ("weights", "alpha_weights"):
                    for arm in ("baseline", "projected"):
                        name = f"{case}_{seed}_{mode}_{arm}.json"
                        output = OUT / "workers" / name
                        command = [
                            sys.executable,
                            str(HERE / "worker.py"),
                            "--case",
                            case,
                            "--seed",
                            str(seed),
                            "--mode",
                            mode,
                            "--arm",
                            arm,
                            "--output",
                            str(output),
                        ]
                        try:
                            proc = subprocess.run(
                                command,
                                cwd=ROOT,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=300,
                                check=False,
                            )
                            status = "ok" if proc.returncode == 0 else "error"
                            error = proc.stderr[-4000:]
                        except subprocess.TimeoutExpired as exc:
                            status = "timeout"
                            error = str(exc)
                        record = {
                            "case": case,
                            "seed": seed,
                            "mode": mode,
                            "arm": arm,
                            "status": status,
                            "error": error,
                            "worker_file": name if status == "ok" else None,
                        }
                        if status == "ok":
                            record.update(
                                json.loads(output.read_text(encoding="utf-8"))
                            )
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                        stream.flush()
                        done += 1
                        print(
                            f"{done}/24 {case} {seed} {mode} {arm}: {status}",
                            flush=True,
                        )
                        if status != "ok":
                            (OUT / f"failure_{name}.json").write_text(
                                json.dumps(record, indent=2) + "\n", encoding="utf-8"
                            )
    assert_frozen()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    parser.add_argument("--start-authorized", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        if not args.start_authorized:
            parser.error("numerical runs require lead authorization")
        run_all()


if __name__ == "__main__":
    main()
