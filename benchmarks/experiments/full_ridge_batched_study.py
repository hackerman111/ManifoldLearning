"""Fresh frozen protocol for batched preparation; reuse the paired fit harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

from experiments import full_ridge_batched as backend
from experiments import full_ridge_fast_study as study

# Each isolated worker binds the shared experimental harness to this backend.
study.OUT = study.ROOT / "benchmarks/results/verification/full-ridge-batched-20261025"
study.SEEDS = {
    "selection": tuple(range(974000, 974004)),
    "validation": tuple(range(974100, 974104)),
}
study.fast = backend


def freeze(phase):
    directory, manifest = study.freeze(phase)
    test = study.ROOT / "tests/test_full_ridge_batched.py"
    manifest["sources"][str(test.relative_to(study.ROOT))] = study.sha(test)
    with zipfile.ZipFile(
        directory / "sources.zip", "a", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(test, test.relative_to(study.ROOT))
    manifest["archive_sha256"] = study.sha(directory / "sources.zip")
    del manifest["fingerprint"]
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    study.write(directory / "manifest.json", manifest)
    return directory, manifest


def verify(directory, manifest):
    for relative, expected in manifest["sources"].items():
        if study.sha(study.ROOT / relative) != expected:
            raise RuntimeError(f"frozen source drift: {relative}")
    if study.sha(directory / "sources.zip") != manifest["archive_sha256"]:
        raise RuntimeError("source archive drift")


def run(phase):
    if phase == "validation":
        if not json.loads((study.OUT / "selection/summary.json").read_text())["passed"]:
            raise RuntimeError("selection failed; validation remains untouched")
        audit = json.loads((study.OUT / "selection/independent_audit.json").read_text())
        if not audit["integrity_ok"] or not audit["audit_passed"]:
            raise RuntimeError("independent selection audit must pass first")
    directory, manifest = freeze(phase)
    if phase == "validation":
        selected = json.loads((study.OUT / "selection/manifest.json").read_text())
        if manifest["sources"] != selected["sources"]:
            raise RuntimeError("source changed after selection")
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    started = time.perf_counter()
    for i, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
        verify(directory, manifest)
        input_path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
        if (
            study.sha(input_path)
            != manifest["inputs"][str(input_path.relative_to(study.ROOT))]
        ):
            raise RuntimeError("frozen input drift")
        name = f"{link}-m{m}-{seed}-{arm}"
        log = directory / "fits" / f"{name}.log"
        result = study.run_process(
            [
                sys.executable,
                "-m",
                "experiments.full_ridge_batched_study",
                "worker",
                phase,
                link,
                str(m),
                str(seed),
                arm,
            ],
            env,
            log,
            timeout=60,
            memory_mib=2048,
            cwd=study.ROOT,
        )
        verify(directory, manifest)
        if result["status"] == "ok":
            try:
                row = json.loads(log.read_text().splitlines()[-1])
            except (ValueError, IndexError) as error:
                row = {"status": "invalid_output", "error": repr(error)}
        else:
            row = {"status": result["status"], "error": log.read_text()[-4000:]}
        row.update(
            link=link,
            m=m,
            seed=seed,
            arm=arm,
            fingerprint=manifest["fingerprint"],
            process_seconds=result["process_seconds"],
            process_peak_rss_mib=result["peak_rss_mib"],
        )
        study.write(directory / "fits" / f"{name}.json", row)
        if i % 12 == 0 or row["status"] != "ok":
            print(
                f"{phase}: {i}/72, {time.perf_counter() - started:.1f}s, {row['status']}",
                flush=True,
            )
    study.write(
        directory / "execution.json",
        {"wall_seconds": time.perf_counter() - started, "recorded": i},
    )
    return study.analyze(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "run":
        run(args.arguments[0])
    elif args.action == "worker":
        phase, link, m, seed, arm = args.arguments
        with np.load(study.OUT / phase / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
            row = study.fit(saved["x"], saved["y"], saved["truth"], int(seed), arm)
        print(
            json.dumps({**row, "status": "ok"}, default=study.default, allow_nan=False)
        )
    elif args.action == "analyze":
        study.analyze(args.arguments[0])
    else:
        x, y, truth = study.arrays(study.case(25, 5, 5000, 0.3, "sine"), 974900)
        for arm in study.ARMS:
            row = study.fit(x, y, truth, 974900, arm)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "fit_seconds": row["fit_seconds"],
                        "stages": len(row["stages"]),
                    },
                    default=study.default,
                )
            )


if __name__ == "__main__":
    main()
