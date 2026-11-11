"""Paired d=50/100 full-fit study for batched full-moment ridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

from experiments import full_ridge_batched_study as harness
from experiments.mave_grassmann_grid import case as base_case

ROOT = harness.study.ROOT
OUT = ROOT / "benchmarks/results/verification/full-ridge-batched-hd-r2-20261025"

# Encode ambient dimension in the link key so the shared frozen harness can
# retain its (link, m, seed, arm) key format without conflating d=50 and d=100.
LINKS = tuple(f"d{d}-{link}" for d in (50, 100)
              for link in ("additive", "quadratic", "sine"))
CASES = tuple((link, m) for link in LINKS for m in (2, 3, 5))
SEEDS = {
    "selection": (975000, 975001),
    "validation": (975100, 975101),
}


def case(d, m, n, noise, link="additive", block="core"):
    if link.startswith("d") and "-" in link:
        encoded_d, actual_link = link.split("-", 1)
        d, link = int(encoded_d[1:]), actual_link
    return base_case(d, m, n, noise, link, block)


def configure():
    harness.study.OUT = OUT
    harness.study.CASES = CASES
    harness.study.SEEDS = SEEDS
    harness.study.case = case

    original_freeze = harness.freeze

    def freeze_with_auditor(phase):
        directory, manifest = original_freeze(phase)
        audit_path = OUT / "audit.py"
        relative = str(audit_path.relative_to(ROOT))
        manifest["sources"][relative] = harness.study.sha(audit_path)
        with zipfile.ZipFile(directory / "sources.zip", "a", zipfile.ZIP_DEFLATED) as archive:
            archive.write(audit_path, relative)
        manifest["archive_sha256"] = harness.study.sha(directory / "sources.zip")
        manifest.pop("fingerprint", None)
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        harness.study.write(directory / "manifest.json", manifest)
        return directory, manifest

    harness.freeze = freeze_with_auditor

    def run(phase):
        if phase == "validation":
            summary_path = OUT / "selection/summary.json"
            audit_path = OUT / "selection/independent_audit.json"
            if not json.loads(summary_path.read_text())["passed"]:
                raise RuntimeError("selection failed; validation remains untouched")
            audit = json.loads(audit_path.read_text())
            if not audit["integrity_ok"] or not audit["audit_passed"]:
                raise RuntimeError("independent selection audit must pass first")
        directory, manifest = harness.freeze(phase)
        if phase == "validation":
            selected = json.loads((OUT / "selection/manifest.json").read_text())
            if manifest["sources"] != selected["sources"]:
                raise RuntimeError("source changed after selection")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        started = time.perf_counter()
        for i, (link, m, seed, arm) in enumerate(manifest["jobs"], 1):
            harness.verify(directory, manifest)
            input_path = directory / "inputs" / f"{link}-m{m}-{seed}.npz"
            if harness.study.sha(input_path) != manifest["inputs"][str(input_path.relative_to(ROOT))]:
                raise RuntimeError("frozen input drift")
            name = f"{link}-m{m}-{seed}-{arm}"
            log = directory / "fits" / f"{name}.log"
            result = harness.study.run_process(
                [sys.executable, "-m", "experiments.full_ridge_batched_hd_study",
                 "worker", phase, link, str(m), str(seed), arm],
                env, log, timeout=60, memory_mib=2048, cwd=ROOT,
            )
            harness.verify(directory, manifest)
            if result["status"] == "ok":
                try:
                    row = json.loads(log.read_text().splitlines()[-1])
                except (ValueError, IndexError) as error:
                    row = {"status": "invalid_output", "error": repr(error)}
            else:
                row = {"status": result["status"], "error": log.read_text()[-4000:]}
            row.update(link=link, m=m, seed=seed, arm=arm,
                       fingerprint=manifest["fingerprint"],
                       process_seconds=result["process_seconds"],
                       process_peak_rss_mib=result["peak_rss_mib"])
            harness.study.write(directory / "fits" / f"{name}.json", row)
            if i % 12 == 0 or row["status"] != "ok":
                print(f"{phase}: {i}/72, {time.perf_counter()-started:.1f}s, {row['status']}",
                      flush=True)
        harness.study.write(directory / "execution.json",
                            {"wall_seconds": time.perf_counter()-started,
                             "recorded": i})
        return harness.study.analyze(phase)

    harness.run = run


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "analyze"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.action == "run":
        harness.run(args.arguments[0])
    elif args.action == "analyze":
        harness.study.analyze(args.arguments[0])
    else:
        phase, link, m, seed, arm = args.arguments
        with np.load(OUT / phase / "inputs" / f"{link}-m{m}-{seed}.npz") as saved:
            row = harness.study.fit(
                saved["x"], saved["y"], saved["truth"], int(seed), arm
            )
        print(json.dumps({**row, "status": "ok"}, default=harness.study.default,
                         allow_nan=False))


if __name__ == "__main__":
    main()
