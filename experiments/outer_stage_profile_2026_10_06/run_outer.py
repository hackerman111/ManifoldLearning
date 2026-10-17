"""Capture existing ADP model stage profiles on matching baseline fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "ADP/engine/common/index_fit.py", "ADP/engine/common/statistic.py",
    "ADP/engine/common/weights.py", "ADP/engine/common/calculus.py",
    "ADP/engine/common/logger.py",
]


def worker(seed: int) -> dict:
    from ADP.core.multi.ADP_multi_index import ADP_multi_index
    from experiments.grassman_stage_profile_2026_10_06.run_profile import profiled_fit

    saved = {}
    original = ADP_multi_index.fit

    def capture(self, *args, **kwargs):
        fitted = original(self, *args, **kwargs)
        saved["profile"] = self.profile_
        return fitted

    ADP_multi_index.fit = capture
    row = profiled_fit(seed, "baseline", None)
    row["model_profile"] = saved["profile"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.seed), allow_nan=False))
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "runs.jsonl"
    if rows_path.exists():
        raise FileExistsError(rows_path)
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("UV_CACHE_DIR", "/tmp/adp-uv-cache")
    rows = []
    with rows_path.open("x", encoding="utf-8") as stream:
        for seed in (0, 1, 2):
            cmd = ["uv", "run", "--no-sync", "python", "-m", "experiments.outer_stage_profile_2026_10_06.run_outer", "--worker", "--seed", str(seed), "--output", str(out)]
            started = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=900)
                if proc.returncode:
                    row = {"seed": seed, "status": "process_error", "returncode": proc.returncode, "stderr": proc.stderr[-4000:]}
                else:
                    row = json.loads(proc.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired as exc:
                row = {"seed": seed, "status": "timeout", "stderr": str(exc)[:2000]}
            row["wall_including_startup_sec"] = time.perf_counter() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if row.get("status") != "ok":
                break
    prior_path = ROOT / "experiments/grassman_stage_profile_2026_10_06/results/runs.jsonl"
    prior = [json.loads(line) for line in prior_path.read_text().splitlines() if line.strip()]
    baselines = {r["seed"]: r for r in prior if r.get("mode") == "baseline" and r.get("status") == "ok"}
    paired = []
    for row in rows:
        old = baselines.get(row.get("seed"), {})
        paired.append({"seed": row.get("seed"), "same_inputs": row.get("input_hashes") == old.get("input_hashes"), "same_basis": row.get("basis_hash") == old.get("basis_hash"), "same_diagnostics": row.get("diagnostics_hash") == old.get("diagnostics_hash")})
    protocol = {"command": "UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python -m experiments.outer_stage_profile_2026_10_06.run_outer", "seeds": [0, 1, 2], "threads": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, "timeout_sec": 900, "paired_parity": paired, "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": [{k: v for k, v in r.items() if k != "model_profile"} for r in rows]}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"paired_parity": paired, "profiles": [{"seed": r.get("seed"), "fit_time_sec": r.get("fit_time_sec"), "model_profile": r.get("model_profile"), "solver_wall_sec": r.get("solver_wall_sec"), "peak_rss_kib": r.get("peak_rss_kib"), "projector_distance_to_truth": r.get("projector_distance_to_truth"), "outer_iterations": r.get("outer_iterations"), "selected_iteration": r.get("selected_iteration"), "fit_stop_reason": r.get("fit_stop_reason")} for r in rows]}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
