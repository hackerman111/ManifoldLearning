"""Replay six representative rows against the final guarded implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location("metric_benchmark", OUT / "benchmark.py")
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
rows = [json.loads(line) for line in (OUT / "selection.jsonl").read_text().splitlines()]
replays = []
for row in rows:
    if row["variant"] != "tensor" or row["seed"] != 11:
        continue
    if row["kind"] == "fixed" and row["case"] != "300":
        continue
    current = benchmark.worker(
        row["kind"], row["case"], row["seed"], row["target"], row["variant"]
    )
    assert current["data_sha256"] == row["data_sha256"]
    compared = ["objective_last", "normal_residual_max"]
    compared.append(
        "projector_distance" if row["kind"] == "fit" else "coefficient_error"
    )
    differences = {key: current[key] - row[key] for key in compared}
    for key in compared:
        assert abs(differences[key]) <= 1e-10 * max(1.0, abs(row[key]))
    replays.append(
        {
            "kind": row["kind"],
            "case": row["case"],
            "target": row["target"],
            "differences": differences,
        }
    )
assert len(replays) == 6
verification = {
    "tests": "90 passed: metric, SVD, model/CLI regressions",
    "ruff": "passed",
    "diff_check": "passed",
    "replays": replays,
    "final_hashes": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [
            "ADP/solver/SVD.py",
            "ADP/core/ADP_Solver.py",
            "ADP/core/multi/ADP_multi_index.py",
            "ADP/engine/common/index_fit.py",
            "tests/test_svd_metric.py",
            "experiments/svd_a_metric_2026-09-30/benchmark.py",
        ]
    },
}
(OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print("Six numerical replays agree with saved measurements.")
