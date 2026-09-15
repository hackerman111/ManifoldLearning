from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np

root = Path("benchmark_outputs/hybrid_optimization")
pairs = {}
for path in (root / "validated").glob("*.json"):
    case, seed, variant = path.stem.rsplit("_", 2)
    pairs.setdefault(case, {}).setdefault(seed, {})[variant] = (
        json.loads(path.read_text()),
        path,
    )

rows = []
for case, seeds in pairs.items():
    row = {"case": case, "pairs": len(seeds), "failed_pairs": []}
    good = []
    max_difference = 0.0
    max_quality_difference = 0.0
    for seed, pair in seeds.items():
        a, ap = pair["reference"]
        b, bp = pair["hybrid"]
        if a["error"] or b["error"]:
            row["failed_pairs"].append(
                {"seed": seed, "reference": a["error"], "hybrid": b["error"]}
            )
            continue
        good.append((a, b))
        A, B = np.load(ap.with_suffix(".npy")), np.load(bp.with_suffix(".npy"))
        if A.ndim == 3:
            residual = B - (B @ A.swapaxes(1, 2)) @ A
            diff = np.sqrt(2 * np.square(residual).sum(axis=(1, 2))).max()
        else:
            diff = np.sqrt(2) * np.linalg.norm(B - (B @ A.T) @ A)
        max_difference = max(max_difference, float(diff))
        max_quality_difference = max(
            max_quality_difference, abs(a["quality"] - b["quality"])
        )
    for mode, position in (("reference", 0), ("hybrid", 1)):
        values = [p[position] for p in good]
        entry = {
            "solver_seconds": median(
                v["profile"]["solver"]["time_seconds"] for v in values
            ),
            "total_seconds": median(
                v["profile"]["total"]["time_seconds"] for v in values
            ),
            "solver_traced_peak_bytes": median(
                v["profile"]["solver"]["traced_peak_bytes"] for v in values
            ),
            "rss_peak_mib": median(v["rss_peak_mib"] for v in values),
        }
        if case.startswith("multi"):
            entry["iterations"] = median(
                sum(
                    t["solver"]["linear_iterations_total"]
                    for t in v["metadata"]["trace"]
                )
                for v in values
            )
            entry["screened_trials"] = median(
                sum(
                    t["solver"].get("linear_screened_trials", 0)
                    for t in v["metadata"]["trace"]
                )
                for v in values
            )
            entry["screening_matvecs"] = median(
                sum(
                    t["solver"].get("linear_screening_matvecs", 0)
                    for t in v["metadata"]["trace"]
                )
                for v in values
            )
        else:
            entry["iterations"] = median(
                sum(t.get("linear_iterations", 0) for t in v["metadata"]["trace"])
                for v in values
            )
            entry["residual_max"] = max(
                t.get("linear_relative_residual_max", 0)
                for v in values
                for t in v["metadata"]["trace"]
            )
        row[mode] = entry
    row["max_projector_frobenius_difference"] = max_difference
    row["max_quality_difference"] = max_quality_difference
    row["solver_speedup"] = (
        row["reference"]["solver_seconds"] / row["hybrid"]["solver_seconds"]
    )
    rows.append(row)

report = {
    "aggregation": "median of paired successful seeds; failed pairs retained",
    "results": sorted(rows, key=lambda r: r["case"]),
}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
