"""Evaluate prespecified gates without tuning fixtures or thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def evaluate(stage):
    if stage == "selection":
        fixed_prefix = "selection"
        output_prefix = "selection_corrected"
    else:
        fixed_prefix = "validation_fixed"
        output_prefix = "validation"
    manifest = json.loads((ROOT / f"{fixed_prefix}_manifest.json").read_text())
    initial_rows = [
        json.loads(line)
        for line in (ROOT / f"{fixed_prefix}_runs.jsonl").read_text().splitlines()
    ]
    rows = [r for r in initial_rows if r["kind"] == "fixed"]
    rows += [
        json.loads(line)
        for line in (ROOT / f"{stage}_fit_runs.jsonl").read_text().splitlines()
    ]
    lookup = {
        (r["kind"], r["d"], r.get("scaled"), r["mode"], r["seed"], r["method"]): r
        for r in rows
    }
    gates = manifest["gates"]
    summary = {
        "stage": stage,
        "rows": len(rows),
        "failures": [r for r in rows if r["status"] != "ok"],
    }
    summary["complete"] = len(rows) == 144 and len(lookup) == 144
    successful = [r for r in rows if r["status"] == "ok"]
    summary["max_relative_matrix_error"] = max(
        r.get("relative_matrix_error", 0) for r in successful
    )
    summary["max_projector_disagreement"] = max(
        r["projector_disagreement"] for r in successful
    )
    comparisons = []
    quality_changes = []
    step_changes = []
    for row in successful:
        if row["method"] == "baseline":
            continue
        baseline = lookup[
            (
                row["kind"],
                row["d"],
                row.get("scaled"),
                row["mode"],
                row["seed"],
                "baseline",
            )
        ]
        if baseline["status"] != "ok":
            continue
        pair = {
            "kind": row["kind"],
            "d": row["d"],
            "scaled": row.get("scaled"),
            "mode": row["mode"],
            "method": row["method"],
            "seed": row["seed"],
            "time_ratio": row["wall_sec"] / baseline["wall_sec"],
        }
        if row["kind"] == "fixed":
            pair["pass_ratio"] = (
                row["diagnostics"]["u_vector_passes"]
                / baseline["diagnostics"]["u_vector_passes"]
            )
        else:
            quality_changes.append(
                abs(row["projector_error"] - baseline["projector_error"])
            )
            step_changes.append(abs(row["outer_steps"] - baseline["outer_steps"]))
        if "traced_peak_bytes" in row:
            pair["default_memory_ok"] = (
                row["traced_peak_bytes"]
                <= 1.1 * baseline["traced_peak_bytes"] + 1024**2
            )
        comparisons.append(pair)
    summary["pairs"] = comparisons
    summary["max_quality_change"] = max(quality_changes, default=0)
    summary["max_outer_step_change"] = max(step_changes, default=0)
    scaled = [
        p
        for p in comparisons
        if p["kind"] == "fixed" and p["scaled"] and p["method"] == "jacobi"
    ]
    isotropic = [
        p
        for p in comparisons
        if p["kind"] == "fixed" and not p["scaled"] and p["method"] == "fused"
    ]
    summary["scaled_median_time_ratio"] = float(
        np.median([p["time_ratio"] for p in scaled])
    )
    summary["scaled_median_pass_ratio"] = float(
        np.median([p["pass_ratio"] for p in scaled])
    )
    summary["default_isotropic_time_ratio"] = float(
        np.median([p["time_ratio"] for p in isotropic])
    )
    summary["default_memory_ok"] = all(
        p.get("default_memory_ok", True) for p in comparisons if p["method"] == "fused"
    )
    fixed = [r for r in successful if r["kind"] == "fixed"]
    summary["objective_and_rank_ok"] = all(
        r["objective_error"] <= 1e-8 * max(1, r["objective"])
        and np.all(
            np.diff(r["diagnostics"]["rank_objective_history"])
            <= 1e-8 * max(1, r["objective"])
        )
        and r["diagnostics"]["effective_rank"] == 2
        and np.isfinite(r["diagnostics"]["v_normal_residual_max"])
        and r["diagnostics"]["v_normal_residual_max"] <= 1e-7
        for r in fixed
    )
    summary["passed"] = (
        summary["complete"]
        and not summary["failures"]
        and summary["max_relative_matrix_error"] <= gates["relative_matrix_error"]
        and summary["max_projector_disagreement"] <= gates["projector_disagreement"]
        and summary["max_quality_change"] <= gates["quality_change"]
        and summary["max_outer_step_change"] == 0
        and summary["scaled_median_time_ratio"] <= gates["scaled_time_and_pass_ratio"]
        and summary["scaled_median_pass_ratio"] <= gates["scaled_time_and_pass_ratio"]
        and summary["default_isotropic_time_ratio"]
        <= gates["default_isotropic_time_ratio"]
        and summary["default_memory_ok"]
        and summary["objective_and_rank_ok"]
    )
    (ROOT / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"pairs", "failures"}
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    raise SystemExit(0 if result["passed"] else 1)
