"""Validate paired localization outputs and summarize per-seed comparisons."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def read_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (OUT / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def main() -> None:
    rows = read_rows()
    indexed = {(row["case"], row["seed"], row["mode"], row["arm"]): row for row in rows}
    audit = []
    paired = []
    for case in ("small", "large"):
        for seed in (93000, 93001, 93002):
            for mode in ("weights", "alpha_weights"):
                baseline = indexed[(case, seed, mode, "baseline")]
                projected = indexed[(case, seed, mode, "projected")]
                same_inputs = baseline["hashes"] == projected["hashes"]
                valid = all(row["status"] == "ok" for row in (baseline, projected))
                audit.append(
                    {
                        "case": case,
                        "seed": seed,
                        "mode": mode,
                        "same_input_hashes": same_inputs,
                        "both_ok": valid,
                        "thread_count_one": all(
                            pool["num_threads"] == 1
                            for row in (baseline, projected)
                            for pool in row.get("threadpools", [])
                        ),
                    }
                )
                if not valid or not same_inputs:
                    continue
                paired.append(
                    {
                        "case": case,
                        "seed": seed,
                        "mode": mode,
                        "baseline_sec": baseline["localization_plus_output_sec"],
                        "projected_sec": projected["localization_plus_output_sec"],
                        "speed_ratio_projected_over_baseline": (
                            projected["localization_plus_output_sec"]
                            / baseline["localization_plus_output_sec"]
                        ),
                        "baseline_weight_only_sec": baseline["weight_only_sec"],
                        "projected_weight_only_sec": projected["weight_only_sec"],
                        "baseline_alpha_sec": baseline["alpha_search_sec"],
                        "projected_alpha_sec": projected["alpha_search_sec"],
                        "baseline_alpha": baseline["alpha"],
                        "projected_alpha": projected["alpha"],
                        "alpha_error": projected["alpha_error"],
                        "max_weight_abs_error": projected["max_weight_abs_error"],
                        "support_mismatches": projected["support_mismatches"],
                        "baseline_nnz": baseline["actual_storage"]["nnz"],
                        "projected_nnz": projected["actual_storage"]["nnz"],
                        "baseline_csr_bytes": baseline["actual_storage"]["csr_bytes"],
                        "projected_csr_bytes": projected["actual_storage"]["csr_bytes"],
                        "baseline_peak_rss_kib": baseline["peak_rss_after_timing_kib"],
                        "projected_peak_rss_kib": projected[
                            "peak_rss_after_timing_kib"
                        ],
                    }
                )
    if (
        len(rows) != 24
        or len(paired) != 12
        or not all(
            item["same_input_hashes"] and item["both_ok"] and item["thread_count_one"]
            for item in audit
        )
    ):
        raise RuntimeError(f"microbenchmark audit failed: {audit}")
    (OUT / "pairing_audit.json").write_text(
        json.dumps({"pass": True, "pairs": audit}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUT / "paired_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = []
    for case in ("small", "large"):
        for mode in ("weights", "alpha_weights"):
            selected = [
                row for row in paired if row["case"] == case and row["mode"] == mode
            ]
            summary.append(
                {
                    "case": case,
                    "mode": mode,
                    "n_pairs": len(selected),
                    "median_speed_ratio": statistics.median(
                        row["speed_ratio_projected_over_baseline"] for row in selected
                    ),
                    "max_weight_error": max(
                        row["max_weight_abs_error"] for row in selected
                    ),
                    "max_support_mismatches": max(
                        row["support_mismatches"] for row in selected
                    ),
                    "max_alpha_error": max(
                        row["alpha_error"] or 0.0 for row in selected
                    ),
                }
            )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
