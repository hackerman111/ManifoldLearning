"""Summarize the paired forced-LSMR preconditioner comparison."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "runs.jsonl").read_text().splitlines()]
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        raise SystemExit(f"worker failures retained in runs.jsonl: {errors}")
    by_key = {(int(row["seed"]), bool(row["precondition_v"])): row for row in rows}
    if len(by_key) != 6:
        raise SystemExit(f"expected 6 unique paired rows, got {len(by_key)}")

    pairs = []
    for seed in (0, 1, 2):
        off, on = by_key[(seed, False)], by_key[(seed, True)]
        pairs.append(
            {
                "seed": seed,
                "time_off_sec": off["fit_time_sec"],
                "time_on_sec": on["fit_time_sec"],
                "time_ratio_on_over_off": on["fit_time_sec"] / off["fit_time_sec"],
                "time_change_percent": 100
                * (on["fit_time_sec"] / off["fit_time_sec"] - 1),
                "projector_error_off": off["projector_distance"],
                "projector_error_on": on["projector_distance"],
                "projector_error_change": on["projector_distance"]
                - off["projector_distance"],
                "outer_iterations_off": off["outer_iterations"],
                "outer_iterations_on": on["outer_iterations"],
                "direct_solves_off": off["direct_solves"],
                "direct_solves_on": on["direct_solves"],
                "lsmr_iterations_off": off["linear_iterations"],
                "lsmr_iterations_on": on["linear_iterations"],
                "lsmr_iteration_ratio_on_over_off": on["linear_iterations"]
                / off["linear_iterations"],
                "lsmr_iteration_change_percent": 100
                * (on["linear_iterations"] / off["linear_iterations"] - 1),
                "flag_enabled_updates_on": on["updates_with_flag_enabled"],
                "peak_rss_mib_off": off["peak_rss_kib"] / 1024,
                "peak_rss_mib_on": on["peak_rss_kib"] / 1024,
            }
        )

    ratios = [pair["time_ratio_on_over_off"] for pair in pairs]
    iteration_ratios = [pair["lsmr_iteration_ratio_on_over_off"] for pair in pairs]
    summary = {
        "rows": len(rows),
        "pairs": pairs,
        "median_time_off_sec": statistics.median(
            row["fit_time_sec"] for row in rows if not row["precondition_v"]
        ),
        "median_time_on_sec": statistics.median(
            row["fit_time_sec"] for row in rows if row["precondition_v"]
        ),
        "median_paired_time_ratio_on_over_off": statistics.median(ratios),
        "median_paired_time_change_percent": 100 * (statistics.median(ratios) - 1),
        "median_paired_lsmr_iteration_ratio_on_over_off": statistics.median(
            iteration_ratios
        ),
        "median_paired_lsmr_iteration_change_percent": 100
        * (statistics.median(iteration_ratios) - 1),
        "median_peak_rss_mib_off": statistics.median(
            row["peak_rss_kib"] / 1024 for row in rows if not row["precondition_v"]
        ),
        "median_peak_rss_mib_on": statistics.median(
            row["peak_rss_kib"] / 1024 for row in rows if row["precondition_v"]
        ),
        "all_projector_errors_equal_within_pair": all(
            pair["projector_error_change"] == 0 for pair in pairs
        ),
        "all_outer_steps_equal_within_pair": all(
            pair["outer_iterations_off"] == pair["outer_iterations_on"]
            for pair in pairs
        ),
        "all_full_fits_exceeded_30_second_budget": all(
            not row["within_time_budget"] for row in rows
        ),
        "all_svd_inner_converged": all(
            row["svd_inner_converged"] for row in rows
        ),
        "all_lsmr_iterations_positive": all(row["linear_iterations"] > 0 for row in rows),
        "all_direct_solves_zero": all(row["direct_solves"] == 0 for row in rows),
        "all_direct_fallbacks_zero": all(row["direct_fallbacks"] == 0 for row in rows),
        "all_preconditioner_cache_diagnostics_present": all(
            "diagnostic_preconditioner_cache_bytes" in row for row in rows
        ),
        "direct_solve_counts_match_within_pair": all(
            pair["direct_solves_off"] == pair["direct_solves_on"] for pair in pairs
        ),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
