"""Audit frozen paired runs and emit descriptive arm and pairwise summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ARMS, CASES, DEV_SEEDS, OUT, ROOT, write_json
from .paired import write_pairwise
from .run import assert_frozen

HASH_KEYS = ("X", "Y", "truth", "initial_basis")


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def read_rows(source: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_pairing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("case")), int(row.get("seed", -1)))].append(row)
    audits = []
    for case in CASES:
        for seed in DEV_SEEDS:
            group = grouped.get((case.name, seed), [])
            by_arm = {str(row.get("arm")): row for row in group}
            valid = [row for row in group if row.get("status") == "ok"]
            reference_row = next(
                (row for row in valid if row.get("arm") == "baseline"),
                valid[0] if valid else {},
            )
            reference = reference_row.get("input_hashes", {})
            mismatches, failures = [], []
            for arm in (item.name for item in ARMS):
                row = by_arm.get(arm)
                if row is None:
                    mismatches.append(arm)
                elif row.get("status") != "ok":
                    failures.append(arm)
                elif any(
                    row.get("input_hashes", {}).get(key) != reference.get(key)
                    for key in HASH_KEYS
                ):
                    mismatches.append(arm)
            audits.append(
                {
                    "case": case.name,
                    "seed": seed,
                    "arm_count": len(by_arm),
                    "expected_arm_count": len(ARMS),
                    "reference_arm": reference_row.get("arm"),
                    "input_and_init_hashes_match": bool(reference) and not mismatches,
                    "mismatched_arms": mismatches,
                    "failed_arms": failures,
                    "coefficient_mapping_valid": all(
                        row.get("coefficient_mapping", {}).get(
                            "row_count_matches_selection", False
                        )
                        for row in valid
                    ),
                }
            )

    frozen = assert_frozen()
    snapshot = ROOT / frozen["source_snapshot"]
    snapshot_matches = hashlib.sha256(snapshot.read_bytes()).hexdigest() == frozen.get(
        "source_snapshot_sha256"
    )
    if not snapshot_matches:
        raise RuntimeError("source snapshot hash mismatch")
    return {
        "source_hashes_match_freeze": True,
        "protocol_hash_matches_freeze": True,
        "source_snapshot_hash_matches": snapshot_matches,
        "freeze_manifest_sha256": hashlib.sha256(
            (OUT / "results" / "freeze_manifest.json").read_bytes()
        ).hexdigest(),
        "case_seed_pairs": audits,
        "all_input_and_init_hashes_match": all(
            item["input_and_init_hashes_match"] for item in audits
        ),
    }


def write_arm_summary(rows: list[dict[str, Any]], destination: Path) -> None:
    columns = (
        "case",
        "arm",
        "n_rows",
        "n_ok",
        "n_all_inner_converged",
        "median_edr_error",
        "median_fit_wall_sec",
        "median_initialization_time_sec",
        "median_bandwidth_time_sec",
        "median_directions_time_sec",
        "median_statistics_time_sec",
        "median_solver_time_sec",
        "median_update_time_sec",
        "median_weight_time_sec",
        "median_projected_time_sec",
        "median_selection_time_sec",
        "median_assembly_time_sec",
        "median_peak_rss_kib",
        "median_outer_iterations",
        "median_active_centers",
        "median_mass_coverage",
        "median_candidate_edges",
        "median_support_edges",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("case")), str(row.get("arm")))].append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for (case, arm), group in sorted(groups.items()):
            ok = [row for row in group if row.get("status") == "ok"]
            last_diagnostics = [
                row.get("trace", [{}])[-1].get("experimental_centers", {}) for row in ok
            ]
            active_centers = [
                float(diag["selected_centers"])
                if diag.get("selected_centers") is not None
                else float(row.get("shape", {}).get("n", 500))
                for row, diag in zip(ok, last_diagnostics, strict=True)
            ]
            fields = [
                (
                    "median_candidate_edges",
                    "candidate_edges",
                ),
                ("median_support_edges", "support_edges"),
            ]
            row_out = {
                "case": case,
                "arm": arm,
                "n_rows": len(group),
                "n_ok": len(ok),
                "n_all_inner_converged": sum(
                    bool(row.get("inner_certificates", {}).get("all_converged"))
                    for row in ok
                ),
                "median_edr_error": _median([float(row["edr_error"]) for row in ok]),
                "median_fit_wall_sec": _median(
                    [float(row["fit_wall_sec"]) for row in ok]
                ),
                "median_peak_rss_kib": _median(
                    [float(row["peak_rss_kib"]) for row in ok]
                ),
                "median_outer_iterations": _median(
                    [float(row["outer_iterations"]) for row in ok]
                ),
                "median_active_centers": _median(active_centers),
                "median_mass_coverage": _median(
                    [float(diag.get("mass_coverage", 1.0)) for diag in last_diagnostics]
                ),
            }
            for stage in (
                "initialization",
                "bandwidth",
                "directions",
                "statistics",
                "solver",
                "update",
            ):
                row_out[f"median_{stage}_time_sec"] = _median(
                    [
                        float(row["profile"]["stages"][stage]["time_seconds"])
                        for row in ok
                        if row.get("profile", {})
                        .get("stages", {})
                        .get(stage, {})
                        .get("time_seconds")
                        is not None
                    ]
                )
            for stage in (
                "weight",
                "projected",
                "selection",
                "assembly",
            ):
                values = []
                for row in ok:
                    seconds = sum(
                        float(
                            entry.get("experimental_centers", {}).get(
                                f"{stage}_time_seconds", 0.0
                            )
                        )
                        for entry in row.get("trace", [])
                    )
                    values.append(seconds)
                row_out[f"median_{stage}_time_sec"] = _median(values)
            for output_key, diag_key in fields:
                row_out[output_key] = _median(
                    [
                        float(diag[diag_key])
                        for diag in last_diagnostics
                        if diag.get(diag_key) is not None
                    ]
                )
            writer.writerow(row_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=OUT / "results" / "fullfit_dev.jsonl"
    )
    args = parser.parse_args()
    rows = read_rows(args.input)
    write_json(OUT / "results" / "pairing_audit.json", verify_pairing(rows))
    write_arm_summary(rows, OUT / "results" / "fullfit_dev_summary.csv")
    write_pairwise(rows, OUT / "results" / "fullfit_dev_pairs.csv")


if __name__ == "__main__":
    main()
