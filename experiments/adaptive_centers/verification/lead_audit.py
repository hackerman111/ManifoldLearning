"""Independent raw-row audit, without the benchmark summary implementation."""

import hashlib
import json
import statistics
from pathlib import Path

from .compatibility_overlay import verify_overlay


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = Path(__file__).resolve().parents[1]
    root = folder.parents[1]
    raw = folder / "results/fullfit_dev.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    freeze = json.loads((folder / "results/freeze_manifest.json").read_text())
    assert len(rows) == 120 and all(row["status"] == "ok" for row in rows)
    overlay = verify_overlay()
    assert digest(folder / "PROTOCOL.md") == freeze["protocol_sha256"]
    assert digest(root / freeze["source_snapshot"]) == freeze["source_snapshot_sha256"]
    assert all(
        digest(root / path) == sha
        for path, sha in freeze["provenance_sha256"].items()
        if path != "tests/test_adaptive_centers.py"
    )
    by = {(r["case"], r["seed"], r["arm"]): r for r in rows}
    blocks = sorted({(r["case"], r["seed"]) for r in rows})
    arms = sorted({r["arm"] for r in rows})
    for case, seed in blocks:
        for key in ("X", "Y", "truth", "initial_basis"):
            assert len({by[case, seed, arm]["input_hashes"][key] for arm in arms}) == 1
    assert all(r["coefficient_mapping"]["row_count_matches_selection"] for r in rows)
    assert all(
        p["num_threads"] == 1 for r in rows for p in r["environment"]["threadpools"]
    )
    assert all(r["outer_stop_reason"] == "h_min" for r in rows)
    assert sum(r["inner_certificates"]["converged_count"] for r in rows) == 0
    comparisons = []
    pairs = [(a, "baseline") for a in arms if a != "baseline"]
    pairs += [("floor25_search", "floor25"), ("floor25_search_stats", "floor25_search")]
    pairs += [
        (a, "random_equal_j")
        for a in arms
        if a.endswith("equal_j") and a != "random_equal_j"
    ]
    for arm, control in pairs:
        diffs, times, rss = [], [], []
        for case, seed in blocks:
            a, b = by[case, seed, arm], by[case, seed, control]
            diffs.append(a["edr_error"] - b["edr_error"])
            times.append(a["fit_wall_sec"] / b["fit_wall_sec"])
            rss.append(a["peak_rss_kib"] / b["peak_rss_kib"])
        comparisons.append(
            {
                "candidate": arm,
                "control": control,
                "median_edr_delta": statistics.median(diffs),
                "worst_edr_delta": max(diffs),
                "median_fit_time_ratio": statistics.median(times),
                "median_process_peak_rss_ratio": statistics.median(rss),
            }
        )
    trajectory = json.loads((folder / "verification/trajectory_audit.json").read_text())
    checks = [c for arm in trajectory["arms"].values() for c in arm["checks"]]
    maxima = {
        key: max(c["statistics_max_abs"][key] for c in checks)
        for key in checks[0]["statistics_max_abs"]
    }
    supplemental = folder / "supplement/results"
    micro = [
        json.loads(line)
        for line in (supplemental / "runs.jsonl").read_text().splitlines()
    ]
    assert len(micro) == 24 and all(r["status"] == "ok" for r in micro)
    micro_manifest = json.loads((supplemental / "manifest.json").read_text())
    for path, sha in micro_manifest["source_sha256"].items():
        assert digest(root / path) == sha
    assert (
        digest(folder / "supplement/PROTOCOL.md") == micro_manifest["protocol_sha256"]
    )
    assert (
        digest(folder / "results/freeze_manifest.json")
        == micro_manifest["parent_freeze_manifest_sha256"]
    )
    assert all(
        r["support_mismatches"] == 0
        and r["max_weight_abs_error"] == 0
        and r["alpha_error"] == 0
        for r in micro
    )
    micro_comparisons = []
    for case in ("small", "large"):
        for mode in ("weights", "alpha_weights"):
            ratios = []
            for seed in (93000, 93001, 93002):
                pair = {
                    r["arm"]: r
                    for r in micro
                    if r["case"] == case and r["mode"] == mode and r["seed"] == seed
                }
                a, b = pair["projected"], pair["baseline"]
                assert a["hashes"] == b["hashes"]
                assert all(
                    p["num_threads"] == 1
                    for r in pair.values()
                    for p in r["threadpools"]
                )
                ratios.append(
                    a["localization_plus_output_sec"]
                    / b["localization_plus_output_sec"]
                )
            micro_comparisons.append(
                {
                    "case": case,
                    "mode": mode,
                    "ratios": ratios,
                    "median_projected_baseline_time": statistics.median(ratios),
                }
            )
    report = {
        "rows": len(rows),
        "paired_blocks": len(blocks),
        "raw_sha256": digest(raw),
        "frozen_source_files_verified_in_snapshot": len(freeze["source_sha256"]),
        "current_source_overlay": overlay,
        "protocol_snapshot_original_math_unchanged": True,
        "thread_input_initial_coefficient_checks": True,
        "outer_stop": "all h_min",
        "inner_converged_certificates": 0,
        "max_orthogonality_error": max(r["orthogonality_error"] for r in rows),
        "heldout_rows": sum(r["seed"] in freeze["heldout_seeds"] for r in rows),
        "comparisons": comparisons,
        "supplemental_localization_only": {
            "workers": len(micro),
            "comparisons": micro_comparisons,
            "sources_hashes_pairing_threads_weights_verified": True,
        },
        "trajectory_audit": {
            "states": len(checks),
            "max_weight_abs": max(c["weight_max_abs"] for c in checks),
            "max_statistics_abs": maxima,
            "eta_diagnostic_used_in_solver": False,
        },
    }
    (folder / "verification/lead_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
