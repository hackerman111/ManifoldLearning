"""Integrity checks and paired analysis for frozen Grassmann pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmarks/results/outputs/r_sdr/grassman-lowrank22-pilot"
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261022, 10_000


def geo(values: list[float]) -> float | None:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        return None
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(statistics.mean(map(math.log, values)))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    point = q * (len(ordered) - 1)
    lo, hi = int(point), min(int(point) + 1, len(ordered) - 1)
    return ordered[lo] + (point - lo) * (ordered[hi] - ordered[lo])


def _finite(row: dict | None, field: str, *, positive: bool = False) -> bool:
    if not row or row.get("status") != "ok":
        return False
    value = row.get(field)
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and (not positive or value > 0)
    )


def _paired_metrics(
    candidate: dict | None, base: dict | None, actual_r: dict | None = None
):
    fields = (
        (candidate, "projector_distance", False),
        (base, "projector_distance", True),
        (candidate, "fit_seconds", True),
        (base, "fit_seconds", True),
        (candidate, "peak_rss_mib", True),
        (base, "peak_rss_mib", True),
        (candidate, "orthogonality_error", False),
    )
    if actual_r is not None:
        fields += (
            (actual_r, "projector_distance", True),
            (actual_r, "fit_seconds", True),
        )
    if not all(_finite(row, key, positive=positive) for row, key, positive in fields):
        return None
    assert candidate is not None and base is not None
    return (
        candidate["projector_distance"] / base["projector_distance"],
        candidate["fit_seconds"] / base["fit_seconds"],
        candidate["peak_rss_mib"] / base["peak_rss_mib"],
        candidate["projector_distance"] - base["projector_distance"],
        candidate.get("orthogonality_error"),
        candidate["projector_distance"] / actual_r["projector_distance"]
        if actual_r is not None
        else None,
    )


def main(output: Path = OUT) -> dict:
    rows = [
        json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()
    ]
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    cases = [case["name"] for case in protocol["cases"]]
    methods = tuple(protocol["methods"])
    seeds = list(range(protocol["seed"], protocol["seed"] + protocol["runs"]))
    if not cases or methods[0] != "baseline" or methods[-1] != "MEANMAVE":
        raise ValueError("expected cases and baseline/candidate/MEANMAVE method order")
    with zipfile.ZipFile(output / "sources.zip") as archive:
        for rel, expected in protocol["sources"].items():
            if hashlib.sha256(archive.read(rel)).hexdigest() != expected:
                raise ValueError(f"frozen source hash mismatch: {rel}")

    by, paired, failures = {}, {}, []
    outer, inner, fallback = Counter(), Counter(), Counter()
    for row in rows:
        key = row["case"], row["seed"], row["method"]
        if (
            key in by
            or row["case"] not in cases
            or row["method"] not in methods
            or row["seed"] not in seeds
        ):
            raise ValueError(f"duplicate or foreign result: {key}")
        by[key] = row
        pairkey = row["case"], row["seed"]
        hashes = {k: row["input_hashes"][k] for k in ("x", "y", "truth")}
        pair = hashes, row["method_seed"]
        if pairkey in paired and paired[pairkey] != pair:
            raise ValueError(f"input/method-seed mismatch: {pairkey}")
        paired[pairkey] = pair
        if row.get("status") != "ok":
            failures.append(
                {
                    k: row.get(k)
                    for k in (
                        "case",
                        "seed",
                        "method",
                        "status",
                        "returncode",
                        "error",
                        "log",
                    )
                }
            )
        outer[(row["method"], row.get("stop_reason"))] += 1
        for diag in row.get("inner_diagnostics", []):
            inner[(row["method"], diag.get("stop_reason"))] += 1
            for name in (
                "rank_guard_fallbacks",
                "workspace_fallbacks",
                "damping_scale_fallbacks",
            ):
                fallback[(row["method"], name)] += diag.get(name, 0)

    planned = len(cases) * len(seeds) * len(methods)
    expected_keys = {(c, s, m) for c in cases for s in seeds for m in methods}
    missing = sorted(expected_keys - by.keys())
    if len(rows) != len(by):
        raise ValueError("duplicate result rows")
    integrity = protocol["sources"].get("ADP/solver/grassman.py") == protocol[
        "sources"
    ].get("benchmarks/experiments/grassman_accuracy_2026_10_22/reference.py")
    results = {
        "rows": len(rows),
        "planned_rows": planned,
        "missing_rows": missing,
        "unique_input_blocks": len(paired),
        "failures": failures,
        "integrity": {
            "source_archive_verified": True,
            "baseline_matches_frozen_reference": integrity,
            "input_hashes_and_method_seeds_paired": True,
        },
        "outer_stops": {f"{m}:{s}": n for (m, s), n in sorted(outer.items())},
        "inner_stops": {f"{m}:{s}": n for (m, s), n in sorted(inner.items())},
        "fallbacks": {f"{m}:{k}": n for (m, k), n in sorted(fallback.items()) if n},
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "unit": "seed blocks; each draw retains all cases",
            "interval": "two-sided percentile 2.5%, 97.5%",
        },
        "methods": {},
    }

    for method in methods[1:-1]:
        case_results, valid_by_seed, all_valid = [], {s: [] for s in seeds}, True
        for case in cases:
            ratios, times, rss, deltas, angles = [], [], [], [], []
            for seed in seeds:
                c, b, r = (
                    by.get((case, seed, m)) for m in (method, "baseline", "MEANMAVE")
                )
                metrics = _paired_metrics(c, b, r)
                if metrics is None:
                    all_valid = False
                    continue
                ratio, tratio, rratio, delta, angle, rratio_actual = metrics
                ratios.append(ratio)
                times.append(tratio)
                rss.append(rratio)
                deltas.append(delta)
                angles.append(angle)
                valid_by_seed[seed].append(
                    (ratio, tratio, rratio, rratio_actual, delta)
                )
            case_results.append(
                {
                    "case": case,
                    "planned_pairs": len(seeds),
                    "valid_pairs": len(ratios),
                    "EDR_ratio_base_median": statistics.median(ratios)
                    if ratios
                    else None,
                    "EDR_delta_median": statistics.median(deltas) if deltas else None,
                    "max_EDR_increase": max(deltas, default=None),
                    "time_ratio_median": statistics.median(times) if times else None,
                    "time_ratio_p90": percentile(times, 0.9),
                    "RSS_ratio_median": statistics.median(rss) if rss else None,
                    "orthogonality_error_max": max(angles, default=None),
                }
            )

        complete_seeds = [s for s in seeds if len(valid_by_seed[s]) == len(cases)]
        pooled = [x for s in complete_seeds for x in valid_by_seed[s]]
        boot = []
        if complete_seeds:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            for indices in rng.integers(
                0, len(complete_seeds), size=(BOOTSTRAP_DRAWS, len(complete_seeds))
            ):
                sampled = [
                    x for i in indices for x in valid_by_seed[complete_seeds[int(i)]]
                ]
                boot.append(geo([x[0] for x in sampled]))
        relevant_methods = {method, "baseline", "MEANMAVE"}
        pair_failures = [f for f in failures if f["method"] in relevant_methods]
        pair_missing = [key for key in missing if key[2] in relevant_methods]
        finite_metrics = bool(pooled) and all_valid
        candidate_result = {
            "planned_pairs": len(cases) * len(seeds),
            "valid_pairs": len(pooled),
            "complete_seed_blocks": len(complete_seeds),
            "gate_complete": finite_metrics and not pair_missing and not pair_failures,
            "EDR_geomean_base": geo([x[0] for x in pooled]),
            "EDR_geomean_base_bootstrap95": [
                percentile(boot, 0.025),
                percentile(boot, 0.975),
            ]
            if boot
            else None,
            "EDR_geomean_R": geo([x[3] for x in pooled]),
            "fit_time_median_base": statistics.median([x[1] for x in pooled])
            if pooled
            else None,
            "fit_time_p90_base": percentile([x[1] for x in pooled], 0.9),
            "RSS_median_base": statistics.median([x[2] for x in pooled])
            if pooled
            else None,
            "max_orthogonality_error": max(
                (
                    c["orthogonality_error_max"]
                    for c in case_results
                    if c["orthogonality_error_max"] is not None
                ),
                default=None,
            ),
            "max_case_median_EDR_increase": max(
                (
                    c["EDR_delta_median"]
                    for c in case_results
                    if c["EDR_delta_median"] is not None
                ),
                default=None,
            ),
            "max_individual_EDR_increase": None,
            "relevant_failures": pair_failures,
            "relevant_missing_rows": pair_missing,
            "cases": case_results,
        }
        case_medians = [
            c["EDR_delta_median"]
            for c in case_results
            if c["EDR_delta_median"] is not None
        ]
        if finite_metrics:
            # The individual-increase gate is over all paired seed/case rows.
            candidate_result["max_individual_EDR_increase"] = max(
                x[4] for seed_rows in valid_by_seed.values() for x in seed_rows
            )
        edr_boot = candidate_result["EDR_geomean_base_bootstrap95"]
        candidate_result["gates"] = {
            "complete_pairs_and_no_failures": candidate_result["gate_complete"],
            "edr_geomean_le_0_98": candidate_result["EDR_geomean_base"] is not None
            and candidate_result["EDR_geomean_base"] <= 0.98,
            "edr_bootstrap_upper_lt_1": edr_boot is not None
            and edr_boot[1] is not None
            and edr_boot[1] < 1.0,
            "median_fit_time_le_1_10": candidate_result["fit_time_median_base"]
            is not None
            and candidate_result["fit_time_median_base"] <= 1.10,
            "p90_fit_time_le_1_25": candidate_result["fit_time_p90_base"] is not None
            and candidate_result["fit_time_p90_base"] <= 1.25,
            "median_rss_le_1_10": candidate_result["RSS_median_base"] is not None
            and candidate_result["RSS_median_base"] <= 1.10,
            "orthogonality_le_1e_10": candidate_result["max_orthogonality_error"]
            is not None
            and candidate_result["max_orthogonality_error"] <= 1e-10,
            "case_median_edr_increase_le_0_005": bool(case_medians)
            and max(case_medians) <= 0.005,
            "individual_edr_increase_le_0_05": candidate_result[
                "max_individual_EDR_increase"
            ]
            is not None
            and candidate_result["max_individual_EDR_increase"] <= 0.05,
        }
        candidate_result["all_gates_pass"] = all(candidate_result["gates"].values())
        results["methods"][method] = candidate_result

    if "budget10" in methods and "projected10" in methods:
        pair_rows, case_rows = [], []
        values_by_seed = {seed: [] for seed in seeds}
        complete = True
        for case in cases:
            ratios, times, rss = [], [], []
            for seed in seeds:
                projected = by.get((case, seed, "projected10"))
                budget = by.get((case, seed, "budget10"))
                metrics = _paired_metrics(projected, budget)
                if metrics is None:
                    complete = False
                    pair_rows.append(
                        {"case": case, "seed": seed, "status": "incomplete"}
                    )
                    continue
                ratio, time_ratio, rss_ratio, _, _, _ = metrics
                ratios.append(ratio)
                times.append(time_ratio)
                rss.append(rss_ratio)
                values_by_seed[seed].append((ratio, time_ratio, rss_ratio))
                pair_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "status": "ok",
                        "EDR_ratio_projected10_vs_budget10": ratio,
                        "fit_time_ratio_projected10_vs_budget10": time_ratio,
                        "RSS_ratio_projected10_vs_budget10": rss_ratio,
                    }
                )
            case_rows.append(
                {
                    "case": case,
                    "planned_pairs": len(seeds),
                    "valid_pairs": len(ratios),
                    "EDR_ratio_median": statistics.median(ratios) if ratios else None,
                    "fit_time_ratio_median": statistics.median(times)
                    if times
                    else None,
                    "fit_time_ratio_p90": percentile(times, 0.9),
                    "RSS_ratio_median": statistics.median(rss) if rss else None,
                }
            )
        full_seeds = [seed for seed in seeds if len(values_by_seed[seed]) == len(cases)]
        boot = []
        if full_seeds:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            for indices in rng.integers(
                0, len(full_seeds), size=(BOOTSTRAP_DRAWS, len(full_seeds))
            ):
                sampled = [
                    value
                    for i in indices
                    for value in values_by_seed[full_seeds[int(i)]]
                ]
                boot.append(geo([value[0] for value in sampled]))
        ratios = [value[0] for seed in full_seeds for value in values_by_seed[seed]]
        times = [value[1] for seed in full_seeds for value in values_by_seed[seed]]
        relevant = {"baseline", "budget10", "projected10", "MEANMAVE"}
        results["comparisons"] = {
            "projected10_vs_budget10": {
                "planned_pairs": len(cases) * len(seeds),
                "valid_pairs": len(ratios),
                "complete_seed_blocks": len(full_seeds),
                "gate_complete": complete
                and not any(key[2] in relevant for key in missing)
                and not any(failure["method"] in relevant for failure in failures),
                "EDR_geomean": geo(ratios),
                "EDR_geomean_bootstrap95": [
                    percentile(boot, 0.025),
                    percentile(boot, 0.975),
                ]
                if boot
                else None,
                "fit_time_median": statistics.median(times) if times else None,
                "fit_time_p90": percentile(times, 0.9),
                "cases": case_rows,
                "pairs": pair_rows,
            }
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(main(args.output), indent=2, sort_keys=True))
