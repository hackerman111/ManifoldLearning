"""Analyze one frozen paired Grassmann run; only write its JSON summary."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BASELINE = "ADP-Grassmann"
HASH_FIELDS = ("data_sha256", "x_csv_sha256", "y_csv_sha256", "method_seed")
LIMITS = {
    "edr_geomean": 0.95,
    "case_edr_delta": 0.005,
    "time_ratio_median": 1.10,
    "time_ratio_p90": 1.25,
    "rss_ratio": 1.10,
    "orthogonality": 1e-10,
}


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def bootstrap(values: list[float], statistic: str) -> list[float | None]:
    if len(values) < 2:
        return [None, None]
    x = np.asarray(values, dtype=float)
    idx = np.random.default_rng(908123).integers(0, len(x), (10_000, len(x)))
    samples = x[idx]
    if statistic == "geomean":
        estimates = np.exp(np.mean(np.log(samples), axis=1))
    else:
        estimates = np.median(samples, axis=1)
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def analyze(
    directory: Path,
    max_edr_ratio: float | None = None,
    *,
    write: bool = True,
    candidate: str | None = None,
    baseline: str = BASELINE,
) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = manifest["protocol"]
    methods = protocol["methods"]
    candidates = [m for m in methods if m.startswith("ADP-Grassmann-")]
    if baseline not in methods or (candidate is None and len(candidates) != 1):
        raise ValueError(
            f"need baseline and exactly one Grassmann candidate: {methods}"
        )
    candidate = candidates[0] if candidate is None else candidate
    if candidate not in methods or candidate == baseline:
        raise ValueError("candidate must name a different recorded method")
    with (directory / "runs.csv").open(newline="") as stream:
        rows: list[dict[str, Any]] = list(csv.DictReader(stream))
    numeric = (
        "seed",
        "fit_seconds",
        "peak_rss_mib",
        "projector_distance",
        "orthogonality_error",
    )
    for row in rows:
        for field in numeric:
            if row.get(field):
                row[field] = float(row[field])

    runs = int(protocol["runs"])
    seeds = range(int(protocol["seed"]), int(protocol["seed"]) + runs)
    cases = [case["name"] for case in protocol["cases"]]
    blocks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    lookup = {}
    for row in rows:
        key = (row["case"], int(row["seed"]))
        blocks[key].append(row)
        lookup[(*key, row["method"])] = row
    bad_blocks, input_hashes = [], []
    for case in cases:
        bad_blocks.extend(
            {"case": case, "seed": seed, "reason": "missing block"}
            for seed in seeds
            if (case, seed) not in blocks
        )
    for (case, seed), block in sorted(blocks.items()):
        if {r["method"] for r in block} != set(methods) or len(block) != len(methods):
            bad_blocks.append({"case": case, "seed": seed, "reason": "methods"})
        hashes = {
            field: sorted({str(r.get(field, "")) for r in block})
            for field in HASH_FIELDS
        }
        for field, values in hashes.items():
            if len(values) != 1 or values[0] in ("", "None"):
                bad_blocks.append({"case": case, "seed": seed, "reason": field})
        input_hashes.append(
            {
                "case": case,
                "seed": seed,
                **{k: v[0] if len(v) == 1 else v for k, v in hashes.items()},
            }
        )

    per_case, all_edr_ratio, all_time_ratio = {}, [], []
    for case in cases:
        pairs = []
        for seed in seeds:
            base = lookup.get((case, seed, baseline))
            cand = lookup.get((case, seed, candidate))
            if base and cand and base["status"] == cand["status"] == "ok":
                pairs.append(
                    (
                        cand["projector_distance"] / base["projector_distance"],
                        cand["projector_distance"] - base["projector_distance"],
                        cand["fit_seconds"] / base["fit_seconds"],
                    )
                )
        er = [p[0] for p in pairs if p[0] > 0]
        delta = [p[1] for p in pairs]
        tr = [p[2] for p in pairs]
        all_edr_ratio.extend(er)
        all_time_ratio.extend(tr)
        method_stats = {}
        for method in methods:
            rr = [r for r in rows if r["case"] == case and r["method"] == method]
            ok = [r for r in rr if r["status"] == "ok"]
            fit = [r["fit_seconds"] for r in ok if r.get("fit_seconds") is not None]
            rss = [r["peak_rss_mib"] for r in ok if r.get("peak_rss_mib") is not None]
            edr = [
                r["projector_distance"]
                for r in ok
                if r.get("projector_distance") is not None
            ]
            metas = []
            for row in ok:
                if row.get("meta"):
                    with contextlib.suppress(OSError, json.JSONDecodeError):
                        metas.append(json.loads((directory / row["meta"]).read_text()))
            calls = [d for m in metas for d in m.get("inner_diagnostics", [])]
            method_stats[method] = {
                "n_ok": len(ok),
                "failures": [
                    {
                        k: r.get(k)
                        for k in ("seed", "status", "returncode", "detail", "log")
                    }
                    for r in rr
                    if r["status"] != "ok"
                ],
                "edr_median": median(edr),
                "fit_seconds_median_p90": [median(fit), quantile(fit, 0.9)],
                "rss_mib_median_p90": [median(rss), quantile(rss, 0.9)],
                "orthogonality_max": max(
                    (
                        r["orthogonality_error"]
                        for r in ok
                        if r.get("orthogonality_error") is not None
                    ),
                    default=None,
                ),
                "stop_reasons": dict(Counter(r.get("stop_reason", "") for r in ok)),
                "inner_calls": len(calls),
                "accepted_steps": sum(d.get("accepted_steps", 0) for d in calls),
                "rank_guard_fallbacks": sum(
                    d.get("rank_guard_fallbacks", 0) for d in calls
                ),
                "gn_solves": sum(d.get("gn_solves", 0) for d in calls),
                "local_rank_loss": sum(d.get("local_rank_loss", 0) for d in calls),
                "stationarity_applicable_calls": sum(
                    bool(d.get("stationarity_applicable")) for d in calls
                ),
                "inner_certified_calls": sum(bool(d.get("converged")) for d in calls),
            }
        geo = math.exp(float(np.mean(np.log(er)))) if er else None
        per_case[case] = {
            "planned_pairs": runs,
            "joint_success_pairs": len(pairs),
            "edr_ratio_geomean": geo,
            "edr_ratio_bootstrap95": bootstrap(er, "geomean"),
            "edr_delta_median": median(delta),
            "edr_delta_bootstrap95": bootstrap(delta, "median"),
            "time_ratio_median_p90": [median(tr), quantile(tr, 0.9)],
            "time_ratio_bootstrap95": bootstrap(tr, "median"),
            "methods": method_stats,
        }

    failures = [
        {
            k: r.get(k)
            for k in ("case", "seed", "method", "status", "returncode", "detail", "log")
        }
        for r in rows
        if r["status"] != "ok"
    ]
    base_rss = [
        r["peak_rss_mib"]
        for r in rows
        if r["method"] == baseline and r["status"] == "ok"
    ]
    cand_rss = [
        r["peak_rss_mib"]
        for r in rows
        if r["method"] == candidate and r["status"] == "ok"
    ]
    orth = [
        r["orthogonality_error"]
        for r in rows
        if r["method"] == candidate and r["status"] == "ok"
    ]
    edr_geo = math.exp(float(np.mean(np.log(all_edr_ratio)))) if all_edr_ratio else None
    tr_med, tr_p90 = median(all_time_ratio), quantile(all_time_ratio, 0.9)
    cand_median, base_median = median(cand_rss), median(base_rss)
    rss_ratio = (
        cand_median / base_median
        if cand_median is not None and base_median is not None
        else None
    )
    summary = {
        "run": str(directory),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "source_hashes": protocol.get("sources", {}),
        "baseline": baseline,
        "candidate": candidate,
        "planned_fits": protocol["planned_fits"],
        "recorded_fits": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "raw_errors": failures,
        "paired_hash_audit": {
            "planned_blocks": len(cases) * runs,
            "observed_blocks": len(blocks),
            "bad_blocks": bad_blocks,
            "input_hashes": input_hashes,
        },
        "thresholds": dict(LIMITS),
        "overall": {
            "joint_success_pairs": len(all_edr_ratio),
            "edr_ratio_geomean": edr_geo,
            "edr_ratio_bootstrap95": bootstrap(all_edr_ratio, "geomean"),
            "time_ratio_median_p90": [tr_med, tr_p90],
            "time_ratio_bootstrap95": bootstrap(all_time_ratio, "median"),
            "median_rss_ratio": rss_ratio,
            "candidate_rss_median": median(cand_rss),
            "baseline_rss_median": median(base_rss),
            "candidate_orthogonality_max": max(orth, default=None),
        },
        "gates": {
            "complete_successful_pairs": len(all_edr_ratio) == len(cases) * runs
            and not failures
            and not bad_blocks,
            "edr_geomean": edr_geo is not None and edr_geo <= LIMITS["edr_geomean"],
            "case_median_edr_delta": all(
                v["edr_delta_median"] is not None
                and v["edr_delta_median"] <= LIMITS["case_edr_delta"]
                for v in per_case.values()
            ),
            "time_ratio_median": tr_med is not None
            and tr_med <= LIMITS["time_ratio_median"],
            "time_ratio_p90": tr_p90 is not None and tr_p90 <= LIMITS["time_ratio_p90"],
            "median_rss_ratio": rss_ratio is not None
            and rss_ratio <= LIMITS["rss_ratio"],
            "finite_and_orthogonal": bool(orth)
            and all(math.isfinite(v) for v in orth)
            and max(orth) <= LIMITS["orthogonality"],
        },
        "per_case": per_case,
    }
    if max_edr_ratio is not None:
        observed = max(all_edr_ratio, default=None)
        summary["thresholds"]["max_individual_edr_ratio"] = max_edr_ratio
        summary["overall"]["max_individual_edr_ratio"] = observed
        summary["gates"]["max_individual_edr_ratio"] = (
            observed is not None and observed <= max_edr_ratio
        )
    if write:
        (directory / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-edr-ratio", type=float)
    args = parser.parse_args()
    result = analyze(args.directory.resolve(), args.max_edr_ratio)
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "overall": result["overall"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
