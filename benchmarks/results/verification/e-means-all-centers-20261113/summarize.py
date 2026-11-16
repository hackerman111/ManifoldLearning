"""Descriptive read-back after frozen campaign; no selection/gate changes."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "campaign"
protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
completion = json.loads((ROOT / "completion.json").read_text())
audit = json.loads((ROOT / "selection/audit.json").read_text())
assert completion["execution_complete"] and audit["pass"]
rows = [json.loads(s) for s in (ROOT / "selection/runs.jsonl").read_text().splitlines()]
keys = {(r["case"], r["seed"], r["method"]) for r in rows}
expected = {
    (c["name"], seed, method)
    for c in protocol["cases"]
    for seed in protocol["selection_seeds"]
    for method in protocol["methods"]
}
assert keys == expected and len(rows) == len(keys)
assert all(r["status"] == "ok" for r in rows), "Keep failures explicit in report"
lookup = {(r["case"], r["seed"], r["method"]): r for r in rows}


def contrast(base, target):
    pairs = [
        (lookup[c["name"], seed, base], lookup[c["name"], seed, target])
        for c in protocol["cases"]
        for seed in protocol["selection_seeds"]
    ]
    delta = np.array([b["edr"] - a["edr"] for a, b in pairs])
    worst = pairs[int(np.argmax(delta))]
    return {
        "base": base,
        "target": target,
        "pairs": len(pairs),
        "mean_edr_delta": float(delta.mean()),
        "worst_edr_delta": float(delta.max()),
        "best_edr_delta": float(delta.min()),
        "median_time_ratio": float(
            np.median([b["fit_seconds"] / a["fit_seconds"] for a, b in pairs])
        ),
        "median_rss_ratio": float(
            np.median([b["peak_rss_mib"] / a["peak_rss_mib"] for a, b in pairs])
        ),
        "edr_better": int(np.count_nonzero(delta < 0)),
        "edr_worse": int(np.count_nonzero(delta > 0)),
        "worst_block": {k: worst[0][k] for k in ("case", "seed")},
        "worst_base_edr": worst[0]["edr"],
        "worst_target_edr": worst[1]["edr"],
    }


summary, by_case = {}, []
for method in protocol["methods"]:
    selected = [r for r in rows if r["method"] == method]
    diagnostics, stages, actual_j = [], [], set()
    for row in selected:
        fit = ROOT / "selection" / f"{row['case']}-{row['seed']}" / method
        trace = json.loads((fit / "trace.json").read_text())
        actual_j.add(trace["N_J"])
        diagnostics.extend(t["solver"] for t in trace["trace"])
        stages.append(json.loads((fit / "profile.json").read_text())["stages"])
    summary[method] = {
        "valid": len(selected),
        "mean_edr": float(np.mean([r["edr"] for r in selected])),
        "median_edr": float(np.median([r["edr"] for r in selected])),
        "recovery_le_0.3": sum(r["edr"] <= 0.3 for r in selected),
        "median_fit_seconds": float(np.median([r["fit_seconds"] for r in selected])),
        "median_rss_mib": float(np.median([r["peak_rss_mib"] for r in selected])),
        "max_rss_mib": max(r["peak_rss_mib"] for r in selected),
        "inner_calls": len(diagnostics),
        "inner_certified": sum(d["converged"] is True for d in diagnostics),
        "stationarity_applicable": sum(
            d["stationarity_applicable"] is True for d in diagnostics
        ),
        "starts": dict(
            Counter(d.get("selected_start", "original") for d in diagnostics)
        ),
        "finalization_rejected": sum(
            d.get("finalization_rejected", False) for d in diagnostics
        ),
        "full_or_finalization_workspace_fallbacks": sum(
            int(d.get("full_finalization", d).get("workspace_fallbacks", 0))
            for d in diagnostics
        ),
        "outer_stop_reasons": dict(Counter(r["stop_reason"] for r in selected)),
        "actual_N_J": sorted(actual_j),
        "median_stage_time_fractions": {
            stage: float(np.median([p[stage]["time_fraction"] for p in stages]))
            for stage in stages[0]
        },
    }
for case in protocol["cases"]:
    out = {"case": case["name"], "n": case["n"], "d": case["d"]}
    for method in protocol["methods"]:
        selected = [
            r for r in rows if r["case"] == case["name"] and r["method"] == method
        ]
        out[f"{method}_mean_edr"] = float(np.mean([r["edr"] for r in selected]))
        out[f"{method}_median_fit_seconds"] = float(
            np.median([r["fit_seconds"] for r in selected])
        )
    by_case.append(out)
comparisons = {
    name: contrast(base, target)
    for name, base, target in (
        ("all_vs_bounded", "original20_bounded", "original20"),
        ("mini_vs_all", "original20", "mini"),
        ("ema_vs_all", "original20", "mini_ema"),
        ("ema_vs_mini", "mini", "mini_ema"),
        ("ema_vs_full6", "full6", "mini_ema"),
    )
}
result = {
    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "summary": summary,
    "comparisons": comparisons,
    "by_case": by_case,
    "completion": completion,
    "max_orthogonality_error": max(r["orthogonality_error"] for r in rows),
    "max_computed_profile_increase": max(r["full_guard_max_increase"] for r in rows),
}
(HERE / "analysis.json").write_text(
    json.dumps(result, indent=2, allow_nan=False) + "\n"
)
with (HERE / "by-case.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(by_case[0]))
    writer.writeheader()
    writer.writerows(by_case)
print(
    json.dumps(
        {k: result[k] for k in ("summary", "comparisons", "completion")}, indent=2
    )
)
