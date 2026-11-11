"""Audit completeness and regenerate paired pilot summary."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
summary = {}
for phase, expected in (("selection", 96), ("validation", 24)):
    rows = [
        json.loads(line) for line in (OUT / (phase + ".jsonl")).read_text().splitlines()
    ]
    assert len(rows) == expected
    keys = [(r["kind"], r["case"], r["target"], r["seed"], r["variant"]) for r in rows]
    assert len(set(keys)) == expected
    groups = []
    for kind in ("fit", "fixed"):
        cases = sorted({r["case"] for r in rows if r["kind"] == kind})
        for target in ("matrix", "correction"):
            for case in [*cases, "all"] if cases else []:
                for variant in ("sqrt", "tensor", "floor"):
                    candidates = [
                        r
                        for r in rows
                        if r["kind"] == kind
                        and r["target"] == target
                        and (r["case"] == case or case == "all")
                        and r["variant"] == variant
                    ]
                    if not candidates:
                        continue
                    differences, times, memory, inner, krylov = [], [], [], [], []
                    for row in candidates:
                        baseline = next(
                            r
                            for r in rows
                            if r["kind"] == kind
                            and r["case"] == row["case"]
                            and r["target"] == target
                            and r["seed"] == row["seed"]
                            and r["variant"] == "frobenius"
                        )
                        assert row["status"] == baseline["status"] == "ok"
                        assert row["data_sha256"] == baseline["data_sha256"]
                        metric = (
                            "projector_distance"
                            if kind == "fit"
                            else "coefficient_error"
                        )
                        differences.append(row[metric] - baseline[metric])
                        times.append(row["seconds"] / baseline["seconds"])
                        memory.append(row["peak_rss_kib"] - baseline["peak_rss_kib"])
                        inner.append(
                            row["inner_iterations"] / baseline["inner_iterations"]
                        )
                        if baseline["lsmr_iterations"]:
                            krylov.append(
                                row["lsmr_iterations"] / baseline["lsmr_iterations"]
                            )
                    groups.append(
                        {
                            "kind": kind,
                            "case": case,
                            "target": target,
                            "variant": variant,
                            "pairs": len(candidates),
                            "median_quality_delta": statistics.median(differences),
                            "max_worsening": max(differences),
                            "median_time_ratio": statistics.median(times),
                            "median_rss_delta_kib": statistics.median(memory),
                            "median_inner_iteration_ratio": statistics.median(inner),
                            "median_lsmr_iteration_ratio": statistics.median(krylov)
                            if krylov
                            else None,
                            "inner_converged_count": sum(
                                r["inner_converged"] for r in candidates
                            ),
                            "gate_pass": kind == "fit"
                            and max(differences) <= 0.02
                            and statistics.median(differences) < 0,
                        }
                    )
    summary[phase] = {
        "rows": len(rows),
        "errors": sum(r["status"] != "ok" for r in rows),
        "inner_converged_count": sum(r["inner_converged"] for r in rows),
        "groups": groups,
    }
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for phase, data in summary.items():
    print(phase, data["rows"], "rows", data["errors"], "errors")
    for g in data["groups"]:
        if g["kind"] == "fit" and g["case"] == "all":
            print(g)
