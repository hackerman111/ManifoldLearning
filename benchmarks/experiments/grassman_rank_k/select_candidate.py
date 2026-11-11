"""Apply predeclared development-only rules and freeze one validation arm."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter

from ._analysis_io import _number
from .common import OUT, _assert_frozen
from .lead_audit import audit


def _metrics(pairs):
    def median(values):
        finite = [v for v in values if _number(v) is not None]
        return statistics.median(finite) if finite else None

    ratios = [
        c["wall_sec"] / b["wall_sec"]
        for c, b in pairs
        if c.get("completed")
        and b.get("completed")
        and _number(c.get("wall_sec")) is not None
        and _number(b.get("wall_sec")) is not None
        and b["wall_sec"] > 0
    ]
    return {
        "median_edr": median(c.get("projector_error") for c, _ in pairs),
        "core_median_edr": median(b.get("projector_error") for _, b in pairs),
        "median_paired_fit_ratio": median(ratios),
        "n_wall_pairs": len(ratios),
        "candidate_failures": sum(not c.get("completed") for c, _ in pairs),
        "baseline_failures": sum(not b.get("completed") for _, b in pairs),
    }


def main() -> None:
    _assert_frozen()
    evidence = audit()
    assert evidence["stages"]["fullfit_dev"]["count"] == 54
    path = OUT / "results" / "fullfit_dev.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert Counter(r["arm"] for r in rows) == Counter(
        dict.fromkeys(
            (
                "core",
                "rank_one",
                "spectral",
                "flat",
                "block1",
                "block_full",
                "block_energy",
                "block_cost",
                "block_spectral",
            ),
            6,
        )
    )
    assert not (OUT / "results" / "fullfit_heldout.jsonl").exists()
    destination = OUT / "results" / "selection.json"
    if destination.exists():
        raise FileExistsError(destination)
    baseline = {(r["scenario"], r["seed"]): r for r in rows if r["arm"] == "core"}
    assert len(baseline) == 6
    scores = []
    for arm in ("block1", "block_full", "block_energy", "block_cost"):
        candidates = [r for r in rows if r["arm"] == arm]
        pairs = [(r, baseline[(r["scenario"], r["seed"])]) for r in candidates]
        completed = all(r.get("finite") and r.get("completed") for r in candidates)
        metrics = _metrics(pairs)
        median_edr, core_edr, ratio = (
            metrics[k]
            for k in ("median_edr", "core_median_edr", "median_paired_fit_ratio")
        )
        eligible = bool(
            completed
            and median_edr is not None
            and core_edr is not None
            and ratio is not None
            and metrics["n_wall_pairs"] == 6
            and median_edr <= core_edr
        )
        scores.append(
            {
                "arm": arm,
                "all_completed_finite": completed,
                **metrics,
                "eligible": eligible,
                "passes_development_20_percent_gate": eligible
                and ratio is not None
                and ratio <= 0.8,
                "per_scenario": {
                    scenario: _metrics(
                        [(c, b) for c, b in pairs if c["scenario"] == scenario]
                    )
                    for scenario in sorted({r["scenario"] for r in candidates})
                },
            }
        )
    eligible = [r for r in scores if r["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda r: (r["median_paired_fit_ratio"], r["median_edr"], r["arm"]),
        )
        if eligible
        else min(
            scores,
            key=lambda r: (
                r["median_edr"] if r["median_edr"] is not None else float("inf"),
                r["median_paired_fit_ratio"]
                if r["median_paired_fit_ratio"] is not None
                else float("inf"),
                r["arm"],
            ),
        )
    )
    result = {
        "selected_arm": winner["arm"],
        "development_runs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_rules_sha256": hashlib.sha256(
            (OUT / "SELECTION_RULES.md").read_bytes()
        ).hexdigest(),
        "selection_scope": "development only; heldout not yet run",
        "scores": scores,
        "selection_is_positive_result": winner["passes_development_20_percent_gate"],
        "rule": (
            "Eligible: finite/completed and pooled median EDR<=core; minimize "
            "median paired fit ratio. If none eligible: minimize EDR."
        ),
    }
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
