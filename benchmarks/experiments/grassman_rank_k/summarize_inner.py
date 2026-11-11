"""Describe rank-k inner CSVs without making benchmark-level claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .summarize_cost import cost_summary


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _record(fields: tuple[str, ...], values: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(fields, values, strict=True))


def summarize(analysis: Path, output: Path) -> dict[str, Any]:
    summary, traces = (
        _read(analysis / "inner_summary.csv"),
        _read(analysis / "inner_trace.csv"),
    )
    equal = _read(analysis / "inner_equal_budgets.csv")
    sweeps, reversals = (
        _read(analysis / "rank_sweeps.csv"),
        _read(analysis / "rank_reversals.csv"),
    )
    regressions = _read(analysis / "lower_objective_worse_edr.csv")
    key_fields = ("m", "d", "J", "seed", "arm", "rank_k", "variant", "budget")
    keys = [tuple(row[field] for field in key_fields) for row in summary]
    main = [r for r in summary if r["J"] == "64" and r["budget"] == "100"]
    endpoint = {(r["m"], r["d"], r["J"], r["seed"], r["arm"]): r for r in main}

    arms = []
    for arm in sorted({r["arm"] for r in main}):
        rows = [r for r in main if r["arm"] == arm]
        reached = [_num(r["time_to_fixed_tol_sec"]) for r in rows]
        reached = [x for x in reached if x is not None]
        arms.append(
            _record(
                (
                    "arm",
                    "n",
                    "completed",
                    "converged",
                    "time_to_tol_reached",
                    "median_time_to_tol_sec",
                    "median_wall_sec",
                    "median_full_gradient_norm",
                    "median_EDR",
                    "median_solver_profile_evaluations",
                    "median_all_exact_profile_trials",
                ),
                (
                    arm,
                    len(rows),
                    sum(r["completed"].lower() == "true" for r in rows),
                    sum(r["converged"].lower() == "true" for r in rows),
                    len(reached),
                    _median(reached),
                    _median([_num(r["wall_sec"]) for r in rows]),
                    _median([_num(r["gradient_norm"]) for r in rows]),
                    _median([_num(r["projector_error"]) for r in rows]),
                    _median([_num(r["solver_objective_evaluations"]) for r in rows]),
                    _median(
                        [
                            _num(r["objective_evaluation_count_all_profile_trials"])
                            for r in rows
                        ]
                    ),
                ),
            )
        )

    budget_groups: dict[tuple[str, str], list[tuple[float | None, float | None]]] = (
        defaultdict(list)
    )
    for r in equal:
        key = (r["m"], r["d"], r["J"], r["seed"], r["arm"])
        if key not in endpoint or (*key[:4], "core") not in endpoint:
            continue
        budget_groups[(r["arm"], r["budget_kind"])].append(
            (_num(r["candidate_minus_core"]), _num(r["candidate_minus_core_edr"]))
        )
    equal_summary = []
    for (arm, kind), values in sorted(budget_groups.items()):
        fs = [f for f, _ in values if f is not None]
        es = [e for _, e in values if e is not None]
        paired = [(f, e) for f, e in values if f is not None and e is not None]
        equal_summary.append(
            _record(
                (
                    "arm",
                    "budget_kind",
                    "n",
                    "F_better",
                    "EDR_worse",
                    "both_F_better_EDR_worse",
                    "median_objective_difference",
                    "median_EDR_difference_matched_trace",
                    "F_win_fraction",
                    "missing_objective",
                    "missing_EDR",
                ),
                (
                    arm,
                    kind,
                    len(values),
                    sum(f < 0 for f in fs),
                    sum(e > 0 for e in es),
                    sum(f < 0 and e > 0 for f, e in paired),
                    _median(fs),
                    _median(es),
                    sum(f < 0 for f in fs) / len(fs) if fs else None,
                    len(values) - len(fs),
                    len(values) - len(es),
                ),
            )
        )

    cost = cost_summary(sweeps, output)
    reversal_summary = []
    for variant in ("optimized", "spectral"):
        rows = [
            r for r in reversals if r["variant"] == variant and r["series"] == "block"
        ]
        reversal_summary.append(
            {
                "variant": variant,
                "n_tasks": len(rows),
                "total_k_improvement_reversals": sum(
                    int(r["k_improvement_reversals"]) for r in rows
                ),
                "total_k_transitions": sum(int(r["k_transitions"]) for r in rows),
            }
        )
    noisy = []
    for row in regressions:
        fa, fb, ea, eb = (_num(row[k]) for k in ("F_a", "F_b", "EDR_a", "EDR_b"))
        if any(value is None for value in (fa, fb, ea, eb)):
            continue
        assert fa is not None and fb is not None and ea is not None and eb is not None
        if fa < fb and ea > eb:
            noisy.append((ea - eb, fb - fa, row))
        elif fb < fa and eb > ea:
            noisy.append((eb - ea, fa - fb, row))
    noisy.sort(reverse=True, key=lambda item: item[0])
    noisy_fields = (
        "m",
        "d",
        "J",
        "seed",
        "arm_a",
        "F_a",
        "EDR_a",
        "arm_b",
        "F_b",
        "EDR_b",
        "reference_EDR",
    )
    noise_summary = {
        "rows": len(regressions),
        "eligible_pairs": len(noisy),
        "strongest_cases": [
            {
                "EDR_increase": gap,
                "F_decrease": drop,
                **{key: row[key] for key in noisy_fields},
            }
            for gap, drop, row in noisy[:2]
        ],
    }

    result = {
        "input_audit": {
            "inner_summary_rows": len(summary),
            "unique_jobs": len(set(keys)),
            "duplicate_jobs": len(keys) - len(set(keys)),
            "main_J64_budget100_rows": len(main),
            "main_n_by_arm": {r["arm"]: r["n"] for r in arms},
            "trace_rows": len(traces),
            "equal_budget_rows": len(equal),
            "rank_sweep_rows": len(sweeps),
            "rank_reversal_rows": len(reversals),
            "lower_objective_worse_edr_rows": len(regressions),
        },
        "main_J64_budget100": arms,
        "equal_budgets_vs_core": equal_summary,
        "block_k_reversals": reversal_summary,
        "lower_objective_worse_edr": noise_summary,
        **cost,
    }
    (output / "inner_descriptive.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize(args.analysis_dir, args.output_dir)


if __name__ == "__main__":
    main()
