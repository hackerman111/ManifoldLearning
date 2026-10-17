"""Full-fit aggregation and paired summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ._analysis_io import _median, _number


def _solver_summary(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("solver_summary", {})
    return value if isinstance(value, dict) else {}


def _split(seed: int) -> str:
    return "development" if seed < 81700 else "heldout"


def fullfit_tables(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    keyed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        scenario, arm = row.get("scenario", "unknown"), row.get("arm", "unknown")
        seed = row.get("seed")
        if isinstance(seed, int):
            split = _split(seed)
            for scenario_key in (str(scenario), "ALL"):
                for split_key in (split, "all"):
                    grouped[(scenario_key, split_key, str(arm))].append(row)
                    keyed[(scenario_key, str(scenario), split_key, seed, str(arm))] = (
                        row
                    )
    arms = []
    for (scenario, split, arm), items in sorted(grouped.items()):
        complete = [
            r for r in items if r.get("completed") is True and r.get("finite") is True
        ]
        summary = [_solver_summary(r) for r in items]
        walls = [v for r in items if (v := _number(r.get("wall_sec"))) is not None]
        solver_walls = [
            v for r in items if (v := _number(r.get("solver_wall_sec"))) is not None
        ]
        rss = [v for r in items if (v := _number(r.get("peak_rss_kib"))) is not None]
        edr = [v for r in items if (v := _number(r.get("projector_error"))) is not None]
        ortho = [
            v for r in items if (v := _number(r.get("orthogonality_error"))) is not None
        ]
        iterations = [
            float(x)
            for s in summary
            for x in s.get("iterations_by_call", [])
            if _number(x) is not None
        ]
        calls = [
            v
            for s in summary
            if (v := _number(s.get("profile_evaluations"))) is not None
        ]
        converged_calls, all_calls = [], []
        for s in summary:
            count = _number(s.get("converged_calls"))
            total = s.get("iterations_by_call")
            if count is not None and isinstance(total, list):
                converged_calls.append(count)
                all_calls.append(float(len(total)))
        total_iterations = [
            sum(float(x) for x in s["iterations_by_call"] if _number(x) is not None)
            for s in summary
            if isinstance(s.get("iterations_by_call"), list)
        ]
        fallback_counts: dict[str, int] = defaultdict(int)
        for s in summary:
            fallbacks = s.get("fallbacks", {})
            if isinstance(fallbacks, dict):
                for name, count in fallbacks.items():
                    if isinstance(count, int):
                        fallback_counts[name] += count
        arms.append(
            {
                "scenario": scenario,
                "split": split,
                "arm": arm,
                "n_rows": len(items),
                "n_completed": len(complete),
                "completed_fraction": len(complete) / len(items),
                "recovery_success_n": sum(
                    r.get("completed") is True
                    and _number(r.get("projector_error")) is not None
                    and r["projector_error"] <= 0.1
                    for r in items
                ),
                "recovery_success_fraction_all": sum(
                    r.get("completed") is True
                    and _number(r.get("projector_error")) is not None
                    and r["projector_error"] <= 0.1
                    for r in items
                )
                / len(items),
                "certificate_fraction_calls": sum(converged_calls) / sum(all_calls)
                if sum(all_calls)
                else None,
                "median_wall_sec": _median(walls),
                "median_solver_wall_sec": _median(solver_walls),
                "median_iterations": _median(iterations),
                "total_iterations_sum": sum(total_iterations),
                "median_profile_evaluations": _median(calls),
                "fallback_counts": json.dumps(fallback_counts, sort_keys=True),
                "median_cost_probe_evaluations": _median(
                    [
                        v
                        for s in summary
                        if (v := _number(s.get("cost_probe_evaluations"))) is not None
                    ]
                ),
                "median_cost_probe_seconds": _median(
                    [
                        v
                        for s in summary
                        if (v := _number(s.get("cost_probe_seconds"))) is not None
                    ]
                ),
                "median_peak_rss_kib": _median(rss),
                "median_projector_error": _median(edr),
                "median_orthogonality_error": _median(ortho),
            }
        )
    paired = []
    scenarios = sorted({key[0] for key in keyed})
    splits = sorted({key[2] for key in keyed})
    for scenario in scenarios:
        for split in splits:
            candidates = sorted(
                {
                    arm
                    for scenario_key, _, split_key, _, arm in keyed
                    if scenario_key == scenario and split_key == split
                }
                - {"core", "spectral"}
            )
            for arm in candidates:
                for baseline in ("core", "spectral"):
                    pairs = [
                        (
                            seed,
                            actual_scenario,
                            keyed[(scenario, actual_scenario, split, seed, arm)],
                            keyed[(scenario, actual_scenario, split, seed, baseline)],
                        )
                        for (
                            scenario_key,
                            actual_scenario,
                            split_key,
                            seed,
                            candidate_arm,
                        ) in keyed
                        if scenario_key == scenario
                        and split_key == split
                        and candidate_arm == arm
                        and (
                            scenario,
                            actual_scenario,
                            split,
                            seed,
                            baseline,
                        )
                        in keyed
                    ]
                    ratios, solver_ratios, edr_diffs = [], [], []
                    for _, _, cand, base in pairs:
                        cw, bw = (
                            _number(cand.get("wall_sec")),
                            _number(base.get("wall_sec")),
                        )
                        both_complete = (
                            cand.get("completed") is True
                            and base.get("completed") is True
                        )
                        if (
                            both_complete
                            and cw is not None
                            and bw is not None
                            and bw > 0
                        ):
                            ratios.append(cw / bw)
                        cs, bs = (
                            _number(cand.get("solver_wall_sec")),
                            _number(base.get("solver_wall_sec")),
                        )
                        if (
                            both_complete
                            and cs is not None
                            and bs is not None
                            and bs > 0
                        ):
                            solver_ratios.append(cs / bs)
                        ce, be = (
                            _number(cand.get("projector_error")),
                            _number(base.get("projector_error")),
                        )
                        if (
                            ce is not None
                            and be is not None
                            and cand.get("completed") is True
                            and base.get("completed") is True
                        ):
                            edr_diffs.append(ce - be)
                    paired.append(
                        {
                            "scenario": scenario,
                            "split": split,
                            "arm": arm,
                            "baseline": baseline,
                            "n_seed_pairs": len(pairs),
                            "n_wall_pairs": len(ratios),
                            "median_fit_wall_ratio": _median(ratios),
                            "median_solver_wall_ratio": _median(solver_ratios),
                            "median_edr_difference": _median(edr_diffs),
                            "max_edr_regression": max(edr_diffs) if edr_diffs else None,
                            "n_both_completed": sum(
                                c.get("completed") is True
                                and b.get("completed") is True
                                for _, _, c, b in pairs
                            ),
                            "candidate_failures": sum(
                                c.get("completed") is not True for _, _, c, _ in pairs
                            ),
                            "baseline_failures": sum(
                                b.get("completed") is not True for _, _, _, b in pairs
                            ),
                        }
                    )
    return arms, paired
