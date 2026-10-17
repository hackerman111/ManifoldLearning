"""Direct rank-sweep comparisons and cost-frontier summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if result == result and abs(result) != float("inf") else None


def _median(values: Sequence[float]) -> float | None:
    import statistics

    return statistics.median(values) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _direct_pairs(sweeps: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    pairs, missing = [], 0
    for row in sweeps:
        if int(row["k"]) <= 1:
            continue
        block_f = _number(row["block_objective"])
        seq_f = _number(row["sequential_rank_one_budget_k_objective"])
        block_t, seq_t = (
            _number(row["block_wall_sec"]),
            _number(row["sequential_rank_one_wall_sec"]),
        )
        block_e = _number(row["block_projector_error"])
        seq_e = _number(row["sequential_rank_one_projector_error"])
        values = (block_f, seq_f, block_t, seq_t, block_e, seq_e)
        if any(value is None for value in values) or seq_t == 0:
            missing += 1
            continue
        assert block_f is not None and seq_f is not None
        assert block_t is not None and seq_t is not None
        assert block_e is not None and seq_e is not None
        pairs.append(
            {
                **row,
                "wall_ratio": block_t / seq_t,
                "F_delta": block_f - seq_f,
                "EDR_delta": block_e - seq_e,
            }
        )
    return pairs, missing


def cost_summary(
    sweeps: list[dict[str, str]],
    output: Path,
) -> dict[str, Any]:
    direct, missing = _direct_pairs(sweeps)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in direct:
        groups[(row["m"], row["d"], row["J"], row["variant"])].append(row)
    frontier = []
    for (m, d, J, variant), rows in sorted(groups.items()):
        ratios = [row["wall_ratio"] for row in rows]
        fs = [row["F_delta"] for row in rows]
        es = [row["EDR_delta"] for row in rows]
        mr, mf = _median(ratios), _median(fs)
        assert mr is not None and mf is not None
        frontier.append(
            {
                "m": m,
                "d": d,
                "J": J,
                "variant": variant,
                "n": len(rows),
                "median_block_over_sequential_wall_ratio": mr,
                "median_block_minus_sequential_F": mf,
                "F_wins": sum(value < 0 for value in fs),
                "EDR_wins": sum(value < 0 for value in es),
                "both_F_and_time_wins": sum(
                    row["F_delta"] < 0 and row["wall_ratio"] < 1 for row in rows
                ),
                "frontier_candidate": mr < 1 and mf <= 0,
            }
        )
    _write_csv(output / "cost_frontier.csv", frontier)

    by_variant = []
    for variant in ("optimized", "spectral"):
        rows = [row for row in direct if row["variant"] == variant]
        if not rows:
            continue
        by_variant.append(
            {
                "variant": variant,
                "n": len(rows),
                "median_wall_ratio": _median([row["wall_ratio"] for row in rows]),
                "F_wins": sum(row["F_delta"] < 0 for row in rows),
                "EDR_wins": sum(row["EDR_delta"] < 0 for row in rows),
                "both_F_and_time_wins": sum(
                    row["F_delta"] < 0 and row["wall_ratio"] < 1 for row in rows
                ),
                "median_objective_difference": _median(
                    [row["F_delta"] for row in rows]
                ),
                "median_EDR_difference": _median([row["EDR_delta"] for row in rows]),
            }
        )

    fixed: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in sweeps:
        fixed[(row["m"], row["d"], row["J"], row["seed"], row["variant"])].append(row)
    adjacent = []
    for rows in fixed.values():
        rows.sort(key=lambda row: int(row["k"]))
        for left, right in pairwise(rows):
            if int(right["k"]) == int(left["k"]) + 1:
                a, b = (
                    _number(left["block_objective"]),
                    _number(right["block_objective"]),
                )
                if a is not None and b is not None:
                    adjacent.append(b - a)
    return {
        "direct_k_gt1_pairs": len(direct),
        "direct_k_gt1_missing_pairs": missing,
        "k_gt1_block_vs_sequential_by_variant": by_variant,
        "cost_frontier_candidates": [
            row for row in frontier if row["frontier_candidate"]
        ],
        "fixed_k_adjacent_improvement": {
            "n_adjacent_pairs": len(adjacent),
            "larger_k_lower_objective": sum(value < 0 for value in adjacent),
            "nonimprovements": sum(value >= 0 for value in adjacent),
        },
    }
