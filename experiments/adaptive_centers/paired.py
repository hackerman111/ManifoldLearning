"""Paired EDR, runtime, recovery, and predeclared gate summaries."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from .common import ARMS, CASES, DEV_SEEDS

CONTROLS = ("baseline", "floor25", "floor25_search", "random_equal_j")


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def paired_rows(rows: list[dict[str, Any]], candidate: str, control: str):
    indexed = {(row.get("case"), row.get("seed"), row.get("arm")): row for row in rows}
    case_names = [case.name for case in CASES]
    scopes: list[tuple[str, list[str]]] = [(case, [case]) for case in case_names]
    scopes.append(("overall", case_names))
    for scope, allowed_cases in scopes:
        deltas, ratios, regressions = [], [], 0
        valid_pairs = 0
        completed_candidate = completed_control = 0
        candidate_recovery = control_recovery = 0
        total_pairs = len(allowed_cases) * len(DEV_SEEDS)
        for case in allowed_cases:
            for seed in DEV_SEEDS:
                left = indexed.get((case, seed, candidate), {})
                right = indexed.get((case, seed, control), {})
                left_ok, right_ok = (
                    left.get("status") == "ok",
                    right.get("status") == "ok",
                )
                completed_candidate += int(left_ok)
                completed_control += int(right_ok)
                if left_ok and float(left["edr_error"]) <= 0.1:
                    candidate_recovery += 1
                if right_ok and float(right["edr_error"]) <= 0.1:
                    control_recovery += 1
                if not (left_ok and right_ok):
                    continue
                valid_pairs += 1
                difference = float(left["edr_error"]) - float(right["edr_error"])
                deltas.append(difference)
                regressions += int(difference > 0.05)
                control_wall = float(right["fit_wall_sec"])
                ratios.append(
                    float(left["fit_wall_sec"]) / control_wall
                    if control_wall > 0
                    else float("inf")
                )
        median_delta, median_ratio = _median(deltas), _median(ratios)
        failure_candidate = (
            1 - completed_candidate / total_pairs if total_pairs else 1.0
        )
        failure_control = 1 - completed_control / total_pairs if total_pairs else 1.0
        quality_ok = (
            median_delta is not None
            and median_delta <= 0.01
            and regressions == 0
            and failure_candidate <= failure_control
        )
        is_p0 = candidate == "floor25" and control == "baseline"
        p0_quality_ok = (
            is_p0
            and median_delta is not None
            and median_delta <= -0.01
            and median_ratio is not None
            and median_ratio <= 1.10
        )
        yield {
            "scope": scope,
            "candidate": candidate,
            "control": control,
            "n_pairs_expected": total_pairs,
            "n_pairs_valid": valid_pairs,
            "median_edr_difference": median_delta,
            "median_fit_time_ratio": median_ratio,
            "regressions_over_0_05": regressions,
            "candidate_completed": completed_candidate,
            "control_completed": completed_control,
            "candidate_recovery_le_0_1": candidate_recovery,
            "control_recovery_le_0_1": control_recovery,
            "speed_gate": bool(
                quality_ok and median_ratio is not None and median_ratio <= 0.90
            ),
            "p0_quality_gate": bool(p0_quality_ok),
        }


def write_pairwise(rows: list[dict[str, Any]], destination: Path) -> None:
    columns = (
        "scope",
        "candidate",
        "control",
        "n_pairs_expected",
        "n_pairs_valid",
        "median_edr_difference",
        "median_fit_time_ratio",
        "regressions_over_0_05",
        "candidate_completed",
        "control_completed",
        "candidate_recovery_le_0_1",
        "control_recovery_le_0_1",
        "speed_gate",
        "p0_quality_gate",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in (arm.name for arm in ARMS):
            for control in CONTROLS:
                for row in paired_rows(rows, candidate, control):
                    writer.writerow(row)
