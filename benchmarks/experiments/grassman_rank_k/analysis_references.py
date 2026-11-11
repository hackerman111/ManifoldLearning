"""Reference-polish validation and rank-law cohort helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from ._analysis_io import _number


def merge_reference_overrides(
    rows: list[dict[str, Any]], overrides: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = ("m", "d", "seed")
    originals = {
        (row.get("m"), row.get("d"), row.get("seed")): row
        for row in rows
        if row.get("arm") == "reference"
    }
    accepted: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for override in overrides:
        key = (override.get("m"), override.get("d"), override.get("seed"))
        original = originals.get(key)
        result = override.get("polish_result")
        status = override.get("worker_status", {})
        if not isinstance(status, dict) or status.get("status") != "ok":
            continue
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        objective = _number(result.get("objective"))
        old_objective = _number(original.get("objective")) if original else None
        if result.get("finite") is False or objective is None or old_objective is None:
            continue
        if objective <= old_objective and (
            key not in accepted or objective < float(accepted[key]["objective"])
        ):
            accepted[key] = result

    merged_rows, used = [], []
    metadata = ("scenario", "m", "d", "J", "seed", "arm", "comparison_kind", "snapshot")
    for original in rows:
        key = (original.get("m"), original.get("d"), original.get("seed"))
        result = accepted.get(key) if original.get("arm") == "reference" else None
        if result is None:
            merged_rows.append(original)
            continue
        merged = {
            **original,
            **result,
            **{k: original[k] for k in metadata if k in original},
        }
        old, new = original.get("starts"), result.get("starts")
        merged.update(
            {
                "original_reference_objective": original.get("objective"),
                "original_reference_normalized_gradient": original.get(
                    "normalized_gradient"
                ),
                "original_reference_converged": original.get("converged"),
                "starts": (old if isinstance(old, list) else [])
                + (new if isinstance(new, list) else []),
                "reference_override_used": True,
            }
        )
        merged_rows.append(merged)
        used.append(
            {**dict(zip(fields, key, strict=True)), "objective": result["objective"]}
        )
    return merged_rows, used


def law_cohort(
    sweeps: list[dict[str, Any]], certified_only: bool
) -> list[dict[str, Any]]:
    return (
        [row for row in sweeps if row.get("reference_converged") is True]
        if certified_only
        else sweeps
    )


def fit_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None, int | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        if (
            row.get("seed") == 81600
            and row.get("reference_gap_valid")
            and _number(row.get("ratio")) is not None
        ):
            groups[(str(row.get("variant")), None, None)].append(row)
            groups[(str(row.get("variant")), int(row["m"]), int(row["d"]))].append(row)
    output = []

    def order(item: tuple[tuple[str, int | None, int | None], list[dict[str, Any]]]):
        key = item[0]
        return (
            key[0],
            -1 if key[1] is None else key[1],
            -1 if key[2] is None else key[2],
        )

    for (variant, m, d), items in sorted(groups.items(), key=order):
        usable = [
            row
            for row in items
            if float(row["ratio"]) > 0 and _number(row.get("x_k_over_r")) is not None
        ]
        x = np.asarray([float(row["x_k_over_r"]) for row in usable])
        y = np.asarray([-math.log(float(row["ratio"])) for row in usable])
        xx = float(x @ x)
        c_ls = max(0.0, float(x @ y / xx)) if len(x) and xx else 0.0
        lower = float(np.min(y / x)) if len(x) else None
        output.append(
            {
                "variant": variant,
                "m": m,
                "d": d,
                "n_train": len(usable),
                "c_least_squares": c_ls,
                "c_conservative_lower_envelope_raw": lower,
                "c_conservative_lower_envelope": max(0.0, lower)
                if lower is not None
                else None,
            }
        )
    return output
