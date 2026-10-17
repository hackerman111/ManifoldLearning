"""Aggregate frozen rank-k raw JSONL into descriptive tables and plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ._analysis_io import _number, _read, _write_csv
from .analysis_fullfit import fullfit_tables
from .analysis_inner import inner_tables
from .analysis_plots import _write_plots
from .analysis_references import fit_c, law_cohort, merge_reference_overrides
from .common import OUT


def analyze(
    results: Path,
    output: Path,
    reference_overrides: Path | None = None,
    certified_reference_only: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    dev = _read(results / "fullfit_dev.jsonl")
    held = _read(results / "fullfit_heldout.jsonl")
    inner_rows = _read(results / "inner.jsonl")
    override_hash = None
    overrides_used: list[dict[str, Any]] = []
    if reference_overrides is not None:
        override_hash = hashlib.sha256(reference_overrides.read_bytes()).hexdigest()
        inner_rows, overrides_used = merge_reference_overrides(
            inner_rows, _read(reference_overrides)
        )
    ff_arms, ff_pairs = fullfit_tables(dev + held)
    inner_summary, sweeps, frozen_regressions, trace_rows, references, equal_budgets = (
        inner_tables(inner_rows)
    )
    law_sweeps = law_cohort(sweeps, certified_reference_only)
    fits = fit_c(law_sweeps)
    laws = []
    invalid_by_seed: dict[int, int] = defaultdict(int)
    for seed in (81600, 81700):
        for variant in ("optimized", "spectral"):
            for m, d in [
                (None, None),
                *sorted(
                    {(r["m"], r["d"]) for r in law_sweeps if r["variant"] == variant}
                ),
            ]:
                fit = next(
                    (
                        r
                        for r in fits
                        if r["variant"] == variant and r["m"] == m and r["d"] == d
                    ),
                    None,
                )
                rows = [
                    r
                    for r in law_sweeps
                    if r["seed"] == seed
                    and r["variant"] == variant
                    and (m is None or (r["m"], r["d"]) == (m, d))
                ]
                valid = [
                    r
                    for r in rows
                    if r.get("reference_gap_valid")
                    and _number(r.get("ratio")) is not None
                    and r["ratio"] > 0
                ]
                invalid = sum(not r.get("reference_gap_valid") for r in rows)
                invalid_by_seed[seed] += invalid if m is None else 0
                c = None if fit is None else fit["c_least_squares"]
                conservative = (
                    None if fit is None else fit["c_conservative_lower_envelope"]
                )
                residuals = (
                    [math.log(r["ratio"]) + c * r["x_k_over_r"] for r in valid]
                    if c is not None
                    else []
                )
                laws.append(
                    {
                        "seed": seed,
                        "split": "development" if seed == 81600 else "heldout",
                        "variant": variant,
                        "m": m,
                        "d": d,
                        "n_valid": len(valid),
                        "n_invalid_gaps": invalid,
                        "c_train_least_squares": c,
                        "c_train_lower_envelope": conservative,
                        "inequality_violation_fraction_ls": sum(
                            r["ratio"] > math.exp(-c * r["x_k_over_r"]) + 1e-12
                            for r in valid
                        )
                        / len(valid)
                        if valid and c is not None
                        else None,
                        "inequality_violation_fraction_lower": sum(
                            r["ratio"]
                            > math.exp(-conservative * r["x_k_over_r"]) + 1e-12
                            for r in valid
                        )
                        / len(valid)
                        if valid and conservative is not None
                        else None,
                        "log_rmse_ls": float(np.sqrt(np.mean(np.square(residuals))))
                        if residuals
                        else None,
                    }
                )
    reversals = []
    by_sweep: dict[tuple[int, int, int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in sweeps:
        by_sweep[(row["m"], row["d"], row["J"], row["seed"], row["variant"])].append(
            row
        )
    for key, group in by_sweep.items():
        group = sorted(group, key=lambda r: r["k"])
        for field, series in (
            ("block_objective", "block"),
            ("sequential_rank_one_budget_k_objective", "sequential_rank_one"),
        ):
            count = sum(
                a[field] is not None and b[field] is not None and b[field] > a[field]
                for a, b in pairwise(group)
            )
            reversals.append(
                {
                    "m": key[0],
                    "d": key[1],
                    "J": key[2],
                    "seed": key[3],
                    "variant": key[4],
                    "series": series,
                    "k_improvement_reversals": count,
                    "k_transitions": max(0, len(group) - 1),
                }
            )
    identifiable = []
    for m, d, J, seed in sorted(
        {(r["m"], r["d"], r["J"], r["seed"]) for r in inner_summary if r["J"] == 64}
    ):
        p = max(12, 2 * m)
        identifiable.append(
            {
                "m": m,
                "d": d,
                "J": J,
                "seed": seed,
                "constraint_count": m * (d - m),
                "observation_residual_dim": J * (p - m),
                "underidentified_by_dimension_count": m * (d - m) > J * (p - m),
            }
        )
    _write_csv(output / "fullfit_arms.csv", ff_arms)
    _write_csv(output / "fullfit_paired.csv", ff_pairs)
    _write_csv(output / "inner_summary.csv", inner_summary)
    _write_csv(output / "inner_trace.csv", trace_rows)
    _write_csv(output / "inner_equal_budgets.csv", equal_budgets)
    _write_csv(output / "reference_summary.csv", references)
    _write_csv(output / "rank_sweeps.csv", sweeps)
    _write_csv(output / "rank_law.csv", fits + laws)
    _write_csv(output / "lower_objective_worse_edr.csv", frozen_regressions)
    _write_csv(output / "rank_reversals.csv", reversals)
    _write_csv(output / "inner_identifiability.csv", identifiable)
    _write_plots(output, ff_arms, ff_pairs, inner_rows, law_sweeps, fits)
    summary = {
        "input_counts": {
            "fullfit_dev_rows": len(dev),
            "fullfit_heldout_rows": len(held),
            "inner_rows": len(inner_rows),
        },
        "fullfit_arm_rows": len(ff_arms),
        "fullfit_paired_rows": len(ff_pairs),
        "valid_rank_sweep_rows": sum(
            bool(r.get("reference_gap_valid")) for r in sweeps
        ),
        "invalid_reference_gap_rows": sum(
            not r.get("reference_gap_valid") for r in sweeps
        ),
        "invalid_gap_counts_by_seed": dict(invalid_by_seed),
        "lower_frozen_objective_worse_edr_pairs": len(frozen_regressions),
        "rank_sweep_reversals": reversals,
        "inner_identifiability_counts": {
            "n": len(identifiable),
            "underidentified": sum(
                r["underidentified_by_dimension_count"] for r in identifiable
            ),
        },
        "rank_law": laws,
        "rank_law_mode": "certified_reference_only"
        if certified_reference_only
        else "all_references",
        "rank_law_rows": len(law_sweeps),
        "reference_override_file_sha256": override_hash,
        "reference_overrides_used": overrides_used,
        "objective_crosscheck_max_abs": max(
            (
                r["objective_crosscheck_abs"]
                for r in inner_summary
                if r.get("objective_crosscheck_abs") is not None
            ),
            default=None,
        ),
        "inner_objective_crosscheck_failures_1e-9": sum(
            r.get("objective_crosscheck_abs") is not None
            and r["objective_crosscheck_abs"]
            > 1e-9 * max(1.0, abs(r["objective"] or 0.0))
            for r in inner_summary
        ),
        "reference_rows": references,
        "note": (
            "Descriptive finite-sample aggregation. F_ref is a numerical upper "
            "reference, not a proven global minimum; adaptive fullfit objectives "
            "are not compared across arms."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Equal-budget summaries distinguish local refit calls from exact "
            "objective evaluations (which also count Schur trials)."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=OUT / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-overrides", type=Path)
    parser.add_argument("--certified-reference-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        name = "analysis"
        if args.reference_overrides is not None:
            name += "_polished"
        if args.certified_reference_only:
            name += "_certified"
        output = OUT / "results" / name
    analyze(
        args.results_dir,
        output,
        args.reference_overrides,
        args.certified_reference_only,
    )


if __name__ == "__main__":
    main()
