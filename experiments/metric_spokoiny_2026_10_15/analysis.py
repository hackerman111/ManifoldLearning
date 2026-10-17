"""Auditable summaries and plots for metric-Spokoiny worker rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def development_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(row["seed"]), row["arm"]): row
        for row in rows
        if row.get("status") == "ok"
        and row.get("mode") == "fullfit"
        and row.get("budget") == 5
        and row.get("run_stage", "fullfit5") == "fullfit5"
    }
    pairs = []
    for seed in range(3):
        core, metric = by_key.get((seed, "core")), by_key.get((seed, "metric2"))
        if core is None or metric is None:
            continue
        hash_keys = (
            "X",
            "y",
            "truth_basis",
            "first_index",
            "first_U",
            "first_I",
            "first_mass",
        )
        left_hashes, right_hashes = (
            core.get("data_hashes", {}),
            metric.get("data_hashes", {}),
        )
        identity_match = all(
            left_hashes.get(key) and left_hashes.get(key) == right_hashes.get(key)
            for key in hash_keys
        )
        pairs.append(
            {
                "seed": seed,
                "identity_match": identity_match,
                "time_ratio_metric_over_core": metric["fit_time_sec"]
                / core["fit_time_sec"],
                "projector_regression": metric["projector_distance"]
                - core["projector_distance"],
                "relative_projector_reduction": (
                    (core["projector_distance"] - metric["projector_distance"])
                    / max(core["projector_distance"], 1e-300)
                ),
            }
        )
    if len(pairs) != 3:
        return {
            "passed": False,
            "reason": "requires all six successful core/metric2 rows",
            "pairs": pairs,
        }
    median_ratio = statistics.median(
        item["time_ratio_metric_over_core"] for item in pairs
    )
    median_reduction = statistics.median(
        item["relative_projector_reduction"] for item in pairs
    )
    if not all(item["identity_match"] for item in pairs):
        return {
            "passed": False,
            "reason": "paired data or initial-statistic hashes differ",
            "pairs": pairs,
        }
    no_regression = max(item["projector_regression"] for item in pairs) <= 1e-4
    passed = no_regression and (
        median_ratio <= 1 / 1.10 or (median_reduction >= 0.20 and median_ratio <= 1.5)
    )
    return {
        "passed": passed,
        "no_per_seed_regression_over_1e-4": no_regression,
        "median_metric_over_core_time_ratio": median_ratio,
        "median_relative_projector_reduction": median_reduction,
        "speed_gate": median_ratio <= 1 / 1.10,
        "quality_gate": median_reduction >= 0.20 and median_ratio <= 1.5,
        "pairs": pairs,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fullfit = [
        row
        for row in rows
        if row.get("mode") == "fullfit" and row.get("status") == "ok"
    ]
    frozen = [
        row for row in rows if row.get("mode") == "frozen" and row.get("status") == "ok"
    ]
    stage_rows = {
        stage: [row for row in fullfit if row.get("run_stage") == stage]
        for stage in ("fullfit5", "budget20", "validation")
    }
    pairs_by_stage = {stage: _fit_pairs(group) for stage, group in stage_rows.items()}
    frozen_pairs = []
    groups = sorted({(row["seed"], row.get("phase"), row["budget"]) for row in frozen})
    for seed, phase, budget in groups:
        by_arm = {
            row["arm"]: row
            for row in frozen
            if row["seed"] == seed
            and row.get("phase") == phase
            and row["budget"] == budget
        }
        if "core" not in by_arm or "metric2" not in by_arm:
            continue
        left, right = by_arm["core"], by_arm["metric2"]
        cert_left = left["smooth"] and left["live_gradient_certificate"] <= 1e-6
        cert_right = right["smooth"] and right["live_gradient_certificate"] <= 1e-6
        frozen_pairs.append(
            {
                "seed": seed,
                "phase": phase,
                "budget": budget,
                "same_certificate": cert_left and cert_right,
                "core_gradient": left["live_gradient_certificate"],
                "metric2_gradient": right["live_gradient_certificate"],
                "core_objective_ratio": left["objective_ratio"],
                "metric2_objective_ratio": right["objective_ratio"],
                "core_projector_distance": left["projector_distance"],
                "metric2_projector_distance": right["projector_distance"],
                "metric2_over_core_time": right["time_sec"] / left["time_sec"],
            }
        )
    certified = [row for row in frozen_pairs if row["same_certificate"]]
    return {
        "development_gate": development_gate(stage_rows["fullfit5"]),
        "fullfit_pairs_by_stage": pairs_by_stage,
        "fullfit_metric2_over_core_median_by_stage": {
            stage: statistics.median(row["metric2_over_core_time"] for row in pairs)
            if pairs
            else None
            for stage, pairs in pairs_by_stage.items()
        },
        "frozen_pairs": frozen_pairs,
        "frozen_same_certificate_pairs": len(certified),
        "frozen_same_certificate_metric2_over_core_median": (
            statistics.median(row["metric2_over_core_time"] for row in certified)
            if certified
            else None
        ),
        "failures": [
            row
            for row in rows
            if row.get("status") != "ok" and row.get("status") != "skipped"
        ],
    }


def _fit_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        by_arm = {row["arm"]: row for row in rows if int(row["seed"]) == seed}
        if "core" not in by_arm or "metric2" not in by_arm:
            continue
        pairs.append(
            {
                "seed": seed,
                "metric2_over_core_time": by_arm["metric2"]["fit_time_sec"]
                / by_arm["core"]["fit_time_sec"],
                "core_projector_distance": by_arm["core"]["projector_distance"],
                "metric2_projector_distance": by_arm["metric2"]["projector_distance"],
                "metric1_projector_distance": by_arm.get("metric1", {}).get(
                    "projector_distance"
                ),
            }
        )
    return pairs


def _plot(output: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = summary["fullfit_pairs_by_stage"]["fullfit5"]
    frozen = summary["frozen_pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if fit:
        axes[0].plot(
            [row["core_projector_distance"] for row in fit],
            [row["metric2_projector_distance"] for row in fit],
            "o",
            label="metric2 vs core",
        )
        upper = max(
            max(row["core_projector_distance"], row["metric2_projector_distance"])
            for row in fit
        )
        axes[0].plot([0, upper], [0, upper], "k--", linewidth=1)
        axes[0].set(xlabel="core projector error", ylabel="metric2 projector error")
        axes[0].legend()
    certified = [row for row in frozen if row["same_certificate"]]
    uncertified = [row for row in frozen if not row["same_certificate"]]
    for subset, marker, label in (
        (certified, "o", "both certified"),
        (uncertified, "x", "not jointly certified"),
    ):
        if subset:
            axes[1].scatter(
                [row["core_gradient"] for row in subset],
                [row["metric2_gradient"] for row in subset],
                marker=marker,
                label=label,
            )
    axes[1].axvline(1e-6, color="k", linestyle="--", linewidth=1)
    axes[1].axhline(1e-6, color="k", linestyle="--", linewidth=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="core live gradient",
        ylabel="metric2 live gradient",
    )
    if frozen:
        axes[1].legend()
    fig.tight_layout()
    path = output / "metric_spokoiny_summary.png"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(output: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output.glob("*.jsonl")):
        rows.extend(read_rows(path))
    summary = summarize(rows)
    summary_path = output / "analysis_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_rows = [
        {"stage": stage, **row}
        for stage, pairs in summary["fullfit_pairs_by_stage"].items()
        for row in pairs
    ] + [{"stage": "frozen", **row} for row in summary["frozen_pairs"]]
    csv_path = output / "paired_results.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        columns = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
