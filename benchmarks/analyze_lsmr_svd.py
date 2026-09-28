"""Summarize the frozen paired full-fit LSMR/SVD benchmark."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

CASE_LABELS = {
    "small": "Малая: n=500, d=20, m=3 (100 пар)",
    "medium": "Средняя: n=900, d=60, m=3 (50 пар)",
    "spokoini_m2_dhigh_30s": "Большая: Spokoiny, n=800, d=50, m=2, N_J=800 (25 пар)",
}


def _median_iqr(values: list[float], scale: float = 1.0) -> str:
    if not values:
        return "—"
    q1, median, q3 = np.quantile(np.asarray(values) / scale, [0.25, 0.5, 0.75])
    return f"{median:.3g} [{q1:.3g}, {q3:.3g}]"


def _bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(20_000, len(data)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _ci_text(ci: tuple[float, float] | None, scale: float = 1.0) -> str:
    if ci is None:
        return "—"
    return f"[{ci[0] / scale:.3g}, {ci[1] / scale:.3g}]"


def analyze(root: Path) -> tuple[dict[str, object], str]:
    rows = [
        json.loads(line)
        for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report: list[str] = [
        "# LSMR vs current SVD: paired full-fit comparison",
        "",
        "## Protocol and interpretation",
        "",
        "Each seed uses identical generated data, truth basis, and initialization. "
        "Fits ran in isolated processes with float64, one BLAS thread, and no "
        "`tracemalloc`. RSS is process high-water memory, including interpreter "
        "and import overhead. Entries use median [Q1, Q3]. Paired 95% intervals "
        "are percentile bootstrap intervals for the median (20,000 resamples); "
        "treat them as descriptive, especially for the 25-seed large case.",
        "",
        "LSMR optimizes the current HPAO correction-penalty objective. SVD uses "
        "the fixed-g rank-constrained matrix objective (`low_rank_target=matrix`, "
        "rank `m-1`); it alternates an approximate low-rank solve and completes "
        "the remaining public basis direction from the prior. Projector recovery "
        "therefore compares end-to-end estimators, not identical inner objectives. "
        "Convergence diagnostics also differ. Small/medium fits use three outer "
        "steps; the large saved schedule runs to `h_min`.",
        "A positive paired error delta (SVD minus LSMR) means LSMR had the lower "
        "projector error. In the diagnostics table, SVD convergence means all "
        "rank-one inner solves converged; LSMR convergence is its HPAO stopping "
        "certificate. These flags are not directly equivalent.",
        "",
        "## Results",
        "",
        "| Workload | Successful pairs | SVD time s | LSMR time s | "
        "Median paired time ratio SVD/LSMR (95% CI) | SVD projector error | "
        "LSMR projector error | Median paired error delta SVD-LSMR (95% CI) | "
        "SVD RSS MiB | LSMR RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    result: dict[str, object] = {"workloads": {}}

    for case_index, case in enumerate(CASE_LABELS):
        group = [row for row in rows if row.get("case") == case]
        by_method = {
            method: [row for row in group if row.get("solver") == method]
            for method in ("svd", "lsmr")
        }
        successful = {
            method: {
                int(row["seed"]): row
                for row in method_rows
                if row.get("status") == "ok"
            }
            for method, method_rows in by_method.items()
        }
        seeds = sorted(set(successful["svd"]) & set(successful["lsmr"]))
        paired = [
            (seed, successful["svd"][seed], successful["lsmr"][seed]) for seed in seeds
        ]
        time_ratios = [
            float(s["fit_time_sec"]) / float(l["fit_time_sec"]) for _, s, l in paired
        ]
        quality_deltas = [
            float(s["projector_distance"]) - float(l["projector_distance"])
            for _, s, l in paired
        ]
        rss_deltas = [
            int(s["peak_rss_kib"]) - int(l["peak_rss_kib"]) for _, s, l in paired
        ]
        seed = 20260929 + case_index
        time_ci = _bootstrap_median_ci(time_ratios, seed)
        quality_ci = _bootstrap_median_ci(quality_deltas, seed + 100)
        rss_ci = _bootstrap_median_ci(rss_deltas, seed + 200)

        medians: dict[str, dict[str, float | None]] = {}
        method_details: dict[str, object] = {}
        for method in ("svd", "lsmr"):
            values = list(successful[method].values())
            method_medians: dict[str, float | None] = {}
            for metric in (
                "fit_time_sec",
                "peak_rss_kib",
                "projector_distance",
                "outer_iterations",
            ):
                nums = [
                    float(row[metric]) for row in values if row.get(metric) is not None
                ]
                method_medians[metric] = (
                    float(statistics.median(nums)) if nums else None
                )
            medians[method] = method_medians
            failures = [row for row in by_method[method] if row.get("status") != "ok"]
            diagnostics = values
            if method == "svd":
                converged = sum(
                    row.get("svd_inner_converged") is True for row in diagnostics
                )
                inner_iterations = [
                    int(row["inner_iterations"])
                    for row in diagnostics
                    if row.get("inner_iterations") is not None
                ]
                linear_iterations = []
                direct_solves = [
                    int(row["direct_solves"])
                    for row in diagnostics
                    if row.get("direct_solves") is not None
                ]
            else:
                converged = sum(
                    row.get("lsmr_solver_converged") is True for row in diagnostics
                )
                inner_iterations = []
                linear_iterations = [
                    int(row["linear_iterations"])
                    for row in diagnostics
                    if row.get("linear_iterations") is not None
                ]
                direct_solves = []
            method_details[method] = {
                "attempts": len(by_method[method]),
                "successes": len(values),
                "failures": [
                    {"seed": row.get("seed"), "error": row.get("error")}
                    for row in failures
                ],
                "inner_converged_fits": converged,
                "outer_stop_reasons": {
                    str(reason): sum(row.get("stop_reason") == reason for row in values)
                    for reason in sorted(
                        {str(row.get("stop_reason")) for row in values}
                    )
                },
                "median_inner_iterations_svd": statistics.median(inner_iterations)
                if inner_iterations
                else None,
                "median_linear_iterations_lsmr": statistics.median(linear_iterations)
                if linear_iterations
                else None,
                "median_direct_solves_svd": statistics.median(direct_solves)
                if direct_solves
                else None,
            }

        paired_summary = {
            "seeds": seeds,
            "complete_pairs": len(paired),
            "svd_faster_pairs": sum(value < 1.0 for value in time_ratios),
            "svd_lower_error_pairs": sum(value < 0.0 for value in quality_deltas),
            "lsmr_lower_error_pairs": sum(value > 0.0 for value in quality_deltas),
            "svd_lower_rss_pairs": sum(value < 0.0 for value in rss_deltas),
            "median_time_ratio_svd_over_lsmr": float(statistics.median(time_ratios))
            if time_ratios
            else None,
            "median_time_ratio_ci95": list(time_ci) if time_ci else None,
            "median_quality_delta_svd_minus_lsmr": float(
                statistics.median(quality_deltas)
            )
            if quality_deltas
            else None,
            "median_quality_delta_ci95": list(quality_ci) if quality_ci else None,
            "median_rss_delta_kib_svd_minus_lsmr": float(statistics.median(rss_deltas))
            if rss_deltas
            else None,
            "median_rss_delta_ci95": list(rss_ci) if rss_ci else None,
        }
        case_summary = {
            "description": CASE_LABELS[case],
            "per_method": method_details,
            "medians": medians,
            "paired": paired_summary,
        }
        result["workloads"][case] = case_summary  # type: ignore[index]

        attempts = max(len(by_method["svd"]), len(by_method["lsmr"]))
        report.append(
            "| "
            + " | ".join(
                [
                    CASE_LABELS[case],
                    f"{len(paired)}/{attempts}",
                    _median_iqr(
                        [
                            float(row["fit_time_sec"])
                            for row in by_method["svd"]
                            if row.get("status") == "ok"
                        ]
                    ),
                    _median_iqr(
                        [
                            float(row["fit_time_sec"])
                            for row in by_method["lsmr"]
                            if row.get("status") == "ok"
                        ]
                    ),
                    f"{statistics.median(time_ratios):.3g} {_ci_text(time_ci)}"
                    if time_ci
                    else "—",
                    _median_iqr(
                        [
                            float(row["projector_distance"])
                            for row in by_method["svd"]
                            if row.get("status") == "ok"
                        ]
                    ),
                    _median_iqr(
                        [
                            float(row["projector_distance"])
                            for row in by_method["lsmr"]
                            if row.get("status") == "ok"
                        ]
                    ),
                    f"{statistics.median(quality_deltas):+.3g} {_ci_text(quality_ci)}"
                    if quality_ci
                    else "—",
                    _median_iqr(
                        [
                            float(row["peak_rss_kib"])
                            for row in by_method["svd"]
                            if row.get("status") == "ok"
                        ],
                        1024,
                    ),
                    _median_iqr(
                        [
                            float(row["peak_rss_kib"])
                            for row in by_method["lsmr"]
                            if row.get("status") == "ok"
                        ],
                        1024,
                    ),
                ]
            )
            + " |"
        )

    report.extend(["", "## Paired outcome counts", ""])
    report.append(
        "| Workload | SVD faster | Lower projector error: SVD / LSMR | SVD lower RSS |"
    )
    report.append("|---|---:|---:|---:|")
    for case in CASE_LABELS:
        paired = result["workloads"][case]["paired"]  # type: ignore[index]
        outcome_cells = [
            f"{paired['svd_faster_pairs']}/{paired['complete_pairs']}",
            f"{paired['svd_lower_error_pairs']}/{paired['complete_pairs']} / "
            f"{paired['lsmr_lower_error_pairs']}/{paired['complete_pairs']}",
            f"{paired['svd_lower_rss_pairs']}/{paired['complete_pairs']}",
        ]
        report.append(
            "| " + CASE_LABELS[case] + " | " + " | ".join(outcome_cells) + " |"
        )
    report.extend(["", "## Solver diagnostics", ""])
    report.append(
        "| Workload | SVD inner-converged fits | "
        "Median SVD inner iterations / direct solves | LSMR converged fits | "
        "Median LSMR iterations | Outer stop reasons (SVD / LSMR) |"
    )
    report.append("|---|---:|---:|---:|---:|---|")
    for case in CASE_LABELS:
        details = result["workloads"][case]["per_method"]  # type: ignore[index]
        svd, lsmr = details["svd"], details["lsmr"]
        svd_iterations = svd["median_inner_iterations_svd"]
        svd_solves = svd["median_direct_solves_svd"]
        lsmr_iterations = lsmr["median_linear_iterations_lsmr"]
        svd_iterations = "—" if svd_iterations is None else svd_iterations
        svd_solves = "—" if svd_solves is None else svd_solves
        lsmr_iterations = "—" if lsmr_iterations is None else lsmr_iterations
        report.append(
            f"| {CASE_LABELS[case]} | "
            f"{svd['inner_converged_fits']}/{svd['successes']} | "
            f"{svd_iterations} / {svd_solves} | "
            f"{lsmr['inner_converged_fits']}/{lsmr['successes']} | "
            f"{lsmr_iterations} | "
            f"{svd['outer_stop_reasons']} / {lsmr['outer_stop_reasons']} |"
        )
    report.append("")
    report.extend(["", "## Failures and incomplete pairs", ""])
    any_failures = False
    for case in CASE_LABELS:
        group = [row for row in rows if row.get("case") == case]
        for method in ("svd", "lsmr"):
            failures = [
                row
                for row in group
                if row.get("solver") == method and row.get("status") != "ok"
            ]
            if failures:
                any_failures = True
                report.extend(
                    [
                        f"- {CASE_LABELS[case]}, {method}, seed {failure.get('seed')}: "
                        f"{failure.get('error', failure.get('status'))}"
                        for failure in failures
                    ]
                )
    if not any_failures:
        report.append("No worker failures or incomplete fits were recorded.")
    report.append("")
    return result, "\n".join(report)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    result, report = analyze(args.experiment_dir)
    (args.experiment_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.experiment_dir / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
