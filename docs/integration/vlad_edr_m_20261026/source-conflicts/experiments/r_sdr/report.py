"""Сводки и парные интервалы; неудачи сохраняются в знаменателях."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_baseline(methods: list[str]) -> str | None:
    if "ADP-Grassmann" in methods:
        return "ADP-Grassmann"
    if "ADP" in methods:
        return "ADP"
    if "ADP-Grassmann-combined" in methods:
        return "ADP-Grassmann-combined"
    return None


def interval(values: list[float], key: str) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(2000, len(values)), replace=True)
    low, high = np.quantile(np.median(sample, axis=1), [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: list[dict], runs: int, methods: list[str]
) -> tuple[list[dict], list[dict]]:
    summaries, pairs = [], []
    baseline_method = paired_baseline(methods)
    for case in sorted({row["case"] for row in rows}):
        case_rows = [r for r in rows if r["case"] == case]
        baseline = {
            r["seed"]: r
            for r in case_rows
            if r["method"] == baseline_method and r["status"] == "ok"
        }
        for method in methods:
            group = [r for r in case_rows if r["method"] == method]
            ok = [r for r in group if r["status"] == "ok"]
            count = sum(r.get("recovered", False) for r in ok)
            p = count / runs
            z = 1.959963984540054
            center = (p + z * z / (2 * runs)) / (1 + z * z / runs)
            radius = (
                z
                * np.sqrt(p * (1 - p) / runs + z * z / (4 * runs * runs))
                / (1 + z * z / runs)
            )
            row = {
                "case": case,
                "method": method,
                "n_planned": runs,
                "n_attempted": len(group),
                "n_ok": len(ok),
                "n_missing": runs - len(group),
                "n_error": sum(r["status"] == "error" for r in group),
                "n_timeout": sum(r["status"] == "timeout" for r in group),
                "n_memory_limit": sum(r["status"] == "memory_limit" for r in group),
                "recovery_rate": p,
                "recovery_wilson_low": max(0.0, center - radius),
                "recovery_wilson_high": min(1.0, center + radius),
                "n_converged": sum(r.get("converged") is True for r in ok),
                "n_convergence_unknown": sum(r.get("converged") is None for r in ok),
                "n_all_inner_certified": sum(
                    r.get("all_inner_certified") is True for r in ok
                ),
            }
            for metric in (
                "fit_seconds",
                "process_seconds",
                "peak_rss_mib",
                "projector_distance",
                "trace_score",
                "max_angle_deg",
                "mean_angle_deg",
                "max_sin_angle",
            ):
                values = [r[metric] for r in ok if r.get(metric) is not None]
                for suffix, value in (
                    ("mean", np.mean(values) if values else None),
                    ("median", np.median(values) if values else None),
                    ("std", np.std(values, ddof=1) if len(values) > 1 else None),
                    ("p25", np.quantile(values, 0.25) if values else None),
                    ("p75", np.quantile(values, 0.75) if values else None),
                    ("p95", np.quantile(values, 0.95) if values else None),
                ):
                    row[f"{metric}_{suffix}"] = None if value is None else float(value)
                low, high = interval(values, f"{case}:{method}:{metric}")
                row[f"{metric}_median_ci_low"] = low
                row[f"{metric}_median_ci_high"] = high
            summaries.append(row)
            if method == baseline_method or baseline_method is None:
                continue
            paired = [(r, baseline[r["seed"]]) for r in ok if r["seed"] in baseline]
            ratios = [
                r["fit_seconds"] / b["fit_seconds"]
                for r, b in paired
                if r["fit_seconds"] > 0 and b["fit_seconds"] > 0
            ]
            deltas = [
                r["projector_distance"] - b["projector_distance"] for r, b in paired
            ]
            paired_row = {
                "case": case,
                "method": method,
                "baseline": baseline_method,
                "n_pairs": len(paired),
                "paired_complete": len(paired) == runs,
                "n_positive_time_pairs": len(ratios),
            }
            for key, values in (("fit_ratio", ratios), ("projector_delta", deltas)):
                paired_row[f"{key}_median"] = (
                    float(np.median(values)) if values else None
                )
                low, high = interval(values, f"paired:{case}:{method}:{key}")
                paired_row[f"{key}_ci_low"], paired_row[f"{key}_ci_high"] = low, high
            pairs.append(paired_row)
    return summaries, pairs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, *, plots: bool = False) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    protocol = manifest["protocol"]
    rows = [
        json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()
    ]
    summaries, pairs = summarize(rows, protocol["runs"], protocol["methods"])
    baseline_method = paired_baseline(protocol["methods"])
    settings = {c["name"]: c for c in protocol["cases"]}
    for row in summaries:
        row.update({k: v for k, v in settings[row["case"]].items() if k != "name"})
    # Also emit not-yet-started cases so a partial report is explicit.
    seen = {r["case"] for r in rows}
    lines = [
        "# ADP / R SDR comparison",
        "",
        f"Recorded fits: {len(rows)} / {protocol['planned_fits']}.",
        "",
        (
            "Recovery = valid basis and projector distance <= configured "
            "threshold; failures and pending runs remain in denominator."
        ),
        (
            "Convergence is separate: ADP/ADP-Grassmann outer convergence is "
            "unknown; all_inner_certified records only inner solves. R references "
            "report outer subspace-step stopping. Package/SIR: not exposed/N-A."
        ),
        (
            "Times/errors below are conditional on valid fits. Paired intervals"
            " use joint successes only; n_pairs is mandatory."
        ),
        (
            "95% intervals: 2000 paired bootstrap samples of median; recovery "
            "uses Wilson intervals. No multiple-testing adjustment."
        ),
        (
            "RSS is the sampled Linux process VmHWM, including "
            "runtime/imports/input loading; failed runs retain sampled RSS in "
            "runs.csv."
        ),
        (
            "R fit clock: proc.time elapsed (millisecond resolution); Python: "
            "perf_counter. Process time includes startup and I/O."
        ),
        (
            "dMAVE/rMAVE are documented R reference variants; timings compare "
            "these implementations, not optimal algorithmic complexity."
        ),
        "",
        "| Case | Method | Valid/planned | Time median s | EDR median | Recovery |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        fit, edr = row["fit_seconds_median"], row["projector_distance_median"]
        lines.append(
            f"| {row['case']} | {row['method']} | {row['n_ok']}/{row['n_planned']} | "
            f"{fit if fit is not None else 'NA'} | "
            f"{edr if edr is not None else 'NA'} | {row['recovery_rate']:.2f} |"
        )
    pending = [c["name"] for c in protocol["cases"] if c["name"] not in seen]
    if pending:
        lines += ["", "Unstarted cases: " + ", ".join(pending)]
    lines += [
        "",
        (
            f"## Paired comparisons with {baseline_method}"
            if baseline_method
            else "## Paired comparisons"
        ),
        "",
        (
            f"Ratio = method fit time / {baseline_method} fit time; delta = "
            f"method EDR - {baseline_method} EDR. "
            "Intervals are 95% bootstrap median intervals, conditional on "
            "joint successes."
            if baseline_method
            else "No paired baseline method was selected."
        ),
        "",
        "| Case | Method | Pairs/planned | Time ratio [CI] | EDR delta [CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in pairs:
        cells = []
        for key in ("fit_ratio", "projector_delta"):
            value, low, high = (
                row[f"{key}_{suffix}"] for suffix in ("median", "ci_low", "ci_high")
            )
            cells.append("NA" if value is None else f"{value:.4g} [{low}, {high}]")
        lines.append(
            f"| {row['case']} | {row['method']} | "
            f"{row['n_pairs']}/{protocol['runs']} | {cells[0]} | {cells[1]} |"
        )
    lines += [
        "",
        (
            "Definitions, estimator limits and source citations: "
            "experiments/r_sdr/README.md."
        ),
        (
            "Full settings/source hashes/environment: manifest.json; immutable "
            "code: sources.zip."
        ),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired.csv", pairs)
    if plots and summaries:
        plot(output, summaries, protocol["methods"])


def plot(output: Path, rows: list[dict], methods: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 6))
    for method in methods:
        group = [r for r in rows if r["method"] == method and r["n_ok"]]
        axis.scatter(
            [r["fit_seconds_median"] for r in group],
            [r["projector_distance_median"] for r in group],
            label=method,
            alpha=0.7,
        )
    axis.set(
        xscale="symlog", xlabel="Median fit seconds", ylabel="Median projector distance"
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "time-quality.png", dpi=150)
    plt.close(fig)
    cases = sorted({r["case"] for r in rows})
    grid = np.full((len(cases), len(methods)), np.nan)
    for row in rows:
        grid[cases.index(row["case"]), methods.index(row["method"])] = row[
            "recovery_rate"
        ]
    fig, axis = plt.subplots(figsize=(8, max(3, len(cases) * 0.25)))
    image = axis.imshow(grid, aspect="auto", vmin=0, vmax=1)
    axis.set_xticks(range(len(methods)), methods)
    axis.set_yticks(range(len(cases)), cases, fontsize=7)
    fig.colorbar(image, ax=axis, label="Recovery / planned fits")
    fig.tight_layout()
    fig.savefig(output / "recovery.png", dpi=150)
    plt.close(fig)
