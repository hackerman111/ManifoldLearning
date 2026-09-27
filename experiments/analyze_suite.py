"""Компактный JSON/Markdown-дайджест сохраненного ADP experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _boolean(value: object) -> bool | None:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    return None


def _quantiles(values: list[float]) -> dict[str, int | float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {"n": len(ordered), "q05": at(0.05), "median": at(0.5), "q95": at(0.95)}


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json_field(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _phase_digest(path: Path) -> dict[str, Any] | None:
    phase_path = path / "phase_summary.csv"
    if not phase_path.is_file():
        return None
    phase = _read_rows(phase_path)

    def compact(row: dict[str, str]) -> dict[str, Any]:
        return {
            "condition": {
                key: row.get(key)
                for key in (
                    "condition_field",
                    "condition_level",
                    "d",
                    "n_over_d",
                    "build",
                )
                if row.get(key)
            },
            "n": int(row["n_total"]),
            "n_converged": int(row["n_converged"]),
            "n_recovered": int(row["n_recovered"]),
            "convergence_rate": _number(row.get("convergence_rate")),
            "quality_pass_rate": _number(row.get("quality_pass_rate")),
            "recovery_rate": _number(row.get("recovery_rate")),
            "recovery_ci95_wilson": [
                _number(row.get("recovery_ci_low")),
                _number(row.get("recovery_ci_high")),
            ],
            "quality_median": _number(row.get("quality_median")),
        }

    by_recovery = sorted(
        phase,
        key=lambda row: (_number(row.get("recovery_rate")) or 0.0, int(row["n_total"])),
        reverse=True,
    )
    boundaries = (
        _read_rows(path / "boundary.csv") if (path / "boundary.csv").is_file() else []
    )
    return {
        "condition_field": phase[0].get("condition_field") if phase else None,
        "top_recovery_conditions": [compact(row) for row in by_recovery[:3]],
        "worst_recovery_conditions": [
            compact(row) for row in list(reversed(by_recovery[-3:]))
        ],
        "boundary_count": len(boundaries),
        "boundaries": [
            {**compact(row), "boundary_kind": row.get("boundary_kind")}
            for row in boundaries[:10]
        ],
    }


def _summarize(rows: list[dict[str, str]], recovery: dict[str, Any]) -> dict[str, Any]:
    metric = str(recovery["metric"])
    threshold = float(recovery["threshold"])
    direction = str(recovery["direction"])
    statuses: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    values: dict[str, list[float]] = {
        "quality": [],
        "quality_converged": [],
        "prediction_rmse": [],
        "fit_time_sec": [],
        "max_stage_traced_peak_mib": [],
        "outer_iterations": [],
        "initial_quality": [],
        "last_quality": [],
        "selected_error": [],
        "solver_loss": [],
        "manifold_trace_objective": [],
        "manifold_trace_penalty": [],
    }
    converged = quality_passed = recovered = quality_available = 0
    converged_quality_available = converged_quality_passed = 0
    convergence_unknown = 0

    for row in rows:
        status = row.get("status") or "unknown"
        statuses[status] += 1
        quality = _number(row.get(metric, row.get("quality")))
        conv = _boolean(row.get("convergence_pass"))
        if conv is None:
            conv = status == "success" if status != "numerical_failure" else False
        if status == "numerical_failure":
            conv = False
        else:
            convergence_unknown += _boolean(row.get("convergence_pass")) is None

        pass_value = _boolean(row.get("quality_pass"))
        if quality is not None:
            quality_available += 1
            values["quality"].append(quality)
            if pass_value is None:
                pass_value = (
                    quality >= threshold
                    if direction == "higher"
                    else quality <= threshold
                )
            quality_passed += pass_value is True
            if conv:
                converged_quality_available += 1
                converged_quality_passed += pass_value is True
                values["quality_converged"].append(quality)
        whole_manifold_pass = None
        if metric == "local_projector_distance":
            whole_manifold_pass = _boolean(row.get("whole_manifold_pass"))
            if whole_manifold_pass is None:
                whole_distance = _number(
                    row.get(
                        "whole_manifold_max_local_projector_distance",
                        row.get("max_principal_sine"),
                    )
                )
                whole_manifold_pass = (
                    whole_distance <= threshold if whole_distance is not None else False
                )
        recovered_value = (
            status != "numerical_failure"
            and pass_value is True
            and (metric != "local_projector_distance" or whole_manifold_pass is True)
        )
        recovered += recovered_value is True
        converged += conv is True

        for field in values.keys() - {
            "quality",
            "quality_converged",
            "solver_loss",
            "manifold_trace_objective",
            "manifold_trace_penalty",
        }:
            if (value := _number(row.get(field))) is not None:
                values[field].append(value)

        diagnostics = _json_field(row.get("solver_diagnostics"))
        solver_loss = None
        if isinstance(diagnostics, dict):
            solver_loss = _number(diagnostics.get("loss"))
            if solver_loss is None and isinstance(
                diagnostics.get("loss_history"), list
            ):
                solver_loss = next(
                    (
                        value
                        for raw in reversed(diagnostics["loss_history"])
                        if (value := _number(raw)) is not None
                    ),
                    None,
                )
        if solver_loss is not None:
            values["solver_loss"].append(solver_loss)

        trace = _json_field(row.get("trace"))
        if isinstance(trace, list):
            for trace_field, metric_field in (
                ("objective", "manifold_trace_objective"),
                ("manifold_penalty", "manifold_trace_penalty"),
            ):
                value = next(
                    (
                        parsed
                        for entry in reversed(trace)
                        if isinstance(entry, dict)
                        and (parsed := _number(entry.get(trace_field))) is not None
                    ),
                    None,
                )
                if value is not None:
                    values[metric_field].append(value)
        stops[row.get("stop_reason") or status] += 1
        failure_modes[row.get("failure_mode") or status] += 1
        if error := row.get("error"):
            errors[error] += 1

    total = len(rows)
    n_quality_pass = min(quality_passed, quality_available)
    return {
        "n_fits": total,
        "status_counts": dict(sorted(statuses.items())),
        "convergence": {
            "n": converged,
            "rate": converged / total if total else None,
            "ci95_wilson": _wilson(converged, total),
            "unclassified": convergence_unknown,
        },
        "quality_pass": {
            "n": n_quality_pass,
            "n_quality_available": quality_available,
            "rate_given_quality": n_quality_pass / quality_available
            if quality_available
            else None,
            "ci95_wilson": _wilson(n_quality_pass, quality_available),
        },
        "quality_among_converged": {
            "n_quality_available": converged_quality_available,
            "n_pass": converged_quality_passed,
            "n_bad_quality": converged_quality_available - converged_quality_passed,
        },
        "recovery": {
            "n": recovered,
            "rate": recovered / total if total else None,
            "ci95_wilson": _wilson(recovered, total),
        },
        "quality_metric": metric,
        "quality_direction": direction,
        "quality_threshold": threshold,
        "quantiles": {name: _quantiles(items) for name, items in values.items()},
        "stop_reasons": dict(sorted(stops.items())),
        "failure_modes": dict(sorted(failure_modes.items())),
        "errors": dict(
            sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:5]
        ),
    }


def analyze_suite(root: Path) -> dict[str, Any]:
    root = root.resolve()
    suite_path = root / "suite.json"
    if not suite_path.is_file():
        raise FileNotFoundError(f"suite.json not found in {root}")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    planned = suite.get("planned", [])
    completed_names = set(suite.get("completed", []))
    series: list[dict[str, Any]] = []
    completed_rows: list[dict[str, str]] = []
    criteria: set[tuple[str, str, float]] = set()
    mismatches: list[str] = []

    for item in planned:
        selector = str(item["selector"])
        directory = selector.replace(".", "_")
        path = root / directory
        manifest_path, runs_path = path / "series.json", path / "runs.csv"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        rows = _read_rows(runs_path) if runs_path.is_file() else []
        expected = int(item["fits"])
        marked_complete = selector in completed_names
        if marked_complete and len(rows) == expected:
            state = "complete"
        elif marked_complete:
            state = "completed_count_mismatch"
            mismatches.append(selector)
        elif not path.exists():
            state = "not_started"
        elif rows:
            state = "partial"
        else:
            state = "started_no_rows"

        entry: dict[str, Any] = {
            "selector": selector,
            "directory": directory,
            "state": state,
            "planned_fits": expected,
            "observed_fits": len(rows),
            "runner_marked_complete": marked_complete,
            "title": manifest.get("title", selector) if manifest else selector,
            "details": [
                f"{directory}/{name}"
                for name in (
                    "summary.csv",
                    "phase_summary.csv",
                    "boundary.csv",
                    "trace_summary.csv",
                    "failures.csv",
                )
                if (path / name).is_file()
            ],
        }
        if manifest is not None:
            recovery = manifest["recovery"]
            entry["provenance"] = {
                "git_commit": manifest.get("git_commit"),
                "git_status": manifest.get("git_status"),
                "seed": manifest.get("seed"),
                "seed_design": manifest.get("seed_design"),
                "dtype": manifest.get("dtype"),
                "python": manifest.get("python"),
                "packages": manifest.get("packages"),
            }
            entry["metrics"] = _summarize(rows, recovery)
            if state == "complete":
                criteria.add(
                    (
                        str(recovery["metric"]),
                        str(recovery["direction"]),
                        float(recovery["threshold"]),
                    )
                )
        else:
            entry["metrics"] = None
        entry["phase"] = _phase_digest(path)
        if state == "complete":
            completed_rows.extend(rows)
        series.append(entry)

    total_fits = len(completed_rows)
    totals: dict[str, Any] = {
        "completed_series": sum(item["state"] == "complete" for item in series),
        "planned_series": len(planned),
        "planned_fits": sum(int(item["fits"]) for item in planned),
        "completed_expected_fits": sum(
            item["planned_fits"] for item in series if item["state"] == "complete"
        ),
        "completed_observed_fits": total_fits,
        "partial_observed_fits": sum(
            item["observed_fits"] for item in series if item["state"] != "complete"
        ),
        "unstarted_series": [
            item["selector"] for item in series if item["state"] == "not_started"
        ],
        "incomplete_series": [
            item["selector"]
            for item in series
            if item["state"] not in {"complete", "not_started"}
        ],
        "completed_coverage_rate": (
            sum(item["planned_fits"] for item in series if item["state"] == "complete")
            / sum(int(item["fits"]) for item in planned)
            if planned
            else None
        ),
    }
    if criteria:
        # Pool quality values only when every series uses the same metric and rule.
        metric, direction, threshold = sorted(criteria)[0]
        totals["metrics"] = _summarize(
            completed_rows,
            {"metric": metric, "direction": direction, "threshold": threshold},
        )
        if len(criteria) > 1:
            totals["metrics"]["quantiles"]["quality"] = None
            totals["metrics"]["quantiles"]["quality_converged"] = None
            totals["metrics"]["quality_pass"] = None
            totals["metrics"]["recovery"] = None
            totals["metrics"]["quality_metric"] = None
            totals["metrics"]["quality_direction"] = None
            totals["metrics"]["quality_threshold"] = None
            totals["metrics"]["quality_pooling_note"] = (
                "Suppressed: series use different quality metrics or thresholds."
            )
    else:
        totals["metrics"] = None

    provenance_fields = ("git_commit", "git_status", "dtype", "python", "seed_design")
    provenance = {
        field: sorted(
            {
                item["provenance"].get(field)
                for item in series
                if item.get("state") == "complete"
                and item.get("provenance", {}).get(field) is not None
            }
        )
        for field in provenance_fields
    }

    return {
        "schema_version": 1,
        "suite_id": root.name,
        "suite_status": suite.get("status", "unknown"),
        "mode": suite.get("mode"),
        "profile": suite.get("profile"),
        "seed": suite.get("seed"),
        "threads": suite.get("threads"),
        "elapsed_sec": suite.get("elapsed_sec"),
        "metric_definitions": {
            "selected_error": (
                "sum((Y[center_indices] - S)^2) at the selected outer step; "
                "not generally held-out prediction loss"
                if suite.get("mode") in {"single", "multi"}
                else "sum((Y - predict(X))^2) on the generated sample; in-sample SSE"
            ),
            "solver_loss": (
                "HPAO inner solver objective from selected solver diagnostics; "
                "not the outer fit-selection error"
            ),
            "manifold_trace_objective": (
                "last recorded manifold trace objective; kept separate from "
                "the manifold penalty and from HPAO solver loss"
            ),
            "manifold_trace_penalty": (
                "last recorded manifold trace penalty; kept separate from "
                "the data objective"
            ),
        },
        "provenance": provenance,
        "totals": totals,
        "count_mismatches": mismatches,
        "series": series,
    }


def _fmt_rate(rate: float | None, interval: list[float] | None = None) -> str:
    if rate is None:
        return "—"
    value = f"{100 * rate:.1f}%"
    if interval is not None:
        value += f" [{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]"
    return value


def _median_text(quantiles: dict[str, dict[str, float] | None], field: str) -> str:
    values = quantiles.get(field)
    return f"{values['median']:.4g}" if values else "—"


def _quantile_range_text(
    quantiles: dict[str, dict[str, float] | None], field: str
) -> str:
    values = quantiles.get(field)
    if values is None:
        return "—"
    return f"{values['median']:.4g} [{values['q05']:.4g}, {values['q95']:.4g}]"


def _median_p95_text(quantiles: dict[str, dict[str, float] | None], field: str) -> str:
    values = quantiles.get(field)
    if values is None:
        return "—"
    return f"{values['median']:.4g}/{values['q95']:.4g}"


def render_markdown(data: dict[str, Any]) -> str:
    totals = data["totals"]
    overall = totals["metrics"]
    coverage = totals["completed_coverage_rate"] or 0
    lines = [
        f"# ADP experiment digest: {data['suite_id']}",
        "",
        f"Suite: **{data['mode']} / {data['profile']} / {data['suite_status']}**.",
        f"Серии: {totals['completed_series']}/{totals['planned_series']} завершены.",
        f"Fits: {totals['completed_observed_fits']}/{totals['planned_fits']} "
        f"({100 * coverage:.1f}% полного покрытия плана).",
        f"Partial series contain {totals['partial_observed_fits']} observed fits; "
        f"not started: {', '.join(totals['unstarted_series']) or 'none'}.",
        "",
        "Recovery требует сходимости и прохождения порога.",
        "Quality rate условен на конечном качестве; CI — Wilson 95%.",
        "`selected_error`, HPAO `solver_loss`, and manifold trace objective are "
        "different quantities and are reported separately.",
        "Memory is `tracemalloc` inside fit (not RSS). "
        "Fit time excludes data generation and reports.",
        "Overall rates use complete series only; incomplete rows are shown separately.",
        "",
    ]
    if overall:
        lines.extend(["## Завершенные серии (сводно)", ""])
        conv = overall["convergence"]
        lines.append(
            f"- Сходимость: {conv['n']}/{overall['n_fits']} "
            f"({_fmt_rate(conv['rate'], conv['ci95_wilson'])})."
        )
        quality_pass = overall.get("quality_pass")
        if quality_pass is not None:
            quality_rate = _fmt_rate(
                quality_pass["rate_given_quality"], quality_pass["ci95_wilson"]
            )
            lines.append(
                f"- Quality `{overall['quality_metric']} "
                f"{overall['quality_direction']} "
                f"{overall['quality_threshold']}`: {quality_pass['n']}/"
                f"{quality_pass['n_quality_available']} ({quality_rate})."
            )
        recovery = overall.get("recovery")
        if recovery is not None:
            lines.append(
                f"- Recovery: {recovery['n']}/{overall['n_fits']} "
                f"({_fmt_rate(recovery['rate'], recovery['ci95_wilson'])})."
            )
        among = overall["quality_among_converged"]
        if among["n_quality_available"]:
            lines.append(
                f"- Quality pass among converged: {among['n_pass']}/"
                f"{among['n_quality_available']}."
            )
        q_median = (overall["quantiles"].get("quality") or {}).get("median")
        if q_median is None:
            lines.append("- Общая медиана quality скрыта: метрики/пороги различаются.")
        else:
            lines.append(
                f"- Quality median [q05, q95]: "
                f"{_quantile_range_text(overall['quantiles'], 'quality')}."
            )
        initial_quality = _median_text(overall["quantiles"], "initial_quality")
        last_quality = _median_text(overall["quantiles"], "last_quality")
        lines.append(
            f"- Median quality initial → last: {initial_quality} → {last_quality}; "
            f"median prediction RMSE: "
            f"{_median_text(overall['quantiles'], 'prediction_rmse')}."
        )
        lines.append(
            f"- Selected error median [q05, q95]: "
            f"{_quantile_range_text(overall['quantiles'], 'selected_error')}; "
            f"HPAO solver loss median: "
            f"{_median_text(overall['quantiles'], 'solver_loss')}; "
            f"manifold trace objective/penalty median: "
            f"{_median_text(overall['quantiles'], 'manifold_trace_objective')}/"
            f"{_median_text(overall['quantiles'], 'manifold_trace_penalty')}."
        )
        lines.append(
            f"- Fit time median/p95 "
            f"{_median_p95_text(overall['quantiles'], 'fit_time_sec')} s; "
            f"traced peak median/p95 "
            f"{_median_p95_text(overall['quantiles'], 'max_stage_traced_peak_mib')} "
            f"MiB; median outer iterations "
            f"{_median_text(overall['quantiles'], 'outer_iterations')}."
        )
        errors = json.dumps(overall["errors"], ensure_ascii=False)
        stops = json.dumps(overall["stop_reasons"], ensure_ascii=False)
        failures = overall["status_counts"].get("numerical_failure", 0)
        lines.append(
            f"- Numerical failures: {failures}; errors {errors}; stops `{stops}`."
        )
        lines.extend(
            [
                "",
                "Overall medians describe a mixture of catalog conditions, "
                "not one setup.",
                "",
            ]
        )

    lines.extend(
        [
            "## По сериям",
            "",
            "| Series | State | Fits | Convergence | Quality pass | Recovery | "
            "Median quality | RMSE | Selected error | HPAO solver loss | "
            "Manifold objective | Manifold penalty | Outer iterations | "
            "Best condition | "
            "Time, s | Memory, MiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in data["series"]:
        metrics = item["metrics"]
        if metrics is None:
            lines.append(
                f"| {item['selector']} | {item['state']} | 0/{item['planned_fits']} "
                "| — | — | — | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        quantiles = metrics["quantiles"]
        phases = item.get("phase")
        best = (phases or {}).get("top_recovery_conditions", [])
        if phases is None:
            best_condition = "—"
        elif not best or not best[0]["recovery_rate"]:
            best_condition = "no recoveries"
        else:
            condition = best[0]["condition"]
            field = condition.get("condition_field")
            level = condition.get("condition_level")
            best_condition = f"{field}={level} ({_fmt_rate(best[0]['recovery_rate'])})"
        lines.append(
            f"| {item['selector']} | {item['state']} "
            f"| {item['observed_fits']}/{item['planned_fits']} "
            f"| {_fmt_rate(metrics['convergence']['rate'])} "
            f"| {_fmt_rate(metrics['quality_pass']['rate_given_quality'])} "
            f"| {_fmt_rate(metrics['recovery']['rate'])} "
            f"| {_median_text(quantiles, 'quality')} "
            f"| {_median_text(quantiles, 'prediction_rmse')} "
            f"| {_median_text(quantiles, 'selected_error')} "
            f"| {_median_text(quantiles, 'solver_loss')} "
            f"| {_median_text(quantiles, 'manifold_trace_objective')} "
            f"| {_median_text(quantiles, 'manifold_trace_penalty')} "
            f"| {_median_text(quantiles, 'outer_iterations')} | {best_condition} "
            f"| {_median_text(quantiles, 'fit_time_sec')} "
            f"| {_median_text(quantiles, 'max_stage_traced_peak_mib')} |"
        )
    boundary_count = sum(
        (item.get("phase") or {}).get("boundary_count", 0) for item in data["series"]
    )
    lines.extend(
        [
            "",
            f"Phase boundary groups: {boundary_count}. Descriptive conditions, "
            "not parameter recommendations.",
            "",
            "See JSON `details` for factor, phase, trace, and failure tables.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="Path to directory with suite.json")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args()
    try:
        result = analyze_suite(args.suite)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.format == "md":
        print(render_markdown(result), end="")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
