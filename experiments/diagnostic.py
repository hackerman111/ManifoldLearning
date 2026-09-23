"""Paired ADP parameter diagnostics with separate selection and validation seeds.

Run ``python -m experiments.diagnostic --dry-run`` before a full sweep.
This benchmark changes only explicit experimental configurations, never ADP defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ADP.cli.experiment_plots import _wilson_interval
from ADP.core.ADP_Config import ADP_Config

from .models import Build, Experiment, ExperimentPoint, ModelMode
from .runner import _experiment_id

SCENARIOS = ("base", "noise", "correlation", "scarce")
TUNED_FIELDS = (
    "N_loc",
    "N_phi",
    "N_J",
    "solver_max_steps",
    "lambda_penalty",
    "N_manifold",
    "lambda_manifold",
    "sync_steps",
)


@dataclass(frozen=True)
class Case:
    mode: ModelMode
    scenario: str
    experiment: Experiment
    candidates: tuple[str, ...]


def _base_point(mode: ModelMode) -> ExperimentPoint:
    if mode == "manifold":
        return ExperimentPoint(
            d=4,
            n_over_d=60,
            n_samples=240,
            mode=mode,
            link="manifold_radial",
            sigma_eps=0.05,
            N_loc=20,
            N_lin=60,
            N_J=24,
            N_phi=10,
            N_manifold=6,
            sync_steps=1,
            lambda_manifold=0.5,
            h_min_factor=10,
        )
    multi = mode == "multi"
    return ExperimentPoint(
        d=6,
        n_over_d=40,
        n_samples=240,
        mode=mode,
        index_dim=2 if multi else 1,
        link="multi_additive" if multi else "sin_scaled",
        link_scale=2,
        sigma_eps=0.2,
        N_loc=20,
        N_lin=30,
        N_J=24,
        N_phi=8,
        outer_steps=5,
        solver_max_steps=50,
        lambda_penalty=0.05,
        index_init="local",
        h_min_factor=3,
    )


def _scenario(point: ExperimentPoint, name: str) -> ExperimentPoint:
    if name == "noise":
        return replace(point, sigma_eps=0.5 if point.mode == "manifold" else 0.8)
    if name == "correlation":
        return replace(point, rho_corr=0.85)
    if name == "scarce":
        n, d = (100, 6) if point.mode == "manifold" else (120, 8)
        return replace(point, n_samples=n, d=d, n_over_d=n / d)
    return point


def make_case(mode: ModelMode, scenario: str) -> Case:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    base = _scenario(_base_point(mode), scenario)
    if mode == "manifold":
        changes = (
            ("N_loc=10", {"N_loc": 10}),
            ("N_loc=30", {"N_loc": 30}),
            ("N_phi=4", {"N_phi": 4}),
            ("N_phi=16", {"N_phi": 16}),
            ("N_manifold=3", {"N_manifold": 3}),
            ("N_manifold=10", {"N_manifold": 10}),
            ("lambda_manifold=0", {"lambda_manifold": 0.0}),
            ("lambda_manifold=2", {"lambda_manifold": 2.0}),
            ("sync_steps=3", {"sync_steps": 3}),
        )
    else:
        changes = (
            ("N_loc=10", {"N_loc": 10}),
            ("N_loc=30", {"N_loc": 30}),
            ("N_phi=4", {"N_phi": 4}),
            ("N_phi=16", {"N_phi": 16}),
            ("N_J=48", {"N_J": 48}),
            ("solver_max_steps=20", {"solver_max_steps": 20}),
            ("solver_max_steps=80", {"solver_max_steps": 80}),
            ("lambda_penalty=0", {"lambda_penalty": 0.0}),
            ("lambda_penalty=0.2", {"lambda_penalty": 0.2}),
        )
    points = (base, *(replace(base, **change) for _, change in changes))
    candidate_names = ("baseline", *(name for name, _ in changes))
    experiment = Experiment(
        selector=f"diagnostic-{mode}-{scenario}",
        title=f"{mode}: {scenario}, local parameter variants",
        smoke=base,
        full=points,
        report_fields=tuple(
            field for field in TUNED_FIELDS if getattr(base, field) is not None
        ),
        full_runs=12,
        quality_threshold={"single": 0.9, "multi": 0.95, "manifold": 0.2}[mode],
        common_random_fields=TUNED_FIELDS,
        hypothesis="Local one-factor changes can alter recovery and cost.",
    )
    return Case(mode, scenario, experiment, candidate_names)


def _number(value: str | None) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _median(rows: list[dict[str, str]], field: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(field))) is not None]
    return float(np.median(values)) if values else None


def _quantile(rows: list[dict[str, str]], field: str, q: float) -> float | None:
    values = [value for row in rows if (value := _number(row.get(field))) is not None]
    return float(np.quantile(values, q)) if values else None


def _summary(rows: list[dict[str, str]], *, expected: int) -> dict[str, Any]:
    if len(rows) != expected:
        raise ValueError(f"incomplete candidate: {len(rows)} of {expected} fits")
    outcomes = Counter(row.get("failure_mode") or "unknown" for row in rows)
    stops = Counter(row.get("stop_reason") or row["status"] for row in rows)
    recovered = outcomes["recovered"]
    low, high = _wilson_interval(recovered, len(rows))
    improvement = [
        float(row["quality"]) - float(row["initial_quality"])
        for row in rows
        if _number(row.get("quality")) is not None
        and _number(row.get("initial_quality")) is not None
    ]
    return {
        "fits": len(rows),
        "converged": sum(row.get("convergence_pass") == "True" for row in rows),
        "recovered": recovered,
        "recovery_rate": recovered / len(rows),
        "recovery_ci_low": max(0.0, low),
        "recovery_ci_high": min(1.0, high),
        "numerical_failures": outcomes["numerical_failure"],
        "nonconverged": outcomes["nonconverged"],
        "converged_bad_quality": outcomes["converged_bad_quality"],
        "quality_median": _median(rows, "quality"),
        "quality_q10": _quantile(rows, "quality", 0.1),
        "quality_count": sum(_number(row.get("quality")) is not None for row in rows),
        "initial_quality_median": _median(rows, "initial_quality"),
        "quality_gain_from_initial_median": (
            float(np.median(improvement)) if improvement else None
        ),
        "time_median_sec": _median(rows, "fit_time_sec"),
        "timed_count": sum(
            _number(row.get("fit_time_sec")) is not None for row in rows
        ),
        "traced_peak_median_mib": _median(rows, "max_stage_traced_peak_mib"),
        "memory_count": sum(
            _number(row.get("max_stage_traced_peak_mib")) is not None for row in rows
        ),
        "outer_iterations_median": _median(rows, "outer_iterations"),
        "stop_reasons": dict(stops),
        "errors": dict(Counter(row["error"] for row in rows if row.get("error"))),
    }


def _paired(
    rows: list[dict[str, str]], baseline: list[dict[str, str]], direction: str
) -> dict[str, Any]:
    base_by_seed = {int(row["run"]): row for row in baseline}
    recovery_deltas: list[int] = []
    quality_deltas: list[float] = []
    time_ratios: list[float] = []
    for row in rows:
        reference = base_by_seed[int(row["run"])]
        recovery_deltas.append(
            int(row.get("recovered") == "True")
            - int(reference.get("recovered") == "True")
        )
        quality = _number(row.get("quality"))
        base_quality = _number(reference.get("quality"))
        if quality is not None and base_quality is not None:
            quality_deltas.append(
                (quality - base_quality) * (1 if direction == "higher" else -1)
            )
        elapsed = _number(row.get("fit_time_sec"))
        base_elapsed = _number(reference.get("fit_time_sec"))
        if elapsed is not None and base_elapsed is not None and base_elapsed > 0:
            time_ratios.append(elapsed / base_elapsed)
    return {
        "paired_recovery_gain": sum(recovery_deltas) / len(recovery_deltas),
        "paired_quality_gain_median": (
            float(np.median(quality_deltas)) if quality_deltas else None
        ),
        "paired_quality_pairs": len(quality_deltas),
        "paired_time_ratio_median": (
            float(np.median(time_ratios)) if time_ratios else None
        ),
        "paired_time_pairs": len(time_ratios),
    }


def _select_candidate(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    """Prefer the incumbent when recovery and quality are practically tied."""
    recovered = max(int(row["recovered"]) for row in rows)
    eligible = [row for row in rows if row["recovered"] == recovered]
    failures = min(int(row["numerical_failures"]) for row in eligible)
    eligible = [row for row in eligible if row["numerical_failures"] == failures]

    def quality(row: dict[str, Any]) -> float:
        value = row["quality_median"]
        return (
            float(value) * (1 if direction == "higher" else -1)
            if value is not None
            else -math.inf
        )

    best = max(quality(row) for row in eligible)
    near = [row for row in eligible if quality(row) >= best - 0.01]
    baseline = next((row for row in near if row["point"] == 0), None)
    if baseline is not None:
        faster = [
            row
            for row in near
            if row["point"] != 0
            and row["paired_time_ratio_median"] is not None
            and row["paired_time_ratio_median"] <= 0.9
        ]
        if not faster:
            return baseline
        near = faster
    return min(
        near,
        key=lambda row: (
            float(row["time_median_sec"])
            if row["time_median_sec"] is not None
            else math.inf,
            int(row["point"]),
        ),
    )


def analyze_case(case: Case, series_dir: Path, runs: int, seed: int) -> dict[str, Any]:
    series = json.loads((series_dir / "series.json").read_text(encoding="utf-8"))
    if series["points"] != [asdict(point) for point in case.experiment.full]:
        raise ValueError("saved points differ from the diagnostic catalog")
    with (series_dir / "runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = runs * len(case.candidates)
    if len(rows) != expected:
        raise ValueError(f"incomplete series {series_dir}: {len(rows)} of {expected}")
    by_point: dict[int, list[dict[str, str]]] = {
        index: [] for index in range(len(case.candidates))
    }
    for row in rows:
        by_point[int(row["point"])].append(row)
    expected_runs = set(range(runs))
    if any(
        len(group) != runs or {int(row["run"]) for row in group} != expected_runs
        for group in by_point.values()
    ):
        raise ValueError("candidate run indices are incomplete or duplicated")
    seed_bundles = {
        run: {
            next(row["seed_bundle"] for row in group if int(row["run"]) == run)
            for group in by_point.values()
        }
        for run in range(runs)
    }
    if any(len(bundles) != 1 for bundles in seed_bundles.values()):
        raise ValueError("candidate seed bundles are not paired")
    split = max(1, runs // 2)
    direction = "lower" if case.mode == "manifold" else "higher"
    summaries: list[dict[str, Any]] = []
    for index, candidate in enumerate(case.candidates):
        for phase, indices in (
            ("selection", range(split)),
            ("validation", range(split, runs)),
        ):
            if not indices:
                continue
            selected = [row for row in by_point[index] if int(row["run"]) in indices]
            baseline = [row for row in by_point[0] if int(row["run"]) in indices]
            summaries.append(
                {
                    "mode": case.mode,
                    "scenario": case.scenario,
                    "candidate": candidate,
                    "point": index,
                    "phase": phase,
                    "quality_metric": by_point[index][0]["quality_metric"],
                    "quality_direction": direction,
                    "requested_config": by_point[index][0]["requested_config"],
                    "effective_config": next(
                        (
                            row["effective_config"]
                            for row in by_point[index]
                            if row["effective_config"]
                        ),
                        "",
                    ),
                    **_summary(selected, expected=len(indices)),
                    **_paired(selected, baseline, direction),
                }
            )
    selection = [row for row in summaries if row["phase"] == "selection"]
    winner = _select_candidate(selection, direction)
    validation = (
        next(
            row
            for row in summaries
            if row["phase"] == "validation" and row["candidate"] == winner["candidate"]
        )
        if runs > 1
        else None
    )
    baseline_validation = next(
        (
            row
            for row in summaries
            if row["phase"] == "validation" and row["point"] == 0
        ),
        None,
    )
    baseline_selection = next(row for row in selection if row["point"] == 0)
    validated = (
        validation is not None
        and baseline_validation is not None
        and validation["fits"] >= 5
        and winner["recovered"] > 0
        and validation["recovery_rate"] >= 0.8
        and validation["numerical_failures"] == 0
        and validation["recovery_rate"] >= baseline_validation["recovery_rate"]
        and (
            winner["point"] == 0
            or validation["recovery_rate"] > baseline_validation["recovery_rate"]
            or (
                validation["paired_quality_gain_median"] is not None
                and validation["paired_quality_gain_median"] >= 0.01
            )
            or (
                validation["paired_quality_gain_median"] is not None
                and validation["paired_quality_gain_median"] >= -0.01
                and validation["paired_time_ratio_median"] is not None
                and validation["paired_time_ratio_median"] <= 0.9
            )
        )
    )
    baseline_reliable = (
        baseline_validation is not None
        and baseline_validation["fits"] >= 5
        and baseline_selection["recovered"] > 0
        and baseline_validation["recovery_rate"] >= 0.8
        and baseline_validation["numerical_failures"] == 0
    )
    recommendation = (
        winner["candidate"] if validated else "baseline" if baseline_reliable else None
    )
    return {
        "mode": case.mode,
        "scenario": case.scenario,
        "series_dir": str(series_dir),
        "candidate_count": len(case.candidates),
        "selection_seeds": list(range(seed, seed + split)),
        "validation_seeds": list(range(seed + split, seed + runs)),
        "selected_candidate": winner["candidate"],
        "recommendation": recommendation,
        "recommendation_status": (
            "preliminary_validation_pass"
            if validated and winner["point"] != 0
            else "baseline_retained"
            if recommendation == "baseline"
            else "no_reliable_candidate"
        ),
        "selection_rule": (
            "max recovery count, min numerical failures; quality within 0.01 "
            "of best is equivalent; retain baseline unless paired time ratio "
            "is <=0.9, otherwise choose fastest; selection seeds only"
        ),
        "rows": summaries,
    }


def _write_report(root: Path, cases: list[dict[str, Any]], *, status: str) -> None:
    output = {
        "status": status,
        "analysis_code_sha256": _code_fingerprint(),
        "cases": cases,
    }
    (root / "diagnostics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [row for case in cases for row in case["rows"]]
    if rows:
        with (root / "diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {
                    **row,
                    "stop_reasons": json.dumps(row["stop_reasons"], ensure_ascii=False),
                    "errors": json.dumps(row["errors"], ensure_ascii=False),
                }
                for row in rows
            )
    lines = [
        "# ADP diagnostic benchmark",
        "",
        " ".join(
            (
                "Параметры сравниваются на одинаковых данных",
                "и random streams внутри сценария.",
            )
        ),
        "Победитель выбирается только по selection seed; validation не "
        "участвует в выборе.",
        "Отбор: максимум восстановлений, минимум численных ошибок; разницу "
        "качества до 0.01 считаем практическим равенством. При равенстве "
        "оставляем baseline, если парное ускорение меньше 10%.",
        "Проверка: не менее 5 validation seed, recovery >= 0.8, без численных "
        "ошибок. Новый вариант рекомендуем при лучшем recovery либо "
        "практическом выигрыше качества/времени относительно baseline.",
        "`baseline_retained` означает, что отобранный вариант не подтвердил "
        "преимущество или сам baseline был лучшим. Рекомендация предварительная.",
        "Время — fit wall-clock, включая ошибочные fits в новых прогонах; "
        "память — tracemalloc внутри успешного fit, не полный RSS. Число "
        "измеренных fits указано в diagnostics.csv.",
        "Quality сравнивается только внутри одного mode/scenario.",
        "",
    ]

    def fmt(value: object) -> str:
        if value is None:
            return "—"
        if not isinstance(value, (int, float)):
            raise TypeError("report metric must be numeric")
        return f"{value:.3g}"

    for case in cases:
        lines.extend(
            [
                f"## {case['mode']} / {case['scenario']}",
                "",
                "Рекомендация: "
                f"**{case['recommendation'] or 'нет надёжного варианта'}** "
                f"(`{case['recommendation_status']}`); выбран на selection: "
                f"`{case['selected_candidate']}`.",
                "",
                "| Вариант | Split | Recovery | Wilson 95% | Fail / Nonconv / "
                "Bad quality | Quality median | Time median, s | Peak, MiB | "
                "Δ recovery | Δ quality | Time ratio |",
                "|---|---|" + "---:|" * 9,
            ]
        )
        for row in case["rows"]:
            quality = row["quality_median"]
            quality_label = "—" if quality is None else f"{float(quality):.6f}"
            lines.append(
                f"| {row['candidate']} | {row['phase']} | "
                f"{row['recovered']}/{row['fits']} | "
                f"[{fmt(row['recovery_ci_low'])}, {fmt(row['recovery_ci_high'])}] | "
                f"{row['numerical_failures']} / {row['nonconverged']} / "
                f"{row['converged_bad_quality']} | "
                f"{quality_label} (n={row['quality_count']}) | "
                f"{fmt(row['time_median_sec'])} | "
                f"{fmt(row['traced_peak_median_mib'])} | "
                f"{fmt(row['paired_recovery_gain'])} | "
                f"{fmt(row['paired_quality_gain_median'])} | "
                f"{fmt(row['paired_time_ratio_median'])} |"
            )
        chosen_validation = next(
            (
                row
                for row in case["rows"]
                if row["candidate"] == case["selected_candidate"]
                and row["phase"] == "validation"
            ),
            None,
        )
        if chosen_validation is not None:
            stops = json.dumps(chosen_validation["stop_reasons"], ensure_ascii=False)
            errors = json.dumps(chosen_validation["errors"], ensure_ascii=False)
            lines.extend(
                [
                    "",
                    "Выбранный вариант на validation: "
                    "initial quality="
                    f"{fmt(chosen_validation['initial_quality_median'])}, "
                    "изменение от initial="
                    f"{fmt(chosen_validation['quality_gain_from_initial_median'])}, "
                    "outer iterations="
                    f"{fmt(chosen_validation['outer_iterations_median'])}.",
                    f"Остановки: `{stops}`; ошибки: `{errors}`.",
                ]
            )
        lines.extend(["", f"Подробности: `{case['series_dir']}`.", ""])
    (root / "diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(
        (
            *root.joinpath("ADP").rglob("*.py"),
            *root.joinpath("experiments").rglob("*.py"),
        )
    )
    for path in (*paths, root / "pyproject.toml"):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _reanalyze(root: Path) -> None:
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    cases = []
    for mode in manifest["modes"]:
        for scenario in manifest["scenarios"]:
            case = make_case(mode, scenario)
            if manifest["profile"] == "smoke":
                case = replace(
                    case,
                    experiment=replace(case.experiment, full=case.experiment.full[:2]),
                    candidates=case.candidates[:2],
                )
            series_dir = root / "series" / case.experiment.selector
            if not series_dir.exists():
                continue
            cases.append(
                analyze_case(case, series_dir, manifest["runs"], manifest["seed"])
            )
    _write_report(root, cases, status=manifest["status"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("single", "multi", "manifold", "all"), default="all"
    )
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analyze", type=Path, help="Пересобрать отчёт из готового run.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/diagnostic")
    )
    args = parser.parse_args(argv)
    if args.analyze is not None:
        _reanalyze(args.analyze)
        print(f"Report: {args.analyze / 'diagnostics.md'}")
        return 0
    if args.threads < 1 or args.seed < 0:
        parser.error("threads must be positive and seed nonnegative")
    runs = (
        args.runs if args.runs is not None else (2 if args.profile == "smoke" else 12)
    )
    if runs < 2:
        parser.error("at least two runs are needed for separate seed splits")
    modes: tuple[ModelMode, ...] = (
        ("single", "multi", "manifold") if args.mode == "all" else (args.mode,)
    )
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    cases = [make_case(mode, scenario) for mode in modes for scenario in scenarios]
    candidates = [
        len(case.candidates) if args.profile == "full" else 2 for case in cases
    ]
    for case, count in zip(cases, candidates, strict=True):
        print(
            f"{case.mode}/{case.scenario}: {count} variants x {runs} seeds "
            f"= {count * runs} fits"
        )
    print(f"Total: {sum(count * runs for count in candidates)} fits")
    if args.dry_run:
        return 0
    try:
        import threadpoolctl
    except ImportError as error:
        parser.error(f"benchmark dependency unavailable: {error}")
    root = args.output_dir / f"{_experiment_id()}-{args.mode}"
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "modes": modes,
        "scenarios": scenarios,
        "runs": runs,
        "seed": args.seed,
        "threads": args.threads,
        "seed_split": {
            "selection": list(range(max(1, runs // 2))),
            "validation": list(range(max(1, runs // 2), runs)),
        },
        "status": "running",
        "completed": [],
        "run_code_sha256": _code_fingerprint(),
    }
    report_cases: list[dict[str, Any]] = []
    failed = False
    try:
        with threadpoolctl.threadpool_limits(limits=args.threads):
            for case in cases:
                if args.profile == "smoke":
                    case = replace(
                        case,
                        experiment=replace(
                            case.experiment, full=case.experiment.full[:2]
                        ),
                        candidates=case.candidates[:2],
                    )
                build = Build(
                    "ADP",
                    ADP_Config(),
                    solver="cg" if case.mode == "manifold" else "lsmr",
                )
                series_dir = case.experiment.run(
                    build,
                    profile="full",
                    runs=runs,
                    seed=args.seed,
                    output_dir=root,
                    plots=False,
                    experiment_id="series",
                    progress=True,
                )
                if _code_fingerprint() != manifest["run_code_sha256"]:
                    raise RuntimeError("source changed during diagnostic benchmark")
                result = analyze_case(case, series_dir, runs, args.seed)
                report_cases.append(result)
                manifest["completed"].append(f"{case.mode}/{case.scenario}")
                failed |= any(row["numerical_failures"] for row in result["rows"])
                (root / "run.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _write_report(root, report_cases, status="running")
        manifest["status"] = "completed_with_failures" if failed else "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        return 130
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "error"
        (root / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_report(root, report_cases, status=str(manifest["status"]))
    print(f"Report: {root / 'diagnostics.md'}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
