"""CLI трёх исследовательских наборов и учёт общего объёма работы."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from time import perf_counter

from ADP.cli.experiment_utils import validate_run_parameters
from ADP.core.ADP_Config import ADP_Config

from . import manifold, multi, single
from .models import Build, Experiment, ExperimentPoint, ModelMode
from .registry import CATALOG, _selected_experiments
from .runner import _experiment_id, _has_numerical_failures
from .suite_report import summarize_series, write_suite_report


@dataclass(frozen=True, slots=True)
class PlannedSeries:
    selector: str
    points: int
    runs: int
    fits: int
    quality_threshold: float | None


def select_suite(mode: ModelMode, selector: str | None) -> tuple[Experiment, ...]:
    """Выбрать семейство и явно задать критерий восстановления для всех серий."""
    module = {"single": single, "multi": multi, "manifold": manifold}[mode]
    selected = _selected_experiments(
        selector or ",".join(module.DEFAULT_SELECTORS), custom=CATALOG["custom"]
    )
    if any(point.mode != mode for item in selected for point in item.full):
        raise ValueError(f"--experiment должен содержать только режим {mode}")
    threshold = {"single": 0.9, "multi": 0.95, "manifold": 0.2}[mode]
    result = []
    for item in selected:
        factors = item.report_fields or tuple(
            field.name
            for field in fields(ExperimentPoint)
            if field.name not in {"d", "n_over_d"}
            and len({getattr(point, field.name) for point in item.full}) > 1
        )
        condition = item.condition_field or (factors[0] if factors else None)
        # Фазовые вероятности сравниваем при фиксированных остальных факторах.
        groups = (
            tuple(
                dict.fromkeys(
                    (
                        *item.condition_group_fields,
                        *(
                            name
                            for name in factors
                            if name not in {condition, "d", "n_over_d"}
                        ),
                    )
                )
            )
            if condition is not None
            else ()
        )
        result.append(
            replace(
                item,
                quality_threshold=item.quality_threshold
                if item.quality_threshold is not None
                else threshold,
                condition_field=condition,
                condition_group_fields=groups,
            )
        )
    return tuple(result)


def main(mode: ModelMode, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"ADP {mode}: исследовательский набор")
    parser.add_argument(
        "--profile", choices=("smoke", "overview", "full"), default="overview"
    )
    parser.add_argument(
        "--experiment", help="Селекторы через запятую; --list показывает семейство"
    )
    parser.add_argument(
        "--runs", type=int, help="Повторы: smoke=1, overview=5, full=по каталогу"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="Число потоков BLAS")
    parser.add_argument(
        "--solver",
        choices=("lsmr", "cg") if mode == "single" else ("lsmr", "cg", "hybrid"),
        default="lsmr",
    )
    parser.add_argument("--solver-max-steps", type=int, default=3)
    parser.add_argument("--cg-maxiter", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/experiments")
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать сетки и число fits без вычислений",
    )
    args = parser.parse_args(argv)
    if args.list:
        for item in CATALOG.values():
            if item.smoke.mode == mode:
                print(
                    f"{item.selector:>24}  points={len(item.full):>5} "
                    f"runs={item.full_runs:>3}  {item.title}"
                )
        return 0
    try:
        experiments = select_suite(mode, args.experiment)
        if args.threads < 1:
            raise ValueError("--threads должен быть положительным")
        build = Build(
            "ADP",
            ADP_Config(),
            solver=args.solver,
            solver_max_steps=args.solver_max_steps,
            cg_maxiter=args.cg_maxiter,
        )
        planned: list[PlannedSeries] = []
        for item in experiments:
            runs = (
                args.runs
                if args.runs is not None
                else (
                    1
                    if args.profile == "smoke"
                    else 5
                    if args.profile == "overview"
                    else item.full_runs
                )
            )
            validate_run_parameters(build.name, None, runs, args.seed)
            count = len(item.points(args.profile))
            planned.append(
                PlannedSeries(
                    item.selector, count, runs, count * runs, item.quality_threshold
                )
            )
            print(
                f"{item.selector}: {count} points x {runs} runs = {count * runs} fits"
            )
        total = sum(item.fits for item in planned)
        print(f"Итого: {len(experiments)} серий, {total} fits; profile={args.profile}")
        if args.dry_run:
            return 0
        import threadpoolctl
    except (ImportError, TypeError, ValueError) as error:
        parser.error(str(error))

    experiment_id = f"{_experiment_id()}-{mode}"
    root = args.output_dir / experiment_id
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "suite_schema_version": 1,
        "mode": mode,
        "profile": args.profile,
        "seed": args.seed,
        "threads": args.threads,
        "planned": [asdict(item) for item in planned],
        "completed": [],
        "status": "running",
    }
    rows: list[dict[str, object]] = []
    failed = False
    started = perf_counter()

    def save() -> None:
        manifest["elapsed_sec"] = perf_counter() - started
        (root / "suite.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_suite_report(root, rows, mode=mode, profile=args.profile)

    save()
    try:
        with threadpoolctl.threadpool_limits(limits=args.threads):
            for item, spec in zip(experiments, planned, strict=True):
                path = item.run(
                    build,
                    profile=args.profile,
                    runs=spec.runs,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    plots=not args.no_plots,
                    experiment_id=experiment_id,
                    progress=True,
                )
                rows.append(summarize_series(path))
                failed = _has_numerical_failures(path) or failed
                manifest["completed"] = [row["experiment"] for row in rows]
                save()
        manifest["status"] = "completed_with_failures" if failed else "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        print(f"Прервано. Сохранённые результаты: {root}")
        return 130
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "error"
        save()
    print(f"Общий отчёт: {root / 'overview.md'}")
    return int(failed)
