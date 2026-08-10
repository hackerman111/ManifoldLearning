import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    load_experiment,
    run_experiment,
    run_experiment_terminal,
)
from ADP.ADP_Config import epanechnikov
from ADP.engine.logger import format_profile
from ADP.experiment import validate_experiment
from ADP.experiment_runner import _build_jobs


def parse_kernel(value: str):
    if value == "epanechnikov":
        return epanechnikov
    try:
        module_name, function_name = value.rsplit(":", 1)
        kernel = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "kernel должен быть 'epanechnikov' или 'module:function'"
        ) from error
    if not callable(kernel):
        raise argparse.ArgumentTypeError("kernel должен быть функцией")
    return kernel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADP single/multi-index experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--index-dim", type=int, default=1)
    parser.add_argument(
        "--solver", choices=("auto", "lsmr", "varpro"), default="auto"
    )
    parser.add_argument("--solver-tol", type=float)
    parser.add_argument("--solver-max-steps", type=int)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--experiment-file", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ADP/experiment_outputs")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reports-only", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--terminal-only", action="store_true")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--N_loc", "--n-loc", dest="N_loc", type=int, default=10)
    parser.add_argument("--N_lin", "--n-lin", dest="N_lin", type=int)
    parser.add_argument("--N_J", "--J", dest="N_J", type=int, default=64)
    parser.add_argument("--N_phi", "--n-phi", dest="N_phi", type=int)
    parser.add_argument(
        "--outer_steps", "--outer-steps", dest="outer_steps", type=int
    )
    parser.add_argument(
        "--lambda_penalty",
        "--lambda-penalty",
        dest="lambda_penalty",
        type=float,
        default=argparse.SUPPRESS,
        help="штраф LSMR (по умолчанию 10000 для multi, 100 для single)",
    )
    parser.add_argument(
        "--local_ridge",
        "--local-ridge",
        dest="local_ridge",
        type=float,
        default=1e-8,
    )
    parser.add_argument("--kernel", type=parse_kernel, default="epanechnikov")
    parser.add_argument("--a", type=float, default=np.sqrt(2))
    parser.add_argument("--h_min", "--h-min", dest="h_min", type=float)
    parser.add_argument(
        "--batch_size", "--batch-size", dest="batch_size", type=int, default=32
    )
    parser.add_argument(
        "--index_init",
        "--index-init",
        dest="index_init",
        choices=("local", "pilot", "random"),
        default=argparse.SUPPRESS,
        help="инициализация индекса (по умолчанию pilot для multi, local для single)",
    )
    return parser


def experiment_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> ADP_Experiment:
    settings = {}
    if args.solver_tol is not None:
        settings["tol"] = args.solver_tol
    if args.solver_max_steps is not None:
        settings["max_steps"] = args.solver_max_steps
    lambda_penalty = getattr(args, "lambda_penalty", None)
    index_init = getattr(args, "index_init", None)
    try:
        config = ADP_Config(
            seed=args.seed,
            N_loc=args.N_loc,
            N_lin=args.N_lin,
            N_J=args.N_J,
            N_phi=args.N_phi,
            outer_steps=args.outer_steps,
            lambda_penalty=(
                lambda_penalty
                if lambda_penalty is not None
                else (10000.0 if args.mode == "multi" else 100.0)
            ),
            local_ridge=args.local_ridge,
            kernel=args.kernel,
            a=args.a,
            h_min=args.h_min,
            batch_size=args.batch_size,
            index_init=index_init or ("pilot" if args.mode == "multi" else "local"),
        )
        return validate_experiment(
            ADP_Experiment(
                name="manual",
                mode=args.mode,
                index_dim=args.index_dim,
                runs=args.runs,
                seed=args.seed,
                points=(ADP_ExperimentPoint("manual", args.n, args.d, args.noise),),
                variants={
                    "default": ADP_ExperimentVariant(config, args.solver, settings)
                },
            )
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def _finite_values(rows, name):
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _median_text(rows, name, spec):
    values = _finite_values(rows, name)
    return format(median(values), spec) if values else "—"


def _aggregate_profile(rows):
    profiles = [
        row.get("profile_stages", {})
        for row in rows
        if isinstance(row.get("profile_stages"), dict)
    ]
    available = list(
        dict.fromkeys(name for profile in profiles for name in profile)
    )
    preferred = ("initialization", "directions", "statistics", "solver", "update")
    names = [name for name in preferred if name in available]
    names.extend(name for name in available if name not in preferred)
    stages = {}
    for name in names:
        values = [
            profile[name]
            for profile in profiles
            if isinstance(profile.get(name), dict)
        ]
        times = _finite_values(values, "time_seconds")
        memories = _finite_values(values, "memory_bytes")
        if times or memories:
            stages[name] = {
                "time_seconds": median(times) if times else 0.0,
                "memory_bytes": median(memories) if memories else 0.0,
            }
    if not stages:
        return None
    total_stage_time = sum(stage["time_seconds"] for stage in stages.values())
    total_stage_memory = sum(stage["memory_bytes"] for stage in stages.values())
    for stage in stages.values():
        stage["time_fraction"] = (
            stage["time_seconds"] / total_stage_time if total_stage_time else 0.0
        )
        stage["memory_fraction"] = (
            stage["memory_bytes"] / total_stage_memory
            if total_stage_memory
            else 0.0
        )
    totals = _finite_values(rows, "algorithm_time_sec")
    peaks = _finite_values(rows, "tracemalloc_peak_mib")
    return {
        "stages": stages,
        "total_time_seconds": median(totals) if totals else total_stage_time,
        "max_memory": max(stage["memory_bytes"] for stage in stages.values()),
        "peak_memory_bytes": median(peaks) * 2**20 if peaks else 0.0,
    }


def _print_terminal_summary(experiment, runs):
    grouped = {}
    for run in runs:
        grouped.setdefault((run["point"], run["variant"]), []).append(run)

    print(
        f"Эксперимент: {experiment.name} | "
        f"режим: {experiment.mode} | jobs: {len(runs)}"
    )
    for point in experiment.points:
        for variant in experiment.variants:
            rows = grouped.get((point.name, variant), [])
            statuses = [row.get("status") for row in rows]
            print(f"\n=== {point.name} / {variant} ===")
            print(f"запусков: {len(rows)}")
            print(
                "статусы: "
                f"success={statuses.count('success')}; "
                f"nonconverged={statuses.count('nonconverged')}; "
                f"numerical_failure={statuses.count('numerical_failure')}"
            )
            print(
                "инициализация индекса: "
                f"{experiment.variants[variant].config.index_init}"
            )
            if experiment.mode == "single":
                print(
                    "косинус в начале, медиана: "
                    f"{_median_text(rows, 'cosine_initial', '.6f')}"
                )
                print(
                    "косинус в конце, медиана: "
                    f"{_median_text(rows, 'cosine_abs', '.6f')}"
                )
            else:
                print(
                    "ошибка проектора в начале, медиана: "
                    f"{_median_text(rows, 'projector_error_initial', '.6f')}"
                )
                print(
                    "ошибка проектора в конце, медиана: "
                    f"{_median_text(rows, 'projector_error', '.6f')}"
                )
            print(
                "количество центров N_J: "
                f"{_median_text(rows, 'effective_N_J', 'g')}"
            )
            print(
                "количество итераций, медиана: "
                f"{_median_text(rows, 'outer_iterations', 'g')}"
            )
            reasons = Counter(
                str(row["stop_reason"])
                for row in rows
                if row.get("stop_reason")
            )
            print(
                "причины остановки: "
                + (
                    "; ".join(f"{reason}={count}" for reason, count in reasons.items())
                    if reasons
                    else "—"
                )
            )
            issues = Counter(
                (
                    str(row.get("status") or "unknown"),
                    str(row.get("error_type") or ""),
                    str(row.get("error_message") or ""),
                )
                for row in rows
                if row.get("status") != "success"
            )
            if issues:
                print("неуспешные запуски:")
                for (status, error_type, message), count in issues.items():
                    detail = ": ".join(
                        value for value in (error_type, message) if value
                    )
                    suffix = f": {detail}" if detail else ""
                    print(f"  {status} ({count}){suffix}")
            profile = _aggregate_profile(rows)
            if profile is None:
                print("\nпрофиль: —")
                continue
            print()
            print("\n".join(format_profile(profile).splitlines()[:-1]))
            total = _median_text(rows, "algorithm_time_sec", ".6f")
            peak = _median_text(rows, "tracemalloc_peak_mib", ".4f")
            rss = _finite_values(rows, "algorithm_rss_max_mib")
            rss_text = f"{max(rss):.1f}" if rss else "—"
            print(
                f"итого, медиана: {total} с; пик fit: {peak} MiB; "
                f"RSS max: {rss_text} MiB"
            )


def main(argv=None) -> int:
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    dont_write_bytecode = sys.dont_write_bytecode
    try:
        if "--terminal-only" in arguments:
            sys.dont_write_bytecode = True
        return _main(parser, parser.parse_args(arguments))
    finally:
        sys.dont_write_bytecode = dont_write_bytecode


def _main(parser, args) -> int:
    if args.reports_only is not None and any(
        (
            args.experiment_file is not None,
            args.resume is not None,
            args.dry_run,
            args.terminal_only,
        )
    ):
        parser.error(
            "--reports-only нельзя сочетать с --experiment-file, --resume, "
            "--dry-run или --terminal-only"
        )
    if args.terminal_only and args.resume is not None:
        parser.error("--terminal-only нельзя сочетать с --resume")

    try:
        if args.reports_only is not None:
            from ADP.experiment_reports import write_reports

            artifacts = write_reports(args.reports_only)
            print(f"отчёты обновлены: {artifacts}")
            return 0

        if args.experiment_file is None:
            experiment = experiment_from_args(args, parser)
        else:
            experiment = load_experiment(args.experiment_file)
        jobs = _build_jobs(experiment)
        if args.dry_run:
            for point in experiment.points:
                print(f"point={point.name} runs={experiment.runs}")
            print(f"variants: {', '.join(experiment.variants)}")
            print(f"total jobs: {len(jobs)}")
            return 0

        if args.terminal_only:
            runs, failures = run_experiment_terminal(
                experiment,
                show_progress=not args.no_progress,
            )
            _print_terminal_summary(experiment, runs)
            return 1 if failures else 0

        series_dir, failures = run_experiment(
            experiment,
            args.output_dir,
            resume=args.resume,
            save_models=not args.no_save_models,
            show_progress=not args.no_progress,
        )
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"серия сохранена: {series_dir}")
    print(f"ошибок: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
