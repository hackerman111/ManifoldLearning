import argparse
import importlib
import sys
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
        default=100.0,
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
        choices=("local", "random"),
        default="local",
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
    try:
        config = ADP_Config(
            seed=args.seed,
            N_loc=args.N_loc,
            N_lin=args.N_lin,
            N_J=args.N_J,
            N_phi=args.N_phi,
            outer_steps=args.outer_steps,
            lambda_penalty=args.lambda_penalty,
            local_ridge=args.local_ridge,
            kernel=args.kernel,
            a=args.a,
            h_min=args.h_min,
            batch_size=args.batch_size,
            index_init=args.index_init,
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


def _print_table(headers, rows, right_aligned):
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def line(values):
        return "  ".join(
            value.rjust(widths[index])
            if index in right_aligned
            else value.ljust(widths[index])
            for index, value in enumerate(values)
        )

    print(line(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def _print_terminal_summary(experiment, runs):
    grouped = {}
    for run in runs:
        grouped.setdefault((run["point"], run["variant"]), []).append(run)

    print(
        f"Эксперимент: {experiment.name} | "
        f"режим: {experiment.mode} | jobs: {len(runs)}"
    )
    print("\nСтатусы")
    status_rows = []
    metric_rows = []
    quality_name = "cosine_abs" if experiment.mode == "single" else "projector_error"
    quality_header = (
        "Медиана cosine"
        if experiment.mode == "single"
        else "Медиана ошибки проектора"
    )
    for point in experiment.points:
        for variant in experiment.variants:
            rows = grouped.get((point.name, variant), [])
            statuses = [row.get("status") for row in rows]
            status_rows.append(
                (
                    point.name,
                    variant,
                    str(len(rows)),
                    str(statuses.count("success")),
                    str(statuses.count("nonconverged")),
                    str(statuses.count("numerical_failure")),
                )
            )
            quality = _finite_values(rows, quality_name)
            wall_time = _finite_values(rows, "fit_wall_time_sec")
            rss = _finite_values(rows, "algorithm_rss_max_mib")
            iterations = _finite_values(rows, "outer_iterations")
            metric_rows.append(
                (
                    point.name,
                    variant,
                    f"{median(quality):.3f}" if quality else "—",
                    f"{median(wall_time):.3f}" if wall_time else "—",
                    f"{max(rss):.1f}" if rss else "—",
                    f"{median(iterations):g}" if iterations else "—",
                )
            )
    _print_table(
        ("Точка", "Вариант", "Jobs", "Успех", "NC", "Ошибки"),
        status_rows,
        {2, 3, 4, 5},
    )
    print("\nХарактеристики")
    _print_table(
        (
            "Точка",
            "Вариант",
            quality_header,
            "Время med, с",
            "RSS max, MiB",
            "Итер. med",
        ),
        metric_rows,
        {2, 3, 4, 5},
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
