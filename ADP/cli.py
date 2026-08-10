import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    load_experiment,
    run_experiment,
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reports_only is not None and any(
        (args.experiment_file is not None, args.resume is not None, args.dry_run)
    ):
        parser.error(
            "--reports-only нельзя сочетать с --experiment-file, --resume или --dry-run"
        )

    try:
        if args.reports_only is not None:
            from ADP.experiment_reports import write_reports

            artifacts = write_reports(args.reports_only)
            print(f"отчёты обновлены: {artifacts}")
            return 0

        experiment = (
            load_experiment(args.experiment_file)
            if args.experiment_file is not None
            else experiment_from_args(args, parser)
        )
        jobs = _build_jobs(experiment)
        if args.dry_run:
            for point in experiment.points:
                print(f"point={point.name} runs={experiment.runs}")
            print(f"variants: {', '.join(experiment.variants)}")
            print(f"total jobs: {len(jobs)}")
            return 0

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
