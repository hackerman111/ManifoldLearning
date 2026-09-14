"""CLI экспериментов; определения, данные и выполнение находятся в experiments/."""

from __future__ import annotations

import argparse
from pathlib import Path

from ADP.core.ADP_Config import ADP_Config
from experiments.data import (
    _generate_data as _generate_data,
    _link as _link,
    _make_seed_bundle as _make_seed_bundle,
)
from experiments.models import (
    Build as Build,
    Experiment as Experiment,
    ExperimentPoint as ExperimentPoint,
)
from experiments.registry import (
    CATALOG as CATALOG,
    _selected_experiments as _selected_experiments,
)
from experiments.runner import (
    _effective_config as _effective_config,
    _experiment_id as _experiment_id,
    _has_numerical_failures as _has_numerical_failures,
    _local_subspace_metrics as _local_subspace_metrics,
    _outcome_fields as _outcome_fields,
    _subspace_metrics as _subspace_metrics,
    run_experiment as run_experiment,
)
from experiments.single import custom_experiment as custom_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal paired ADP experiments")
    parser.add_argument("--experiment", default="custom")
    parser.add_argument(
        "--profile", choices=("smoke", "overview", "full"), default="smoke"
    )
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/experiments")
    )
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--solver", choices=("lsmr", "cg", "hybrid"), default="lsmr")
    parser.add_argument("--solver-max-steps", type=int, default=3)
    parser.add_argument("--cg-maxiter", type=int)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        for experiment in CATALOG.values():
            print(
                f"{experiment.selector:>24}  "
                f"points={len(experiment.full):>5} runs={experiment.full_runs:>3}  "
                f"{experiment.title}"
            )
        return 0
    failed = False
    try:
        experiments = _selected_experiments(
            args.experiment,
            custom=custom_experiment(args.n, args.d, args.noise),
        )
        experiment_id = _experiment_id()
        build = Build(
            "ADP",
            ADP_Config(),
            solver=args.solver,
            solver_max_steps=args.solver_max_steps,
            cg_maxiter=args.cg_maxiter,
        )
        for experiment in experiments:
            path = experiment.run(
                build,
                profile=args.profile,
                runs=args.runs,
                seed=args.seed,
                output_dir=args.output_dir,
                plots=not args.no_plots,
                experiment_id=experiment_id,
                progress=True,
            )
            print(f"{experiment.selector}: {path}")
            failed = _has_numerical_failures(path) or failed
    except (ImportError, TypeError, ValueError) as error:
        parser.error(str(error))
    return int(failed)


__all__ = [
    "CATALOG",
    "Build",
    "Experiment",
    "ExperimentPoint",
    "custom_experiment",
    "run_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
