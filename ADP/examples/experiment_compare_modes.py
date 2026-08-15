import sys
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    box_kernel,
    run_experiment,
)
from ADP.experiment import validate_experiment

RUNS = 10
SEED = 7
POINTS = (ADP_ExperimentPoint("n500_d10_noise005", 1000, 25, 0.05),)
SOLVER_SETTINGS = {"max_steps": 10}

single = ADP_Config(
    N_loc=10,
    N_lin=20,
    N_J=64,
    N_phi=10,
    lambda_penalty=100,
    local_ridge=1e-8,
)
multi = replace(single, index_init="pilot", lambda_penalty=10_000)


def comparison(name, mode, base, variant_name, variant, *, index_dim=1):
    return ADP_Experiment(
        name=name,
        mode=mode,
        index_dim=index_dim,
        runs=RUNS,
        seed=SEED,
        points=POINTS,
        variants={
            "ordinary": ADP_ExperimentVariant(base, "lsmr", SOLVER_SETTINGS),
            variant_name: ADP_ExperimentVariant(
                variant, "lsmr", SOLVER_SETTINGS
            ),
        },
    )


EXPERIMENTS = (
    comparison(
        "single_ordinary_vs_smart_weights",
        "single",
        single,
        "smart_weights",
        replace(single, smart_weights=True),
    ),
    comparison(
        "single_ordinary_vs_kernel_box",
        "single",
        single,
        "kernel_box",
        replace(single, kernel=box_kernel),
    ),
    comparison(
        "multi_ordinary_vs_smart_weights",
        "multi",
        multi,
        "smart_weights",
        replace(multi, smart_weights=True),
        index_dim=2,
    ),
    comparison(
        "multi_ordinary_vs_kernel_box",
        "multi",
        multi,
        "kernel_box",
        replace(multi, kernel=box_kernel),
        index_dim=2,
    ),
)


def main(argv=None):
    parser = ArgumentParser(description="Compare ordinary, smart and box ADP")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ADP/experiment_outputs")
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        for experiment in EXPERIMENTS:
            validate_experiment(experiment)
        print("4 experiment configurations are valid")
        return 0

    failures = 0
    for experiment in EXPERIMENTS:
        series_dir, count = run_experiment(
            experiment,
            args.output_dir,
            show_progress=not args.no_progress,
        )
        failures += count
        print(f"{experiment.name}: {series_dir}; failures={count}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
