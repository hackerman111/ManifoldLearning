import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
)

N_VALUES = (500, 1000, 2000)
D_VALUES = (10, 25, 50)
RUNS = 25
SEED = 7


def make_experiment(mode: str, m: int) -> ADP_Experiment:
    if not (mode == "single" and m == 1 or mode == "multi" and m > 1):
        raise ValueError("single requires m=1; multi requires m>1")

    config = ADP_Config(
        N_loc=10,
        N_lin=100,
        N_J=200,
        N_phi=max(10, 2 * m),
        lambda_penalty=100 if mode == "single" else 10_000,
        local_ridge=1e-8,
        index_init="local" if mode == "single" else "pilot",
    )
    variant = ADP_ExperimentVariant(config, "lsmr", {"max_steps": 10})
    points = tuple(
        ADP_ExperimentPoint(
            f"n{n}_d{d}_m{m}",
            n,
            d,
            0.05,
            {"m": m},
        )
        for n in N_VALUES
        for d in D_VALUES
        if m < d
    )
    return ADP_Experiment(
        name=f"{mode}_m{m}_ordinary_vs_smart_weights",
        mode=mode,
        index_dim=m,
        runs=RUNS,
        seed=SEED,
        points=points,
        variants={
            "ordinary": variant,
            "smart_weights": replace(
                variant,
                config=replace(config, smart_weights=True),
            ),
        },
    )


def run_file(path: str) -> int:
    from ADP.cli import main

    return main(["--experiment-file", path, *sys.argv[1:]])
