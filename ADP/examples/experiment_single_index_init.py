from dataclasses import replace

from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
)


POINTS = (
    ADP_ExperimentPoint("n250_d25", 250, 25),
    ADP_ExperimentPoint("n500_d25", 500, 25),
    ADP_ExperimentPoint("n1000_d25", 1000, 25),
    ADP_ExperimentPoint("n2000_d25", 2000, 25),
    ADP_ExperimentPoint("n4000_d25", 4000, 25),
    ADP_ExperimentPoint("n2000_d5", 2000, 5),
    ADP_ExperimentPoint("n2000_d10", 2000, 10),
    ADP_ExperimentPoint("n2000_d50", 2000, 50),
    ADP_ExperimentPoint("n2000_d100", 2000, 100),
)

common = ADP_Config(
    N_loc=32,
    N_lin=128,
    N_J=128,
    N_phi=10,
    lambda_penalty=100,
    local_ridge=1e-8,
    index_init="local",
)

experiment = ADP_Experiment(
    name="single_index_local_vs_random",
    mode="single",
    runs=25,
    seed=0,
    points=POINTS,
    variants={
        "local": ADP_ExperimentVariant(common, "lsmr", {"max_steps": 10}),
        "random": ADP_ExperimentVariant(
            replace(common, index_init="random"),
            "lsmr",
            {"max_steps": 10},
        ),
    },
)
