from ADP import (
    ADP_Config,
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
)

common = ADP_Config(
    N_loc=10,
    N_lin=20,
    N_J=64,
    N_phi=10,
    lambda_penalty=100,
    local_ridge=1e-8,
)

experiment = ADP_Experiment(
    name="lsmr_vs_varpro",
    mode="single",
    runs=10,
    seed=7,
    points=(
        ADP_ExperimentPoint("noise_005", 500, 10, 0.05, {"sigma_eps": 0.05}),
        ADP_ExperimentPoint("noise_020", 500, 10, 0.20, {"sigma_eps": 0.20}),
    ),
    variants={
        "lsmr": ADP_ExperimentVariant(common, "lsmr", {"max_steps": 5}),
        "varpro": ADP_ExperimentVariant(common, "varpro", {"max_steps": 100}),
    },
)
