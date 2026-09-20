from dataclasses import replace

from ADP import ADP_Experiment, ADP_ExperimentPoint, ADP_ExperimentVariant
from experiment_single_index_init import POINTS, common


NOISES = (("020", 0.2), ("050", 0.5), ("100", 1.0))

experiment = ADP_Experiment(
    name="single_index_local_vs_random_noisy",
    mode="single",
    runs=25,
    seed=0,
    points=tuple(
        ADP_ExperimentPoint(
            f"{point.name}_noise{tag}",
            point.n,
            point.d,
            noise,
            {"sigma_eps": noise},
        )
        for tag, noise in NOISES
        for point in POINTS
    ),
    variants={
        "local": ADP_ExperimentVariant(common, "lsmr", {"max_steps": 10}),
        "random": ADP_ExperimentVariant(
            replace(common, index_init="random"),
            "lsmr",
            {"max_steps": 10},
        ),
    },
)
