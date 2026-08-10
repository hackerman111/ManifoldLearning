from dataclasses import replace

from .ADP_Solver import ADP_solver
from .experiment import ADP_Experiment, ADP_ExperimentVariant, _solver_method
from .multi_index.ADP_multi_index import ADP_multi_index
from .single_index.ADP_single_index import ADP_single_index


def _build_model(
    experiment: ADP_Experiment,
    variant: ADP_ExperimentVariant,
    seed: int,
):
    config = replace(variant.config, seed=seed)
    model = (
        ADP_single_index(config)
        if experiment.mode == "single"
        else ADP_multi_index(experiment.index_dim, config)
    )
    if variant.solver == "auto":
        if variant.solver_settings:
            settings = {**model.solver.settings, **dict(variant.solver_settings)}
            model.solver = ADP_solver(model.solver.method, **settings)
        return model
    model.solver = ADP_solver(
        _solver_method(experiment.mode, variant.solver),
        **dict(variant.solver_settings),
    )
    return model


def _solver_name(method) -> str:
    if method is _solver_method("single", "lsmr"):
        return "lsmr"
    if method is _solver_method("single", "varpro"):
        return "varpro"
    return f"{method.__module__}:{method.__qualname__}"
