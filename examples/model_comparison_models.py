"""Ready-to-edit model factories for ``run_benchmarks.py compare``."""

from adp import ADP, ADPConfig, StageRegistry
from examples.compare_adp_solvers import DirectBetaSolver


def _config() -> ADPConfig:
    return ADPConfig(
        show_progress=False,
        record_telemetry=True,
        renew_directions=False,
        statistics_workers=1,
        random_state=0,
    )


def baseline():
    return ADP.create("new", _config())


def zero_intercept():
    return ADP.create(
        "new",
        _config(),
        stages={"local_solver": "zero_intercept"},
    )


def direct_beta_solver():
    registry = StageRegistry.with_defaults()
    registry.register(
        "beta_solver",
        "direct",
        lambda context: DirectBetaSolver(context.config),
    )
    return ADP.create(
        "new",
        _config(),
        stages={"beta_solver": "direct"},
        registry=registry,
    )


MODELS = {
    "baseline": baseline,
    "zero_intercept": zero_intercept,
    "direct_beta_solver": direct_beta_solver,
}
