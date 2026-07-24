"""Models for the initial bandwidth experiment.

All variants use the same response-aware ``ridge_1e-4`` beta_ref. Run with
``run_benchmarks.py compare --use-model-initializers`` to activate it.
"""

from adp import ADP, ADPConfig


def _config() -> ADPConfig:
    return ADPConfig(
        min_neighbors=4.0,
        show_progress=False,
        record_telemetry=True,
        renew_directions=False,
        statistics_workers=1,
        random_state=0,
    )


def _model(bandwidth_selector: str):
    return ADP.create(
        "new",
        _config(),
        stages={
            "beta_initializer": "ridge_1e-4",
            "bandwidth_selector": bandwidth_selector,
        },
    )


def mean_mass_control():
    return _model("local_mass_mean")


def local_mass_q0():
    return _model("local_mass_q0")


def local_mass_q05():
    return _model("local_mass_q05")


def local_mass_q10():
    return _model("local_mass_q10")


def local_mass_q25():
    return _model("local_mass_q25")


def knn_q90_k1():
    return _model("knn_q90_k1")


def knn_q90_k2():
    return _model("knn_q90_k2")


def knn_q90_k4():
    return _model("knn_q90_k4")


MODELS = {
    "mean_mass_control": mean_mass_control,
    "local_mass_q0": local_mass_q0,
    "local_mass_q05": local_mass_q05,
    "local_mass_q10": local_mass_q10,
    "local_mass_q25": local_mass_q25,
    "knn_q90_k1": knn_q90_k1,
    "knn_q90_k2": knn_q90_k2,
    "knn_q90_k4": knn_q90_k4,
}
