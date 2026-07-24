"""Models for the beta_ref initialization experiment.

Run with ``run_benchmarks.py compare --use-model-initializers`` so the
comparison runner does not override these stages with a shared external beta0.
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


def _model(initializer: str):
    return ADP.create(
        "new",
        _config(),
        stages={"beta_initializer": initializer},
    )


def e1_control():
    return _model("e1")


def pca():
    return _model("pca")


def ridge_eta_0():
    return _model("ridge_0")


def ridge_eta_1e_4():
    return _model("ridge_1e-4")


def ridge_eta_1e_5():
    return _model("ridge_1e-5")


def ridge_eta_1e_6():
    return _model("ridge_1e-6")


def ridge_eta_1e_7():
    return _model("ridge_1e-7")


def ridge_eta_1e_2():
    return _model("ridge_1e-2")


MODELS = {
    "e1_control": e1_control,
    "pca": pca,
    "ridge_eta_0": ridge_eta_0,
    "ridge_eta_1e-4": ridge_eta_1e_4,
    "ridge_eta_1e-5": ridge_eta_1e_5,
    "ridge_eta_1e-6": ridge_eta_1e_6,
    "ridge_eta_1e-7": ridge_eta_1e_7,
    "ridge_eta_1e-2": ridge_eta_1e_2,
}
