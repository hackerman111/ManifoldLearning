"""Four-model comparison of the control direction and small ridge scales.

Run with ``run_benchmarks.py compare --use-model-initializers``.
"""

from examples.initial_direction_models import (
    e1_control,
    ridge_eta_1e_4,
    ridge_eta_1e_5,
    ridge_eta_1e_6,
)

MODELS = {
    "e1_control": e1_control,
    "ridge_eta_1e-4": ridge_eta_1e_4,
    "ridge_eta_1e-5": ridge_eta_1e_5,
    "ridge_eta_1e-6": ridge_eta_1e_6,
}
