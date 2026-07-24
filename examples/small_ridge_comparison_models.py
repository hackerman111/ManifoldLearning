"""Control direction versus the two smallest ridge direction scales."""

from examples.initial_direction_models import (
    e1_control,
    ridge_eta_1e_6,
    ridge_eta_1e_7,
)


MODELS = {
    "e1_control": e1_control,
    "ridge_eta_1e-6": ridge_eta_1e_6,
    "ridge_eta_1e-7": ridge_eta_1e_7,
}
