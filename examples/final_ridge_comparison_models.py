"""Final comparison of the e1 control and ridge eta=1e-6 directions."""

from examples.initial_direction_models import (
    e1_control,
    ridge_eta_1e_6,
)


MODELS = {
    "e1_control": e1_control,
    "ridge_eta_1e-6": ridge_eta_1e_6,
}
