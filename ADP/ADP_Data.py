from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ADP_Data:
    X: np.ndarray
    Y: np.ndarray
    true_index: np.ndarray
