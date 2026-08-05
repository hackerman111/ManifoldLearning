from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ADP_Data:
    X: np.ndarray
    Y: np.ndarray
    noise: np.ndarray
    beta: np.ndarray
