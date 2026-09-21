from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from .ADP_multi_index_utils import projectors_for_result


@dataclass(slots=True)
class ADP_multi_index_result:
    beta_init: np.ndarray
    beta_final: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    trace: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    dist_init: float | None = None
    dist_final: float | None = None

    def Calculate_dist(self) -> None:
        """Вычислить projector-distance между истинным и оценённым basis."""
        true_projector, shape = projectors_for_result(self.beta_true, "beta_true")
        init_projector, _ = projectors_for_result(self.beta_init, "beta_init", shape)
        final_projector, _ = projectors_for_result(self.beta_final, "beta_final", shape)
        scale = sqrt(2.0 * shape[1])
        self.dist_init = float(
            np.linalg.norm(true_projector - init_projector, ord="fro") / scale
        )
        self.dist_final = float(
            np.linalg.norm(true_projector - final_projector, ord="fro") / scale
        )

    def Set_beta_init(self, beta_init: np.ndarray) -> None:
        """Сохранить начальный multi-index basis."""
        self.beta_init = beta_init

    def Set_beta_true(self, beta_true: np.ndarray) -> None:
        """Сохранить reference multi-index basis."""
        self.beta_true = beta_true

    def Set_beta_final(self, beta_final: np.ndarray) -> None:
        """Сохранить оценённый multi-index basis."""
        self.beta_final = beta_final

    def Set_stop_reason(self, stop_reason: str) -> None:
        """Сохранить причину остановки outer-цикла."""
        self.stop_reason = stop_reason


__all__ = ["ADP_multi_index_result"]
