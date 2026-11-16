from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ADP_single_index_utils import require_result_vectors


@dataclass(slots=True)
class ADP_single_index_result:
    beta_init: np.ndarray
    beta_final: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    trace: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    cosine_init: float | None = None
    cosine_final: float | None = None

    def Calculate_cosine(self) -> None:
        """Вычислить sign-invariant cosine similarity истинного и оценённого индекса."""
        beta_true, beta_final = require_result_vectors(self)
        self.cosine_init = float(abs(np.dot(self.beta_init, beta_true)))
        self.cosine_final = float(abs(np.dot(beta_final, beta_true)))

    def Set_beta_init(self, beta_init: np.ndarray) -> None:
        """Сохранить начальный single-index."""
        self.beta_init = beta_init

    def Set_beta_true(self, beta_true: np.ndarray) -> None:
        """Сохранить reference single-index."""
        self.beta_true = beta_true

    def Set_beta_final(self, beta_final: np.ndarray) -> None:
        """Сохранить оценённый single-index."""
        self.beta_final = beta_final

    def Set_stop_reason(self, stop_reason: str) -> None:
        """Сохранить причину остановки outer-цикла."""
        self.stop_reason = stop_reason


__all__ = ["ADP_single_index_result"]
