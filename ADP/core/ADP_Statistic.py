from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass(frozen=True, slots=True)
class ADP_Statistics:
    _fields: ClassVar[tuple[str, ...]] = (
        "I",
        "U",
        "mass",
        "mean",
        "n_eff",
        "eta",
        "S",
        "S",
    )

    I: np.ndarray
    U: np.ndarray
    mass: np.ndarray
    mean: np.ndarray
    n_eff: np.ndarray
    eta: np.ndarray
    S: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __getitem__(self, name: str) -> np.ndarray:
        """Вернуть поле статистики по имени, сохраняя mapping API."""
        if name not in self._fields:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        """Итерировать имена доступных полей статистики."""
        return iter(self._fields)
