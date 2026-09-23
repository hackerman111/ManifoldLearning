# ruff: noqa: RUF002

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import numpy as np


def epanechnikov(value: np.ndarray) -> np.ndarray:
    """Вычислить профиль ядра Епанечникова ``max(1 - value**2, 0)``.

    Формула — компактное ядро из локальной ядерной регрессии: на входе
    безразмерное расстояние или его нормированное значение, на выходе массив
    неотрицательных весов той же формы. Функция не меняет входной массив.
    """
    result = np.square(value, dtype=float)
    np.subtract(1.0, result, out=result)
    np.maximum(result, 0.0, out=result)
    return result


@dataclass(frozen=True, slots=True)
class ADP_Config:
    seed: int = 42
    N_loc: int = 10
    N_lin: int | None = None
    N_J: int | None = None
    N_phi: int | None = None
    outer_steps: int | None = None
    lambda_penalty: float = 0.05
    local_ridge: float = 1e-8
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov
    a: float = sqrt(2)
    h_min: float = 1.0
    batch_size: int = 32
    index_init: str = "local"
    estimator: str = "new"
    direction_mode: str = "auto"
    multi_tensor: str = "orthogonal"
    select_step: str = "best"
    center_displacement: float = 0.0
    training_set: str = "all"
    redraw_directions: bool = True
    # Совместимость для перенесённых single/multi моделей.
    smart_weights: bool = False
    gpu: bool = False
    gpu_solver: bool = False

    def __post_init__(self) -> None:
        """Проверить границы конфигурации до запуска численного алгоритма.

        Здесь проверяются размеры, параметры регуляризации и допустимые режимы;
        результатом является либо пригодный для ``core`` объект конфигурации,
        либо явная ошибка на границе API.
        """
        integer_fields = (
            "seed",
            "N_loc",
            "N_lin",
            "N_J",
            "N_phi",
            "outer_steps",
            "batch_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1):
                requirement = "nonnegative" if name == "seed" else "positive"
                raise ValueError(f"{name} must be {requirement}")

        for name in ("lambda_penalty", "local_ridge", "center_displacement"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        if not np.isfinite(self.a) or self.a <= 1:
            raise ValueError("a must be finite and exceed one")
        if self.h_min is not None and (not np.isfinite(self.h_min) or self.h_min <= 0):
            raise ValueError("h_min must be finite and positive")
        if not callable(self.kernel):
            raise TypeError("kernel must be callable")
        if self.index_init not in {"local", "local-cv", "pilot", "random"}:
            raise ValueError(
                "index_init must be 'local', 'local-cv', 'pilot', or 'random'"
            )
        if self.estimator not in {"new", "legacy"}:
            raise ValueError("estimator must be 'new' or 'legacy'")
        if self.direction_mode not in {"auto", "isotropic", "localized"}:
            raise ValueError(
                "direction_mode must be 'auto', 'isotropic', or 'localized'"
            )
        if self.multi_tensor not in {"orthogonal", "full"}:
            raise ValueError("multi_tensor must be 'orthogonal' or 'full'")
        if self.select_step not in {"best", "last"}:
            raise ValueError("select_step must be 'best' or 'last'")
        if self.training_set not in {"all", "exclude_centers"}:
            raise ValueError("training_set must be 'all' or 'exclude_centers'")
        if not isinstance(self.redraw_directions, bool):
            raise TypeError("redraw_directions must be boolean")
        if not isinstance(self.smart_weights, (bool, np.bool_)):
            raise TypeError("smart_weights must be boolean")
        if not isinstance(self.gpu, (bool, np.bool_)):
            raise TypeError("gpu must be boolean")
        if not isinstance(self.gpu_solver, (bool, np.bool_)):
            raise TypeError("gpu_solver must be boolean")
        if self.gpu_solver and not self.gpu:
            raise ValueError("gpu_solver requires gpu=True")
