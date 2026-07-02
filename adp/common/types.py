from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import linalg


KernelName = Literal["epanechnikov", "quartic", "gaussian"]
VariantName = Literal["new", "old"]
BackendName = Literal["numpy"]


@dataclass(slots=True)
class ADPConfig:
    """Настройки Average Derivative Procedure.

    Смысл:
        Хранит все численные параметры обучения. Сейчас реализован один
        EDR-вектор beta, а поле target_dim оставлено для будущего multi-index
        расширения из TeX-файлов.
    Вход:
        Значения полей dataclass.
    Выход:
        Объект конфигурации, который передается в ADP.create(...).
    """

    n_centers: int | None = None
    n_directions: int = 10
    target_dim: int = 1
    min_neighbors: float = 10.0
    lambda_penalty: float | None = None
    outer_steps: int = 4
    inner_steps: int = 20
    tol: float = 1e-6
    bandwidth_decay: float = math.sqrt(2.0)
    anisotropy_min: float | None = None
    kernel: KernelName = "epanechnikov"
    backend: BackendName = "numpy"
    dtype: str = "float64"
    center_noise_scale: float = 1.0
    renew_directions: bool = True
    chunk_size: int = 64
    ridge: float = 1e-10
    show_progress: bool = True
    random_state: int | None = None
    use_neighbor_index: bool = True

    def __post_init__(
        self,  # Текущая конфигурация ADP.
    ) -> None:
        """Проверяет поддерживаемость backend.

        Вход:
            self: текущая конфигурация.
        Выход:
            None; при неподдержанном backend выбрасывает ValueError.
        """

        if self.backend != "numpy":
            raise ValueError("Only numpy backend is supported")

    def resolved_lambda(
        self,  # Текущая конфигурация ADP.
    ) -> float:
        """Возвращает штраф регуляризации для beta.

        Вход:
            self: текущая конфигурация.
        Выход:
            lambda_penalty, если он задан, иначе min_neighbors.
        """

        if self.lambda_penalty is None:
            return float(self.min_neighbors)
        return float(self.lambda_penalty)


@dataclass(slots=True)
class ADPData:
    """Сгенерированные данные single-index модели.

    Вход:
        Поля dataclass после генерации данных.
    Выход:
        Контейнер с X, y, истинным beta, центрами и направлениями.
    """

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    centers: np.ndarray
    directions: np.ndarray | None
    noise: np.ndarray
    link_name: str


@dataclass(slots=True)
class LocalStatistics:
    """Локальные суммы, которые входят в objective ADP.

    Вход:
        Поля dataclass после вычисления локальных статистик.
    Выход:
        Контейнер со статистиками конкретного варианта new или old.
    """

    variant: VariantName
    imav: np.ndarray
    centers: np.ndarray
    h: float
    weights_mean: float
    directions: np.ndarray | None = None
    S: np.ndarray | None = None
    U: np.ndarray | None = None
    N: np.ndarray | None = None
    VP: np.ndarray | None = None
    anisotropy: float | None = None
    b: float | None = None


@dataclass(slots=True)
class TrainingStep:
    """Одна внутренняя итерация alternating solver.

    Вход:
        Поля dataclass после внутреннего шага.
    Выход:
        Запись истории обучения.
    """

    outer: int
    inner: int
    objective: float
    beta_delta: float
    h: float
    anisotropy: float | None
    elapsed: float


@dataclass(slots=True)
class ADPResult:
    """Итог обучения ADP.

    Вход:
        Поля dataclass после model.fit(...).
    Выход:
        Контейнер с beta, локальными коэффициентами и диагностикой.
    """

    beta: np.ndarray
    intercepts: np.ndarray
    slopes: np.ndarray
    statistics: LocalStatistics
    history: list[TrainingStep]
    progress: list[dict[str, Any]]
    objective: float
    backend: str
    timings: dict[str, float] = field(default_factory=dict)
    diagnostic_plots: dict[str, Path] = field(default_factory=dict)

    @property
    def projector(
        self,  # Результат обучения с найденным beta.
    ) -> np.ndarray:
        """Строит ортогональный проектор на найденное EDR-направление.

        Вход:
            self: результат ADP.
        Выход:
            Матрица beta beta^T размера d x d.
        """

        beta = np.asarray(self.beta, dtype=float).reshape(-1)
        beta = beta / max(np.linalg.norm(beta), np.finfo(float).eps)
        return np.outer(beta, beta)

    @property
    def basis(
        self,  # Результат обучения с найденным beta.
    ) -> np.ndarray:
        """Возвращает EDR-базис через eig/SVD-совместимый API.

        Вход:
            self: результат ADP.
        Выход:
            Матрица размера 1 x d с ведущим направлением.
        """

        values, vectors = linalg.eigh(self.projector)
        order = np.argsort(values)[::-1][:1]
        return vectors[:, order].T
