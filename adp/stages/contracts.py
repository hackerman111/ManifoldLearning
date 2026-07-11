from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from ..common.types import ADPConfig, LocalStatistics, TrainingStep


@dataclass(frozen=True, slots=True)
class StageContext:
    """Неизменяемые зависимости фабрик этапов ADP."""

    config: ADPConfig
    backend: Any
    rng: np.random.Generator
    model: Any


@dataclass(slots=True)
class ADPState:
    """Полное изменяемое состояние одного запуска `ADPAlgorithm.fit`."""

    X: np.ndarray
    y: np.ndarray
    centers: np.ndarray | None = None
    beta: np.ndarray | None = None
    prior: np.ndarray | None = None
    h: float | None = None
    anisotropy: float | None = None
    directions: np.ndarray | None = None
    statistics: LocalStatistics | None = None
    intercepts: np.ndarray | None = None
    slopes: np.ndarray | None = None
    history: list[TrainingStep] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)


class BetaInitializer(Protocol):
    def initialize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class CenterSelector(Protocol):
    def select(self, X: np.ndarray) -> np.ndarray: ...


class BandwidthSelector(Protocol):
    def select_initial(self, X: np.ndarray, centers: np.ndarray, index: Any = None) -> float: ...

    def select_anisotropy(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
    ) -> float: ...


class DirectionSampler(Protocol):
    def prepare(
        self,
        centers: np.ndarray,
        d: int,
        directions: np.ndarray | None,
        *,
        beta: np.ndarray | None = None,
        anisotropy: float | None = None,
    ) -> np.ndarray | None: ...


class StatisticsBuilder(Protocol):
    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        h: float,
        beta: np.ndarray,
        directions: np.ndarray | None,
        anisotropy: float | None,
    ) -> LocalStatistics: ...


class LocalSolver(Protocol):
    def solve(self, statistics: LocalStatistics, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class BetaSolver(Protocol):
    def solve(
        self,
        statistics: LocalStatistics,
        intercepts: np.ndarray,
        slopes: np.ndarray,
        prior: np.ndarray,
        lambda_penalty: float,
        x0: np.ndarray | None = None,
    ) -> np.ndarray: ...


class StopRule(Protocol):
    def should_stop(
        self,
        phase: str,
        state: ADPState,
        *,
        step: TrainingStep | None = None,
        **metrics: Any,
    ) -> bool: ...


StageFactory = Callable[[StageContext], Any]


class StageExecutionError(RuntimeError):
    """Ошибка выполнения конкретной реализации этапа ADP."""

    def __init__(
        self,
        category: str,
        implementation: str,
        message: str,
        *,
        outer: int | None = None,
        inner: int | None = None,
    ) -> None:
        location = []
        if outer is not None:
            location.append(f"outer={outer}")
        if inner is not None:
            location.append(f"inner={inner}")
        suffix = f" ({', '.join(location)})" if location else ""
        super().__init__(f"Этап {category!r} ({implementation!r}){suffix}: {message}")
        self.category = category
        self.implementation = implementation
        self.outer = outer
        self.inner = inner
