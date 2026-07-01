from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core import ADPConfig


BenchmarkMethod = Literal[
    "adp_new",
    "adp_old",
    "statsmodels_sir",
    "statsmodels_save",
    "statsmodels_phd",
    "sklearn_pls",
]


@dataclass(slots=True)
class BenchmarkScenario:
    """Один воспроизводимый сценарий для проверки EDR-методов."""

    name: str
    n: int
    d: int
    link: str = "linear"
    noise: float = 0.05
    corr: float = 0.5
    sigma_x: float = 1.0
    n_centers: int | None = None
    n_directions: int = 8
    min_neighbors: float = 10.0
    outer_steps: int = 3
    inner_steps: int = 8
    trials: int = 3

    def adp_config(self, *, random_state: int, show_progress: bool) -> ADPConfig:
        return ADPConfig(
            n_centers=self.n_centers or min(self.n, max(20, self.n // 4)),
            n_directions=self.n_directions,
            min_neighbors=self.min_neighbors,
            outer_steps=self.outer_steps,
            inner_steps=self.inner_steps,
            show_progress=show_progress,
            random_state=random_state,
        )


def default_scenarios(*, quick: bool = False) -> list[BenchmarkScenario]:
    """Набор сценариев, закрывающий несколько типичных режимов."""

    trials = 1 if quick else 5
    scale = 0.55 if quick else 1.0

    def n(value: int) -> int:
        return max(80, int(value * scale))

    return [
        BenchmarkScenario(
            name="linear_low_noise",
            n=n(240),
            d=8,
            link="linear",
            noise=0.03,
            corr=0.2,
            n_centers=n(60),
            n_directions=8,
            min_neighbors=8,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
        BenchmarkScenario(
            name="sin_correlated",
            n=n(280),
            d=10,
            link="sin",
            noise=0.08,
            corr=0.55,
            n_centers=n(70),
            n_directions=10,
            min_neighbors=10,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
        BenchmarkScenario(
            name="quadratic_symmetric",
            n=n(320),
            d=10,
            link="quadratic",
            noise=0.05,
            corr=0.35,
            n_centers=n(80),
            n_directions=12,
            min_neighbors=10,
            outer_steps=2 if quick else 5,
            inner_steps=5 if quick else 12,
            trials=trials,
        ),
        BenchmarkScenario(
            name="dimension_stress",
            n=n(360),
            d=18,
            link="linear",
            noise=0.05,
            corr=0.4,
            n_centers=n(90),
            n_directions=12,
            min_neighbors=12,
            outer_steps=2 if quick else 4,
            inner_steps=5 if quick else 10,
            trials=trials,
        ),
    ]


def grid_scenarios(
    *,
    d_values: tuple[int, ...] = (10, 25, 50, 100, 200),
    direction_values: tuple[int, ...] = (5, 10, 20, 40),
    n: int = 360,
    n_centers: int = 90,
    outer_steps: int = 4,
    inner_steps: int = 10,
    trials: int = 5,
    link: str = "linear",
    noise: float = 0.05,
    corr: float = 0.4,
    min_neighbors: float = 12.0,
) -> list[BenchmarkScenario]:
    """Строит строгую сетку сценариев по размерности d и числу направлений P."""

    scenarios: list[BenchmarkScenario] = []
    for d in d_values:
        for n_directions in direction_values:
            scenarios.append(
                BenchmarkScenario(
                    name=f"grid_d{d}_p{n_directions}",
                    n=n,
                    d=d,
                    link=link,
                    noise=noise,
                    corr=corr,
                    n_centers=n_centers,
                    n_directions=n_directions,
                    min_neighbors=min_neighbors,
                    outer_steps=outer_steps,
                    inner_steps=inner_steps,
                    trials=trials,
                )
            )
    return scenarios
