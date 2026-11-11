from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import Literal

from .models import Experiment, ExperimentPoint, ModelMode


def _scaling_point(mode: Literal["single", "multi"], n: int, d: int) -> ExperimentPoint:
    """Фиксированная конфигурация для сопоставимых SI/MI scaling-сеток."""
    multi = mode == "multi"
    return ExperimentPoint(
        d=d,
        n_over_d=n / d,
        n_samples=n,
        sigma_eps=0.2,
        tau=0.0,
        link="multi_additive" if multi else "sin_scaled",
        mode=mode,
        index_dim=2 if multi else 1,
        N_loc=15,
        N_lin=d + 60,
        N_J=math.ceil(n / 5),
        N_phi=20,
        outer_steps=5,
        lambda_penalty=0.05,
        a=math.sqrt(2),
        h_min_factor=3,
        index_init="pilot" if multi else "local",
        direction_mode="isotropic" if multi else "localized",
        select_step="best",
        solver_max_steps=8,
    )


def _smoke_point(point: ExperimentPoint) -> ExperimentPoint:
    """Уменьшить одну точку, сохранив исследуемый estimator-вариант."""
    multi = point.mode == "multi"
    return replace(
        point,
        d=4,
        n_over_d=12 if multi else 10,
        n_samples=48 if multi else 40,
        index_dim=2 if multi else 1,
        basis_pool_dim=3 if point.basis_pool_dim is not None else None,
        N_loc=6,
        N_lin=10,
        N_J=8,
        N_phi=4 if multi else 3,
        outer_steps=1,
        h_min_factor=3,
        solver_max_steps=2,
    )


def _parameter_scaling_catalog(
    mode: Literal["single", "multi"],
) -> tuple[Experiment, ...]:
    """Парные SI/MI-сетки зависимости N_phi, N_loc и N_J от (n, d)."""
    shapes = tuple(product((500, 1000, 2000), (10, 25, 50)))
    n_phi_values = {
        "single": (5, 10, 20, 30, 40, 60),
        "multi": (3, 5, 10, 20, 30, 40),
    }
    n_loc_values = (7, 10, 15, 20, 30, 45)
    center_multipliers = (1, 2, 4, 8, 12)
    baseline_n_loc = 15

    def make(
        mode: ModelMode,
        parameter: str,
        points: tuple[ExperimentPoint, ...],
        hypothesis: str,
    ) -> Experiment:
        prefix = "mi" if mode == "multi" else "si"
        parameter_slug = parameter.lower().replace("_", "")
        return Experiment(
            f"scale-{prefix}-{parameter_slug}",
            f"{prefix.upper()}: зависимость {parameter} от n и d",
            _smoke_point(points[0]),
            points,
            report_fields=("n_samples", "d", parameter),
            full_runs=5,
            quality_threshold=0.95 if mode == "multi" else 0.9,
            condition_field=parameter,
            common_random_fields=(parameter,),
            hypothesis=hypothesis,
        )

    experiments: list[Experiment] = []
    bases = tuple(_scaling_point(mode, n, d) for n, d in shapes)
    multi = mode == "multi"
    experiments.extend(
        (
            make(
                mode,
                "N_phi",
                tuple(
                    replace(point, N_phi=value)
                    for point in bases
                    for value in n_phi_values[mode]
                ),
                (
                    "N_phi > m. "
                    "Это обязательная граница; качество насыщается около 20--30 "
                    "направлений. Зависимость от объёма выборки слабая, "
                    "стоимость при этом продолжает расти."
                    if multi
                    else "N_phi. "
                    "Плато качества ожидается около 20--30 "
                    "направлений. Большая размерность может сдвинуть это плато "
                    "вправо, объём выборки влияет слабее."
                ),
            ),
            make(
                mode,
                "N_loc",
                tuple(
                    replace(point, N_loc=value)
                    for point in bases
                    for value in n_loc_values
                ),
                "Эффективная локальная масса растёт медленнее объёма выборки "
                "и слабо зависит от размерности; малые значения "
                "нестабильны, большие увеличивают смещение.",
            ),
            make(
                mode,
                "N_J",
                tuple(
                    replace(
                        point,
                        N_J=math.ceil(multiplier * point.n / baseline_n_loc),
                    )
                    for point in bases
                    for multiplier in center_multipliers
                ),
                "N_J = c*n/N_loc. "
                "Коэффициент покрытия почти постоянен; "
                "после значений около 4--8 стоимость растёт быстрее качества.",
            ),
        )
    )
    return tuple(experiments)
