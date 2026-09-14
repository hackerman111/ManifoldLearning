"""manifold: сетки и самостоятельный запуск исследовательского набора."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.models import Experiment, ExperimentPoint


def _manifold_catalog() -> tuple[Experiment, ...]:
    """Базовые однофакторные ESTIMATOR-сетки manifold ADP."""
    smoke = ExperimentPoint(
        d=3,
        n_over_d=80 / 3,
        n_samples=80,
        sigma_eps=0.02,
        link="manifold_radial",
        mode="manifold",
        index_dim=1,
        N_loc=15,
        N_lin=30,
        N_J=12,
        N_phi=6,
        N_manifold=5,
        sync_steps=1,
        lambda_manifold=0.5,
        a=2.0,
        h_min_factor=10.0,
    )
    base = replace(
        smoke,
        d=4,
        n_over_d=60,
        n_samples=240,
        sigma_eps=0.05,
        N_loc=20,
        N_lin=60,
        N_J=24,
        N_phi=10,
        N_manifold=6,
    )

    def sweep(
        selector: str,
        title: str,
        field: str,
        values: tuple[int | float, ...],
        hypothesis: str,
        *,
        common_random_fields: tuple[str, ...] | None = None,
    ) -> Experiment:
        def point(value: int | float) -> ExperimentPoint:
            updates = {field: value}
            if field in {"n_samples", "d"}:
                n = int(value) if field == "n_samples" else base.n
                d = int(value) if field == "d" else base.d
                updates["n_over_d"] = n / d
            return replace(base, **updates)

        return Experiment(
            selector,
            title,
            smoke,
            tuple(point(value) for value in values),
            report_fields=(field,),
            full_runs=5,
            quality_threshold=0.2,
            condition_field=field,
            common_random_fields=common_random_fields or (field,),
            hypothesis=hypothesis,
        )

    joint = Experiment(
        "manifold",
        "Manifold ADP: локально меняющееся радиальное направление",
        smoke,
        tuple(
            replace(
                smoke,
                d=d,
                n_over_d=n / d,
                n_samples=n,
                N_lin=linear_mass,
                N_J=centers,
                N_manifold=neighbors,
                N_loc=local_mass,
                N_phi=directions,
                sigma_eps=noise,
            )
            for (
                n,
                d,
                centers,
                neighbors,
                local_mass,
                directions,
                linear_mass,
                noise,
            ) in (
                (120, 3, 16, 6, 15, 6, 30, 0.02),
                (240, 4, 24, 8, 15, 6, 30, 0.05),
                (400, 6, 32, 10, 25, 12, 60, 0.10),
            )
        ),
        report_fields=("n_samples", "d", "sigma_eps"),
        full_runs=5,
        quality_threshold=0.2,
        hypothesis=(
            "Локальные проекторы восстанавливают радиальное направление при "
            "росте n и умеренном шуме."
        ),
    )
    return (
        joint,
        sweep(
            "manifold-n",
            "Manifold ADP: объём выборки",
            "n_samples",
            (120, 240, 360, 480),
            "Ошибка локальных проекторов уменьшается при росте n.",
            common_random_fields=("n_samples", "n_over_d"),
        ),
        sweep(
            "manifold-d",
            "Manifold ADP: размерность пространства",
            "d",
            (2, 3, 4, 6, 8, 10),
            "Восстановление ухудшается при росте внешней размерности d.",
            common_random_fields=("d", "n_over_d"),
        ),
        sweep(
            "manifold-noise",
            "Manifold ADP: шум отклика",
            "sigma_eps",
            (0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
            "При увеличении шума отклика ошибка локальных проекторов растёт.",
        ),
        sweep(
            "manifold-scale",
            "Manifold ADP: масштаб признаков",
            "sigma_x",
            (0.5, 1.0, 2.0, 4.0),
            "Масштаб признаков не меняет качество после адаптации bandwidth.",
        ),
        sweep(
            "manifold-corr",
            "Manifold ADP: корреляция признаков",
            "rho_corr",
            (0.0, 0.25, 0.5, 0.75, 0.9),
            "Сильная корреляция признаков ухудшает локальную идентификацию.",
        ),
        sweep(
            "manifold-nlin",
            "Manifold ADP: масса пилотной регрессии",
            "N_lin",
            (10, 20, 40, 60, 80),
            "Слишком малая или большая N_lin ухудшает пилотные градиенты.",
        ),
        sweep(
            "manifold-nloc",
            "Manifold ADP: локальная масса",
            "N_loc",
            (10, 15, 20, 30, 40),
            "N_loc задаёт компромисс между вариативностью и локальным смещением.",
        ),
        sweep(
            "manifold-centers",
            "Manifold ADP: число центров",
            "N_J",
            (12, 18, 24, 36, 48),
            "После достаточного покрытия рост N_J даёт убывающий выигрыш.",
        ),
        sweep(
            "manifold-nphi",
            "Manifold ADP: число случайных направлений",
            "N_phi",
            (2, 4, 6, 10, 16, 24),
            "Рост N_phi стабилизирует локальные статистики до насыщения.",
        ),
        sweep(
            "manifold-neighbors",
            "Manifold ADP: соседи manifold-графа",
            "N_manifold",
            (2, 4, 6, 8, 10),
            "N_manifold задаёт компромисс между связностью и локальностью графа.",
        ),
        sweep(
            "manifold-lambda",
            "Manifold ADP: регуляризация согласованности",
            "lambda_manifold",
            (0.0, 0.1, 0.5, 1.0, 5.0),
            "Умеренная lambda_manifold улучшает согласованность локальных chart.",
        ),
        sweep(
            "manifold-sync",
            "Manifold ADP: шаги синхронизации",
            "sync_steps",
            (1, 2, 3, 5),
            "Дополнительные шаги синхронизации улучшают качество до насыщения.",
        ),
    )


def catalog() -> tuple[Experiment, ...]:
    """Исходные сетки семейства, включая дорогие полные сетки."""
    return (*_manifold_catalog(),)


DEFAULT_SELECTORS = (
    "manifold",
    "manifold-n",
    "manifold-d",
    "manifold-noise",
    "manifold-scale",
    "manifold-corr",
    "manifold-nlin",
    "manifold-nloc",
    "manifold-centers",
    "manifold-nphi",
    "manifold-neighbors",
    "manifold-lambda",
    "manifold-sync",
)


def main(argv: list[str] | None = None) -> int:
    from experiments.suite import main as run_suite

    return run_suite("manifold", argv)


if __name__ == "__main__":
    raise SystemExit(main())
