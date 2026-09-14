"""single: сетки и самостоятельный запуск исследовательского набора."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ADP.cli.experiment_utils import validate_custom_parameters
from experiments._grids import _parameter_scaling_catalog, _scaling_point, _smoke_point
from experiments.models import Experiment, ExperimentPoint


def _legacy_catalog() -> tuple[Experiment, ...]:
    experiments = (
        Experiment(
            "1",
            "Проверка корректности",
            ExperimentPoint(4, 5, link="linear", sigma_eps=0),
            tuple(
                ExperimentPoint(d, ratio, link=link, sigma_eps=0)
                for d, ratio, link in product(
                    (5, 25), (5.0, 10.0), ("linear", "quadratic")
                )
            ),
        ),
        Experiment(
            "2",
            "Масштабирование",
            ExperimentPoint(4, 2),
            tuple(
                ExperimentPoint(d, ratio)
                for d, ratio in product(
                    (5, 25, 50, 100),
                    (1.15, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
                )
            ),
            report_fields=("d", "n_over_d"),
            full_runs=25,
            quality_threshold=0.9,
        ),
        Experiment(
            "3",
            "Устойчивость к шуму",
            ExperimentPoint(4, 5, sigma_eps=1),
            tuple(
                ExperimentPoint(d, ratio, sigma_eps=noise)
                for d, ratio, noise in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (
                        0.0,
                        0.2,
                        0.316,
                        0.4,
                        0.5,
                        0.6,
                        0.707,
                        0.8,
                        1.0,
                        1.414,
                        2.0,
                    ),
                )
            ),
            report_fields=("d", "n_over_d", "sigma_eps"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
        ),
        Experiment(
            "4",
            "Коррелированные признаки",
            ExperimentPoint(4, 5, rho_corr=0.5),
            tuple(
                ExperimentPoint(d, ratio, rho_corr=rho)
                for d, ratio, rho in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99),
                )
            ),
            report_fields=("d", "n_over_d", "rho_corr"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="rho_corr",
            common_random_fields=("rho_corr",),
        ),
        Experiment(
            "5",
            "Масштаб признаков",
            ExperimentPoint(
                4,
                5,
                sigma_x=2,
                normalize_link_by_sigma_x=True,
            ),
            tuple(
                ExperimentPoint(
                    d,
                    ratio,
                    sigma_x=scale,
                    normalize_link_by_sigma_x=True,
                )
                for d, ratio, scale in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                )
            ),
            report_fields=("d", "n_over_d", "sigma_x"),
            full_runs=25,
            quality_threshold=0.9,
            condition_field="sigma_x",
            common_random_fields=("sigma_x",),
        ),
        Experiment(
            "6",
            "Функции связи",
            ExperimentPoint(4, 5, link="sin"),
            tuple(
                ExperimentPoint(d, ratio, link=link)
                for d, ratio, link in product(
                    (25, 100),
                    (2.0, 5.0, 10.0),
                    ("linear", "quadratic", "square", "sin", "tanh", "oscillating"),
                )
            ),
            condition_field="link",
            common_random_fields=("link",),
        ),
        Experiment(
            "7.1",
            "Распределения признаков",
            ExperimentPoint(4, 5, x_distribution="uniform"),
            tuple(
                ExperimentPoint(d, ratio, x_distribution=distribution)
                for d, ratio, distribution in product(
                    (25, 100), (2.0, 5.0), ("gaussian", "uniform", "student_t5")
                )
            ),
            condition_field="x_distribution",
            common_random_fields=("x_distribution",),
        ),
        Experiment(
            "7.2",
            "Распределения шума",
            ExperimentPoint(4, 5, noise_distribution="student_t5"),
            tuple(
                ExperimentPoint(d, ratio, noise_distribution=distribution)
                for d, ratio, distribution in product(
                    (25, 100), (2.0, 5.0), ("gaussian", "student_t5", "student_t3")
                )
            ),
            condition_field="noise_distribution",
            common_random_fields=("noise_distribution",),
        ),
        Experiment(
            "8.1",
            "Гетероскедастичность",
            ExperimentPoint(4, 5, heteroscedastic=True),
            tuple(
                ExperimentPoint(d, ratio, heteroscedastic=value)
                for d, ratio, value in product((25, 100), (2.0, 5.0), (False, True))
            ),
            condition_field="heteroscedastic",
            common_random_fields=("heteroscedastic",),
        ),
        Experiment(
            "8.2",
            "Выбросы",
            ExperimentPoint(4, 5, outlier_fraction=0.01, outlier_scale=5),
            tuple(
                ExperimentPoint(
                    d,
                    ratio,
                    outlier_fraction=fraction,
                    outlier_scale=scale,
                )
                for d, ratio, (fraction, scale) in product(
                    (25, 100),
                    (2.0, 5.0),
                    (
                        (0.0, 1.0),
                        (0.005, 5.0),
                        (0.01, 5.0),
                        (0.01, 10.0),
                        (0.025, 7.5),
                        (0.05, 5.0),
                        (0.05, 10.0),
                    ),
                )
            ),
            common_random_fields=("outlier_fraction", "outlier_scale"),
        ),
        Experiment(
            "8.3",
            "Нарушение одноиндексной модели",
            ExperimentPoint(4, 5, delta=0.1),
            tuple(
                ExperimentPoint(d, ratio, delta=delta)
                for d, ratio, delta in product(
                    (25, 100),
                    (2.0, 5.0),
                    (0.0, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5),
                )
            ),
            condition_field="delta",
            common_random_fields=("delta",),
        ),
    )
    return experiments


def custom_experiment(
    n: int = 240,
    d: int = 5,
    noise: float = 0.05,
) -> Experiment:
    validate_custom_parameters(n, d, noise)
    point = ExperimentPoint(d=d, n_over_d=n / d, sigma_eps=noise, link="sin")
    return Experiment("custom", "Пользовательский эксперимент", point, (point,))


def _report_catalog() -> tuple[Experiment, ...]:
    si = ExperimentPoint(
        d=100,
        n_over_d=10,
        n_samples=1000,
        sigma_eps=0.2,
        tau=0.4,
        link="x_sin",
        link_scale=3,
        N_loc=20,
        N_lin=160,
        N_J=500,
        N_phi=20,
        outer_steps=3,
        lambda_penalty=0.05,
        a=math.sqrt(2),
        h_min_factor=3,
        index_init="local",
        direction_mode="localized",
        center_displacement=0.1,
        training_set="all",
        redraw_directions=True,
        solver_max_steps=3,
    )

    n_values = (800, 1000, 1200, 2000)

    n_sweep_values = (800, 900, 1000, 1200, 1400, 1600, 2000)

    d_values = tuple(range(10, 101, 10))

    noise_values = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)

    tau_values = (0.0, 0.2, 0.4, 0.8)

    tau_sweep_values = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)

    scales = (1.0, 2.0, 3.0, 4.0)

    scale_sweep_values = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    sigma_x_values = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)

    n_lin_scales = tuple(range(1, 8))

    n_loc_values = (7, 9, 10, 12, 15, 17, 20)

    lambdas = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

    displacements = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)

    ao_steps = tuple(range(1, 8))

    single_directions = (10, 15, 20, 30, 40, 50, 60)

    single_a = tuple(2 ** (1 / 4 + step / 8) for step in range(7))

    h_min_factors = (1.0, 4 / 3, 5 / 3, 2.0, 7 / 3, 8 / 3, 3.0)

    center_counts = tuple(
        scale * si.n // (si.N_loc or 1) for scale in (1, 2, 3, 5, 10, 15, 20)
    )

    def make(
        selector: str,
        title: str,
        points: tuple[ExperimentPoint, ...],
        *report_fields: str,
        full_runs: int = 1,
    ) -> Experiment:
        condition_field = report_fields[0] if len(report_fields) == 1 else None
        return Experiment(
            selector,
            title,
            _smoke_point(points[0]),
            points,
            report_fields,
            full_runs,
            condition_field=condition_field,
            common_random_fields=(condition_field,) if condition_field else (),
        )

    single = (
        make(
            "si-n",
            "Single-index: объём выборки",
            tuple(replace(si, n_samples=n, N_J=n // 2) for n in n_sweep_values),
            "n_samples",
        ),
        make(
            "si-d",
            "Single-index: размерность",
            tuple(replace(si, d=d, N_lin=d + 60) for d in d_values),
            "d",
        ),
        make(
            "si-noise",
            "Single-index: шум",
            tuple(replace(si, sigma_eps=value) for value in noise_values),
            "sigma_eps",
        ),
        make(
            "si-tau",
            "Single-index: общий фактор",
            tuple(replace(si, tau=value) for value in tau_sweep_values),
            "tau",
        ),
        make(
            "si-link",
            "Single-index: функция связи",
            tuple(replace(si, link=link) for link in ("sin_scaled", "x_sin")),
            "link",
        ),
        make(
            "si-frequency-sin",
            "Single-index: частота sin(sx)",
            tuple(
                replace(si, link="sin_scaled", link_scale=s) for s in scale_sweep_values
            ),
            "link_scale",
        ),
        make(
            "si-frequency-xsin",
            "Single-index: частота x sin(sx)",
            tuple(replace(si, link="x_sin", link_scale=s) for s in scale_sweep_values),
            "link_scale",
        ),
        make(
            "si-scale",
            "Single-index: масштаб признаков при d=10",
            tuple(
                replace(si, d=10, N_lin=70, sigma_x=value) for value in sigma_x_values
            ),
            "sigma_x",
        ),
        make(
            "si-nlin",
            "Single-index: локальная линейная масса",
            tuple(replace(si, N_lin=si.d + s * 20) for s in n_lin_scales),
            "N_lin",
        ),
        make(
            "si-centers",
            "Single-index: число центров",
            tuple(replace(si, N_J=value) for value in center_counts),
            "N_J",
        ),
        make(
            "si-displacement",
            "Single-index: смещение центров",
            tuple(replace(si, center_displacement=value) for value in displacements),
            "center_displacement",
        ),
        make(
            "si-training",
            "Single-index: обучающая выборка",
            tuple(
                replace(si, training_set=value) for value in ("all", "exclude_centers")
            ),
            "training_set",
        ),
        make(
            "si-kmax",
            "Single-index: число внутренних AO-шагов",
            tuple(replace(si, solver_max_steps=value) for value in ao_steps),
            "solver_max_steps",
        ),
        make(
            "si-nloc",
            "Single-index: локальная масса",
            tuple(replace(si, N_loc=value) for value in n_loc_values),
            "N_loc",
        ),
        make(
            "si-nphi",
            "Single-index: число направлений",
            tuple(replace(si, N_phi=value) for value in single_directions),
            "N_phi",
        ),
        make(
            "si-lambda",
            "Single-index: регуляризация",
            tuple(replace(si, lambda_penalty=value) for value in lambdas),
            "lambda_penalty",
        ),
        make(
            "si-a",
            "Single-index: коэффициент уменьшения h",
            tuple(replace(si, a=value) for value in single_a),
            "a",
        ),
        make(
            "si-hmin",
            "Single-index: нижняя граница h",
            tuple(replace(si, h_min_factor=value) for value in h_min_factors),
            "h_min_factor",
        ),
    )

    hypothesis_base = replace(
        si,
        d=30,
        n_over_d=20,
        n_samples=600,
        N_lin=90,
        N_J=200,
        N_phi=20,
        outer_steps=5,
        solver_max_steps=8,
        link_scale=1.0,
    )

    function_classes = (
        "linear",
        "quadratic",
        "square",
        "cubic",
        "quartic",
        "sin_scaled",
        "cos_scaled",
        "x_sin",
        "tanh_scaled",
        "absolute",
        "relu",
        "gaussian_bump",
    )

    parity_scales = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

    support_scales = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

    init_links = (
        "linear",
        "square",
        "cos_scaled",
        "tanh_scaled",
        "relu",
        "gaussian_bump",
    )

    single_hypotheses = (
        Experiment(
            "si-function-classes",
            "Single-index: классы функций связи",
            _smoke_point(replace(hypothesis_base, link=function_classes[0])),
            tuple(replace(hypothesis_base, link=link) for link in function_classes),
            report_fields=("link",),
            full_runs=3,
            common_random_fields=("link",),
            hypothesis=(
                "Функции, имеющие информативную производную на широкой области "
                "значений "
                "индекса восстанавливаются лучше локализованных, насыщаемых и "
                "осциллирующих функций."
            ),
        ),
        Experiment(
            "si-parity-frequency",
            "Single-index: чётность и частота",
            _smoke_point(replace(hypothesis_base, link="sin_scaled", link_scale=0.5)),
            tuple(
                replace(hypothesis_base, link=link, link_scale=scale)
                for link, scale in product(("sin_scaled", "cos_scaled"), parity_scales)
            ),
            report_fields=("link", "link_scale"),
            full_runs=3,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "При одинаковой частоте чётность функции не должна существенно "
                "влиять на восстановление направления; рост частоты должен ухудшать "
                "локальную оценку обеих функций."
            ),
        ),
        Experiment(
            "si-gradient-support",
            "Single-index: ширина информативной области производной",
            _smoke_point(replace(hypothesis_base, link="tanh_scaled", link_scale=0.25)),
            tuple(
                replace(hypothesis_base, link=link, link_scale=scale)
                for link, scale in product(
                    ("tanh_scaled", "gaussian_bump"), support_scales
                )
            ),
            report_fields=("link", "link_scale"),
            full_runs=3,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "Увеличение масштаба сосредоточивает производную tanh и gaussian bump "
                "на меньшей доле выборки, поэтому качество и вероятность сходимости "
                "должны снижаться."
            ),
        ),
        Experiment(
            "si-init-by-function",
            "Single-index: инициализация по классам функций",
            _smoke_point(
                replace(hypothesis_base, link=init_links[0], index_init="local")
            ),
            tuple(
                replace(hypothesis_base, link=link, index_init=index_init)
                for link, index_init in product(init_links, ("local", "random"))
            ),
            report_fields=("link", "index_init"),
            full_runs=3,
            common_random_fields=("link", "index_init"),
            hypothesis=(
                "Локальная градиентная инициализация превосходит случайную, причём "
                "её преимущество сильнее для чётных, негладких и локализованных "
                "функций."
            ),
        ),
    )

    focused_dimensions = (40, 44, 48, 50, 52, 56, 60)

    focused_frequencies = (2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0)

    focused_tau = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)

    focused_displacements = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)

    single_focus = (
        Experiment(
            "si-focus-d",
            "Single-index: повтор границы размерности",
            _smoke_point(
                replace(
                    si,
                    d=focused_dimensions[0],
                    N_lin=focused_dimensions[0] + 60,
                )
            ),
            tuple(replace(si, d=d, N_lin=d + 60) for d in focused_dimensions),
            report_fields=("d",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="d",
            common_random_fields=("d", "N_lin"),
            hypothesis=(
                "Граница устойчивого восстановления проходит внутри диапазона d=40--60."
            ),
        ),
        Experiment(
            "si-focus-frequency",
            "Single-index: повтор границы частоты",
            _smoke_point(
                replace(si, link="sin_scaled", link_scale=focused_frequencies[0])
            ),
            tuple(
                replace(si, link=link, link_scale=scale)
                for link, scale in product(("sin_scaled", "x_sin"), focused_frequencies)
            ),
            report_fields=("link", "link_scale"),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "Для sin(sx) качество резко падает около s=3--3.5, тогда как "
                "x sin(sx) остаётся более трудной и немонотонной функцией."
            ),
        ),
        Experiment(
            "si-focus-tau",
            "Single-index: повтор границы общего фактора",
            _smoke_point(replace(si, tau=focused_tau[0])),
            tuple(replace(si, tau=value) for value in focused_tau),
            report_fields=("tau",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="tau",
            common_random_fields=("tau",),
            hypothesis=(
                "Резкий переход к качественному восстановлению происходит между "
                "tau=0.7 и tau=0.8."
            ),
        ),
        Experiment(
            "si-focus-displacement",
            "Single-index: повтор границы смещения центров",
            _smoke_point(replace(si, center_displacement=focused_displacements[0])),
            tuple(
                replace(si, center_displacement=value)
                for value in focused_displacements
            ),
            report_fields=("center_displacement",),
            full_runs=5,
            quality_threshold=0.9,
            condition_field="center_displacement",
            common_random_fields=("center_displacement",),
            hypothesis=(
                "Умеренное смещение улучшает восстановление, тогда как приближение "
                "к единице повышает риск пустой компактной поддержки."
            ),
        ),
    )

    link_cases = tuple(
        (link, scale) for link in ("sin_scaled", "x_sin") for scale in scales
    )

    single_breaking = tuple(
        replace(
            si,
            n_samples=n,
            N_J=n // 2,
            d=d,
            N_lin=d + 60,
            sigma_eps=noise,
            tau=tau,
            link=link,
            link_scale=scale,
        )
        for n, d, noise, tau, (link, scale) in product(
            n_values,
            d_values,
            noise_values,
            tau_values,
            link_cases,
        )
    )

    return (
        *single,
        *single_hypotheses,
        *single_focus,
        make(
            "si-breaking",
            "Single-index: полная breaking-dimension сетка",
            single_breaking,
            "n_samples",
            "d",
            "sigma_eps",
            "tau",
            "link",
            "link_scale",
            full_runs=100,
        ),
    )


def _nd_catalog() -> tuple[Experiment, ...]:
    si_points = tuple(
        replace(_scaling_point("single", n, d), N_loc=n_loc)
        for n, d, n_loc in product(
            (400, 700, 1000, 1500, 2200),
            (10, 20, 30, 40, 50, 60),
            (10, 15, 20, 25, 30, 40, 50),
        )
    )

    return (
        Experiment(
            "si-nloc-nd",
            "Single-index: N_loc в зависимости от n и d",
            _smoke_point(si_points[0]),
            si_points,
            report_fields=("n_samples", "d", "N_loc"),
            full_runs=10,
            quality_threshold=0.9,
            condition_field="N_loc",
            common_random_fields=("N_loc",),
            hypothesis=(
                "Минимальный N_loc для устойчивой сходимости растёт при росте d; "
                "после насыщения качества дальнейший рост оплачивается временем."
            ),
        ),
    )


def catalog() -> tuple[Experiment, ...]:
    """Исходные сетки семейства, включая дорогие полные сетки."""
    return (
        *_legacy_catalog(),
        *_report_catalog(),
        *_parameter_scaling_catalog("single"),
        *_nd_catalog(),
    )


DEFAULT_SELECTORS = (
    "1",
    "si-n",
    "si-d",
    "si-noise",
    "si-tau",
    "si-scale",
    "si-function-classes",
    "si-parity-frequency",
    "si-gradient-support",
    "si-init-by-function",
    "7.1",
    "7.2",
    "8.1",
    "8.2",
    "8.3",
    "si-nlin",
    "si-centers",
    "si-displacement",
    "si-training",
    "si-kmax",
    "si-nloc",
    "si-nphi",
    "si-lambda",
    "si-a",
    "si-hmin",
)


def main(argv: list[str] | None = None) -> int:
    from experiments.suite import main as run_suite

    return run_suite("single", argv)


if __name__ == "__main__":
    raise SystemExit(main())
