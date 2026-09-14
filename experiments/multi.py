"""multi: сетки и самостоятельный запуск исследовательского набора."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._grids import _parameter_scaling_catalog, _scaling_point, _smoke_point
from experiments.models import Experiment, ExperimentPoint


def _detailed_multi_catalog() -> tuple[Experiment, ...]:
    """Крупные MI-сетки границы восстановления."""
    base = ExperimentPoint(
        d=25,
        n_over_d=2,
        mode="multi",
        index_dim=2,
        link="multi_additive",
        link_scale=3,
        solver_max_steps=5,
    )
    dimensions = (25, 50)
    ratios = (2.0, 5.0, 10.0)

    def make(
        selector: str,
        title: str,
        points: tuple[ExperimentPoint, ...],
        report_fields: tuple[str, ...],
        *,
        condition_field: str | None = None,
        common_random_fields: tuple[str, ...] = (),
        condition_group_fields: tuple[str, ...] = (),
    ) -> Experiment:
        return Experiment(
            selector,
            title,
            _smoke_point(points[0]),
            points,
            report_fields=report_fields,
            full_runs=25,
            quality_threshold=0.95,
            condition_field=condition_field,
            common_random_fields=common_random_fields,
            condition_group_fields=condition_group_fields,
        )

    scaling = tuple(
        replace(base, d=d, n_over_d=ratio)
        for d, ratio in product(
            (5, 25, 50),
            (1.15, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        )
    )
    noise = tuple(
        replace(base, d=d, n_over_d=ratio, sigma_eps=value)
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.0, 0.2, 0.316, 0.4, 0.5, 0.6, 0.707, 0.8, 1.0, 1.414, 2.0),
        )
    )
    correlation = tuple(
        replace(base, d=d, n_over_d=ratio, rho_corr=value)
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99),
        )
    )
    scale = tuple(
        replace(
            base,
            d=d,
            n_over_d=ratio,
            sigma_x=value,
            normalize_link_by_sigma_x=True,
        )
        for d, ratio, value in product(
            dimensions,
            ratios,
            (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        )
    )
    index_dimension = tuple(
        replace(
            base,
            d=d,
            n_over_d=ratio,
            index_dim=index_dim,
            link=link,
            link_scale=link_scale,
            basis_pool_dim=10,
        )
        for d, ratio, index_dim, link, link_scale in product(
            (25, 50),
            ratios,
            (2, 3, 5, 7, 10),
            ("multi_additive", "multi_multiplicative"),
            (1.0, 2.0, 3.0, 4.0),
        )
    )
    return (
        make(
            "mi-1",
            "Multi-index: граница по размеру задачи",
            scaling,
            ("d", "n_over_d"),
        ),
        make(
            "mi-2",
            "Multi-index: устойчивость к шуму",
            noise,
            ("d", "n_over_d", "sigma_eps"),
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
        ),
        make(
            "mi-3",
            "Multi-index: коррелированные признаки",
            correlation,
            ("d", "n_over_d", "rho_corr"),
            condition_field="rho_corr",
            common_random_fields=("rho_corr",),
        ),
        make(
            "mi-4",
            "Multi-index: масштаб признаков",
            scale,
            ("d", "n_over_d", "sigma_x"),
            condition_field="sigma_x",
            common_random_fields=("sigma_x",),
        ),
        make(
            "mi-5",
            "Multi-index: размер подпространства и link",
            index_dimension,
            ("d", "n_over_d", "index_dim", "link", "link_scale"),
            condition_field="index_dim",
            common_random_fields=("index_dim", "link", "link_scale"),
            condition_group_fields=("link", "link_scale"),
        ),
    )


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

    mi = replace(
        si,
        mode="multi",
        index_dim=2,
        link="multi_additive",
        N_phi=40,
        direction_mode="isotropic",
        multi_tensor="orthogonal",
        solver_max_steps=5,
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

    multi_directions = (3, 5, 10, 20, 30, 40, 60)

    multi_a = tuple(2 ** (1 / 4 + step / 24) for step in range(7))

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

    multi = (
        make(
            "mi-n",
            "Multi-index: объём выборки",
            tuple(replace(mi, n_samples=n, N_J=n // 2) for n in n_sweep_values),
            "n_samples",
        ),
        make(
            "mi-d",
            "Multi-index: размерность",
            tuple(replace(mi, d=d, N_lin=d + 60) for d in d_values),
            "d",
        ),
        make(
            "mi-noise",
            "Multi-index: шум",
            tuple(replace(mi, sigma_eps=value) for value in noise_values),
            "sigma_eps",
        ),
        make(
            "mi-tau",
            "Multi-index: общий фактор",
            tuple(replace(mi, tau=value) for value in tau_sweep_values),
            "tau",
        ),
        make(
            "mi-link",
            "Multi-index: функция связи",
            tuple(
                replace(mi, link=link)
                for link in ("multi_additive", "multi_multiplicative")
            ),
            "link",
        ),
        make(
            "mi-frequency-additive",
            "Multi-index: частота additive link",
            tuple(replace(mi, link_scale=s) for s in scale_sweep_values),
            "link_scale",
        ),
        make(
            "mi-frequency-multiplicative",
            "Multi-index: частота multiplicative link",
            tuple(
                replace(mi, link="multi_multiplicative", link_scale=s)
                for s in scale_sweep_values
            ),
            "link_scale",
        ),
        make(
            "mi-scale",
            "Multi-index: масштаб признаков при d=10",
            tuple(
                replace(mi, d=10, N_lin=70, sigma_x=value) for value in sigma_x_values
            ),
            "sigma_x",
        ),
        make(
            "mi-nlin",
            "Multi-index: локальная линейная масса",
            tuple(replace(mi, N_lin=mi.d + s * 20) for s in n_lin_scales),
            "N_lin",
        ),
        make(
            "mi-centers",
            "Multi-index: число центров",
            tuple(replace(mi, N_J=value) for value in center_counts),
            "N_J",
        ),
        make(
            "mi-displacement",
            "Multi-index: смещение центров",
            tuple(replace(mi, center_displacement=value) for value in displacements),
            "center_displacement",
        ),
        make(
            "mi-training",
            "Multi-index: обучающая выборка",
            tuple(
                replace(mi, training_set=value) for value in ("all", "exclude_centers")
            ),
            "training_set",
        ),
        make(
            "mi-init",
            "Multi-index: инициализация",
            tuple(replace(mi, index_init=value) for value in ("local", "random")),
            "index_init",
        ),
        make(
            "mi-kmax",
            "Multi-index: число внутренних AO-шагов",
            tuple(replace(mi, solver_max_steps=value) for value in ao_steps),
            "solver_max_steps",
        ),
        make(
            "mi-nloc",
            "Multi-index: локальная масса",
            tuple(replace(mi, N_loc=value) for value in n_loc_values),
            "N_loc",
        ),
        make(
            "mi-nphi",
            "Multi-index: число направлений",
            tuple(replace(mi, N_phi=value) for value in multi_directions),
            "N_phi",
        ),
        make(
            "mi-lambda",
            "Multi-index: регуляризация",
            tuple(replace(mi, lambda_penalty=value) for value in lambdas),
            "lambda_penalty",
        ),
        make(
            "mi-a",
            "Multi-index: коэффициент уменьшения h",
            tuple(replace(mi, a=value) for value in multi_a),
            "a",
        ),
        make(
            "mi-hmin",
            "Multi-index: нижняя граница h",
            tuple(replace(mi, h_min_factor=value) for value in h_min_factors),
            "h_min_factor",
        ),
        make(
            "mi-tensor",
            "Multi-index: тензор локализации",
            tuple(replace(mi, multi_tensor=value) for value in ("orthogonal", "full")),
            "multi_tensor",
        ),
        make(
            "mi-direction-law",
            "Multi-index: закон направлений",
            tuple(
                replace(mi, direction_mode=value)
                for value in ("isotropic", "localized")
            ),
            "direction_mode",
        ),
        make(
            "mi-direction-refresh",
            "Multi-index: обновление направлений",
            tuple(replace(mi, redraw_directions=value) for value in (False, True)),
            "redraw_directions",
        ),
    )

    multi_focus_base = replace(
        mi,
        d=25,
        n_over_d=40,
        n_samples=1000,
        N_loc=12,
        N_lin=85,
        N_J=500,
        N_phi=30,
        link_scale=3.0,
        multi_tensor="full",
    )

    d_boundary = (20, 22, 24, 25, 26, 28, 30)

    additive_frequencies = (2.0, 2.15, 2.3, 2.5, 2.7, 2.85, 3.0)

    focused_directions = (20, 24, 28, 30, 32, 36, 40)

    high_noise = (0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)

    multi_focus = (
        Experiment(
            "mi-focus-d",
            "Multi-index: уточнение границы размерности",
            _smoke_point(
                replace(
                    multi_focus_base,
                    d=d_boundary[0],
                    n_over_d=1000 / d_boundary[0],
                    N_lin=d_boundary[0] + 60,
                )
            ),
            tuple(
                replace(
                    multi_focus_base,
                    d=d,
                    n_over_d=1000 / d,
                    N_lin=d + 60,
                )
                for d in d_boundary
            ),
            report_fields=("d",),
            full_runs=3,
            quality_threshold=0.95,
            condition_field="d",
            common_random_fields=("d", "n_over_d", "N_lin"),
            hypothesis=(
                "После настройки остальных параметров граница восстановления "
                "остаётся внутри диапазона d=20--30."
            ),
        ),
        Experiment(
            "mi-focus-frequency-additive",
            "Multi-index: уточнение частоты additive-link",
            _smoke_point(replace(multi_focus_base, link_scale=additive_frequencies[0])),
            tuple(
                replace(multi_focus_base, link_scale=scale)
                for scale in additive_frequencies
            ),
            report_fields=("link_scale",),
            full_runs=3,
            quality_threshold=0.95,
            condition_field="link_scale",
            common_random_fields=("link_scale",),
            hypothesis=(
                "Увеличение частоты additive-link от 2 до 3 монотонно ухудшает "
                "восстановление подпространства."
            ),
        ),
        Experiment(
            "mi-focus-nphi",
            "Multi-index: уточнение числа направлений",
            _smoke_point(replace(multi_focus_base, N_phi=focused_directions[0])),
            tuple(
                replace(multi_focus_base, N_phi=value) for value in focused_directions
            ),
            report_fields=("N_phi",),
            full_runs=3,
            quality_threshold=0.95,
            condition_field="N_phi",
            common_random_fields=("N_phi",),
            hypothesis=(
                "Рост N_phi улучшает качество примерно до 30 направлений, после "
                "чего дополнительная память не даёт заметного выигрыша."
            ),
        ),
        Experiment(
            "mi-focus-tensor",
            "Multi-index: повторная проверка тензора",
            _smoke_point(replace(multi_focus_base, multi_tensor="orthogonal")),
            tuple(
                replace(multi_focus_base, multi_tensor=value)
                for value in ("orthogonal", "full")
            ),
            report_fields=("multi_tensor",),
            full_runs=3,
            common_random_fields=("multi_tensor",),
            hypothesis=(
                "Полный тензор локализации даёт меньшую ошибку проектора, чем "
                "ортогональный вариант."
            ),
        ),
        Experiment(
            "mi-focus-init",
            "Multi-index: повторная проверка инициализации",
            _smoke_point(replace(multi_focus_base, index_init="local")),
            tuple(
                replace(multi_focus_base, index_init=value)
                for value in ("local", "random")
            ),
            report_fields=("index_init",),
            full_runs=3,
            common_random_fields=("index_init",),
            hypothesis=(
                "Локальная градиентная инициализация устойчивее случайной при "
                "одинаковых данных и алгоритмической случайности."
            ),
        ),
        Experiment(
            "mi-focus-noise",
            "Multi-index: высокая шумовая граница",
            _smoke_point(replace(multi_focus_base, sigma_eps=high_noise[0])),
            tuple(replace(multi_focus_base, sigma_eps=value) for value in high_noise),
            report_fields=("sigma_eps",),
            full_runs=3,
            quality_threshold=0.95,
            condition_field="sigma_eps",
            common_random_fields=("sigma_eps",),
            hypothesis=(
                "При sigma_eps от 0.6 до 1.0 ошибка проектора возрастает, причём "
                "ADP перестаёт улучшать начальное подпространство."
            ),
        ),
    )

    multi_link_cases = tuple(
        (link, scale)
        for link in ("multi_additive", "multi_multiplicative")
        for scale in scales
    )

    multi_breaking = tuple(
        replace(
            mi,
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
            multi_link_cases,
        )
    )

    return (
        *multi,
        *multi_focus,
        make(
            "mi-breaking",
            "Multi-index: полная breaking-dimension сетка",
            multi_breaking,
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
    mi_points = tuple(
        _scaling_point("multi", n, d)
        for n, d in product(
            (400, 600, 800, 1000, 1400, 1800, 2400),
            (10, 15, 20, 25, 30, 35, 40, 50, 60),
        )
    )

    return (
        Experiment(
            "mi-boundary-nd",
            "Multi-index: подробная граница по n и d",
            _smoke_point(mi_points[0]),
            mi_points,
            report_fields=("n_samples", "d", "n_over_d"),
            full_runs=10,
            quality_threshold=0.95,
            hypothesis=(
                "При фиксированном estimator граница trace_score >= 0.95 "
                "зависит и от n/d, и от абсолютных n и d."
            ),
        ),
    )


def catalog() -> tuple[Experiment, ...]:
    """Исходные сетки семейства, включая дорогие полные сетки."""
    return (
        *_detailed_multi_catalog(),
        *_report_catalog(),
        *_parameter_scaling_catalog("multi"),
        *_nd_catalog(),
    )


DEFAULT_SELECTORS = (
    "mi-n",
    "mi-d",
    "mi-noise",
    "mi-tau",
    "mi-scale",
    "mi-3",
    "mi-5",
    "mi-link",
    "mi-frequency-additive",
    "mi-frequency-multiplicative",
    "mi-nlin",
    "mi-centers",
    "mi-displacement",
    "mi-training",
    "mi-init",
    "mi-kmax",
    "mi-nloc",
    "mi-nphi",
    "mi-lambda",
    "mi-a",
    "mi-hmin",
    "mi-tensor",
    "mi-direction-law",
    "mi-direction-refresh",
)


def main(argv: list[str] | None = None) -> int:
    from experiments.suite import main as run_suite

    return run_suite("multi", argv)


if __name__ == "__main__":
    raise SystemExit(main())
