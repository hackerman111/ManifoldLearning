"""Multi v2 experiment catalog with at least seven points per series."""

from __future__ import annotations

import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import multi
from experiments.models import Experiment

_NOISE_LEVELS = (0.0, 0.2, 0.6, 1.0)
_LINK_SCALES = (1.0, 2.0, 3.0, 4.0)
_EXCLUDED_SOURCE_SELECTORS = frozenset({"mi-3", "mi-5"})

_BINARY_STRESS = {
    "mi-training": (
        "training_set",
        "Проверяется, меняет ли обучение на всех наблюдениях или без центров "
        "уровень шума, на котором падает восстановление.",
    ),
    "mi-init": (
        "index_init",
        "Проверяется, сдвигает ли локальная и случайная инициализация шумовую "
        "границу восстановления.",
    ),
    "mi-tensor": (
        "multi_tensor",
        "Проверяется, различаются ли шумовые границы восстановления для полного "
        "и ортогонального тензора.",
    ),
    "mi-direction-law": (
        "direction_mode",
        "Проверяется, различаются ли шумовые границы восстановления для "
        "изотропного и локализованного закона направлений.",
    ),
    "mi-direction-refresh": (
        "redraw_directions",
        "Проверяется, меняет ли обновление направлений шумовую границу восстановления.",
    ),
    "mi-focus-tensor": (
        "multi_tensor",
        "При одинаковых данных проверяется, сохраняется ли преимущество полного "
        "тензора при росте шума.",
    ),
    "mi-focus-init": (
        "index_init",
        "При одинаковых данных проверяется, остаётся ли локальная инициализация "
        "устойчивее случайной при росте шума.",
    ),
}


def _expand_short_grid(experiment: Experiment) -> Experiment:
    """Расширить исходные короткие сравнения независимым стресс-фактором."""
    if experiment.selector == "mi-link":
        links = tuple(dict.fromkeys(point.link for point in experiment.full))
        points = tuple(
            replace(experiment.full[0], link=link, link_scale=scale)
            for link, scale in product(links, _LINK_SCALES)
        )
        return replace(
            experiment,
            full=points,
            report_fields=("link", "link_scale"),
            condition_field="link_scale",
            common_random_fields=("link", "link_scale"),
            condition_group_fields=("link",),
            hypothesis=(
                "Проверяется, как тип link меняет частотную границу "
                "trace_score >= 0.95 при link_scale от 1 до 4."
            ),
        )

    factor, hypothesis = _BINARY_STRESS[experiment.selector]
    points = tuple(
        replace(point, sigma_eps=noise)
        for point in experiment.full
        for noise in _NOISE_LEVELS
    )
    return replace(
        experiment,
        full=points,
        report_fields=(factor, "sigma_eps"),
        condition_field="sigma_eps",
        common_random_fields=(factor, "sigma_eps"),
        condition_group_fields=(factor,),
        hypothesis=hypothesis,
    )


def _v2_selector(selector: str) -> str:
    return f"multiv2-{selector}"


def catalog() -> tuple[Experiment, ...]:
    """Return the retained multi series with at least seven grid points each."""
    result = []
    for original in multi.catalog():
        if original.selector in _EXCLUDED_SOURCE_SELECTORS:
            continue
        experiment = original
        if len(experiment.full) < 7:
            experiment = _expand_short_grid(experiment)
        if len(experiment.full) < 7:
            raise ValueError(
                f"{original.selector}: Multi v2 требует не менее семи точек"
            )
        title = experiment.title.removeprefix("Multi-index: ")
        result.append(
            replace(
                experiment,
                selector=_v2_selector(original.selector),
                title=f"Multi v2: {title}",
            )
        )
    return tuple(result)


CATALOG = {experiment.selector: experiment for experiment in catalog()}
DEFAULT_SELECTORS = tuple(
    _v2_selector(name)
    for name in multi.DEFAULT_SELECTORS
    if name not in _EXCLUDED_SOURCE_SELECTORS
)


def main(argv: list[str] | None = None) -> int:
    from experiments.suite import main as run_suite

    return run_suite(
        "multi",
        argv,
        catalog_override=CATALOG,
        default_selectors=DEFAULT_SELECTORS,
        run_label="multiv2",
    )


if __name__ == "__main__":
    raise SystemExit(main())
