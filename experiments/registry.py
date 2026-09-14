"""Объединение каталогов и совместимые селекторы CLI."""

from __future__ import annotations

from collections.abc import Mapping

from ADP.cli.experiment_utils import select_experiments

from . import manifold, multi, single
from .models import Experiment


def _catalog() -> dict[str, Experiment]:
    si = single.catalog()
    mi = multi.catalog()
    by_name = {item.selector: item for item in (*si, *mi, *manifold.catalog())}
    # Сохраняем исторический порядок селекторов и общих запусков.
    names = (
        *(item.selector for item in si if item.selector[0].isdigit()),
        *(f"mi-{i}" for i in range(1, 6)),
        *(
            item.selector
            for item in si
            if item.selector.startswith("si-")
            and item.selector not in {"si-breaking", "si-nloc-nd"}
        ),
        *(
            item.selector
            for item in mi
            if item.selector.startswith("mi-")
            and item.selector
            not in {"mi-breaking", "mi-boundary-nd", *(f"mi-{i}" for i in range(1, 6))}
        ),
        "si-breaking",
        "mi-breaking",
        *(name for name in by_name if name.startswith("scale-")),
        "mi-boundary-nd",
        "si-nloc-nd",
        *(name for name in by_name if name.startswith("manifold")),
    )
    return {
        **{name: by_name[name] for name in names},
        "custom": single.custom_experiment(),
    }


CATALOG: Mapping[str, Experiment] = _catalog()


def _selected_experiments(
    value: str,
    *,
    custom: Experiment,
) -> tuple[Experiment, ...]:
    catalog = {**CATALOG, "custom": custom}
    single_hypotheses = (
        "si-function-classes",
        "si-parity-frequency",
        "si-gradient-support",
        "si-init-by-function",
    )
    focused_single = (
        "si-focus-d",
        "si-focus-frequency",
        "si-focus-tau",
        "si-focus-displacement",
    )
    focused_multi = (
        "mi-focus-d",
        "mi-focus-frequency-additive",
        "mi-focus-nphi",
        "mi-focus-tensor",
        "mi-focus-init",
        "mi-focus-noise",
    )
    detailed_nd = ("mi-boundary-nd", "si-nloc-nd")
    parameter_scaling = tuple(name for name in catalog if name.startswith("scale-"))
    manifold_basic = tuple(
        name for name in catalog if name == "manifold" or name.startswith("manifold-")
    )
    aliases = {
        "all": tuple(catalog),
        "tex-all": tuple(
            name
            for name in catalog
            if name.startswith(("si-", "mi-"))
            and name not in {"mi-1", "mi-2", "mi-3", "mi-4", "mi-5"}
            and name not in single_hypotheses
            and name not in focused_single
            and name not in focused_multi
            and name not in detailed_nd
            and not name.endswith("-breaking")
        ),
        "si-hypotheses": single_hypotheses,
        "batch-2": (*focused_multi, *single_hypotheses),
        "batch-3": focused_single,
        "parameter-scaling": parameter_scaling,
        "parameter-scaling-si": tuple(
            name for name in parameter_scaling if name.startswith("scale-si-")
        ),
        "parameter-scaling-mi": tuple(
            name for name in parameter_scaling if name.startswith("scale-mi-")
        ),
        "nd-detail": detailed_nd,
        "manifold-basic": manifold_basic,
        "tex-tuning": (
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
        ),
        "report": tuple(
            name
            for name in catalog
            if name.startswith(("si-", "mi-")) and not name.endswith("-breaking")
        ),
        "si": tuple(
            name
            for name in catalog
            if name.startswith("si-") and not name.endswith("-breaking")
        ),
        "mi": tuple(
            name
            for name in catalog
            if name.startswith("mi-") and not name.endswith("-breaking")
        ),
    }
    return select_experiments(value, catalog, aliases)
