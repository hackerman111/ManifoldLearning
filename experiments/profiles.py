"""Обзорная подвыборка сетки без изменения параметров отдельных точек."""

from __future__ import annotations

from dataclasses import fields

from .models import Experiment, ExperimentPoint


def overview_points(experiment: Experiment) -> tuple[ExperimentPoint, ...]:
    """Оставить крайние и средний уровни числовых факторов, все категории.

    Отбираем целые исходные точки. Связанные N_lin(d), N_J(n) и seed-протокол
    сохраняются; smoke-точки не используются для научного сравнения.
    """
    names = experiment.report_fields or tuple(
        field.name
        for field in fields(ExperimentPoint)
        if len({getattr(point, field.name) for point in experiment.full}) > 1
    )
    allowed: dict[str, set[object]] = {}
    for name in names:
        values = {getattr(point, name) for point in experiment.full}
        if len(values) > 3 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            levels = sorted(values)
            allowed[name] = {levels[0], levels[len(levels) // 2], levels[-1]}
    points = tuple(
        point
        for point in experiment.full
        if all(getattr(point, name) in values for name, values in allowed.items())
    )
    if not points:
        raise ValueError(
            f"{experiment.selector}: обзорная сетка пуста; используйте --profile full"
        )
    return points
