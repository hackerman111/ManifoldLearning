"""Операции, специфичные для ориентации single-index линии."""

# ruff: noqa: RUF002

from __future__ import annotations

import numpy as np


def orient_index(index: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Выбрать непрерывную ориентацию индексной линии.

    Направление ADP задаёт линию, поэтому ``beta`` и ``-beta`` эквивалентны.
    Если скалярное произведение с предыдущим ``reference`` отрицательно,
    возвращается ``-index``; иначе возвращается исходный массив.
    """
    if np.dot(index, reference) < 0:
        return -index
    return index


__all__ = ["orient_index"]
