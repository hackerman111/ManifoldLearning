"""Командные интерфейсы ADP и воспроизводимые эксперименты."""

from __future__ import annotations

from importlib import import_module

from .main import (
    _quality as _quality,
    _run as _run,
    _solver_index as _solver_index,
    build_parser,
    main,
    parse_kernel,
)


def __getattr__(name: str) -> object:
    """Сохранить доступ к прежним внутренним именам ``ADP.cli``."""
    return getattr(import_module(".main", __name__), name)


__all__ = ["build_parser", "main", "parse_kernel"]
