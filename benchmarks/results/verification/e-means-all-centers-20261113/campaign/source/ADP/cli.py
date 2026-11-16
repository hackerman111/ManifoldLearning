"""Совместимый запуск CLI; реализация находится в :mod:`ADP.cli.main`."""

from __future__ import annotations

from ADP.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
