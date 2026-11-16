"""Compatibility package path for experiment modules stored under benchmarks."""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).parent / "benchmarks" / "experiments")]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__
