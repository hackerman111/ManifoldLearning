from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    model: Any


def load_model_specs(path: str | Path) -> tuple[ModelSpec, ...]:
    """Load an ordered ``MODELS`` mapping and evaluate its factories."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"model file does not exist: {source}")
    module_name = f"_adp_comparison_models_{abs(hash(source))}"
    module_spec = importlib.util.spec_from_file_location(
        module_name,
        source,
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot import model file: {source}")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(source.parent))
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.path.pop(0)
    factories = getattr(module, "MODELS", None)
    if not isinstance(factories, Mapping):
        raise ValueError(f"{source} must define a MODELS mapping")
    if len(factories) < 2:
        raise ValueError("MODELS must contain at least two model factories")

    names = tuple(str(name).strip() for name in factories)
    if any(not name for name in names):
        raise ValueError("model names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("model names must be distinct after trimming")

    loaded: list[ModelSpec] = []
    for name, factory in zip(names, factories.values(), strict=True):
        if not callable(factory):
            raise ValueError(f"model factory {name!r} must be callable")
        try:
            model = factory()
        except Exception as exc:
            raise ValueError(
                f"model factory {name!r} failed: {exc}"
            ) from exc
        if not callable(getattr(model, "fit", None)):
            raise ValueError(
                f"model factory {name!r} must return an object with fit()"
            )
        try:
            cloudpickle.dumps(model)
        except Exception as exc:
            raise ValueError(
                f"model {name!r} cannot be serialized: {exc}"
            ) from exc
        loaded.append(ModelSpec(name=name, model=model))
    return tuple(loaded)


__all__ = ["ModelSpec", "load_model_specs"]
