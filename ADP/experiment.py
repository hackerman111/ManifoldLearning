from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .engine.utils import _finite_real_array, _prepare_xy
from .single_index.solvers.LSMR import solve as solve_lsmr
from .single_index.solvers.VarPro import solve as solve_varpro

MetadataValue = str | int | float | bool | None
SolverName = Literal["auto", "lsmr", "varpro"]
DataFactory = Callable[["ADP_ExperimentPoint", np.random.Generator], ADP_Data]


@dataclass(frozen=True, slots=True)
class ADP_ExperimentPoint:
    name: str
    n: int
    d: int
    noise: float = 0.05
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ADP_ExperimentVariant:
    config: ADP_Config
    solver: SolverName = "auto"
    solver_settings: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ADP_Experiment:
    name: str
    mode: Literal["single", "multi"]
    points: tuple[ADP_ExperimentPoint, ...]
    variants: Mapping[str, ADP_ExperimentVariant]
    runs: int = 1
    seed: int = 7
    index_dim: int = 1
    data_factory: DataFactory | None = None


def _safe_name(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{field_name} must be a non-empty path-safe string")
    return value


def _solver_method(mode: str, name: str):
    if name == "varpro":
        if mode == "multi":
            raise ValueError("varpro is not available in multi-index mode")
        return solve_varpro
    if name in {"auto", "lsmr"}:
        return solve_lsmr
    raise ValueError("solver must be 'auto', 'lsmr', or 'varpro'")


def _validate_solver_settings(mode: str, variant: ADP_ExperimentVariant) -> None:
    if not isinstance(variant.solver_settings, Mapping):
        raise TypeError("solver_settings must be a mapping")
    method = _solver_method(mode, variant.solver)
    forbidden = {"statistics", "beta", "lambda_penalty", "local_ridge"}
    keys = set(variant.solver_settings)
    unknown = keys - set(inspect.signature(method).parameters)
    overlap = keys & forbidden
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unknown solver settings: {names}")
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"problem parameters cannot be solver settings: {names}")
    try:
        json.dumps(dict(variant.solver_settings), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("solver_settings must be finite JSON values") from error


def validate_experiment(experiment: ADP_Experiment) -> ADP_Experiment:
    if type(experiment) is not ADP_Experiment:
        raise TypeError("experiment file must export ADP_Experiment as 'experiment'")
    _safe_name(experiment.name, "experiment.name")
    if experiment.mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'")
    if (
        isinstance(experiment.runs, bool)
        or not isinstance(experiment.runs, int)
        or experiment.runs < 1
    ):
        raise ValueError("runs must be a positive integer")
    if (
        isinstance(experiment.seed, bool)
        or not isinstance(experiment.seed, int)
        or experiment.seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(experiment.points, tuple):
        raise TypeError("points must be a tuple")
    if not experiment.points:
        raise ValueError("points must not be empty")
    if isinstance(experiment.index_dim, bool) or not isinstance(
        experiment.index_dim, int
    ):
        raise TypeError("index_dim must be an integer")
    if not isinstance(experiment.variants, Mapping):
        raise TypeError("variants must be a mapping")
    if not 1 <= len(experiment.variants) <= 2:
        raise ValueError("variants must contain one or two entries")

    point_names = []
    for point in experiment.points:
        if type(point) is not ADP_ExperimentPoint:
            raise TypeError("points must contain ADP_ExperimentPoint values")
        point_names.append(_safe_name(point.name, "point.name"))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (point.n, point.d)
        ):
            raise TypeError(f"point {point.name}: n and d must be integers")
        if point.d < 1 or point.n <= point.d + 1:
            raise ValueError(f"point {point.name}: require d >= 1 and n > d + 1")
        if isinstance(point.noise, bool) or not isinstance(point.noise, (int, float)):
            raise TypeError(f"point {point.name}: noise must be numeric")
        if not np.isfinite(point.noise) or point.noise < 0:
            raise ValueError(f"point {point.name}: noise must be finite and nonnegative")
        if not isinstance(point.metadata, Mapping):
            raise TypeError(f"point {point.name}: metadata must be a mapping")
        for key, value in point.metadata.items():
            _safe_name(key, "metadata key")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"metadata {key} must be a JSON scalar")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"metadata {key} must be finite")
        if experiment.mode == "multi" and not 1 <= experiment.index_dim < point.d:
            raise ValueError(f"point {point.name}: require 1 <= index_dim < d")
    if len(point_names) != len(set(point_names)):
        raise ValueError("point names must be unique")
    if experiment.mode == "single" and experiment.index_dim != 1:
        raise ValueError("single-index experiments require index_dim=1")

    variant_names = []
    for name, variant in experiment.variants.items():
        variant_names.append(_safe_name(name, "variant name"))
        if type(variant) is not ADP_ExperimentVariant:
            raise TypeError("variant values must be ADP_ExperimentVariant")
        if not isinstance(variant.config, ADP_Config):
            raise TypeError("variant.config must be ADP_Config")
        if experiment.mode == "single" and variant.config.index_init == "pilot":
            raise ValueError("pilot initialization is multi-index only")
        if variant.config.gpu and variant.solver == "varpro":
            raise ValueError("GPU mode requires the built-in LSMR solver")
        _validate_solver_settings(experiment.mode, variant)
    if len(variant_names) != len(set(variant_names)):
        raise ValueError("variant names must be unique")
    if experiment.data_factory is not None and not callable(experiment.data_factory):
        raise TypeError("data_factory must be callable")
    return experiment


def load_experiment(path: str | Path) -> ADP_Experiment:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("_adp_experiment_file", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment file: {source}")
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    if not hasattr(module, "experiment"):
        raise ValueError("experiment file must export 'experiment'")
    return validate_experiment(module.experiment)


def _orient_basis(basis: np.ndarray) -> np.ndarray:
    columns = np.arange(basis.shape[1])
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), columns])
    basis *= np.where(signs == 0, 1.0, signs)
    return basis


def make_data(
    experiment: ADP_Experiment,
    point: ADP_ExperimentPoint,
    seed: int,
) -> ADP_Data:
    rng = np.random.default_rng(seed)
    if experiment.data_factory is not None:
        data = experiment.data_factory(point, rng)
    else:
        X = rng.normal(size=(point.n, point.d))
        if experiment.mode == "single":
            true_index = rng.normal(size=point.d)
            true_index /= np.linalg.norm(true_index)
            signal = np.sin(X @ true_index)
        else:
            true_index, _ = np.linalg.qr(
                rng.normal(size=(point.d, experiment.index_dim)),
                mode="reduced",
            )
            true_index = _orient_basis(true_index)
            signal = np.sin(X @ true_index).sum(axis=1)
        Y = signal + point.noise * rng.normal(size=point.n)
        data = ADP_Data(X, Y, true_index)
    return validate_data(data, experiment, point)


def validate_data(
    data: ADP_Data,
    experiment: ADP_Experiment,
    point: ADP_ExperimentPoint,
) -> ADP_Data:
    if not isinstance(data, ADP_Data):
        raise TypeError("data_factory must return ADP_Data")
    X, Y = _prepare_xy(data.X, data.Y)
    if X.shape != (point.n, point.d):
        raise ValueError(f"point {point.name}: data must have shape {(point.n, point.d)}")
    if data.true_index is None:
        return ADP_Data(X, Y, None)
    truth = _finite_real_array(data.true_index, "true_index")
    expected = (
        (point.d,)
        if experiment.mode == "single"
        else (point.d, experiment.index_dim)
    )
    if truth.shape != expected:
        raise ValueError(f"true_index must have shape {expected}")
    if experiment.mode == "single":
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(truth)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("true_index norm must be finite and nonzero")
    if (
        experiment.mode == "multi"
        and np.linalg.matrix_rank(truth) != experiment.index_dim
    ):
        raise ValueError("true_index must have full column rank")
    return ADP_Data(X, Y, truth)
