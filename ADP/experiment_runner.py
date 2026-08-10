from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .ADP_Solver import ADP_solver
from .experiment import (
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    _solver_method,
)
from .multi_index.ADP_multi_index import ADP_multi_index
from .single_index.ADP_single_index import ADP_single_index


@dataclass(frozen=True, slots=True)
class _Job:
    point: ADP_ExperimentPoint
    run_index: int
    seed: int
    variant_index: int
    variant_name: str
    variant: ADP_ExperimentVariant

    @property
    def run_id(self) -> str:
        return (
            f"{len(self.point.name)}-{self.point.name}__seed-{self.seed}__"
            f"{len(self.variant_name)}-{self.variant_name}"
        )


def _build_jobs(experiment: ADP_Experiment) -> tuple[_Job, ...]:
    variants = tuple(experiment.variants.items())
    index_by_name = {name: index for index, (name, _) in enumerate(variants)}
    jobs = []
    for point in experiment.points:
        for run_index in range(experiment.runs):
            ordered = variants if run_index % 2 == 0 else tuple(reversed(variants))
            for name, variant in ordered:
                jobs.append(
                    _Job(
                        point=point,
                        run_index=run_index,
                        seed=experiment.seed + run_index,
                        variant_index=index_by_name[name],
                        variant_name=name,
                        variant=variant,
                    )
                )
    return tuple(jobs)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _qualified_name(value) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError(f"{type(value).__name__} has no stable qualified name")
    return f"{module}:{qualname}"


def _spec_value(value):
    if isinstance(value, partial):
        return {
            "function": _qualified_name(value.func),
            "args": _spec_value(value.args),
            "keywords": _spec_value(value.keywords or {}),
        }
    if callable(value):
        return _qualified_name(value)
    if isinstance(value, np.ndarray):
        return _spec_value(value.tolist())
    if isinstance(value, np.longdouble):
        if not np.isfinite(value):
            raise ValueError("np.longdouble must be finite")
        return {
            "__numpy_scalar__": "longdouble",
            "dtype": str(value.dtype),
            "value": np.format_float_scientific(value, unique=True, trim="k"),
        }
    if isinstance(value, np.generic):
        item = value.item()
        if type(item) not in (bool, int, float, str):
            raise TypeError(f"NumPy scalar {value.dtype} is not JSON-compatible")
        return item
    if isinstance(value, dict):
        return {
            str(key): _spec_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_spec_value(item) for item in value]
    return value


def _job_spec(job: _Job, experiment: ADP_Experiment) -> dict:
    requested_config = {
        item.name: _spec_value(getattr(job.variant.config, item.name))
        for item in fields(ADP_Config)
    }
    effective_config = {**requested_config, "seed": job.seed}
    return {
        "schema_version": 1,
        "experiment": experiment.name,
        "mode": experiment.mode,
        "index_dim": experiment.index_dim,
        "point": {
            "name": job.point.name,
            "n": job.point.n,
            "d": job.point.d,
            "noise": job.point.noise,
            "metadata": _spec_value(dict(job.point.metadata)),
        },
        "run_index": job.run_index,
        "seed": job.seed,
        "variant_index": job.variant_index,
        "variant": job.variant_name,
        "requested_config": requested_config,
        "effective_config": effective_config,
        "solver": job.variant.solver,
        "solver_settings": _spec_value(dict(job.variant.solver_settings)),
    }


class _SeriesStore:
    def __init__(self, series_dir: Path):
        self.series_dir = series_dir
        self.data_dir = series_dir / "data"
        self.commit_dir = series_dir / "commits"
        self.model_dir = series_dir / "models"

    @classmethod
    def create(cls, output_dir: Path, experiment: ADP_Experiment) -> _SeriesStore:
        parent = Path(output_dir) / experiment.name
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        series_dir = parent / stamp
        suffix = 1
        while series_dir.exists():
            series_dir = parent / f"{stamp}-{suffix}"
            suffix += 1
        series_dir.mkdir()
        store = cls(series_dir)
        store.data_dir.mkdir()
        store.commit_dir.mkdir()
        store.model_dir.mkdir()
        return store

    @classmethod
    def resume(cls, series_dir: Path) -> _SeriesStore:
        path = Path(series_dir)
        if not path.is_dir() or not (path / "commits").is_dir():
            raise ValueError(f"not an ADP series directory: {path}")
        return cls(path)

    def data_path(self, point: ADP_ExperimentPoint, seed: int) -> Path:
        return self.data_dir / point.name / f"seed_{seed}.npz"

    def save_data(
        self,
        point: ADP_ExperimentPoint,
        seed: int,
        data: ADP_Data,
    ) -> Path:
        values = {
            "X": data.X,
            "Y": data.Y,
            "has_true_index": np.asarray(data.true_index is not None),
        }
        if data.true_index is not None:
            values["true_index"] = data.true_index
        path = self.data_path(point, seed)
        _atomic_npz(path, **values)
        return path

    def load_data(self, point: ADP_ExperimentPoint, seed: int) -> ADP_Data:
        with np.load(self.data_path(point, seed), allow_pickle=False) as archive:
            truth = (
                archive["true_index"]
                if bool(archive["has_true_index"])
                else None
            )
            return ADP_Data(archive["X"], archive["Y"], truth)

    def commit_path(self, job: _Job) -> Path:
        return self.commit_dir / f"{job.run_id}.json"

    def read_commits(self) -> list[dict]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.commit_dir.glob("*.json"))
        ]

    def is_complete(self, job: _Job, experiment: ADP_Experiment) -> bool:
        path = self.commit_path(job)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("spec") != _job_spec(job, experiment):
            raise ValueError(f"resume specification differs for {job.run_id}")
        return True

    def commit(
        self,
        job: _Job,
        experiment: ADP_Experiment,
        outcome: dict,
    ) -> None:
        path = self.commit_path(job)
        payload = {"spec": _job_spec(job, experiment), **outcome}
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("spec") != payload["spec"]:
                raise ValueError(f"resume specification differs for {job.run_id}")
            return
        _atomic_json(path, payload)


def _build_model(
    experiment: ADP_Experiment,
    variant: ADP_ExperimentVariant,
    seed: int,
):
    config = replace(variant.config, seed=seed)
    model = (
        ADP_single_index(config)
        if experiment.mode == "single"
        else ADP_multi_index(experiment.index_dim, config)
    )
    if variant.solver == "auto":
        if variant.solver_settings:
            settings = {**model.solver.settings, **dict(variant.solver_settings)}
            model.solver = ADP_solver(model.solver.method, **settings)
        return model
    model.solver = ADP_solver(
        _solver_method(experiment.mode, variant.solver),
        **dict(variant.solver_settings),
    )
    return model


def _solver_name(method) -> str:
    if method is _solver_method("single", "lsmr"):
        return "lsmr"
    if method is _solver_method("single", "varpro"):
        return "varpro"
    return f"{method.__module__}:{method.__qualname__}"
