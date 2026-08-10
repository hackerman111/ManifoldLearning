from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, fields, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from .ADP_Config import ADP_Config
from .ADP_Data import ADP_Data
from .ADP_Solver import ADP_solver
from .experiment import (
    ADP_Experiment,
    ADP_ExperimentPoint,
    ADP_ExperimentVariant,
    _solver_method,
    make_data,
    validate_experiment,
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

    def validate_data_artifacts(self) -> None:
        series_dir = self.series_dir.resolve()
        for commit in self.read_commits():
            artifact = commit.get("run", {}).get("data_artifact", "")
            if not artifact:
                continue
            path = (self.series_dir / artifact).resolve()
            try:
                path.relative_to(series_dir)
            except ValueError as error:
                raise ValueError(
                    f"data artifact escapes series directory: {artifact}"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(
                    f"referenced data artifact does not exist: {artifact}"
                )

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


class _RSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> None:
        self.samples.append(self.process.memory_info().rss / 2**20)

    def _sample_loop(self) -> None:
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self._sample()

    def summary(self) -> dict:
        values = np.asarray(self.samples, dtype=float)
        return {
            "algorithm_rss_start_mib": float(values[0]),
            "algorithm_rss_min_mib": float(values.min()),
            "algorithm_rss_mean_mib": float(values.mean()),
            "algorithm_rss_max_mib": float(values.max()),
            "algorithm_rss_peak_delta_mib": float(values.max() - values[0]),
            "algorithm_memory_samples": int(values.size),
            "algorithm_memory_source": "psutil.Process.memory_info().rss",
        }


def _projector_error(estimated, truth) -> float:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.ndim == 1:
        estimated = estimated[:, None]
    if truth.ndim == 1:
        truth = truth[:, None]
    estimated, _ = np.linalg.qr(estimated, mode="reduced")
    truth, _ = np.linalg.qr(truth, mode="reduced")
    return float(
        np.linalg.norm(
            estimated @ estimated.T - truth @ truth.T,
            ord="fro",
        )
        / np.sqrt(2.0 * truth.shape[1])
    )


def _json_safe(value):
    value = _spec_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return value if np.isfinite(value) else ""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _metric(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else ""
    return ""


def _compact_json(value) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _json_array(value) -> str:
    return _compact_json(np.asarray(value).tolist())


def _normalized_cosine(left, right) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if (
        left.shape != right.shape
        or not np.isfinite(left_norm)
        or not np.isfinite(right_norm)
        or left_norm == 0
        or right_norm == 0
    ):
        raise RuntimeError("cannot calculate cosine for invalid indices")
    return abs(float((left / left_norm) @ (right / right_norm)))


def _outer_rows(job, experiment, data, model, series_id):
    rows = []
    structural = {
        "k",
        "h",
        "rho",
        "alpha",
        "beta",
        "basis",
        "eigenvalues",
        "mean_mass",
        "stop_reason",
    }
    for item in model.result_.trace:
        estimate = item.get("beta", item.get("basis"))
        diagnostics = {
            key: _json_safe(value)
            for key, value in item.items()
            if key not in structural
        }
        cosine = ""
        projector = ""
        if data.true_index is not None and estimate is not None:
            projector = _projector_error(estimate, data.true_index)
            if experiment.mode == "single":
                cosine = _normalized_cosine(estimate, data.true_index)
        rows.append(
            {
                "schema_version": 1,
                "series_id": series_id,
                "run_id": job.run_id,
                "experiment": experiment.name,
                "point": job.point.name,
                "variant": job.variant_name,
                "seed": job.seed,
                "local_solver": _solver_name(model.solver.method),
                "mode": experiment.mode,
                "outer_k": _metric(item.get("k")),
                "h_k": _metric(item.get("h")),
                "rho_k": _metric(item.get("rho")),
                "alpha_k": _metric(item.get("alpha")),
                "beta_k": (
                    _json_array(item["beta"]) if "beta" in item else ""
                ),
                "basis_k": (
                    _json_array(item["basis"]) if "basis" in item else ""
                ),
                "eigenvalues": (
                    _json_array(item["eigenvalues"])
                    if "eigenvalues" in item
                    else ""
                ),
                "cosine_abs": cosine,
                "projector_error": projector,
                "beta_delta": _metric(item.get("beta_delta")),
                "objective_after": _metric(item.get("objective")),
                "inner_iterations": _metric(item.get("inner_iterations")),
                "linear_solver_iterations": _metric(
                    item.get("lsmr_iterations")
                ),
                "solver_diagnostics": _compact_json(diagnostics),
                "local_mass_mean": _metric(item.get("mean_mass")),
                "stop_reason": item.get("stop_reason", "") or "",
            }
        )
    return rows


def _base_run(
    job: _Job,
    experiment: ADP_Experiment,
    series_id: str,
) -> dict:
    requested_config = {
        item.name: _json_safe(getattr(job.variant.config, item.name))
        for item in fields(ADP_Config)
    }
    effective = replace(job.variant.config, seed=job.seed)
    effective_config = {
        item.name: _json_safe(getattr(effective, item.name))
        for item in fields(ADP_Config)
    }
    return {
        "schema_version": 1,
        "series_id": series_id,
        "run_id": job.run_id,
        "experiment": experiment.name,
        "point": job.point.name,
        "variant": job.variant_name,
        "variant_index": job.variant_index,
        "mode": experiment.mode,
        "index_dim": experiment.index_dim,
        "seed": job.seed,
        "n": job.point.n,
        "d": job.point.d,
        "n_over_d": job.point.n / job.point.d,
        "noise": job.point.noise,
        "requested_config": requested_config,
        "effective_config": effective_config,
        "requested_solver": job.variant.solver,
        "requested_solver_settings": _compact_json(
            dict(job.variant.solver_settings)
        ),
        "local_solver": "",
        "effective_solver_settings": "",
        "h_initial": "",
        "h_final": "",
        "rho_final": "",
        "alpha_final": "",
        "outer_iterations": 0,
        "cosine_abs": "",
        "projector_error": "",
        "fit_wall_time_sec": "",
        "algorithm_time_sec": "",
        "algorithm_rss_start_mib": "",
        "algorithm_rss_min_mib": "",
        "algorithm_rss_mean_mib": "",
        "algorithm_rss_max_mib": "",
        "algorithm_rss_peak_delta_mib": "",
        "algorithm_memory_samples": 0,
        "algorithm_memory_source": "",
        "tracemalloc_peak_mib": "",
        "profile_stages": {},
        "stop_reason": "",
        "status": "numerical_failure",
        "error_type": "",
        "error_message": "",
        "error_traceback": "",
        "data_artifact": "",
        "model_artifact": "",
        "outer_row_count": 0,
        "inner_row_count": 0,
        "local_row_count": 0,
        "solver_row_count": 0,
        **{
            f"adp_{name}": value for name, value in requested_config.items()
        },
        **{
            f"effective_{name}": value
            for name, value in effective_config.items()
        },
    }


def _profile_fields(model) -> dict:
    profile = getattr(model, "profile_", {})
    if not isinstance(profile, dict):
        return {}
    stages = profile.get("stages", {})
    result = {
        "algorithm_time_sec": _metric(profile.get("total_time_seconds")),
        "tracemalloc_peak_mib": (
            _metric(profile.get("peak_memory_bytes")) / 2**20
            if _metric(profile.get("peak_memory_bytes")) != ""
            else ""
        ),
        "profile_stages": _json_safe(stages if isinstance(stages, dict) else {}),
    }
    if isinstance(stages, dict):
        for name, values in stages.items():
            if not isinstance(values, dict):
                continue
            result[f"stage_{name}_time_sec"] = _metric(
                values.get("time_seconds")
            )
            memory = _metric(values.get("memory_bytes"))
            result[f"stage_{name}_memory_mib"] = (
                memory / 2**20 if memory != "" else ""
            )
    return result


def _final_index(model, experiment: ADP_Experiment, d: int) -> np.ndarray:
    name = "beta_" if experiment.mode == "single" else "basis_"
    if not hasattr(model, name):
        raise RuntimeError(f"fitted model is missing {name}")
    if experiment.mode == "multi" and not hasattr(model, "eigenvalues_"):
        raise RuntimeError("fitted multi-index model is missing eigenvalues_")
    value = getattr(model, name)
    index = np.asarray(value, dtype=float)
    expected = (d,) if experiment.mode == "single" else (d, experiment.index_dim)
    if index.shape != expected or not np.all(np.isfinite(index)):
        raise RuntimeError(f"fitted index must have finite shape {expected}")
    if experiment.mode == "single":
        if np.linalg.norm(index) == 0:
            raise RuntimeError("fitted index must be nonzero")
    elif np.linalg.matrix_rank(index) != experiment.index_dim:
        raise RuntimeError("fitted basis must have full column rank")
    if experiment.mode == "multi":
        eigenvalues = np.asarray(model.eigenvalues_, dtype=float)
        if (
            eigenvalues.shape != (experiment.index_dim,)
            or not np.all(np.isfinite(eigenvalues))
        ):
            raise RuntimeError("fitted eigenvalues are invalid")
    coefficients = getattr(model, "coefficients_", None)
    if coefficients is not None and not np.all(
        np.isfinite(np.asarray(coefficients, dtype=float))
    ):
        raise RuntimeError("fitted coefficients are non-finite")
    return index


def _save_model(store: _SeriesStore, job: _Job, model) -> str:
    values = {}
    if hasattr(model, "beta_"):
        values["index"] = np.asarray(model.beta_)
    if hasattr(model, "eigenvalues_"):
        values["eigenvalues"] = np.asarray(model.eigenvalues_)
    if getattr(model, "coefficients_", None) is not None:
        values["coefficients"] = np.asarray(model.coefficients_)
    path = store.model_dir / f"{job.run_id}.npz"
    _atomic_npz(path, **values)
    return str(path.relative_to(store.series_dir))


def _execute_job(
    store,
    experiment,
    job,
    data,
    save_models,
    progress_callback,
):
    run = _base_run(job, experiment, store.series_dir.name)
    run["data_artifact"] = str(
        store.data_path(job.point, job.seed).relative_to(store.series_dir)
    )
    model = None
    sampler = None
    fit_wall = ""
    outer = []
    try:
        model = _build_model(experiment, job.variant, job.seed)
        run["local_solver"] = _solver_name(model.solver.method)
        run["effective_solver_settings"] = _compact_json(model.solver.settings)
        with threadpool_limits(limits=1):
            with _RSSSampler() as sampler:
                started_at = time.perf_counter()
                try:
                    model.fit(data.X, data.Y, progress=progress_callback)
                finally:
                    fit_wall = time.perf_counter() - started_at
        run.update(sampler.summary())
        run["fit_wall_time_sec"] = fit_wall
        run.update(_profile_fields(model))

        result = model.result_
        trace = list(result.trace)
        stop_reason = getattr(result, "stop_reason", None) or ""
        index = _final_index(model, experiment, job.point.d)
        if stop_reason not in {"h_min", "local_mass_limit"}:
            raise RuntimeError(f"invalid stop reason: {stop_reason or 'missing'}")

        effective = {
            item.name: _json_safe(getattr(model.config, item.name))
            for item in fields(ADP_Config)
        }
        effective.update(
            {
                str(name): _json_safe(value)
                for name, value in getattr(
                    model,
                    "effective_parameters_",
                    {},
                ).items()
            }
        )
        run["effective_config"] = effective
        run.update({f"effective_{key}": value for key, value in effective.items()})
        run.update(
            {
                "h_initial": _metric(trace[0].get("h")) if trace else "",
                "h_final": _metric(trace[-1].get("h")) if trace else "",
                "rho_final": _metric(trace[-1].get("rho")) if trace else "",
                "alpha_final": _metric(trace[-1].get("alpha")) if trace else "",
                "outer_iterations": len(trace),
                "projector_error": (
                    _projector_error(index, data.true_index)
                    if data.true_index is not None
                    else ""
                ),
                "cosine_abs": (
                    _normalized_cosine(index, data.true_index)
                    if experiment.mode == "single" and data.true_index is not None
                    else ""
                ),
                "stop_reason": stop_reason,
                "status": "success",
            }
        )
        outer = _outer_rows(
            job,
            experiment,
            data,
            model,
            store.series_dir.name,
        )
        run["outer_row_count"] = len(outer)
        if save_models:
            run["model_artifact"] = _save_model(store, job, model)
    except Exception as error:
        if sampler is not None and sampler.samples:
            run.update(sampler.summary())
        if fit_wall != "":
            run["fit_wall_time_sec"] = fit_wall
        if model is not None:
            run.update(_profile_fields(model))
            result = getattr(model, "result_", None)
            run["stop_reason"] = getattr(result, "stop_reason", "") or ""
        run.update(
            {
                "status": (
                    "nonconverged"
                    if isinstance(error, RuntimeError)
                    and str(error)
                    == "outer_steps exhausted before reaching h_min"
                    else "numerical_failure"
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_traceback": traceback.format_exc(),
                "model_artifact": "",
            }
        )
    return {"run": _json_safe(run), "outer": _json_safe(outer)}


def _failure_outcome(store, experiment, job, error):
    run = _base_run(job, experiment, store.series_dir.name)
    run.update(
        {
            "status": "numerical_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_traceback": "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            ),
        }
    )
    return {"run": _json_safe(run), "outer": []}


def run_experiment(
    experiment: ADP_Experiment,
    output_dir: str | Path,
    *,
    resume: str | Path | None = None,
    save_models: bool = True,
    show_progress: bool = True,
) -> tuple[Path, int]:
    experiment = validate_experiment(experiment)
    jobs = _build_jobs(experiment)
    if resume is None:
        store = _SeriesStore.create(Path(output_dir), experiment)
    else:
        store = _SeriesStore.resume(Path(resume))
        store.validate_data_artifacts()
    pending = [job for job in jobs if not store.is_complete(job, experiment)]
    data_cache = {}
    progress_disabled = not show_progress or not sys.stderr.isatty()
    outer = tqdm(
        total=len(jobs),
        initial=len(jobs) - len(pending),
        desc=experiment.name,
        unit="fit",
        dynamic_ncols=True,
        disable=progress_disabled,
    )
    try:
        for job in pending:
            key = (job.point.name, job.seed)
            if key not in data_cache:
                path = store.data_path(job.point, job.seed)
                try:
                    data_cache[key] = (
                        store.load_data(job.point, job.seed)
                        if path.exists()
                        else make_data(experiment, job.point, job.seed)
                    )
                    if not path.exists():
                        store.save_data(job.point, job.seed, data_cache[key])
                except Exception as error:
                    data_cache[key] = error
            data = data_cache[key]
            if isinstance(data, Exception):
                outcome = _failure_outcome(store, experiment, job, data)
            else:
                inner = tqdm(
                    desc=f"{job.point.name}/{job.variant_name}",
                    unit="outer",
                    leave=False,
                    position=1,
                    disable=progress_disabled,
                )

                def advance(item):
                    inner.set_postfix(
                        h=item.get("h"),
                        localization=item.get("rho", item.get("alpha")),
                        refresh=False,
                    )
                    inner.update(1)

                try:
                    outcome = _execute_job(
                        store,
                        experiment,
                        job,
                        data,
                        save_models,
                        advance,
                    )
                finally:
                    inner.close()
            store.commit(job, experiment, outcome)
            outer.set_postfix(
                point=job.point.name,
                variant=job.variant_name,
                status=outcome["run"]["status"],
            )
            outer.update(1)
            if show_progress and not sys.stderr.isatty():
                print(f"{job.run_id}: {outcome['run']['status']}")
    finally:
        outer.close()
    failures = sum(
        commit["run"]["status"] != "success"
        for commit in store.read_commits()
    )
    return store.series_dir, failures
