"""Shared inputs and reporting helpers for the block-rank experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DEV_SEEDS = (81600, 81601, 81602)
HOLDOUT_SEEDS = (81700, 81701, 81702, 81703, 81704)
ARMS = (
    "core",
    "rank_one",
    "spectral",
    "flat",
    "block1",
    "block_full",
    "block_energy",
    "block_cost",
    "block_spectral",
)
SCENARIOS = ("spokoiny_m2_d50_n800", "multi_additive_m3_d25_n250")
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SETTINGS = {
    "max_steps": 5,
    "tol": 1e-6,
    "lambda_prox": 0.05,
    "local_ridge": 0.0,
    "max_angle": 0.5,
    "workspace_bytes": 16 * 1024**2,
    "block_max_steps": 2,
}


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(repr(array.shape).encode())
    if array.flags.c_contiguous:
        h.update(memoryview(array).cast("B"))
    else:
        # Bound hashing scratch instead of materializing a full contiguous copy.
        iterator = np.nditer(
            array, flags=["external_loop", "buffered"], order="C", buffersize=1 << 17
        )
        for block in iterator:
            h.update(memoryview(np.ascontiguousarray(block)).cast("B"))
    return h.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        "benchmarks/grassman_benchmark.py",
        "benchmarks/experiments/data.py",
        "benchmarks/experiments/multi.py",
        "benchmarks/experiments/_grids.py",
        "benchmarks/experiments/models.py",
        "benchmarks/experiments/runner.py",
        "ADP/solver/grassman.py",
        "ADP/solver/grassman_exp.py",
        "ADP/solver/LSMR.py",
        "ADP/solver/_grassman_schur.py",
        "ADP/solver/_multi_operator.py",
        "ADP/core/ADP_Config.py",
        "ADP/core/ADP_Solver.py",
        "ADP/core/ADP_Statistic.py",
        "ADP/core/multi/ADP_multi_index.py",
        "ADP/engine/common/index_fit.py",
        "ADP/engine/common/statistic.py",
        "ADP/engine/common/weights.py",
        "ADP/engine/common/initialize.py",
        "ADP/engine/common/calculus.py",
        "ADP/engine/common/box_kernel.py",
        "ADP/engine/common/ADP_Statistic_engine.py",
        "ADP/engine/multi_index/ADP_multi_index_engine.py",
        "ADP/core/multi/ADP_multi_index_utils.py",
        "benchmarks/experiments/grassman_rank_k/PROTOCOL.md",
        "benchmarks/experiments/grassman_rank_k/MATH.md",
        "benchmarks/experiments/grassman_rank_k/common.py",
        "benchmarks/experiments/grassman_rank_k/inner.py",
        "benchmarks/experiments/grassman_rank_k/fullfit.py",
        "benchmarks/experiments/grassman_rank_k/run.py",
        "benchmarks/experiments/grassman_rank_k/run_inner.py",
        "benchmarks/experiments/grassman_rank_k/reference.py",
    )
    return {name: file_hash(ROOT / name) for name in paths if (ROOT / name).is_file()}


def scenario_metadata(scenario: str) -> dict[str, Any]:
    if scenario == SCENARIOS[0]:
        from benchmarks.grassman_benchmark import PARAMS

        point = {**PARAMS, "selector": "mi-spokoini-m2-dhigh"}
    elif scenario == SCENARIOS[1]:
        from experiments.multi import _detailed_multi_catalog

        experiment = next(
            item for item in _detailed_multi_catalog() if item.selector == "mi-5"
        )
        point_obj = next(
            p
            for p in experiment.full
            if p.d == 25
            and p.n_over_d == 10
            and p.index_dim == 3
            and p.link == "multi_additive"
            and p.link_scale == 3.0
        )
        point = asdict(point_obj)
        point["selector"] = "mi-5"
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    _, _, _, config = make_problem(scenario, DEV_SEEDS[0])
    return {
        "point": json_value(point),
        "effective_config_seed81600": json_value(asdict(config)),
    }


def git_state() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("dirty", ["status", "--porcelain"]),
    ):
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            result[key] = (
                "true"
                if key == "dirty" and proc.stdout
                else ("false" if key == "dirty" else proc.stdout.strip())
            )
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    return result


def make_problem(scenario: str, seed: int):
    """Return data, truth basis and config from the existing scenario generators."""
    from ADP import ADP_Config
    from benchmarks.grassman_benchmark import _make_problem as hard_problem
    from experiments.data import _generate_data, _make_seed_bundle
    from experiments.runner import _effective_config

    if scenario == SCENARIOS[0]:
        return hard_problem(seed)
    if scenario != SCENARIOS[1]:
        raise ValueError(f"unknown scenario: {scenario}")
    from experiments.multi import _detailed_multi_catalog

    experiment = next(
        item for item in _detailed_multi_catalog() if item.selector == "mi-5"
    )
    point = next(
        p
        for p in experiment.full
        if p.d == 25
        and p.n_over_d == 10
        and p.index_dim == 3
        and p.link == "multi_additive"
        and p.link_scale == 3.0
    )
    selector = "mi-5"
    bundle = _make_seed_bundle(selector, point, seed)
    generated = _generate_data(selector, point, bundle, seed)
    config = _effective_config(ADP_Config(), point, bundle.init)
    return generated.X, generated.Y, generated.beta, config


def environment() -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV},
        "threadpools": threadpool_info(),
    }


def projector_metrics(index: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    overlap = index @ truth
    singular = np.linalg.svd(overlap, compute_uv=False)
    sine2 = np.maximum(0.0, 1.0 - np.minimum(1.0, singular) ** 2)
    return float(np.sqrt(np.sum(sine2) / truth.shape[1])), float(
        np.linalg.norm(index @ index.T - np.eye(index.shape[0]))
    )


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(json_value(row), sort_keys=True, allow_nan=False) + "\n"
        )


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value


def _run_child(module: str, arguments: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(dict.fromkeys(THREAD_ENV, "1"))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", module, *arguments]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            return {
                "status": "error",
                "returncode": result.returncode,
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-1000:],
            }
        return {"status": "ok"}
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "timeout_sec": timeout,
            "stdout": str(exc.stdout)[-1000:],
            "stderr": str(exc.stderr)[-4000:],
        }


def _read_row(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def _assert_frozen() -> dict[str, Any]:
    path = OUT / "freeze_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("run --stage freeze first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    current = source_hashes()
    if current != manifest.get("source_sha256"):
        raise RuntimeError(
            "source hashes changed since freeze; create a reviewed new freeze"
        )
    return manifest


def _fresh_worker(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
