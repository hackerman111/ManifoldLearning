"""Shared definitions for the adaptive-center paired full-fit experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DEV_SEEDS = (92800, 92801, 92802)
HELDOUT_SEEDS = (92900, 92901, 92902, 92903, 92904)
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    d: int
    link: str

    @property
    def n(self) -> int:
        return 500

    @property
    def index_dim(self) -> int:
        return 2


CASES = (
    Case("additive30", 30, "additive"),
    Case("multiplicative30", 30, "multiplicative"),
    Case("aniso_curv40", 40, "aniso_curv"),
    Case("additive50", 50, "additive"),
)


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    floor: float | None
    search: bool = False
    projected_statistics: bool = False
    selection: str = "all"
    budget: str | int | None = None
    quadrature: bool = False


ARMS = (
    Arm("baseline", None),
    Arm("floor25", 0.25),
    Arm("floor50", 0.50),
    Arm("floor25_search", 0.25, search=True),
    Arm("floor25_search_stats", 0.25, search=True, projected_statistics=True),
    Arm("random_equal_j", 0.25, True, True, "random", "fixed"),
    Arm("score_equal_j", 0.25, True, True, "score", "fixed"),
    Arm("logdet_equal_j", 0.25, True, True, "logdet", "fixed"),
    Arm("logdet_coverage_equal_j", 0.25, True, True, "logdet_coverage", "fixed"),
    Arm(
        "logdet_coverage_quad_equal_j",
        0.25,
        True,
        True,
        "logdet_coverage",
        "fixed",
        quadrature=True,
    ),
)

FIT_SETTINGS: dict[str, Any] = {
    "N_J": 500,
    "N_phi": 10,
    "N_loc": 20,
    "N_lin_factor": 2,
    "a": math.sqrt(2.0),
    "h_min": 1.0,
    "index_init": "local",
    "estimator": "new",
    "select_step": "last",
    "timeout_sec": 120,
}


def budget_for(case: Case) -> int:
    return max(4 * case.index_dim, 2 * case.d)


def arm_config(arm: Arm, case: Case):
    from ADP import AdaptiveCentersConfig

    if arm.floor is None:
        return None
    budget = budget_for(case) if arm.budget == "fixed" else None
    return AdaptiveCentersConfig(
        lambda_floor=arm.floor,
        projection_search=arm.search,
        projected_statistics=arm.projected_statistics,
        selection=arm.selection,
        budget=budget,
        quadrature=arm.quadrature,
        coverage_weight=1.0,
        score_ridge=1e-6,
        logdet_ridge=1.0,
    )


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    iterator = np.nditer(
        array, flags=["external_loop", "buffered"], order="C", buffersize=1 << 16
    )
    for block in iterator:
        digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "ADP").rglob("*.py")) + sorted(OUT.glob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def provenance_hashes() -> dict[str, str]:
    names = (
        "pyproject.toml",
        "tests/test_adaptive_centers.py",
        "benchmarks/experiments/adaptive_centers/MATH.md",
        "docs/SVD/adaptive_centers_anisotropic_adp_edr_report.tex",
    )
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in names
        if (ROOT / name).is_file()
    }


def environment() -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV},
        "threadpools": threadpool_info(),
    }


def git_state() -> dict[str, str | None]:
    state: dict[str, str | None] = {}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("dirty", ["status", "--porcelain"]),
    ):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            state[key] = (
                ("true" if result.stdout else "false")
                if key == "dirty"
                else result.stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            state[key] = None
    return state


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
