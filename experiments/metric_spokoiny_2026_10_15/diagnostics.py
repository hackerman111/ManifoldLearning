"""Small serialization, provenance, and subspace helpers for this study."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_hash(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_dict(config: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if callable(value):
            value = f"{value.__module__}.{value.__qualname__}"
        values[field.name] = json_value(value)
    return values


def environment(
    root: Path, protocol: Path, source_paths: tuple[str, ...]
) -> dict[str, Any]:
    import numpy as np
    import scipy

    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
    except ImportError:
        pools = []
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    ).stdout
    sources = {
        path: sha256_bytes((root / path).read_bytes())
        for path in source_paths
        if (root / path).is_file()
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threadpools": pools,
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "git_dirty_sha256": sha256_bytes(dirty.encode()),
        "source_sha256": sources,
        "protocol_sha256": sha256_bytes(protocol.read_bytes()),
        "kernel_formula": "K(t)=max(1-t^2,0)",
        "kernel_argument": (
            "t=(alpha^2*orthogonal_distance^2+principal_distance^2)/h^2; no sqrt"
        ),
    }


def projector_metrics(basis: Any, truth: Any) -> dict[str, Any]:
    import numpy as np

    basis, truth = np.asarray(basis), np.asarray(truth)
    singular = np.linalg.svd(truth.T @ basis, compute_uv=False)
    sine = np.sqrt(np.maximum(0.0, 1.0 - np.clip(singular, 0.0, 1.0) ** 2))
    return {
        "principal_angle_sines": sine.tolist(),
        "projector_distance": float(np.sqrt(np.mean(sine**2))),
        "orthogonality_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]))
        ),
    }


def solver_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "profile_objective",
        "iterations",
        "accepted_steps",
        "converged",
        "stop_reason",
        "riemannian_gradient",
        "gradient_norm",
        "local_rank_loss",
        "stationarity_applicable",
        "rank_guard_fallbacks",
        "workspace_fallbacks",
        "profile_evaluations",
        "operator_projects",
        "operator_adjoints",
        "full_gradient_evaluations",
        "newton_iterations",
        "recycled_steps",
        "gn_solves",
    )
    return {key: json_value(diagnostics.get(key)) for key in keys if key in diagnostics}


_MATCH_REFERENCE: Any = None


def metric_direction_matched(
    G: Any, coefficients: Any, mass: Any, rank: int, metric_floor: float = 1e-6
) -> Any:
    """Isotropic rank-q proposal scaled to the metric proposal's spectral norm."""
    import numpy as np

    from ADP.solver._grassman_schur import metric_direction

    reference = _MATCH_REFERENCE or metric_direction
    Z = reference(G, coefficients, mass, rank, metric_floor)
    left, singular, right = np.linalg.svd(G, full_matrices=False)
    E = -(left[:, :rank] * singular[:rank]) @ right[:rank]
    norm_e = float(np.linalg.norm(E, ord=2))
    if norm_e == 0:
        return np.zeros_like(G)
    return E * (float(np.linalg.norm(Z, ord=2)) / norm_e)


def metric_patch(arm: str):
    global _MATCH_REFERENCE
    from ADP.solver import _grassman_schur, grassman_research

    original = _grassman_schur.metric_direction
    if arm != "isotropic2_matched":
        return None
    _MATCH_REFERENCE = original
    grassman_research.metric_direction = metric_direction_matched
    return original


def restore_metric(original: Any) -> None:
    global _MATCH_REFERENCE
    if original is None:
        return
    from ADP.solver import grassman_research

    grassman_research.metric_direction = original
    _MATCH_REFERENCE = None
