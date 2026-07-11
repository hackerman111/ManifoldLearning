from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.types import ADPData
from ...common.utils import link_function, normalize_rows, unit_vector
from .types import SingleIndexJob


class DatasetUnavailable(RuntimeError):
    """A required real dataset is absent from the explicit local cache."""


@dataclass(frozen=True, slots=True)
class RealDataset:
    X: np.ndarray
    y: np.ndarray
    path: Path
    sha256: str
    source: str


_OPENML_NAMES = {
    "D01": "airfoil_self_noise",
    "D02": "concrete_compressive_strength",
    "D03": "wine-quality-white",
    "D04": "superconduct",
}


def generate_synthetic_data(job: SingleIndexJob) -> ADPData:
    scenario = job.scenario
    data = scenario.data
    algorithm = scenario.algorithm
    n = int(data.get("n", 0))
    d = int(data.get("d", 0))
    n_centers = int(algorithm.get("n_centers", 0))
    n_directions = int(algorithm.get("n_directions", 0))
    noise = float(data.get("noise", 0.0))
    sigma_x = float(data.get("sigma_x", 1.0))
    corr = float(data.get("corr", 0.0))
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    if n_centers <= 0 or n_directions <= 0:
        raise ValueError("n_centers and n_directions must be positive")
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError("noise must be finite and nonnegative")
    if not np.isfinite(sigma_x) or sigma_x <= 0.0:
        raise ValueError("sigma_x must be finite and positive")
    if not 0.0 <= corr < 1.0:
        raise ValueError("corr must be in [0, 1)")

    beta_rng = np.random.default_rng(job.seeds.beta)
    beta = unit_vector(beta_rng.normal(size=d))
    data_rng = np.random.default_rng(job.seeds.data)
    shared = data_rng.normal(size=(n, 1))
    individual = data_rng.normal(size=(n, d))
    X = sigma_x * (
        np.sqrt(corr) * shared + np.sqrt(1.0 - corr) * individual
    )
    eps = data_rng.normal(scale=noise, size=n)
    link, link_name = link_function(str(data.get("link", "tanh")))
    y = np.asarray(link(X @ beta) + eps, dtype=float)

    center_rng = np.random.default_rng(job.seeds.centers)
    center_count = min(n_centers, n)
    selected = center_rng.choice(n, size=center_count, replace=False)
    center_noise_scale = float(algorithm.get("center_noise_scale", 0.0))
    centers = X[selected].copy()
    if center_noise_scale:
        centers += center_noise_scale * sigma_x * center_rng.normal(
            size=centers.shape
        )

    direction_rng = np.random.default_rng(job.seeds.directions)
    directions = normalize_rows(
        direction_rng.normal(size=(center_count, n_directions, d))
    )
    return ADPData(
        X=np.asarray(X, dtype=float),
        y=y,
        beta=np.asarray(beta, dtype=float),
        centers=np.asarray(centers, dtype=float),
        directions=np.asarray(directions, dtype=float),
        noise=np.asarray(eps, dtype=float),
        link_name=link_name,
    )


def load_cached_real_dataset(
    dataset_id: str,
    data_dir: str | Path,
    *,
    allow_download: bool,
) -> RealDataset:
    if dataset_id not in _OPENML_NAMES:
        raise ValueError(f"unknown real dataset: {dataset_id}")
    root = Path(data_dir)
    path = root / f"{dataset_id}.csv"
    if not path.exists():
        if not allow_download:
            raise DatasetUnavailable(
                f"dataset {dataset_id} is missing at {path}; enable --allow-download"
            )
        _download_openml(dataset_id, path, root)
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"dataset {dataset_id} must contain features and target")
    target_column = "target" if "target" in frame else frame.columns[-1]
    y = frame.pop(target_column).to_numpy(dtype=float)
    X = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(f"dataset {dataset_id} contains non-finite values")
    return RealDataset(
        X=X,
        y=y,
        path=path,
        sha256=_sha256(path),
        source=f"openml:{_OPENML_NAMES[dataset_id]}",
    )


def _download_openml(dataset_id: str, path: Path, data_dir: Path) -> None:
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:
        raise DatasetUnavailable("scikit-learn is required for OpenML download") from exc
    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        name=_OPENML_NAMES[dataset_id],
        as_frame=True,
        data_home=str(data_dir / "openml"),
    )
    frame = bunch.data.copy()
    frame["target"] = pd.to_numeric(bunch.target)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
