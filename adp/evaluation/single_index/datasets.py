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


_REAL_DATASET_IDS = {"D01", "D02", "D03", "D04"}
_MANIFEST_COLUMNS = {
    "id",
    "file",
    "rows",
    "features",
    "target",
    "sha256",
    "official_page",
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
    if dataset_id not in _REAL_DATASET_IDS:
        raise ValueError(f"unknown real dataset: {dataset_id}")
    root = Path(data_dir)
    manifest_path = root / "dataset_manifest.csv"
    if not manifest_path.is_file():
        raise DatasetUnavailable(
            f"dataset {dataset_id} manifest is missing at {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path, dtype=str)
    missing_columns = sorted(_MANIFEST_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise ValueError(
            "dataset manifest is missing columns: " + ", ".join(missing_columns)
        )
    duplicated_ids = manifest.loc[manifest["id"].duplicated(keep=False), "id"]
    if not duplicated_ids.empty:
        duplicates = ", ".join(sorted(set(duplicated_ids.astype(str))))
        raise ValueError(f"dataset manifest contains duplicate ids: {duplicates}")
    selected = manifest.loc[manifest["id"] == dataset_id]
    if selected.empty:
        raise DatasetUnavailable(
            f"dataset {dataset_id} is absent from manifest {manifest_path}"
        )
    row = selected.iloc[0]
    prepared = (root / "prepared").resolve()
    relative = Path(str(row["file"]))
    if relative.is_absolute():
        raise ValueError("dataset manifest file must be relative to prepared")
    path = (prepared / relative).resolve()
    try:
        path.relative_to(prepared)
    except ValueError as exc:
        raise ValueError("dataset manifest file must stay inside prepared") from exc
    if not path.is_file():
        raise DatasetUnavailable(f"dataset {dataset_id} is missing at {path}")
    actual_sha256 = _sha256(path)
    expected_sha256 = str(row["sha256"]).lower()
    if actual_sha256.lower() != expected_sha256:
        raise ValueError(
            f"dataset {dataset_id} checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    frame = pd.read_csv(path)
    target_column = str(row["target"])
    if target_column not in frame:
        raise ValueError(
            f"dataset {dataset_id} target column is missing: {target_column}"
        )
    try:
        expected_rows = int(str(row["rows"]))
        expected_features = int(str(row["features"]))
    except ValueError as exc:
        raise ValueError("dataset manifest rows and features must be integers") from exc
    if len(frame) != expected_rows:
        raise ValueError(
            f"dataset {dataset_id} rows mismatch: "
            f"expected {expected_rows}, got {len(frame)}"
        )
    if frame.shape[1] - 1 != expected_features:
        raise ValueError(
            f"dataset {dataset_id} features mismatch: "
            f"expected {expected_features}, got {frame.shape[1] - 1}"
        )
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dataset {dataset_id} contains nonnumeric values") from exc
    y = numeric.pop(target_column).to_numpy(dtype=float)
    X = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(f"dataset {dataset_id} contains non-finite values")
    return RealDataset(
        X=X,
        y=y,
        path=path,
        sha256=actual_sha256,
        source=str(row["official_page"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
