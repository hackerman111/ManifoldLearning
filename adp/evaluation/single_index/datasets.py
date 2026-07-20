from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.experiment_log import Scalar
from ...common.types import ADPData
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


@dataclass(frozen=True, slots=True)
class GeneratedSingleIndexData:
    data: ADPData
    signal: np.ndarray
    ordinary_noise: np.ndarray
    gamma: np.ndarray | None
    metadata: dict[str, Scalar]


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


def generate_synthetic_data(job: SingleIndexJob) -> GeneratedSingleIndexData:
    parameters = job.parameters
    n = parameters.n
    d = parameters.d
    n_centers = parameters.n_centers
    n_directions = max(4, min(d, 32))

    beta = _unit_normal_vector(np.random.default_rng(job.seeds.beta), d, "beta")
    X = _generate_features(job)
    index = X @ beta
    link_index_divisor = (
        parameters.sigma_x if job.experiment == "5" else 1.0
    )
    raw_signal = _apply_link(index / link_index_divisor, parameters.link)
    signal, link_mean, link_std = _standardize_sample(
        raw_signal,
        f"{parameters.link} link",
    )

    ordinary_noise = _generate_ordinary_noise(job, index)
    final_noise, outlier_count = _replace_outliers(job, ordinary_noise)

    gamma: np.ndarray | None = None
    misspecification_signal: np.ndarray | None = None
    misspecification_mean: float | None = None
    misspecification_std: float | None = None
    if parameters.delta > 0.0:
        gamma = _generate_gamma(job, beta)
        misspecification_index = X @ gamma
        raw_misspecification = (
            misspecification_index + 0.5 * misspecification_index**2
        )
        (
            misspecification_signal,
            misspecification_mean,
            misspecification_std,
        ) = _standardize_sample(
            raw_misspecification,
            "misspecification link",
        )

    y = signal.copy()
    if misspecification_signal is not None:
        y = y + parameters.delta * misspecification_signal
    y = y + final_noise

    center_rng = np.random.default_rng(job.seeds.centers)
    selected = center_rng.choice(n, size=n_centers, replace=False)
    centers = X[selected].copy()

    directions_rng = np.random.default_rng(job.seeds.directions)
    directions = directions_rng.normal(size=(n_centers, n_directions, d))
    direction_norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    if (
        not np.all(np.isfinite(direction_norms))
        or np.any(direction_norms <= np.finfo(float).eps)
    ):
        raise ValueError("directions contain a degenerate vector")
    directions = directions / direction_norms

    data = ADPData(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        beta=np.asarray(beta, dtype=float),
        centers=np.asarray(centers, dtype=float),
        directions=np.asarray(directions, dtype=float),
        noise=np.asarray(final_noise, dtype=float),
        link_name=parameters.link,
    )
    _require_finite_arrays(
        X=data.X,
        y=data.y,
        beta=data.beta,
        centers=data.centers,
        directions=data.directions,
        noise=data.noise,
        signal=signal,
        ordinary_noise=ordinary_noise,
        gamma=gamma,
    )

    metadata: dict[str, Scalar] = {
        "effective_p": d,
        "effective_n": n,
        "n_over_d": parameters.n_over_d,
        "effective_J": n_centers,
        "effective_n_directions": n_directions,
        "snr": (
            math.inf
            if parameters.sigma_eps == 0.0
            else 1.0 / parameters.sigma_eps**2
        ),
        "link": parameters.link,
        "link_mean": link_mean,
        "link_std": link_std,
        "x_distribution": parameters.x_distribution,
        "noise_distribution": parameters.noise_distribution,
        "effective_noise_distribution": (
            "gaussian"
            if parameters.heteroscedastic
            else parameters.noise_distribution
        ),
        "heteroscedastic": parameters.heteroscedastic,
        "outliers_enabled": outlier_count > 0,
        "misspecified": parameters.delta > 0.0,
        "outlier_count": outlier_count,
        "effective_outlier_fraction": outlier_count / n,
        "link_index_divisor": link_index_divisor,
        "sigma_x": parameters.sigma_x,
        "rho_corr": parameters.rho_corr,
        "effective_rho_corr": (
            parameters.rho_corr
            if parameters.x_distribution == "gaussian"
            else 0.0
        ),
        "sigma_eps": parameters.sigma_eps,
        "outlier_fraction": parameters.outlier_fraction,
        "outlier_scale": parameters.outlier_scale,
        "delta": parameters.delta,
        "center_fraction": parameters.center_fraction,
    }
    for seed_field in fields(job.seeds):
        metadata[f"seed_{seed_field.name}"] = int(
            getattr(job.seeds, seed_field.name)
        )
    if misspecification_mean is not None and misspecification_std is not None:
        metadata["misspecification_mean"] = misspecification_mean
        metadata["misspecification_std"] = misspecification_std

    return GeneratedSingleIndexData(
        data=data,
        signal=np.asarray(signal, dtype=float),
        ordinary_noise=np.asarray(ordinary_noise, dtype=float),
        gamma=None if gamma is None else np.asarray(gamma, dtype=float),
        metadata=metadata,
    )


def _generate_features(job: SingleIndexJob) -> np.ndarray:
    parameters = job.parameters
    rng = np.random.default_rng(job.seeds.features)
    shape = (parameters.n, parameters.d)
    if parameters.x_distribution == "gaussian":
        factor = _ar1_factor(parameters.d, parameters.rho_corr)
        features = rng.normal(size=shape) @ factor.T
    elif parameters.x_distribution == "uniform":
        bound = math.sqrt(3.0)
        features = rng.uniform(-bound, bound, size=shape)
    elif parameters.x_distribution == "student_t5":
        features = rng.standard_t(df=5, size=shape) * math.sqrt(3.0 / 5.0)
    else:  # pragma: no cover - validated by ExperimentParameters
        raise ValueError(
            f"unknown feature distribution: {parameters.x_distribution}"
        )
    return np.asarray(parameters.sigma_x * features, dtype=float)


def _generate_ordinary_noise(
    job: SingleIndexJob,
    index: np.ndarray,
) -> np.ndarray:
    parameters = job.parameters
    if parameters.sigma_eps == 0.0:
        return np.zeros(parameters.n, dtype=float)
    rng = np.random.default_rng(job.seeds.noise)
    if parameters.heteroscedastic:
        unit_noise = rng.normal(size=parameters.n)
        scale = parameters.sigma_eps * np.sqrt((0.25 + index**2) / 1.25)
        return np.asarray(scale * unit_noise, dtype=float)
    if parameters.noise_distribution == "gaussian":
        unit_noise = rng.normal(size=parameters.n)
    elif parameters.noise_distribution == "student_t5":
        unit_noise = rng.standard_t(df=5, size=parameters.n) * math.sqrt(3.0 / 5.0)
    elif parameters.noise_distribution == "student_t3":
        unit_noise = rng.standard_t(df=3, size=parameters.n) * math.sqrt(1.0 / 3.0)
    else:  # pragma: no cover - validated by ExperimentParameters
        raise ValueError(
            f"unknown noise distribution: {parameters.noise_distribution}"
        )
    return np.asarray(parameters.sigma_eps * unit_noise, dtype=float)


def _replace_outliers(
    job: SingleIndexJob,
    ordinary_noise: np.ndarray,
) -> tuple[np.ndarray, int]:
    parameters = job.parameters
    if parameters.outlier_fraction == 0.0:
        return ordinary_noise.copy(), 0
    count = min(
        parameters.n,
        math.ceil(parameters.outlier_fraction * parameters.n),
    )
    indices = np.random.default_rng(job.seeds.outliers).permutation(
        parameters.n
    )[:count]
    replacements = np.random.default_rng(job.seeds.outlier_noise).normal(
        scale=parameters.outlier_scale * parameters.sigma_eps,
        size=count,
    )
    final_noise = ordinary_noise.copy()
    final_noise[indices] = replacements
    return final_noise, count


def _generate_gamma(job: SingleIndexJob, beta: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(job.seeds.gamma)
    threshold = np.finfo(float).eps
    gamma: np.ndarray | None = None
    for _ in range(256):
        candidate = rng.normal(size=beta.size)
        candidate = candidate - beta * float(candidate @ beta)
        norm = float(np.linalg.norm(candidate))
        if np.isfinite(norm) and norm > threshold:
            gamma = candidate / norm
            break
    if gamma is None:
        raise ValueError("cannot generate a direction orthogonal to beta")

    orientation_rng = np.random.default_rng(job.seeds.misspecification)
    orientation = -1.0 if orientation_rng.integers(0, 2) == 0 else 1.0
    return np.asarray(orientation * gamma, dtype=float)


def _unit_normal_vector(
    rng: np.random.Generator,
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = rng.normal(size=dimension)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError(f"{name} has a degenerate norm")
    return np.asarray(vector / norm, dtype=float)


def _ar1_factor(d: int, rho: float) -> np.ndarray:
    coordinates = np.arange(d)
    covariance = rho ** np.abs(np.subtract.outer(coordinates, coordinates))
    return np.linalg.cholesky(covariance)


def _apply_link(index: np.ndarray, name: str) -> np.ndarray:
    if name == "linear":
        return index
    if name == "quadratic":
        return index + 0.5 * index**2
    if name == "square":
        return index**2
    if name == "sin":
        return np.sin(1.5 * index)
    if name == "tanh":
        return np.tanh(2.0 * index)
    if name == "oscillating":
        return index * np.sin(math.sqrt(5.0) * index)
    raise ValueError(f"unknown link: {name}")


def _standardize_sample(
    values: np.ndarray,
    name: str,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if (
        not np.isfinite(mean)
        or not np.isfinite(scale)
        or scale <= np.finfo(float).eps
    ):
        raise ValueError(f"{name} has degenerate sample variance")
    standardized = (values - mean) / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError(f"{name} has degenerate sample variance")
    return np.asarray(standardized, dtype=float), mean, scale


def _require_finite_arrays(**arrays: np.ndarray | None) -> None:
    for name, array in arrays.items():
        if array is None:
            continue
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must have a floating dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")


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
