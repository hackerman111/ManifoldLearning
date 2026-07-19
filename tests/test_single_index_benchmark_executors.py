from __future__ import annotations

import hashlib
import math
import time
from dataclasses import FrozenInstanceError, asdict, fields, replace

import numpy as np
import pandas as pd
import pytest

import adp.evaluation.single_index.datasets as dataset_module
import adp.evaluation.single_index.executors as executors
from adp.evaluation.single_index.datasets import (
    DatasetUnavailable,
    generate_synthetic_data,
    load_cached_real_dataset,
)
from adp.evaluation.single_index.telemetry import (
    diagnose_local_systems,
    encode_beta,
    summarize_weights,
    timing_remainder,
)
from adp.evaluation.single_index.types import (
    ExperimentParameters,
    SeedBundle,
    SingleIndexJob,
    SingleIndexSeriesConfig,
)


DEFAULT_SEEDS = SeedBundle(
    beta=101,
    features=102,
    noise=103,
    centers=104,
    directions=105,
    init=106,
    outliers=107,
    outlier_noise=108,
    gamma=109,
    misspecification=110,
)


def make_job(*, seeds: SeedBundle = DEFAULT_SEEDS, **updates: object) -> SingleIndexJob:
    parameters = ExperimentParameters(
        d=5,
        n_over_d=40,
        sigma_x=1.0,
        rho_corr=0.35,
        sigma_eps=0.5,
        link="quadratic",
        x_distribution="gaussian",
        noise_distribution="gaussian",
        heteroscedastic=False,
        outlier_fraction=0.0,
        outlier_scale=4.0,
        delta=0.0,
        center_fraction=0.2,
    )
    parameters = replace(parameters, **updates)
    return SingleIndexJob(
        experiment="1",
        parameters=parameters,
        seed=7,
        seeds=seeds,
        run_id="run-generator-test",
    )


def generated_arrays(generated: object) -> dict[str, np.ndarray | None]:
    data = generated.data
    return {
        "X": data.X,
        "y": data.y,
        "beta": data.beta,
        "centers": data.centers,
        "directions": data.directions,
        "noise": data.noise,
        "signal": generated.signal,
        "ordinary_noise": generated.ordinary_noise,
        "gamma": generated.gamma,
    }


def assert_same_arrays(first: object, second: object, *names: str) -> None:
    first_arrays = generated_arrays(first)
    second_arrays = generated_arrays(second)
    for name in names:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])


def test_weight_and_local_diagnostics_match_fixed_arrays():
    weights = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])

    summary = summarize_weights(weights)

    np.testing.assert_allclose(summary.sum_w, [1.5, 0.0])
    np.testing.assert_allclose(summary.sum_w2, [1.25, 0.0])
    np.testing.assert_allclose(summary.ess, [1.8, 0.0])
    np.testing.assert_array_equal(summary.nonzero, [2, 0])
    np.testing.assert_allclose(summary.min_weight, [0.0, 0.0])
    np.testing.assert_allclose(summary.max_weight, [1.0, 0.0])

    diagnostics = diagnose_local_systems(
        S=np.array([[1.0, 2.0]]),
        U=np.array([[[2.0], [1.0]]]),
        imav=np.array([[2.0, 3.0]]),
        beta=np.array([1.0]),
        intercepts=np.array([0.0]),
        slopes=np.array([1.0]),
        regularization=1e-10,
    )

    assert diagnostics[0].determinant == pytest.approx(9.0)
    assert diagnostics[0].lambda_min == pytest.approx(1.0)
    assert diagnostics[0].lambda_max == pytest.approx(9.0)
    assert diagnostics[0].condition == pytest.approx(9.0)
    assert diagnostics[0].rank == 2
    assert diagnostics[0].residual == pytest.approx(2.0)
    assert diagnostics[0].regularization == pytest.approx(1e-10)
    assert diagnostics[0].singular is False


def test_zero_local_system_and_exact_singular_threshold_are_classified():
    zero = diagnose_local_systems(
        S=np.zeros((1, 2)),
        U=np.zeros((1, 2, 1)),
        imav=np.zeros((1, 2)),
        beta=np.ones(1),
        intercepts=np.zeros(1),
        slopes=np.zeros(1),
        regularization=0.0,
    )[0]

    assert zero.determinant == 0.0
    assert zero.rank == 0
    assert math.isinf(zero.condition)
    assert zero.singular is True

    eps = np.finfo(np.float64).eps
    at_threshold = diagnose_local_systems(
        S=np.array([[math.sqrt(eps), math.sqrt(eps), 0.0]]),
        U=np.array([[[0.0], [0.0], [1.0]]]),
        imav=np.zeros((1, 3)),
        beta=np.ones(1),
        intercepts=np.zeros(1),
        slopes=np.zeros(1),
        regularization=0.0,
    )[0]
    above_threshold = diagnose_local_systems(
        S=np.array([[2.0 * math.sqrt(eps), 0.0, 0.0]]),
        U=np.array([[[0.0], [1.0], [0.0]]]),
        imav=np.zeros((1, 3)),
        beta=np.ones(1),
        intercepts=np.zeros(1),
        slopes=np.zeros(1),
        regularization=0.0,
    )[0]

    assert at_threshold.lambda_min == pytest.approx(2.0 * eps)
    assert at_threshold.singular is True
    assert at_threshold.rank == 1
    assert above_threshold.singular is False
    assert above_threshold.rank == 2


@pytest.mark.parametrize(
    ("dtype", "precision"),
    [(np.float64, 17), (np.float32, 9)],
)
def test_beta_encoding_uses_dtype_specific_roundtrip_precision(dtype, precision):
    beta = np.array([np.nextafter(dtype(1.0), dtype(2.0)), dtype(1.0 / 3.0)], dtype=dtype)

    encoded = encode_beta(beta)

    assert encoded == "|".join(format(float(value), f".{precision}g") for value in beta)
    np.testing.assert_array_equal(
        np.array([float(value) for value in encoded.split("|")], dtype=dtype),
        beta,
    )


def test_timing_remainder_is_exact_and_never_negative():
    assert timing_remainder(1.0, 0.2, 0.3) == pytest.approx(0.5)
    assert timing_remainder(1.0, 0.6, 0.5) == 0.0


def test_generated_data_wrapper_is_repeatable_frozen_and_slotted():
    job = make_job(delta=0.4, outlier_fraction=0.1)

    first = generate_synthetic_data(job)
    repeated = generate_synthetic_data(job)

    assert_same_arrays(first, repeated, *generated_arrays(first))
    assert first.metadata == repeated.metadata
    assert not hasattr(first, "__dict__")
    with pytest.raises(FrozenInstanceError):
        first.signal = np.zeros_like(first.signal)


@pytest.mark.parametrize(
    ("seed_name", "changed_name", "unchanged_names"),
    [
        (
            "beta",
            "beta",
            ("X", "centers", "directions", "noise", "ordinary_noise"),
        ),
        (
            "features",
            "X",
            ("beta", "directions", "noise", "ordinary_noise"),
        ),
        (
            "noise",
            "ordinary_noise",
            ("X", "beta", "centers", "directions", "signal", "gamma"),
        ),
        (
            "centers",
            "centers",
            ("X", "beta", "directions", "noise", "signal", "gamma"),
        ),
        (
            "directions",
            "directions",
            ("X", "beta", "centers", "noise", "signal", "gamma"),
        ),
        (
            "outliers",
            "noise",
            (
                "X",
                "beta",
                "centers",
                "directions",
                "signal",
                "ordinary_noise",
                "gamma",
            ),
        ),
        (
            "outlier_noise",
            "noise",
            (
                "X",
                "beta",
                "centers",
                "directions",
                "signal",
                "ordinary_noise",
                "gamma",
            ),
        ),
        (
            "gamma",
            "gamma",
            (
                "X",
                "beta",
                "centers",
                "directions",
                "noise",
                "signal",
                "ordinary_noise",
            ),
        ),
    ],
)
def test_each_generation_subseed_is_isolated(
    seed_name: str,
    changed_name: str,
    unchanged_names: tuple[str, ...],
):
    job = make_job(delta=0.6, outlier_fraction=0.2)
    changed_seeds = replace(
        job.seeds,
        **{seed_name: getattr(job.seeds, seed_name) + 10_000},
    )

    original = generate_synthetic_data(job)
    changed = generate_synthetic_data(replace(job, seeds=changed_seeds))

    assert not np.array_equal(
        generated_arrays(original)[changed_name],
        generated_arrays(changed)[changed_name],
    )
    assert_same_arrays(original, changed, *unchanged_names)


def test_misspecification_seed_only_changes_orientation_and_response():
    job = make_job(delta=0.6)
    original = generate_synthetic_data(job)
    candidates = [
        generate_synthetic_data(
            replace(
                job,
                seeds=replace(job.seeds, misspecification=seed),
            )
        )
        for seed in range(1_000, 1_016)
    ]
    changed = next(
        candidate
        for candidate in candidates
        if not np.array_equal(original.gamma, candidate.gamma)
    )

    assert_same_arrays(
        original,
        changed,
        "X",
        "beta",
        "centers",
        "directions",
        "noise",
        "signal",
        "ordinary_noise",
    )
    np.testing.assert_array_equal(original.gamma, -changed.gamma)
    assert not np.array_equal(original.data.y, changed.data.y)


def test_init_seed_does_not_change_generated_arrays():
    job = make_job(delta=0.6, outlier_fraction=0.2)
    changed = replace(job, seeds=replace(job.seeds, init=999_999))

    original_data = generate_synthetic_data(job)
    changed_data = generate_synthetic_data(changed)

    assert_same_arrays(original_data, changed_data, *generated_arrays(original_data))
    assert original_data.metadata["seed_init"] == job.seeds.init
    assert changed_data.metadata["seed_init"] == changed.seeds.init


def test_beta_centers_and_directions_follow_their_exact_streams():
    job = make_job(d=7, n_over_d=8, center_fraction=0.25)

    generated = generate_synthetic_data(job)
    data = generated.data

    expected_beta = np.random.default_rng(job.seeds.beta).normal(size=7)
    expected_beta /= np.linalg.norm(expected_beta)
    np.testing.assert_array_equal(data.beta, expected_beta)
    assert np.count_nonzero(data.beta) == 7
    assert np.linalg.norm(data.beta) == pytest.approx(1.0, abs=1e-15)

    selected = np.random.default_rng(job.seeds.centers).choice(
        job.parameters.n,
        size=job.parameters.n_centers,
        replace=False,
    )
    np.testing.assert_array_equal(data.centers, data.X[selected])

    n_directions = max(4, min(job.parameters.d, 32))
    expected_directions = np.random.default_rng(job.seeds.directions).normal(
        size=(job.parameters.n_centers, n_directions, job.parameters.d)
    )
    expected_directions /= np.linalg.norm(
        expected_directions,
        axis=-1,
        keepdims=True,
    )
    np.testing.assert_array_equal(data.directions, expected_directions)
    np.testing.assert_allclose(
        np.linalg.norm(data.directions, axis=-1),
        1.0,
        atol=1e-15,
    )


def test_gaussian_features_use_ar1_covariance_and_sigma_x_scale():
    job = make_job(
        d=6,
        n_over_d=2_000,
        sigma_x=1.7,
        rho_corr=0.75,
        sigma_eps=0.0,
        center_fraction=0.01,
    )

    X = generate_synthetic_data(job).data.X
    empirical = np.corrcoef(X, rowvar=False)

    assert empirical[0, 1] == pytest.approx(0.75, abs=0.025)
    assert empirical[0, 2] == pytest.approx(0.75**2, abs=0.03)
    assert empirical[0, 3] == pytest.approx(0.75**3, abs=0.035)
    np.testing.assert_allclose(X.var(axis=0), 1.7**2, rtol=0.06)


@pytest.mark.parametrize(
    ("distribution", "variance_tolerance"),
    [("uniform", 0.06), ("student_t5", 0.12)],
)
def test_nongaussian_features_are_standardized_and_independent(
    distribution: str,
    variance_tolerance: float,
):
    job = make_job(
        d=4,
        n_over_d=10_000,
        sigma_x=1.8,
        rho_corr=0.9,
        x_distribution=distribution,
        sigma_eps=0.0,
        center_fraction=0.0025,
    )

    X = generate_synthetic_data(job).data.X

    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.035)
    np.testing.assert_allclose(
        X.var(axis=0),
        1.8**2,
        atol=variance_tolerance * 1.8**2,
    )
    correlation = np.corrcoef(X, rowvar=False)
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
    assert np.max(np.abs(off_diagonal)) < 0.025


@pytest.mark.parametrize(
    ("distribution", "df", "unit_scale", "relative_tolerance"),
    [
        ("student_t5", 5, math.sqrt(3.0 / 5.0), 0.06),
        ("student_t3", 3, math.sqrt(1.0 / 3.0), 0.16),
    ],
)
def test_student_noise_has_theoretical_variance_and_exact_scaling(
    distribution: str,
    df: int,
    unit_scale: float,
    relative_tolerance: float,
):
    job = make_job(
        d=1,
        n_over_d=250_000,
        sigma_eps=1.7,
        noise_distribution=distribution,
        center_fraction=0.0004,
    )

    generated = generate_synthetic_data(job)
    expected = (
        np.random.default_rng(job.seeds.noise).standard_t(
            df=df,
            size=job.parameters.n,
        )
        * unit_scale
        * job.parameters.sigma_eps
    )

    np.testing.assert_array_equal(generated.ordinary_noise, expected)
    assert generated.ordinary_noise.var() == pytest.approx(
        job.parameters.sigma_eps**2,
        rel=relative_tolerance,
    )


LINKS = {
    "linear": lambda z: z,
    "quadratic": lambda z: z + 0.5 * z**2,
    "square": lambda z: z**2,
    "sin": lambda z: np.sin(1.5 * z),
    "tanh": lambda z: np.tanh(2.0 * z),
    "oscillating": lambda z: z * np.sin(math.sqrt(5.0) * z),
}


@pytest.mark.parametrize(("link_name", "raw_link"), LINKS.items())
def test_each_link_uses_exact_formula_and_realized_normalization(
    link_name: str,
    raw_link,
):
    generated = generate_synthetic_data(make_job(link=link_name, sigma_eps=0.0))
    z = generated.data.X @ generated.data.beta
    raw = raw_link(z)
    raw_mean = float(raw.mean())
    raw_std = float(raw.std(ddof=0))

    np.testing.assert_allclose(
        generated.signal,
        (raw - raw_mean) / raw_std,
        rtol=0.0,
        atol=2e-15,
    )
    assert generated.metadata["link_mean"] == pytest.approx(raw_mean, abs=1e-15)
    assert generated.metadata["link_std"] == pytest.approx(raw_std, abs=1e-15)
    assert generated.signal.mean() == pytest.approx(0.0, abs=2e-15)
    assert generated.signal.var() == pytest.approx(1.0, abs=2e-15)
    assert generated.data.link_name == link_name


def test_zero_noise_has_infinite_snr_and_zero_errors():
    generated = generate_synthetic_data(make_job(sigma_eps=0.0))

    np.testing.assert_array_equal(
        generated.ordinary_noise,
        np.zeros_like(generated.ordinary_noise),
    )
    np.testing.assert_array_equal(generated.data.noise, generated.ordinary_noise)
    np.testing.assert_array_equal(generated.data.y, generated.signal)
    assert generated.metadata["snr"] == math.inf


def test_heteroscedastic_noise_uses_exact_gaussian_formula():
    job = make_job(
        sigma_eps=0.7,
        heteroscedastic=True,
        noise_distribution="student_t3",
    )

    generated = generate_synthetic_data(job)
    z = generated.data.X @ generated.data.beta
    xi = np.random.default_rng(job.seeds.noise).normal(size=job.parameters.n)
    expected = 0.7 * np.sqrt((0.25 + z**2) / 1.25) * xi

    np.testing.assert_array_equal(generated.ordinary_noise, expected)
    np.testing.assert_array_equal(generated.data.noise, expected)
    np.testing.assert_allclose(generated.data.y, generated.signal + expected)


def test_outliers_replace_exact_selected_errors_with_independent_gaussians():
    job = make_job(
        n_over_d=50,
        sigma_eps=0.6,
        outlier_fraction=0.121,
        outlier_scale=7.0,
    )

    generated = generate_synthetic_data(job)
    count = math.ceil(job.parameters.outlier_fraction * job.parameters.n)
    indices = np.random.default_rng(job.seeds.outliers).choice(
        job.parameters.n,
        size=count,
        replace=False,
    )
    replacements = np.random.default_rng(job.seeds.outlier_noise).normal(
        scale=job.parameters.outlier_scale * job.parameters.sigma_eps,
        size=count,
    )
    expected = generated.ordinary_noise.copy()
    expected[indices] = replacements

    np.testing.assert_array_equal(generated.data.noise, expected)
    np.testing.assert_array_equal(
        np.flatnonzero(generated.data.noise != generated.ordinary_noise),
        np.sort(indices),
    )
    assert generated.metadata["outlier_count"] == count


def test_misspecification_gamma_is_orthogonal_and_response_uses_plus_sign():
    job = make_job(delta=0.8, sigma_eps=0.35)

    generated = generate_synthetic_data(job)

    assert generated.gamma is not None
    assert np.linalg.norm(generated.gamma) == pytest.approx(1.0, abs=1e-14)
    assert float(generated.data.beta @ generated.gamma) == pytest.approx(
        0.0,
        abs=1e-14,
    )
    misspec_index = generated.data.X @ generated.gamma
    raw = misspec_index + 0.5 * misspec_index**2
    misspec_signal = (
        raw - generated.metadata["misspecification_mean"]
    ) / generated.metadata["misspecification_std"]
    np.testing.assert_allclose(
        generated.data.y,
        generated.signal
        + job.parameters.delta * misspec_signal
        + generated.data.noise,
        rtol=0.0,
        atol=3e-15,
    )


def test_delta_zero_does_not_generate_misspecification_direction():
    generated = generate_synthetic_data(make_job(delta=0.0))

    assert generated.gamma is None
    assert "misspecification_mean" not in generated.metadata
    assert "misspecification_std" not in generated.metadata


def test_standardization_rejects_degenerate_and_nonfinite_samples():
    with pytest.raises(ValueError, match="constant link has degenerate sample variance"):
        dataset_module._standardize_sample(np.ones(8), "constant link")
    with pytest.raises(ValueError, match="bad link has degenerate sample variance"):
        dataset_module._standardize_sample(
            np.array([0.0, np.nan, 1.0]),
            "bad link",
        )


def test_metadata_distinguishes_requested_and_effective_generator_settings():
    overridden = generate_synthetic_data(
        make_job(
            x_distribution="uniform",
            rho_corr=0.9,
            noise_distribution="student_t3",
            heteroscedastic=True,
        )
    ).metadata

    assert overridden["rho_corr"] == 0.9
    assert overridden["noise_distribution"] == "student_t3"
    assert overridden["effective_rho_corr"] == 0.0
    assert overridden["effective_noise_distribution"] == "gaussian"

    ordinary = generate_synthetic_data(
        make_job(
            x_distribution="gaussian",
            rho_corr=0.65,
            noise_distribution="student_t5",
            heteroscedastic=False,
        )
    ).metadata

    assert ordinary["rho_corr"] == ordinary["effective_rho_corr"] == 0.65
    assert (
        ordinary["noise_distribution"]
        == ordinary["effective_noise_distribution"]
        == "student_t5"
    )


def test_metadata_is_scalar_only_complete_and_reports_effective_values():
    job = make_job(
        d=8,
        n_over_d=12.5,
        sigma_x=1.4,
        rho_corr=0.2,
        sigma_eps=0.4,
        x_distribution="student_t5",
        noise_distribution="student_t3",
        heteroscedastic=False,
        outlier_fraction=0.05,
        outlier_scale=6.0,
        delta=0.3,
        center_fraction=0.25,
    )

    generated = generate_synthetic_data(job)
    metadata = generated.metadata
    required = {
        "effective_p",
        "effective_n",
        "effective_J",
        "effective_n_directions",
        "snr",
        "link",
        "link_mean",
        "link_std",
        "x_distribution",
        "noise_distribution",
        "effective_noise_distribution",
        "heteroscedastic",
        "outliers_enabled",
        "misspecified",
        "outlier_count",
        "sigma_x",
        "rho_corr",
        "effective_rho_corr",
        "sigma_eps",
        "outlier_fraction",
        "outlier_scale",
        "delta",
        "center_fraction",
        "misspecification_mean",
        "misspecification_std",
        *(f"seed_{field.name}" for field in fields(SeedBundle)),
    }

    assert required <= metadata.keys()
    assert metadata["effective_p"] == job.parameters.d
    assert metadata["effective_n"] == job.parameters.n
    assert metadata["n_over_d"] == job.parameters.n_over_d
    assert metadata["effective_J"] == job.parameters.n_centers
    assert metadata["effective_n_directions"] == 8
    assert metadata["snr"] == pytest.approx(1.0 / job.parameters.sigma_eps**2)
    assert metadata["outliers_enabled"] is True
    assert metadata["misspecified"] is True
    for field in fields(SeedBundle):
        assert metadata[f"seed_{field.name}"] == getattr(job.seeds, field.name)
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in metadata.values()
    )
    arrays = generated_arrays(generated).values()
    assert all(
        np.issubdtype(array.dtype, np.floating)
        for array in arrays
        if array is not None
    )
    assert all(
        np.all(np.isfinite(array))
        for array in generated_arrays(generated).values()
        if array is not None
    )


def write_d1_package(tmp_path, *, manifest_updates=None, duplicate=False):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    path = prepared / "D01_airfoil_self_noise.csv"
    pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0],
            "x2": [2.0, 1.0, 0.0],
            "sound": [0.5, 1.0, 1.5],
        }
    ).to_csv(path, index=False)
    row = {
        "id": "D01",
        "file": path.name,
        "rows": 3,
        "features": 2,
        "target": "sound",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "official_page": "https://example.test/airfoil",
    }
    row.update(manifest_updates or {})
    rows = [row, dict(row)] if duplicate else [row]
    pd.DataFrame(rows).to_csv(tmp_path / "dataset_manifest.csv", index=False)
    return path


def test_cached_real_dataset_uses_d1_manifest_and_prepared_file(tmp_path):
    with pytest.raises(DatasetUnavailable, match="D01"):
        load_cached_real_dataset("D01", tmp_path, allow_download=False)

    path = write_d1_package(tmp_path)

    dataset = load_cached_real_dataset("D01", tmp_path, allow_download=False)

    assert dataset.X.shape == (3, 2)
    assert dataset.y.shape == (3,)
    np.testing.assert_array_equal(dataset.y, [0.5, 1.0, 1.5])
    assert dataset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.path == path
    assert dataset.source == "https://example.test/airfoil"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sha256": "0" * 64}, "checksum"),
        ({"rows": 4}, "rows"),
        ({"features": 3}, "features"),
    ],
)
def test_d1_manifest_rejects_integrity_mismatch(tmp_path, updates, message):
    write_d1_package(tmp_path, manifest_updates=updates)

    with pytest.raises(ValueError, match=message):
        load_cached_real_dataset("D01", tmp_path, allow_download=False)


def test_d1_manifest_rejects_duplicate_id_and_path_traversal(tmp_path):
    write_d1_package(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_cached_real_dataset("D01", tmp_path, allow_download=False)

    pd.read_csv(tmp_path / "dataset_manifest.csv").iloc[:1].assign(
        file="../outside.csv"
    ).to_csv(tmp_path / "dataset_manifest.csv", index=False)
    with pytest.raises(ValueError, match="prepared"):
        load_cached_real_dataset("D01", tmp_path, allow_download=False)


def test_d1_loader_never_uses_network_fallback(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("network fallback called")

    monkeypatch.setattr(dataset_module, "_download_openml", fail_download, raising=False)

    with pytest.raises(DatasetUnavailable, match="manifest"):
        load_cached_real_dataset("D01", tmp_path, allow_download=True)


def make_config(*, diagnostic_seeds: tuple[int, ...] = (0,)) -> SingleIndexSeriesConfig:
    return SingleIndexSeriesConfig(
        profile="smoke",
        experiments=("1",),
        jobs=1,
        seeds=(0,),
        diagnostic_seeds=diagnostic_seeds,
        center_fraction=0.2,
    )


def test_execute_job_returns_all_normalized_row_groups():
    job = replace(
        make_job(d=4, n_over_d=20, center_fraction=0.2),
        seed=0,
        run_id="run-executor-contract",
        diagnostic=True,
    )

    outcome = executors.execute_job(job, make_config())

    assert outcome.run_row["status"] in {"success", "nonconverged"}
    assert outcome.outer_rows
    assert outcome.inner_rows
    assert outcome.local_rows
    assert outcome.solver_rows
    for rows in (
        outcome.outer_rows,
        outcome.inner_rows,
        outcome.local_rows,
        outcome.solver_rows,
    ):
        assert all(row["run_id"] == job.run_id for row in rows)
    assert outcome.run_row["run_id"] == job.run_id
    assert outcome.run_row["statistics_workers"] == 1


def test_execute_job_logs_disjoint_phase_times_resources_and_full_adp_config(
    monkeypatch,
):
    job = replace(
        make_job(d=4, n_over_d=20, center_fraction=0.2),
        seed=0,
        run_id="run-resource-contract",
        diagnostic=True,
    )
    generated = generate_synthetic_data(job)
    adp_config = executors._benchmark_adp_config(job)
    model = executors.ADP.create("new", adp_config)
    result = model.fit(
        generated.data.X,
        generated.data.y,
        centers=generated.data.centers,
        directions=generated.data.directions,
    )
    result.timings["total"] = 999.0
    result.resource_usage = {
        "algorithm_time_sec": 1.25,
        "algorithm_rss_start_mib": 100.0,
        "algorithm_rss_min_mib": 99.0,
        "algorithm_rss_mean_mib": 105.0,
        "algorithm_rss_max_mib": 112.0,
        "algorithm_rss_peak_delta_mib": 12.0,
        "algorithm_memory_samples": 7,
        "algorithm_memory_source": "test-reader",
    }

    def delayed_generation(_job):
        time.sleep(0.03)
        return generated

    monkeypatch.setattr(executors, "generate_synthetic_data", delayed_generation)
    monkeypatch.setattr(executors, "_fit_adp", lambda *args, **kwargs: result)

    row = executors.execute_job(job, make_config()).run_row

    assert row["algorithm_time_sec"] == 1.25
    assert row["algorithm_rss_max_mib"] == 112.0
    assert row["algorithm_rss_peak_delta_mib"] == 12.0
    assert row["algorithm_memory_samples"] == 7
    assert row["algorithm_memory_source"] == "test-reader"
    assert row["data_generation_time_sec"] >= 0.02
    assert 0.0 <= row["fit_wall_time_sec"] < row["data_generation_time_sec"]
    assert row["telemetry_serialization_time_sec"] >= 0.0
    assert row["job_wall_time_sec"] >= (
        row["data_generation_time_sec"]
        + row["fit_wall_time_sec"]
        + row["telemetry_serialization_time_sec"]
    )
    for name, value in asdict(adp_config).items():
        assert row[f"adp_{name}"] == value


def test_failed_fit_without_partial_result_retains_algorithm_resources(
    monkeypatch,
):
    job = replace(
        make_job(d=4, n_over_d=20, center_fraction=0.2),
        seed=4,
        run_id="run-failed-resource-contract",
        diagnostic=False,
    )

    class FailingModel:
        result_ = None
        last_resource_usage_: dict[str, float | int | str] = {}

        def fit(self, *args, **kwargs):
            self.last_resource_usage_ = {
                "algorithm_time_sec": 1.23,
                "algorithm_rss_start_mib": 40.0,
                "algorithm_rss_min_mib": 39.5,
                "algorithm_rss_mean_mib": 43.0,
                "algorithm_rss_max_mib": 45.6,
                "algorithm_rss_peak_delta_mib": 5.6,
                "algorithm_memory_samples": 9,
                "algorithm_memory_source": "failure-reader",
            }
            raise ValueError("forced numerical failure")

    failing_model = FailingModel()
    monkeypatch.setattr(
        executors.ADP,
        "create",
        lambda *args, **kwargs: failing_model,
    )

    row = executors.execute_job(job, make_config(diagnostic_seeds=())).run_row

    assert row["status"] == "numerical_failure"
    assert row["error_type"] == "ValueError"
    assert row["algorithm_time_sec"] == 1.23
    assert row["algorithm_rss_start_mib"] == 40.0
    assert row["algorithm_rss_max_mib"] == 45.6
    assert row["algorithm_rss_peak_delta_mib"] == 5.6
    assert row["algorithm_memory_samples"] == 9
    assert row["algorithm_memory_source"] == "failure-reader"
    assert row["fit_wall_time_sec"] >= 0.0
    assert row["job_wall_time_sec"] >= row["fit_wall_time_sec"]


def test_nonfinite_result_is_numerical_failure_and_keeps_partial_rows(monkeypatch):
    job = replace(
        make_job(d=4, n_over_d=20, center_fraction=0.2),
        seed=3,
        run_id="run-nonfinite-contract",
        diagnostic=False,
    )
    original_fit = executors._fit_adp

    def fake_nonfinite_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        result.beta = np.full_like(result.beta, np.nan)
        return result

    monkeypatch.setattr(executors, "_fit_adp", fake_nonfinite_fit)

    outcome = executors.execute_job(job, make_config(diagnostic_seeds=()))

    assert outcome.run_row["status"] == "numerical_failure"
    assert outcome.run_row["invalid_value_count"] > 0
    assert outcome.outer_rows
    assert outcome.inner_rows
    assert outcome.local_rows
