import time

import numpy as np
import pytest

from adp import ADP, ADPConfig, StageExecutionError
from adp.common.types import LocalStatistics
from adp.common.utils import normalize_rows, unit_vector


def _float32_statistics() -> LocalStatistics:
    return LocalStatistics(
        variant="new",
        imav=np.array([[1.2, -0.4], [0.3, 0.8]], dtype=np.float32),
        centers=np.zeros((2, 3), dtype=np.float32),
        h=1.0,
        weights_mean=3.0,
        S=np.array([[0.7, -0.2], [0.1, 0.6]], dtype=np.float32),
        U=np.array(
            [
                [[0.9, 0.2, -0.1], [0.3, -0.7, 0.4]],
                [[-0.4, 0.6, 0.1], [0.8, 0.2, -0.3]],
            ],
            dtype=np.float32,
        ),
    )


def test_unit_vector_normalizes_huge_float32_without_overflow():
    vector = np.array([2.0e38, -2.0e38, 1.0e38], dtype=np.float32)

    normalized = unit_vector(vector)

    assert normalized.dtype == np.dtype("float32")
    assert np.all(np.isfinite(normalized))
    assert np.linalg.norm(normalized) == pytest.approx(1.0, rel=2e-7)


def test_normalize_rows_is_scale_safe_and_preserves_float32():
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    rows = np.array(
        [[2.0e38, -2.0e38, 1.0e38], [tiny, tiny, -tiny]],
        dtype=np.float32,
    )

    normalized = normalize_rows(rows)

    assert normalized.dtype == np.dtype("float32")
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=1),
        np.ones(2, dtype=np.float32),
        rtol=2e-7,
        atol=0.0,
    )


def test_normalize_rows_rejects_a_zero_direction():
    with pytest.raises(ValueError, match="нулев"):
        normalize_rows(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))


def test_alternating_solver_normalizes_huge_float32_beta(monkeypatch):
    model = ADP.create(
        "new",
        ADPConfig(
            dtype="float32",
            inner_steps=1,
            ridge=1.0,
            show_progress=False,
        ),
    )
    stats = _float32_statistics()

    monkeypatch.setattr(
        model,
        "_solve_local_coefficients",
        lambda statistics, beta: (
            np.zeros(2, dtype=np.float32),
            np.full(2, 1.0e-39, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        model,
        "_solve_beta",
        lambda *args, **kwargs: np.array(
            [2.0e38, -2.0e38, 1.0e38],
            dtype=np.float32,
        ),
    )

    beta, _, slopes, history = model._alternating_solve(
        stats,
        beta_start=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        lambda_penalty=0.5,
        outer=0,
        outer_started=time.perf_counter(),
    )

    assert beta.dtype == np.dtype("float32")
    assert np.all(np.isfinite(beta))
    assert np.linalg.norm(beta) == pytest.approx(1.0, rel=2e-7)
    assert np.all(np.isfinite(slopes))
    assert np.isfinite(history[-1].pre_normalization_beta_norm)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("min_neighbors", 0.0),
        ("min_neighbors", -1.0),
        ("min_neighbors", np.nan),
        ("min_neighbors", np.inf),
        ("ridge", -1.0),
        ("ridge", np.nan),
        ("ridge", np.inf),
        ("lambda_penalty", -1.0),
        ("lambda_penalty", np.nan),
        ("lambda_penalty", np.inf),
    ],
)
def test_config_rejects_invalid_regularization_and_neighbor_values(name, value):
    with pytest.raises(ValueError, match=name):
        ADPConfig(**{name: value})


def test_config_requires_positive_effective_beta_regularization():
    with pytest.raises(ValueError, match="lambda_penalty.*ridge"):
        ADPConfig(lambda_penalty=0.0, ridge=0.0)


def test_config_allows_zero_explicit_lambda_with_positive_ridge():
    config = ADPConfig(lambda_penalty=0.0, ridge=1.0e-10)

    assert config.resolved_lambda() == 0.0


@pytest.mark.parametrize(
    ("candidate_kind", "info", "expected_status"),
    [
        ("zero", 0, "invalid_solution"),
        ("nan", 0, "invalid_solution"),
        ("finite", -1, "breakdown"),
    ],
)
def test_invalid_or_broken_cg_raises_stage_failure_after_telemetry(
    monkeypatch,
    candidate_kind,
    info,
    expected_status,
):
    import adp.variants.random_projection as random_projection

    def fake_cg(operator, rhs, **kwargs):
        if candidate_kind == "zero":
            candidate = np.zeros_like(rhs)
        elif candidate_kind == "nan":
            candidate = np.full_like(rhs, np.nan)
        else:
            candidate = np.asarray(kwargs["x0"]).copy()
        return candidate, info

    monkeypatch.setattr(random_projection, "cg", fake_cg)
    model = ADP.create(
        "new",
        ADPConfig(dtype="float32", inner_steps=2, show_progress=False),
    )

    with pytest.raises(StageExecutionError, match="beta_solver.*CG"):
        model.algorithm._alternating_solve(
            _float32_statistics(),
            beta_start=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            lambda_penalty=0.5,
            outer=0,
            outer_started=time.perf_counter(),
        )

    assert model._last_solver_telemetry is not None
    assert model._last_solver_telemetry["linear_solver_status"] == expected_status


def test_float32_local_lstsq_fallback_keeps_augmented_operands_in_dtype(
    monkeypatch,
):
    import adp.variants.random_projection as random_projection

    model = ADP.create(
        "new",
        ADPConfig(dtype="float32", ridge=0.8, show_progress=False),
    )
    observed_dtypes = []

    def fail_batched_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("forced fallback")

    def recording_lstsq(design, response, **kwargs):
        observed_dtypes.append((design.dtype, response.dtype))
        return np.zeros(2, dtype=design.dtype), None, None, None

    monkeypatch.setattr(random_projection.np.linalg, "solve", fail_batched_solve)
    monkeypatch.setattr(random_projection.np.linalg, "lstsq", recording_lstsq)

    intercepts, slopes = model._solve_local_coefficients(
        _float32_statistics(),
        np.array([0.4, -0.3, 0.8], dtype=np.float32),
    )

    assert observed_dtypes
    assert set(observed_dtypes) == {(np.dtype("float32"), np.dtype("float32"))}
    assert intercepts.dtype == np.dtype("float32")
    assert slopes.dtype == np.dtype("float32")
