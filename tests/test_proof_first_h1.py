"""Малый reference для изолированного H1-скетча."""

from __future__ import annotations

import numpy as np
import pytest

from ADP.engine.common.statistic import calculate_statistics
from ADP.engine.manifol_engine.optimisation import local_slopes
from experiments.proof_first_h1 import _summarize, orthogonal_directions


def test_haar_blocks_and_compact_gram_match_direct_reference() -> None:
    d, P, J = 5, 12, 3
    direct = orthogonal_directions(np.random.default_rng(13), J, P, d)
    compact = np.empty((J, d + P % d, d))
    compact[:, :d] = np.sqrt(P // d) * np.eye(d)
    compact[:, d:] = direct[:, 2 * d :]
    for j in range(J):
        for start, width in ((0, d), (d, d), (2 * d, P % d)):
            rows = direct[j, start : start + width]
            np.testing.assert_allclose(rows @ rows.T, np.eye(width), atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0)
        np.testing.assert_allclose(direct[j].T @ direct[j], compact[j].T @ compact[j])

    generated_compact = orthogonal_directions(
        np.random.default_rng(17), J, P, d, compact_full=True
    )
    np.testing.assert_allclose(
        generated_compact[0].T @ generated_compact[0],
        2 * np.eye(d) + generated_compact[0, d:].T @ generated_compact[0, d:],
    )


def test_moment_identity_and_all_target_loss_with_unequal_mass() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=8)
    W = np.array([[8, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 2, 1, 1, 0, 0]], dtype=float)
    P = 7
    direct = orthogonal_directions(np.random.default_rng(21), 2, P, 3)
    compact = np.empty((2, 4, 3))
    compact[:, :3] = np.sqrt(2) * np.eye(3)
    compact[:, 3:] = direct[:, 6:]
    stats = calculate_statistics(X, Y, W, direct, normalized=True)
    compact_stats = calculate_statistics(X, Y, W, compact, normalized=True)

    for j in range(2):
        weights = W[j] / W[j].sum()
        centered = X - weights @ X
        c = centered.T @ (weights * (Y - weights @ Y))
        C = centered.T @ (weights[:, None] * centered)
        np.testing.assert_allclose(stats.I[j], direct[j] @ c, atol=1e-14)
        np.testing.assert_allclose(stats.U[j], direct[j] @ C, atol=1e-14)
        np.testing.assert_allclose(compact_stats.I[j], compact[j] @ c, atol=1e-14)
        np.testing.assert_allclose(compact_stats.U[j], compact[j] @ C, atol=1e-14)

    for _ in range(3):
        B = rng.normal(size=(2, 3))
        slopes = rng.normal(size=(2, 2))
        residual = stats.I - np.einsum("jpd,jd->jp", stats.U, slopes @ B)
        compact_residual = compact_stats.I - np.einsum(
            "jpd,jd->jp", compact_stats.U, slopes @ B
        )
        np.testing.assert_allclose(
            np.einsum("j,jp,jp->", stats.mass, residual, residual),
            np.einsum(
                "j,jp,jp->", compact_stats.mass, compact_residual, compact_residual
            ),
            atol=1e-13,
        )


def test_direction_budget_rejects_large_full_blocks() -> None:
    with pytest.raises(MemoryError, match="256 MiB"):
        orthogonal_directions(np.random.default_rng(0), 200, 1000, 1000)


def test_self_only_neighborhood_still_fails_manifold_rank() -> None:
    X = np.array([[1e8, -1e8, 0.0], [1e8 + 1, -1e8, 0.0]])
    Y = np.array([1.0, 2.0])
    W = np.array([[1.0, 0.0]])
    directions = orthogonal_directions(
        np.random.default_rng(4), 1, 4, 3, compact_full=True
    )
    stat = calculate_statistics(X, Y, W, directions, normalized=True)
    np.testing.assert_array_equal(stat.I, 0.0)
    np.testing.assert_array_equal(stat.U, 0.0)
    with pytest.raises(RuntimeError, match="rank-deficient local slope"):
        local_slopes(stat.I, stat.U, np.array([[1.0, 0.0, 0.0]]), 0)


def test_pair_gate_rejects_one_quality_loss_even_with_speed_gain() -> None:
    rows = []
    for seed in range(6):
        for variant, quality, seconds in (("iid", 0.9, 1.0), ("h1", 0.91, 0.7)):
            rows.append(
                {
                    "case": "multi_d10",
                    "variant": variant,
                    "seed": seed,
                    "error": None,
                    "quality": quality,
                    "fit_seconds": seconds,
                    "process_peak_rss_mib": 100.0,
                    "convergence": True,
                    "recovered": False,
                }
            )
    assert _summarize(rows, 6)["any_passed"]
    rows[-1]["quality"] = 0.9 - 2e-10
    assert not _summarize(rows, 6)["any_passed"]
