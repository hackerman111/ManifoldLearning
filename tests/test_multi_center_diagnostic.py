from __future__ import annotations

import numpy as np

from experiments.multi_center_diagnostic import _local_spectrum, _weight_metrics
from experiments.multi_center_select import _cross_covariances, _greedy


def test_pilot_metrics_detect_rank_loss_without_ridge() -> None:
    X = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    weights = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    mass, n_eff, support = _weight_metrics(weights)
    rank, condition = _local_spectrum(X, weights, mass)

    np.testing.assert_array_equal(mass, [4.0, 2.0])
    np.testing.assert_array_equal(n_eff, [4.0, 2.0])
    np.testing.assert_array_equal(support, [4, 2])
    np.testing.assert_array_equal(rank, [2, 1])
    np.testing.assert_allclose(condition, [1.0, 0.0], atol=1e-14)


def test_cross_covariance_matches_direct_weighted_reference() -> None:
    X = np.array([[1.0, 2.0], [3.0, -1.0], [-1.0, 0.0]])
    Y = np.array([2.0, -1.0, 4.0])
    W = np.array([[1.0, 2.0, 0.5], [0.5, 0.0, 2.0]])
    expected = []
    for w in W:
        p = w / w.sum()
        mean_x = p @ X
        mean_y = p @ Y
        expected.append(
            np.sum(p[:, None] * (X - mean_x) * (Y - mean_y)[:, None], axis=0)
        )
    np.testing.assert_allclose(_cross_covariances(X, Y, W), expected, atol=1e-14)


def test_greedy_coverage_and_logdet_match_small_dense_reference() -> None:
    vectors = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.1, 0.1]])
    support = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
            [True, False, True, False],
            [False, False, False, True],
        ]
    )
    ids = np.array([5, 1, 3, 7])
    selected, checkpoints, reason = _greedy(vectors, support, ids, 4)
    assert reason == "selected"
    np.testing.assert_array_equal(selected, [1, 5])
    reference = np.eye(2) + vectors.T @ vectors
    log_reference = np.linalg.slogdet(reference)[1]
    for row in checkpoints:
        k = int(row["J"])
        chosen = np.array([1, 0, 2, 3][:k])
        info = np.eye(2) + 4 / k * vectors[chosen].T @ vectors[chosen]
        np.testing.assert_allclose(
            row["information_ratio"], np.linalg.slogdet(info)[1] / log_reference
        )
