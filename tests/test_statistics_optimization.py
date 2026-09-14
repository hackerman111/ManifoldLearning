from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ADP.ADP_Manifold import ADP_Manifold
from ADP.engine.statistic import calculate_statistics

_spec = importlib.util.spec_from_file_location(
    "statistics_reference",
    Path(__file__).resolve().parents[1] / "benchmarks/statistics_reference.py",
)
assert _spec is not None and _spec.loader is not None
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
reference_statistics = _reference.calculate_statistics
manifold_statistics = _reference.manifold_statistics


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_statistics_reference(normalized: bool, sparse: bool, batch_size: int) -> None:
    rng = np.random.default_rng(25)
    X = rng.normal(size=(96, 7))
    X[:, -1] = X[:, 0] + 1e-12 * X[:, -1]
    X += 1e10
    Y = rng.normal(size=96)
    directions = rng.normal(size=(9, 5, 7))
    W = rng.random((9, 96))
    if sparse:
        for j, row in enumerate(W):
            row[j + 1 :] = 0  # Неравные соседства, включая одноточечное.
    W *= np.geomspace(1e-12, 1e12, 9)[:, None]
    originals = [a.copy() for a in (X, Y, W, directions)]
    expected = reference_statistics(X, Y, W, directions, normalized=normalized)
    result = calculate_statistics(
        X, Y, W, directions, normalized=normalized, batch_size=batch_size
    )
    # Ошибку ненормированных моментов оцениваем в единицах локальной массы.
    for name in result:
        scale = 1
        if name in ("I", "U") and not normalized:
            scale = result.mass.reshape((-1,) + (1,) * (result[name].ndim - 1))
        np.testing.assert_allclose(
            result[name] / scale, expected[name] / scale, rtol=3e-13, atol=3e-14
        )
    blocks = iter((j, W[j : j + 1]) for j in range(len(W)))
    streamed = calculate_statistics(X, Y, blocks, directions, normalized=normalized)
    for name in ("I", "U"):
        scale = result.mass.reshape((-1,) + (1,) * (result[name].ndim - 1))
        np.testing.assert_allclose(
            streamed[name] / scale, result[name] / scale, rtol=3e-13, atol=3e-14
        )
    for value, original in zip((X, Y, W, directions), originals, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_statistics_invalid_neighborhood(bad: float) -> None:
    W = np.ones((2, 8))
    W[1] = bad
    with pytest.raises(ValueError):
        calculate_statistics(np.ones((8, 3)), np.ones(8), W, np.ones((2, 4, 3)))


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_manifold_statistics_reference(batch_size: int) -> None:
    rng = np.random.default_rng(18)
    X = rng.normal(size=(97, 8))
    X[:, -1] = X[:, 0] + 1e-12 * X[:, -1]
    Y = rng.normal(size=97)
    W = rng.random((9, 97))
    W[W < 0.8] = 0
    W *= np.geomspace(1e-10, 1e10, 9)[:, None]
    directions = rng.normal(size=(9, 5, 8))
    model = SimpleNamespace(
        batch_size=batch_size,
        _weight_block=lambda *args: W[args[-1] : args[-1] + batch_size],
    )
    args = (model, X, Y, X[:9], directions, None, None, 1.0, 1.0)
    expected = manifold_statistics(*args)
    actual = ADP_Manifold._calculate_statistics(*args)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("mode", ["multi", "manifold"])
def test_fit_reference(seed: int, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = importlib.import_module("ADP.cli.main")
    args = runner.build_parser().parse_args(
        [
            "--mode",
            mode,
            "--n",
            "120",
            "--d",
            "4",
            "--index-dim",
            "2",
            "--N_loc",
            "28",
            "--N_lin",
            "50",
            "--N_J",
            "14",
            "--N_phi",
            "12",
            "--N_manifold",
            "8",
            "--sync-steps",
            "1",
            "--h_min",
            "1000000",
            "--outer_steps",
            "2",
            "--solver",
            "hybrid",
            "--solver-max-steps",
            "3",
            "--seed",
            "11",
            "--data-seed",
            str(seed),
        ]
    )
    actual, truth, _, _ = runner._run(args)
    monkeypatch.setattr(runner, "calculate_statistics", reference_statistics)
    monkeypatch.setattr(ADP_Manifold, "_calculate_statistics", manifold_statistics)
    expected, _, _, _ = runner._run(args)
    np.testing.assert_allclose(
        np.swapaxes(actual, -1, -2) @ actual,
        np.swapaxes(expected, -1, -2) @ expected,
        rtol=1e-10,
        atol=1e-10,
    )
    actual_quality = runner._quality(mode, actual, truth)
    expected_quality = runner._quality(mode, expected, truth)
    np.testing.assert_allclose(actual_quality, expected_quality, rtol=1e-10, atol=1e-10)
