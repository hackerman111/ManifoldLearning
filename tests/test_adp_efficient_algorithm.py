from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ADP_DIR = ROOT / "ADP"
if str(ADP_DIR) not in sys.path:
    sys.path.insert(0, str(ADP_DIR))

from algorithm import step0, stepk
from data import generation as data
from pipeline import main as adp_pipeline


def test_squared_distances_gram_matches_naive_pairwise_distances():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(17, 11))
    centers = rng.normal(size=(5, 11))

    squared = step0.SquaredDistancesGram(X, centers)
    expected = np.sum((X[None, :, :] - centers[:, None, :]) ** 2, axis=2)

    assert squared.shape == (5, 17)
    assert np.allclose(squared, expected)
    assert np.allclose(
        step0.ComputeWeight(X, centers, 2.5),
        step0.Kernel(expected / 2.5**2),
    )


def test_pipeline_uses_projection_outer_steps_from_efficient_adp():
    X, Y, beta = data.MakeData(
        n=140,
        d=5,
        function="linear",
        noise_std=0.0,
        seed=77,
    )

    result = adp_pipeline.RunADP(
        X,
        Y,
        n_J=36,
        n_min=10,
        n_directions=6,
        outer_steps=2,
        inner_steps=4,
        seed=77,
        show_progress=False,
    )

    assert result["algorithm"] == "projection_average_derivative"
    assert result["directions"].shape == (36, 6, 5)
    assert result["local_gradients"].shape == (36, 5)
    assert len(result["history"]) >= 2
    assert len(result["h_history"]) >= 2
    assert stepk.CosineSimilarity(result["beta"], beta) > 0.8
