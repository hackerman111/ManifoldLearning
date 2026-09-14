from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ADP.engine.initialize import _centered_ridge_gradient
from benchmarks.localization import environment, measure


def _case(k: int, d: int) -> dict[str, object]:
    rng = np.random.default_rng(82)
    X = rng.normal(size=(k, d))
    Y = rng.normal(size=k)
    root = np.sqrt(rng.uniform(0.1, 1, size=k))
    ridge = 1e-8
    rcond = np.finfo(float).eps * (10000 + d)

    def reference() -> np.ndarray:
        penalty = np.zeros((d, d + 1))
        penalty[:, 1:] = np.sqrt(ridge) * np.eye(d)
        design = np.column_stack((np.ones(k), X)) * root[:, None]
        return np.linalg.lstsq(
            np.vstack((design, penalty)),
            np.concatenate((Y * root, np.zeros(d))),
            rcond=rcond,
        )[0][1:]

    def centered() -> np.ndarray:
        result = _centered_ridge_gradient(X, Y, root, ridge, rcond)
        assert result is not None
        return result

    expected, actual = reference(), centered()
    return {
        "k": k,
        "d": d,
        "population_n_for_cutoff": 10000,
        "ridge": ridge,
        "seed": 82,
        "dtype": "float64",
        "relative_error": float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        "reference": measure(lambda: float(np.linalg.norm(reference())), 3),
        "centered": measure(lambda: float(np.linalg.norm(centered())), 3),
    }


def main() -> None:
    """Ridge-SVD против augmented LS: время отдельно от traced peak."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [_case(k, d) for k, d in ((300, 100), (300, 1000), (2000, 1000))]
    report = {**environment(), "results": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
