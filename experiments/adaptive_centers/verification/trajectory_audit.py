"""Replay the divergent pair; compare localization at identical states."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ADP import ADP_Config, ADP_multi_index
from ADP.engine.common.calculus import (
    calculate_alpha_k,
    generate_isotropic_proj,
    pairwise_distance2,
)
from ADP.engine.common.experimental_geometry import projected_components
from ADP.engine.common.statistic import calculate_statistics
from ADP.engine.common.weights import calculate_multi_weight
from experiments.adaptive_centers.common import ARMS, CASES, arm_config, array_hash
from experiments.adaptive_centers.data import generate


def main():
    root = Path(__file__).resolve().parents[1]
    raw = root / "results/fullfit_dev.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    case = next(c for c in CASES if c.name == "multiplicative30")
    problem = generate(case, 92802)
    distance = pairwise_distance2(problem.X, problem.X)
    source = root.parents[1] / "ADP/engine/common/index_fit.py"
    report = {
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "index_fit_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "arms": {},
    }
    trajectories = []
    for name in ("floor25", "floor25_search"):
        expected = next(
            r
            for r in rows
            if r["case"] == case.name and r["seed"] == 92802 and r["arm"] == name
        )
        settings = expected["effective_config"].copy()
        settings.pop("kernel")
        settings["experimental_centers"] = arm_config(
            next(a for a in ARMS if a.name == name), case
        )
        config = ADP_Config(**settings)
        trace = []
        model = ADP_multi_index(2, config).fit(
            problem.X, problem.Y, progress=trace.append
        )
        hashes = {
            "initial_basis": array_hash(model.result_.beta_init),
            "final_basis": array_hash(model.basis_),
        }
        assert all(expected["input_hashes"][k] == v for k, v in hashes.items())
        basis = np.asarray(model.result_.beta_init)
        spectrum = np.ones(2)
        rng = np.random.default_rng(np.random.SeedSequence(92802).spawn(3)[2])
        checks = []
        for row in trace:
            phi = generate_isotropic_proj(rng, 500, 10, 30)
            lam = np.maximum(spectrum, 0.25)
            h, alpha = row["h"], row["factor"]
            geometry = projected_components(
                problem.X, problem.X, basis, lam, h, distance
            )
            sparse = geometry.weights(alpha)
            dense = np.vstack(
                [
                    w
                    for _, w in calculate_multi_weight(
                        problem.X,
                        problem.X,
                        basis,
                        lam,
                        h,
                        alpha,
                        config.kernel,
                        config.batch_size,
                        distance2=distance,
                    )
                ]
            )
            assert np.array_equal(dense > 0, sparse.toarray() > 0)
            weight_error = float(np.max(np.abs(dense - sparse.toarray())))
            np.testing.assert_allclose(dense, sparse.toarray(), atol=2e-13, rtol=2e-12)
            original = calculate_statistics(
                problem.X, problem.Y, dense, phi, normalized=True
            )
            candidate = calculate_statistics(
                problem.X, problem.Y, sparse, phi, normalized=True
            )
            errors = {}
            for key in ("I", "U", "mass", "S", "mean", "n_eff", "eta"):
                left, right = getattr(original, key), getattr(candidate, key)
                errors[key] = float(np.max(np.abs(left - right)))
                if key != "eta":
                    np.testing.assert_allclose(left, right, atol=2e-12, rtol=2e-10)
            dense_alpha = calculate_alpha_k(
                problem.X,
                problem.X,
                basis,
                lam,
                h,
                config.N_loc,
                config.kernel,
                distance2=distance,
                block_size=config.batch_size,
            )
            sparse_alpha = geometry.alpha(config.N_loc)
            assert dense_alpha == sparse_alpha
            checks.append(
                {
                    "iteration": row["iteration"],
                    "weight_max_abs": weight_error,
                    "statistics_max_abs": errors,
                    "same_state_alpha": dense_alpha,
                    "support_edges": sparse.nnz,
                }
            )
            basis = np.asarray(row["basis"])
            spectrum = np.asarray(row["eigenvalues"])
        report["arms"][name] = {
            "hashes_match_raw": True,
            "hashes": hashes,
            "checks": checks,
        }
        trajectories.append(trace)
    report["trajectory_gaps"] = [
        {
            "iteration": a["iteration"],
            "basis_frobenius": float(np.linalg.norm(a["basis"] - b["basis"])),
            "alpha_abs": abs(a["factor"] - b["factor"]),
        }
        for a, b in zip(*trajectories, strict=True)
    ]
    report["interpretation"] = (
        "Same-state support and alpha agree; weights/statistics agree to roundoff. "
        "Full adaptive trajectories are not bitwise equivalent. This audit does not "
        "prove which iterative solver decision amplifies the perturbation."
    )
    (root / "verification/trajectory_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["trajectory_gaps"]))


if __name__ == "__main__":
    main()
