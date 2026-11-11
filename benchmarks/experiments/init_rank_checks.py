"""Supplementary noise-only and QR controls; descriptive, no selection gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.engine.common.calculus import pairwise_distance2, search_bandwidth
from ADP.engine.common.initialize import initialize_basis_local_with_spectrum
from ADP.solver import grassman
from experiments.init_rank_diagnosis import OUT, ROOT, SAVED, geometry, input_data


def run():
    """First20 seeds, decided after initial replay; not held-out evidence."""
    destination = OUT / "supplement.json"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "source_sha256": source_hash,
        "m": [2, 3, 4, 5, 8],
        "runs": 20,
        "purpose": "post-hoc descriptive noise-only initializer and m2 QR/SVD controls",
    }
    (OUT / "supplement_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    qr = []
    for m in manifest["m"]:
        for offset in range(20):
            seed = (412000 if m in (2, 3, 5) else 413000) + offset
            x, y, truth = input_data(m, seed)
            method_seed = int(
                np.random.SeedSequence([seed, 1]).generate_state(1)[0]
            ) % (2**31 - 1)
            config = ADP_Config(seed=method_seed, N_lin=50, N_J=300, N_phi=12, N_loc=20)
            stream = np.random.SeedSequence(method_seed).spawn(3)[0]
            centers = x[
                np.random.default_rng(stream).choice(len(x), size=300, replace=False)
            ]
            distance2 = pairwise_distance2(x, centers)
            h = search_bandwidth(
                distance2, 50, config.kernel, lower=np.finfo(float).eps
            )
            weights = config.kernel(distance2 / h**2)
            mass = weights.sum(axis=1)
            support = np.count_nonzero(weights, axis=1)
            effective = mass**2 / np.sum(weights**2, axis=1)
            z = x @ truth
            signal = np.sin(z[:, 0]) + 0.5 * z[:, 1] ** 2
            if m > 2:
                signal += 0.35 * np.tanh(z[:, 2:]).sum(axis=1)
            clean = (signal - signal.mean()) / signal.std()
            basis, spectrum = initialize_basis_local_with_spectrum(
                x,
                clean,
                centers,
                distance2,
                50,
                config.kernel,
                config.local_ridge,
                m,
                mass_weighted=True,
            )
            row = {
                "m": m,
                "seed": seed,
                "clean_init": geometry(basis, truth),
                "clean_spectrum": (spectrum / spectrum[0]).tolist(),
                "h_lin": h,
                "mass_quantiles": np.quantile(mass, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                "support_quantiles": np.quantile(
                    support, [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
                "effective_n_quantiles": np.quantile(
                    effective, [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
                "centers_support_le_d": int(np.count_nonzero(support <= x.shape[1])),
                "effective_centers": float(mass.sum() ** 2 / (mass @ mass)),
                "data_sha256": hashlib.sha256(
                    x.tobytes() + y.tobytes() + truth.tobytes()
                ).hexdigest(),
            }
            rows.append(row)
            if m == 2:
                original = grassman._profile
                checks = []

                def svd_only(M, I, mass, tau):
                    fast = original(M, I, mass, tau)
                    reference = grassman._profile_svd(M, I, mass, tau)
                    if fast.triangular is not None:
                        checks.append(
                            {
                                "objective": abs(fast.value - reference.value),
                                "coefficients": float(
                                    np.max(
                                        np.abs(
                                            fast.coefficients - reference.coefficients
                                        )
                                    )
                                ),
                            }
                        )
                    return reference

                model = ADP_multi_index(
                    m,
                    config=config,
                    solver=ADP_solver(grassman.solve, max_steps=20, tol=1e-6),
                )
                with patch.object(grassman, "_profile", svd_only):
                    model.fit(x, y)
                old_rows = [
                    json.loads(line)
                    for line in (SAVED / "m2/runs.jsonl").read_text().splitlines()
                ]
                old = next(
                    r
                    for r in old_rows
                    if r["seed"] == seed and r["method"] == "ADP-Grassmann-combined"
                )
                assert row["data_sha256"] == old["data_sha256"]
                edr = geometry(model.basis_, truth)["edr"]
                qr.append(
                    {
                        "seed": seed,
                        "svd_edr": edr,
                        "old_edr": old["projector_distance"],
                        "delta_edr": edr - old["projector_distance"],
                        "safe_qr_calls": len(checks),
                        "max_objective_difference": max(
                            (c["objective"] for c in checks), default=0.0
                        ),
                        "max_coefficients_difference": max(
                            (c["coefficients"] for c in checks), default=0.0
                        ),
                    }
                )
            destination.write_text(
                json.dumps(
                    {"manifest": manifest, "initializers": rows, "qr_svd": qr}, indent=2
                )
            )
        print(f"m={m}: 20 noise-only initializers complete", flush=True)
    assert len(rows) == 100 and len(qr) == 20
    print(f"saved {destination.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    run()
