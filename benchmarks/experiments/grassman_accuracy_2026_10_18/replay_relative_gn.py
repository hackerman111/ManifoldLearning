"""Проверка public port на всех сохранённых dev/heldout fit, без retuning."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ADP.core.ADP_Config import ADP_Config
from ADP.core.ADP_Solver import ADP_solver
from ADP.core.multi.ADP_multi_index import ADP_multi_index
from ADP.solver import grassman
from experiments.r_sdr.protocol import ROOT, digest, source_files

RUNS = (
    "grassman-relative-dev-repaired-2026-10-22",
    "grassman-relative-heldout-2026-10-22",
)


def run() -> None:
    output = Path(__file__).with_name("relative_gn_public_replay.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(k) != "1" for k in keys):
        raise ValueError("set all BLAS thread environment variables to 1")
    fits: list[dict[str, object]] = []
    result: dict[str, object] = {
        "note": "exact port equivalence; verification time is not benchmark timing",
        "current_source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in source_files()
        },
        "fits": fits,
    }
    for name in RUNS:
        directory = ROOT / "benchmarks/results/outputs/r_sdr" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        protocol = manifest["protocol"]
        for path, expected in protocol["sources"].items():
            assert digest(directory / "source" / path) == expected, path
        assert protocol["relative_solver"] == {
            "max_steps": 10,
            "tol": 1e-6,
            "local_ridge": 0.0,
            "smooth_max_steps": 5,
        }
        with (directory / "runs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["method"] not in ("ADP-Grassmann", "ADP-Grassmann-relative"):
                continue
            assert row["status"] == "ok"
            path = directory / "inputs" / f"{row['case']}-{row['seed']}.npz"
            with np.load(path) as data:
                x, y, truth = data["x"], data["y"], data["truth"]
            payload = hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes())
            assert payload.hexdigest() == row["data_sha256"], path
            relative = row["method"] == "ADP-Grassmann-relative"
            settings = dict(protocol["grassman_solver"])
            settings.update(
                relative_gn=False,
                quadratic_trials=0,
                constant_centers=False,
                smooth_max_steps=5,
            )
            if relative:
                settings = {
                    k: v
                    for k, v in protocol["relative_solver"].items()
                    if k != "smooth_max_steps"
                }
                settings["relative_gn"] = True
                settings.update(
                    quadratic_trials=0, constant_centers=False, smooth_max_steps=5
                )
            model = ADP_multi_index(
                int(row["m"]),
                config=ADP_Config(
                    seed=int(row["method_seed"]), **protocol["adp_config"][row["case"]]
                ),
                solver=ADP_solver(grassman.solve, **settings),
            )
            model.fit(x, y)
            assert model.basis_ is not None
            meta_path = directory / row["meta"]
            saved = np.loadtxt(meta_path.with_name("basis.csv"), delimiter=",", ndmin=2)
            old = json.loads(meta_path.read_text())
            diagnostics = json.loads(
                json.dumps(model.solver_diagnostics_, default=str, allow_nan=False)
            )
            check = {
                "run": name,
                "case": row["case"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "input_sha256": row["data_sha256"],
                "basis_equal": bool(np.array_equal(model.basis_, saved)),
                "basis_max_abs_error": float(np.max(np.abs(model.basis_ - saved))),
                "all_inner_diagnostics_equal": diagnostics == old["inner_diagnostics"],
                "outer_stop_equal": model.result_.stop_reason == old["stop_reason"],
                "manifest_fingerprint": manifest["fingerprint"],
            }
            fits.append(check)
            passed = (
                check["basis_equal"]
                and check["all_inner_diagnostics_equal"]
                and check["outer_stop_equal"]
            )
            if not passed:
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                raise AssertionError(check)
            if len(fits) % 20 == 0:
                print(f"{len(fits)}/120 fits exactly equal", flush=True)
    assert len(fits) == 120
    result["all_equal"] = True
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
