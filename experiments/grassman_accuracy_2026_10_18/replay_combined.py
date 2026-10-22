"""Точный full-fit replay четырёх ADP arms после смены public defaults."""

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


def run() -> None:
    output = Path(__file__).with_name("combined_public_replay.json")
    if output.exists():
        raise FileExistsError(output)
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(k) != "1" for k in keys):
        raise ValueError("set BLAS threads to1")
    directory = ROOT / "benchmark_outputs/r_sdr/grassman-combined-heldout-2026-10-22"
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = manifest["protocol"]
    for name, sha in protocol["sources"].items():
        assert digest(directory / "source" / name) == sha, name
    source = {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
    config_keys = {
        "ADP-Grassmann": "grassman_solver",
        "ADP-Grassmann-relative": "relative_solver",
        "ADP-Grassmann-combined10": "combined10_solver",
        "ADP-Grassmann-combined": "combined_solver",
    }
    with (directory / "runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fits = []
    result: dict[str, object] = {
        "source_sha256": source,
        "reference_manifest": manifest["fingerprint"],
        "note": "default combined20; explicit controls; verification only",
        "fits": fits,
    }
    for row in rows:
        if row["method"] not in config_keys:
            continue
        assert row["status"] == "ok"
        with np.load(directory / "inputs" / f"{row['case']}-{row['seed']}.npz") as data:
            x, y, truth = data["x"], data["y"], data["truth"]
        assert (
            hashlib.sha256(x.tobytes() + y.tobytes() + truth.tobytes()).hexdigest()
            == row["data_sha256"]
        )
        settings = (
            {}
            if row["method"] == "ADP-Grassmann-combined"
            else protocol[config_keys[row["method"]]]
        )
        model = ADP_multi_index(
            int(row["m"]),
            config=ADP_Config(
                seed=int(row["method_seed"]), **protocol["adp_config"][row["case"]]
            ),
            solver=ADP_solver(grassman.solve, **settings),
        )
        model.fit(x, y)
        meta_path = directory / row["meta"]
        saved = json.loads(meta_path.read_text())
        basis = np.loadtxt(meta_path.with_name("basis.csv"), delimiter=",", ndmin=2)
        diagnostics = json.loads(
            json.dumps(model.solver_diagnostics_, default=str, allow_nan=False)
        )
        check = {
            "case": row["case"],
            "seed": int(row["seed"]),
            "method": row["method"],
            "data_sha256": row["data_sha256"],
            "basis_equal": bool(np.array_equal(model.basis_, basis)),
            "diagnostics_equal": diagnostics == saved["inner_diagnostics"],
            "stop_equal": model.result_.stop_reason == saved["stop_reason"],
        }
        fits.append(check)
        if not all(
            check[k] for k in ("basis_equal", "diagnostics_equal", "stop_equal")
        ):
            output.write_text(json.dumps(result, indent=2) + "\n")
            raise AssertionError(check)
        if len(fits) % 20 == 0:
            print(f"{len(fits)}/120 exact", flush=True)
    assert len(fits) == 120
    assert source == {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
    result["all_equal"] = True
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
