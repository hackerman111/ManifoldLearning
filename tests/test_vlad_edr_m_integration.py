from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ADP import ADP_multi_index, ADP_single_index
from ADP.cli.main import build_parser
from ADP.core.solver_common import HPAOResult
from ADP.solver.LSMR import HPAOResult as ExistingResult, solve
from experiments.r_sdr.design import catalog
from experiments.r_sdr.report import paired_baseline
from experiments.r_sdr.worker import solver_for


def test_imported_result_matches_existing_engine_contract() -> None:
    assert HPAOResult is ExistingResult


def test_existing_defaults_and_solver_choices_are_preserved() -> None:
    parser = build_parser()
    assert parser.parse_args([]).solver == "lsmr"
    for name in ("lsmr", "cg", "hybrid", "grassman"):
        assert parser.parse_args(["--solver", name]).solver == name
    assert ADP_single_index().solver.method is solve
    assert ADP_multi_index(2).solver.method is solve
    assert solver_for("ADP") is solve


@pytest.mark.parametrize("profile", ["d25-n5000", "high-dim-dm"])
def test_imported_comparison_profiles(profile: str) -> None:
    cases = catalog(profile)
    assert {case.m for case in cases} == {2, 3, 5}
    if profile == "d25-n5000":
        assert len(cases) == 3
        assert all((case.n, case.d) == (5000, 25) for case in cases)
    else:
        assert len(cases) == 9
        assert all(case.n == 10 * case.d * case.m for case in cases)


def test_combined_comparison_baseline_preserves_adp_priority() -> None:
    assert paired_baseline(["ADP-Grassmann-combined", "MAVE"]) == (
        "ADP-Grassmann-combined"
    )
    assert paired_baseline(["ADP", "ADP-Grassmann-combined"]) == "ADP"


def test_single_grassman_cli_uses_rank_one_matrix() -> None:
    from ADP.cli.main import _run

    args = build_parser().parse_args(
        [
            "--mode",
            "single",
            "--solver",
            "grassman",
            "--n",
            "48",
            "--d",
            "4",
            "--N_J",
            "8",
            "--N_phi",
            "3",
            "--N_loc",
            "6",
            "--N_lin",
            "10",
            "--outer_steps",
            "1",
            "--solver-max-steps",
            "2",
            "--index_init",
            "random",
            "--h_min",
            "1000000",
        ]
    )
    index, _, _, metadata = _run(args)
    assert index.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(index), 1.0)
    assert metadata["solver"] == "grassman"
    assert metadata["solver_max_steps"] == 2


@pytest.mark.parametrize("study_name", ["parameter_search", "gaussian_fourier_study"])
def test_imported_workers_retain_grassman_baseline(
    study_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from ADP.solver.grassman import solve as grassman

    study = importlib.import_module(f"experiments.{study_name}")
    config = {
        "N_J": 8,
        "N_phi": 3,
        "N_loc": 6,
        "N_lin": 10,
        "outer_steps": 1,
        "index_init": "random",
        "h_min": 1000000,
    }
    original_fit = ADP_multi_index.fit
    observed = []

    def bounded_fit(model, x, y):
        observed.append((model.solver.method, model.solver.settings))
        # Gaussian runner fixes a large grid; exercise its real fit on a tiny case.
        model.config = replace(model.config, **config)
        return original_fit(model, x, y)

    monkeypatch.setattr(ADP_multi_index, "fit", bounded_fit)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(48, 4))
    truth = np.eye(4)[:, :2]
    y = np.sin(x[:, 0]) + x[:, 1] ** 2
    if study_name == "parameter_search":
        input_file, output_file = tmp_path / "input.npz", tmp_path / "row.json"
        np.savez(input_file, x=x, y=y, truth=truth)
        job = {
            "phase": "test",
            "case": "tiny",
            "seed": 7,
            "arm": "base",
            "fingerprint": "test",
            "config": config,
            "input": str(input_file),
            "result": str(output_file),
        }
        job_file = tmp_path / "job.json"
        job_file.write_text(json.dumps(job))
        study.worker(job_file)
    else:
        (tmp_path / "inputs").mkdir()
        (tmp_path / "fits").mkdir()
        (tmp_path / "manifest.json").write_text(
            json.dumps({"jobs": [["original", 2, 7, "base"]]})
        )
        np.savez(tmp_path / "inputs/7.npz", X=x, basis=truth, noise=np.zeros(48))
        study.worker(tmp_path, 0)
        output_file = tmp_path / "fits/0.json"
    assert observed == [(grassman, {"max_steps": 20, "tol": 1e-6})]
    assert json.loads(output_file.read_text())["status"] == "ok"
