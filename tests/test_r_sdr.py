from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


def design():
    return importlib.import_module("experiments.r_sdr.design")


def test_metrics_match_dense_projectors_and_reject_rank_loss():
    module = design()
    rng = np.random.default_rng(8)
    truth = np.linalg.qr(rng.normal(size=(12, 3)))[0]
    estimate = rng.normal(size=(12, 3))
    q = np.linalg.qr(estimate)[0]
    expected = np.linalg.norm(q @ q.T - truth @ truth.T) / np.sqrt(6)
    assert module.metrics(estimate, truth)["projector_distance"] == pytest.approx(
        expected, abs=1e-14
    )
    assert (
        module.metrics(truth @ np.diag([-2, 4, 0.2]), truth)["projector_distance"]
        < 1e-14
    )
    with pytest.raises(ValueError, match="rank"):
        module.metrics(np.zeros_like(truth), truth)


def test_catalog_and_seed_pairing():
    module = design()
    cases = module.catalog("full")
    assert len({case.name for case in cases}) == len(cases)
    assert min(case.d for case in cases) <= 5
    assert max(case.d for case in cases) >= 500
    assert max(case.n for case in cases) >= 5000
    assert {1, 2, 3} <= {case.m for case in cases}
    for case in module.catalog("smoke"):
        x, y, truth = module.generate(case, 91)
        for first, second in zip((x, y, truth), module.generate(case, 91), strict=True):
            np.testing.assert_array_equal(first, second)
        assert x.shape == (case.n, case.d)
        assert truth.shape == (case.d, case.m)
        np.testing.assert_allclose(truth.T @ truth, np.eye(case.m), atol=1e-14)
        assert not np.array_equal(x, module.generate(case, 92)[0])


@pytest.mark.parametrize(
    "method", ["ADP-Grassmann-combined10", "ADP-Grassmann-combined"]
)
def test_combined_methods_dispatch_public_solver_and_json_meta(tmp_path, method):
    from ADP.solver import grassman
    from experiments.r_sdr import runner, worker

    assert method in design().SUPPORTED_METHODS
    assert method not in design().METHODS
    assert worker.solver_for(method) is grassman.solve
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"fit_seconds": 0.1, "iterations": 2}))
    assert runner.read_worker_meta(method, meta)["iterations"] == 2


def test_multi_rank_profile_has_three_paired_rank_cases():
    module = design()
    cases = module.catalog("multi-rank")
    assert [(case.name, case.n, case.d, case.m, case.link) for case in cases] == [
        ("rank-3", 500, 20, 3, "additive"),
        ("rank-4", 500, 20, 4, "additive"),
        ("rank-5", 500, 20, 5, "additive"),
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "multi-rank",
            "--methods",
            "ADP,MAVE,MAVE-package",
            "--runs",
            "10",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "3 cases x 10 repetitions x 3 methods = 90 fits" in result.stdout


def test_multi_rank_profile_accepts_grassman_without_changing_default_methods():
    module = design()
    assert "ADP-Grassmann" not in module.METHODS
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "multi-rank",
            "--methods",
            "ADP-Grassmann,MAVE,MAVE-package",
            "--grassman-steps",
            "5",
            "--runs",
            "10",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "3 cases x 10 repetitions x 3 methods = 90 fits" in result.stdout


def test_accurate_grassman_dispatch_and_snapshot_sources():
    module = design()
    assert "ADP-Grassmann-accurate" not in module.METHODS
    assert "ADP-Grassmann-accurate" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-constant" not in module.METHODS
    assert "ADP-Grassmann-constant" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-armijo" not in module.METHODS
    assert "ADP-Grassmann-armijo" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-quadratic" not in module.METHODS
    assert "ADP-Grassmann-quadratic" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-intrinsic" not in module.METHODS
    assert "ADP-Grassmann-intrinsic" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-warmup" not in module.METHODS
    assert "ADP-Grassmann-warmup" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-conjugate" not in module.METHODS
    assert "ADP-Grassmann-conjugate" in module.SUPPORTED_METHODS
    assert "ADP-Grassmann-relative" not in module.METHODS
    assert "ADP-Grassmann-relative" in module.SUPPORTED_METHODS
    worker = importlib.import_module("experiments.r_sdr.worker")
    lazy_import = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import experiments.r_sdr.worker; "
            "assert 'experiments.grassman_accuracy_2026_10_18.candidate' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.constant_centers' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.backtracking' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.quadratic' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.intrinsic' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.warmup' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.conjugate_guard' "
            "not in sys.modules; assert "
            "'experiments.grassman_accuracy_2026_10_18.relative_gn' "
            "not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert lazy_import.returncode == 0, lazy_import.stderr
    assert worker.solver_for("ADP") is worker.solve_lsmr
    assert worker.solver_for("ADP-Grassmann") is worker.solve_grassman
    candidate = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.candidate"
    )
    assert worker.solver_for("ADP-Grassmann-accurate") is candidate.solve
    constant = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.constant_centers"
    )
    assert worker.solver_for("ADP-Grassmann-constant") is constant.solve
    armijo = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.backtracking"
    )
    assert worker.solver_for("ADP-Grassmann-armijo") is armijo.solve
    quadratic = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.quadratic"
    )
    assert worker.solver_for("ADP-Grassmann-quadratic") is quadratic.solve
    intrinsic = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.intrinsic"
    )
    assert worker.solver_for("ADP-Grassmann-intrinsic") is intrinsic.solve
    warmup = importlib.import_module("experiments.grassman_accuracy_2026_10_18.warmup")
    assert worker.solver_for("ADP-Grassmann-warmup") is warmup.solve
    conjugate = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.conjugate_guard"
    )
    assert worker.solver_for("ADP-Grassmann-conjugate") is conjugate.solve
    relative = importlib.import_module(
        "experiments.grassman_accuracy_2026_10_18.relative_gn"
    )
    assert worker.solver_for("ADP-Grassmann-relative") is worker.solve_grassman
    assert callable(relative.solve)  # Сохранённый независимый прототип.
    protocol = importlib.import_module("experiments.r_sdr.protocol")
    sources = protocol.source_files()
    assert any(path.name == "quadratic.py" for path in sources)
    assert any(path.name == "intrinsic.py" for path in sources)
    assert any(path.name == "warmup.py" for path in sources)
    assert any(path.name == "conjugate_guard.py" for path in sources)
    assert any(path.name == "relative_gn.py" for path in sources)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "multi-rank",
            "--methods",
            "ADP,ADP-Grassmann-accurate,ADP-Grassmann-constant,ADP-Grassmann-armijo,ADP-Grassmann-quadratic,ADP-Grassmann-intrinsic,ADP-Grassmann-warmup,ADP-Grassmann-conjugate,ADP-Grassmann-relative",
            "--runs",
            "10",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "3 cases x 10 repetitions x 9 methods = 270 fits" in result.stdout


@pytest.mark.parametrize(
    "method",
    [
        "ADP-Grassmann-armijo",
        "ADP-Grassmann-quadratic",
        "ADP-Grassmann-intrinsic",
        "ADP-Grassmann-warmup",
        "ADP-Grassmann-conjugate",
        "ADP-Grassmann-relative",
    ],
)
def test_experimental_grassmann_worker_meta_uses_json_parser(tmp_path, method):
    runner = importlib.import_module("experiments.r_sdr.runner")
    expected = {"solver_variant": method, "fit_seconds": 0.25}
    meta_path = tmp_path / "meta.txt"
    meta_path.write_text(json.dumps(expected))
    assert runner.read_worker_meta(method, meta_path) == expected


@pytest.mark.parametrize(
    ("method", "config_key", "description"),
    [
        ("ADP-Grassmann-conjugate", "conjugate_solver", "PR+"),
        ("ADP-Grassmann-relative", "relative_solver", "relative damping"),
        ("ADP-Grassmann-combined10", "combined10_solver", "quadratic3"),
        ("ADP-Grassmann-combined", "combined_solver", "quadratic3"),
    ],
)
def test_proposal_protocol_records_frozen_solver_and_semantics(
    tmp_path, monkeypatch, method, config_key, description
):
    runner = importlib.import_module("experiments.r_sdr.runner")
    protocol = {}

    class StopAfterProtocol(Exception):
        pass

    monkeypatch.setattr(runner, "environment", lambda *_: ({}, {}))

    def capture(_output, value, *, resume):
        protocol.update(value)
        raise StopAfterProtocol

    monkeypatch.setattr(runner, "ensure_manifest", capture)
    args = SimpleNamespace(
        output=tmp_path,
        rscript="Rscript",
        threads=1,
        adp_config=None,
        runs=10,
        seed=103800,
        timeout=300,
        memory_mib=4096,
        recovery_threshold=0.1,
        r_iterations=25,
        r_tolerance=1e-5,
        sir_slices=10,
        keep_inputs=False,
        solver_steps=80,
        grassman_steps=5,
        resume=False,
    )
    with pytest.raises(StopAfterProtocol):
        runner.run(
            args,
            [design().Case("rank-3", 500, 20, 3)],
            [method],
        )
    expected = {
        "max_steps": 10,
        "tol": 1e-6,
        "local_ridge": 0.0,
        "smooth_max_steps": 5,
    }
    if method == "ADP-Grassmann-relative":
        expected.update(relative_gn=True, quadratic_trials=0, constant_centers=False)
    if method.startswith("ADP-Grassmann-combined"):
        expected.update(relative_gn=True, quadratic_trials=3, constant_centers=True)
        if method == "ADP-Grassmann-combined":
            expected.update(max_steps=20, smooth_max_steps=None)
    assert protocol[config_key] == expected
    assert description in protocol["implementations"][method]
    assert "certificate" in protocol["convergence_definitions"][method]

    # Реальный fit_current: mass/lambda_prox передаёт сама ADP-модель.
    np = importlib.import_module("numpy")
    module = design()
    x, y, _ = module.generate(module.Case("tiny-multi", 80, 5), 90800)
    x_path, y_path = tmp_path / "x.csv", tmp_path / "y.csv"
    np.savetxt(x_path, x, delimiter=",")
    np.savetxt(y_path, y, delimiter=",")
    meta = tmp_path / "worker-meta.json"
    worker = importlib.import_module("experiments.r_sdr.worker")
    actual_solver = worker.solver_for(method)
    received_damping = []

    def recording_solver(*args, **kwargs):
        received_damping.append(kwargs["lambda_prox"])
        return actual_solver(*args, **kwargs)

    monkeypatch.setattr(worker, "solver_for", lambda _: recording_solver)
    worker.run(
        {
            "method": method,
            "x": str(x_path),
            "y": str(y_path),
            "basis": str(tmp_path / "basis.csv"),
            "meta": str(meta),
            "m": 2,
            "seed": 90800,
            "config": {"N_lin": 10, "N_J": 20, "N_phi": 5, "N_loc": 10},
            "solver": protocol[config_key],
        }
    )
    metadata = json.loads(meta.read_text())
    assert metadata["inner_diagnostics"]
    assert received_damping and all(value == 0.05 for value in received_damping)


def test_r_references():
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable")
    result = subprocess.run(
        ["Rscript", "--vanilla", "tests/r_sdr_reference.R"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_r_user_library_available_to_vanilla_workers():
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable")
    module = importlib.import_module("experiments.r_sdr.protocol")
    info, env = module.environment("Rscript", 1)
    assert "MAVE" in info["r_session"]
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", "library(MAVE)"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_cli_rejects_fewer_than_ten_runs():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.r_sdr", "--runs", "9", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "at least 10" in result.stderr


def test_process_timeout_retains_log(tmp_path):
    runner = importlib.import_module("experiments.r_sdr.runner")
    result = runner.run_process(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(10)",
        ],
        os.environ.copy(),
        tmp_path / "log.txt",
        timeout=0.3,
        memory_mib=4096,
    )
    assert result["status"] == "timeout"
    assert result["process_seconds"] < 5
    assert "started" in (tmp_path / "log.txt").read_text()


def test_resume_rejects_changed_protocol_and_preserves_manifest(tmp_path):
    module = importlib.import_module("experiments.r_sdr.protocol")
    module.ensure_manifest(tmp_path, {"runs": 10}, resume=False)
    module.ensure_manifest(tmp_path, {"runs": 10}, resume=True)
    with pytest.raises(ValueError, match="differs"):
        module.ensure_manifest(tmp_path, {"runs": 11}, resume=True)
    with pytest.raises(ValueError, match="resume"):
        module.ensure_manifest(tmp_path, {"runs": 10}, resume=False)
    assert json.loads((tmp_path / "manifest.json").read_text())["protocol"] == {
        "runs": 10
    }


def test_report_retains_failure_denominator_and_pairs_by_seed():
    module = importlib.import_module("experiments.r_sdr.report")
    rows = [
        {
            "case": "a",
            "seed": seed,
            "method": method,
            "status": "ok",
            "fit_seconds": 2.0 if method == "ADP" else 1.0,
            "process_seconds": 3.0,
            "peak_rss_mib": 50.0,
            "projector_distance": 0.05,
            "recovered": True,
        }
        for seed in range(10)
        for method in ("ADP", "SIR")
    ]
    rows[-1].update(status="timeout", recovered=False)
    rows[-1].pop("fit_seconds")
    summaries, pairs = module.summarize(
        list(reversed(rows)), runs=10, methods=["ADP", "SIR"]
    )
    sir = next(row for row in summaries if row["method"] == "SIR")
    assert sir["recovery_rate"] == pytest.approx(0.9)
    assert sir["n_ok"] == 9
    assert sir["n_timeout"] == 1
    assert pairs[0]["n_pairs"] == 9
    assert pairs[0]["fit_ratio_median"] == 0.5
    assert pairs[0]["paired_complete"] is False


def test_report_pairs_r_methods_to_grassman_when_requested():
    module = importlib.import_module("experiments.r_sdr.report")
    rows = [
        {
            "case": "rank-3",
            "seed": seed,
            "method": method,
            "status": "ok",
            "fit_seconds": 2.0 if method == "ADP-Grassmann" else 10.0,
            "projector_distance": 0.2 if method == "ADP-Grassmann" else 0.3,
            "recovered": False,
        }
        for seed in range(10)
        for method in ("ADP-Grassmann", "MAVE", "MAVE-package")
    ]
    _, pairs = module.summarize(
        rows, runs=10, methods=["ADP-Grassmann", "MAVE", "MAVE-package"]
    )
    mave = next(row for row in pairs if row["method"] == "MAVE")
    assert mave["baseline"] == "ADP-Grassmann"
    assert mave["n_pairs"] == 10
    assert mave["fit_ratio_median"] == 5.0
    assert mave["projector_delta_median"] == pytest.approx(0.1)


def test_adp_inner_certification_is_not_outer_convergence(tmp_path):
    module = design()
    x, y, _ = module.generate(module.catalog("smoke")[0], 34)
    np.savetxt(tmp_path / "x.csv", x, delimiter=",")
    np.savetxt(tmp_path / "y.csv", y, delimiter=",")
    worker = importlib.import_module("experiments.r_sdr.worker")
    worker.run(
        {
            "x": str(tmp_path / "x.csv"),
            "y": str(tmp_path / "y.csv"),
            "basis": str(tmp_path / "basis.csv"),
            "meta": str(tmp_path / "meta.json"),
            "m": 1,
            "seed": 5,
            "config": {"outer_steps": 1},
            "solver": {"max_steps": 80, "tol": 1e-6},
        }
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["converged"] is None
    assert isinstance(meta["all_inner_certified"], bool)
    assert meta["stop_reason"] == "outer_steps"


def test_grassman_worker_dispatches_the_real_solver(tmp_path):
    module = design()
    case = module.Case("grassman-smoke", 80, 6, 2)
    x, y, _ = module.generate(case, 34)
    np.savetxt(tmp_path / "x.csv", x, delimiter=",")
    np.savetxt(tmp_path / "y.csv", y, delimiter=",")
    worker = importlib.import_module("experiments.r_sdr.worker")
    worker.run(
        {
            "method": "ADP-Grassmann",
            "x": str(tmp_path / "x.csv"),
            "y": str(tmp_path / "y.csv"),
            "basis": str(tmp_path / "basis.csv"),
            "meta": str(tmp_path / "meta.json"),
            "m": 2,
            "seed": 5,
            "config": {
                "N_lin": 12,
                "N_J": 80,
                "N_phi": 3,
                "N_loc": 10,
                "index_init": "local",
                "select_step": "best",
                "outer_steps": 1,
            },
            "solver": {"max_steps": 2, "tol": 1e-6, "local_ridge": 0.0},
        }
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    basis = np.loadtxt(tmp_path / "basis.csv", delimiter=",", ndmin=2)
    assert meta["solver_variant"] == "ADP-Grassmann"
    assert meta["converged"] is None
    assert meta["n_inner_total"] == 1
    assert meta["inner_diagnostics"][0]["linear_solver"] == "grassman"
    assert np.isfinite(basis).all()
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)


def test_memory_limit_keeps_failed_process_measurements(tmp_path):
    runner = importlib.import_module("experiments.r_sdr.runner")
    result = runner.run_process(
        [sys.executable, "-c", "import time; x=bytearray(64*1024**2); time.sleep(10)"],
        os.environ.copy(),
        tmp_path / "memory.log",
        timeout=5,
        memory_mib=32,
    )
    assert result["status"] == "memory_limit"
    assert result["peak_rss_mib"] > 32


def test_cli_dry_run_creates_no_output(tmp_path):
    output = tmp_path / "never-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.r_sdr",
            "--profile",
            "full",
            "--dry-run",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "3000 fits" in result.stdout
    assert not output.exists()


def test_actual_r_run_resumes_after_interruption_without_duplicate_rows(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable")
    command = [
        sys.executable,
        "-m",
        "experiments.r_sdr",
        "--profile",
        "smoke",
        "--cases",
        "tiny-linear",
        "--methods",
        "SIR",
        "--output",
        str(tmp_path / "run"),
    ]
    rows_path = tmp_path / "run" / "runs.jsonl"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and process.poll() is None:
            if rows_path.exists() and len(rows_path.read_text().splitlines()) >= 2:
                break
            time.sleep(0.025)
        assert process.poll() is None, process.communicate()
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=10)
        assert process.returncode == 130
        before = rows_path.read_text().splitlines()
        assert 2 <= len(before) < 10
        result = subprocess.run(
            [*command, "--resume"], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
        after = rows_path.read_text().splitlines()
        assert after[: len(before)] == before
        assert len(after) == 10
        assert len({json.loads(line)["seed"] for line in after}) == 10
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
