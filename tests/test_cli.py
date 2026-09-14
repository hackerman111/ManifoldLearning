from __future__ import annotations

import numpy as np
import pytest

from ADP.cli.main import _quality, _run, build_parser, main


@pytest.mark.parametrize(
    ("mode", "n", "d", "index_dim", "metric", "solver"),
    (
        ("single", "40", "3", "1", "cosine_abs=", "lsmr"),
        ("multi", "48", "4", "2", "trace_score=", "cg"),
        ("multi", "48", "4", "2", "trace_score=", "hybrid"),
    ),
)
def test_cli_smoke(
    capsys: pytest.CaptureFixture[str],
    mode: str,
    n: str,
    d: str,
    index_dim: str,
    metric: str,
    solver: str,
) -> None:
    assert (
        main(
            [
                "--mode",
                mode,
                "--n",
                n,
                "--d",
                d,
                "--index-dim",
                index_dim,
                "--noise",
                "0.01",
                "--data-seed",
                "1",
                "--seed",
                "2",
                "--N_loc",
                "6",
                "--N_lin",
                "10",
                "--N_J",
                "8",
                "--N_phi",
                "3",
                "--outer_steps",
                "1",
                "--lambda_penalty",
                "0.2",
                "--local_ridge",
                "1e-6",
                "--kernel",
                "epanechnikov",
                "--a",
                "1.41421356237",
                "--h_min",
                "1000000",
                "--batch_size",
                "4",
                "--index_init",
                "random",
                "--solver-tol",
                "1e-6",
                "--solver",
                solver,
                "--solver-max-steps",
                "2",
                "--theta",
                "0.2",
                "--trust-radius",
                "0.5",
                "--lsmr-maxiter",
                "100",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"mode={mode}" in output
    assert metric in output
    assert "estimator=new" in output
    assert f"solver={solver}" in output
    assert "step=0" in output
    assert "statistics:" in output
    assert "process RSS peak:" in output


@pytest.mark.parametrize("solver", ["cg", "hybrid"])
def test_cli_manifold_smoke(capsys: pytest.CaptureFixture[str], solver: str) -> None:
    assert (
        main(
            [
                "--mode",
                "manifold",
                "--solver",
                solver,
                "--n",
                "80",
                "--d",
                "4",
                "--index-dim",
                "2",
                "--noise",
                "0.01",
                "--N_loc",
                "20",
                "--N_lin",
                "30",
                "--N_J",
                "12",
                "--N_phi",
                "8",
                "--N_manifold",
                "6",
                "--sync-steps",
                "1",
                "--lambda-manifold",
                "0.5",
                "--h_min",
                "1000000",
                "--batch-size",
                "4",
                "--solver-tol",
                "1e-7",
                "--cg-maxiter",
                "200",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "mode=manifold" in output
    assert "local_projector_distance=" in output
    assert "prediction_rmse=" in output
    assert "phase=sync" in output


def test_cli_trace_selects_the_minimum_fit_step() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "single",
            "--n",
            "80",
            "--d",
            "4",
            "--N_loc",
            "8",
            "--N_lin",
            "12",
            "--N_J",
            "12",
            "--N_phi",
            "4",
            "--outer_steps",
            "2",
            "--h_min",
            "0.01",
            "--index_init",
            "random",
            "--solver-max-steps",
            "2",
            "--batch_size",
            "4",
            "--seed",
            "2",
            "--data-seed",
            "1",
        ]
    )

    index, true_basis, _, metadata = _run(args)
    trace = metadata["trace"]
    errors = [step["err"] for step in trace]
    selected = int(np.argmin(errors))

    assert metadata["selected_iteration"] == selected
    assert metadata["selected_error"] == min(errors)
    assert _quality("single", index, true_basis) == trace[selected]["quality"]


def test_multi_quality_is_normalized_projector_trace() -> None:
    true_basis = np.eye(4)[:, :2]
    angle = np.deg2rad(60.0)
    index = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angle), np.sin(angle), 0.0],
        ]
    )

    # tr(P_hat P_true) / m = (1 + cos(theta)^2) / 2.
    assert _quality("multi", index, true_basis) == pytest.approx(0.625, abs=1e-14)

    rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert _quality("multi", rotation @ index, true_basis) == pytest.approx(
        0.625, abs=1e-14
    )
    assert _quality("multi", true_basis.T, true_basis) == pytest.approx(1.0, abs=1e-14)


def test_cli_center_split_and_fixed_directions_are_recorded() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "single",
            "--n",
            "40",
            "--d",
            "3",
            "--N_loc",
            "6",
            "--N_lin",
            "8",
            "--N_J",
            "8",
            "--N_phi",
            "3",
            "--outer_steps",
            "2",
            "--h_min",
            "0.01",
            "--center-displacement",
            "0.1",
            "--training-set",
            "exclude_centers",
            "--fixed-directions",
            "--solver-max-steps",
            "2",
            "--batch_size",
            "4",
            "--seed",
            "2",
            "--data-seed",
            "1",
        ]
    )

    _, _, _, metadata = _run(args)

    assert metadata["training_set"] == "exclude_centers"
    assert metadata["training_size"] == 32
    assert metadata["center_displacement"] == 0.1
    assert metadata["center_displacement_scale"] > 0
    assert metadata["redraw_directions"] is False
    assert len(metadata["initial_eigenvalues"]) == 2
