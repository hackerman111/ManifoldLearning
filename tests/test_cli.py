from __future__ import annotations

import numpy as np
import pytest

from ADP.cli import _quality, _run, build_parser, main


@pytest.mark.parametrize(
    ("mode", "n", "d", "index_dim", "metric", "solver"),
    (
        ("single", "40", "3", "1", "cosine_abs=", "lsmr"),
        ("multi", "48", "4", "2", "projector_distance=", "cg"),
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
