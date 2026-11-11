from __future__ import annotations

import numpy as np
import pytest

from ADP import ADP_multi_index, ADP_solver
from ADP.cli.main import _config, _run, _synthetic_data, build_parser
from ADP.solver.grassman import solve


@pytest.mark.parametrize("steps", [None, 2])
def test_grassman_cli_matches_public_model(steps: int | None) -> None:
    argv = [
        "--mode",
        "multi",
        "--solver",
        "grassman",
        "--n",
        "48",
        "--d",
        "4",
        "--index-dim",
        "2",
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
        "--index_init",
        "random",
        "--h_min",
        "1000000",
    ]
    if steps is not None:
        argv.extend(["--solver-max-steps", str(steps)])
    args = build_parser().parse_args(argv)
    data = _synthetic_data(args.n, args.d, args.index_dim, args.noise, args.data_seed)
    index, _, _, metadata = _run(args, data=data)
    model = ADP_multi_index(
        2, _config(args), ADP_solver(solve, max_steps=steps or 20, tol=args.solver_tol)
    ).fit(data[0], data[1])
    np.testing.assert_array_equal(index.T, model.basis_)
    assert metadata["solver_max_steps"] == (steps or 20)


@pytest.mark.parametrize("mode", ["manifold"])
def test_grassman_cli_requires_multi(mode: str) -> None:
    args = build_parser().parse_args(["--mode", mode, "--solver", "grassman"])
    with pytest.raises(ValueError, match="requires solver cg"):
        _run(args)
