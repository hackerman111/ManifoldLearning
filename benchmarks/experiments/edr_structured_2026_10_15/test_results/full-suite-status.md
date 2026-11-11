# Full-suite status

The full suite did not collect cleanly. Logs are retained verbatim:

- `full-pytest.log`: `rtk ... uv run pytest -q` exited 2 with 4 collection
  errors. The console entry point omitted the root namespace import for
  `benchmarks`; three more collection errors were the `multiv2` catalog
  `KeyError: 'mi-spokoini-m2-n'`.
- `full-pytest-module.log`: `rtk ... uv run python -m pytest -q` exited 2 with
  the same catalog `KeyError` in `test_experiment.py`,
  `test_multi_center_diagnostic.py`, and `test_multiv2_quality.py`.
- `pytest-excluding-catalog-failures.log`: the remaining suite, with exactly
  those three collection-blocked modules excluded, ran 479 passed, 7 failed,
  and 28 skipped in 20.53 s. CUDA skips state that no CUDA device is present.

The seven failures are outside this task's source scope:

- `test_cli_manifold_smoke[cg]`, `test_cli_manifold_smoke[hybrid]`,
  `test_hybrid.py::test_manifold_hybrid_fit_matches_cg`, and
  `test_statistics_optimization.py::test_fit_reference[manifold-1/2/3]`
  raise `ValueError: local_quadratic estimator requires index_dim=1`.
- `test_inverse_moment_init.py::test_recovery_gate_retains_failures_and_checks_each_pair`
  raises `KeyError: 'mi-spokoini-m2-n'`.

`rtk git diff --exit-code -- experiments/multiv2.py
experiments/multiv2_quality.py experiments/multi_center_diagnostic.py` exited
0, confirming the implicated catalog files are unchanged in this worktree.
The focused new and existing Grassmann gate remains green: 73 passed.

Fresh-process isolation selected exactly the seven failed node IDs above (no
new EDR tests): `isolated-existing-failures.log` records 7 failed in 0.59 s.
The same six manifold cases fail with `local_quadratic estimator requires
index_dim=1`, and the recovery-gate case fails with the catalog
`KeyError: 'mi-spokoini-m2-n'`. This rules out side effects from the new public
smoke test as the cause of these failures.
