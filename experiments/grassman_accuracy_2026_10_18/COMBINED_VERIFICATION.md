# Combined solver verification

## Numerical and integration checks

Focused Grassmann/profile/rank/relative/quadratic/constant/combined/R-SDR
suite: **247 passed**. New numerical paths had failing tests before the port;
default20 was separately tested red before promotion. Direct geometry,
minimum-norm augmented LS, huge constant loss, original gradient scale,
positive ridge, zero/all-zero/no-zero inputs and m1/2/3 comparisons are covered.
Historical scalar/polish prototypes explicitly pin old defaults; references
for old rank-one behavior do the same.

Ruff check and format check:17 modified Python files pass. Pyright:14 files,
zero errors/warnings. git diff --check passes.

## Exact full-fit public replay

`replay_combined.py` and `combined_public_replay.json`: **120/120 exactly
equal** bases, all inner diagnostics and outer stops, using all30 validation
datasets and four ADP arms. Combined20 uses `ADP_solver(grassman.solve)` with
no solver settings; the three controls use frozen explicit settings. Inputs
and frozen source hashes verified. Current source set matches replay hashes.
Grassman SHA256:
6610d885741dac4d604fe49e3e73120cc08c71b55cd75afea7ae185771cebfd7.
This is port/default equivalence on consumed data, not extra validation or
a new timing measurement. An initial verification run was interrupted for a
docstring lint correction; only the complete saved120-fit run is evidence.

## Whole-repository test boundary

`python -m pytest -q --continue-on-collection-errors --tb=short`:
717 passed,28 skipped (CUDA unavailable),7 failed,3 collection errors.
Final repeated output is retained in `full_pytest.txt`.

Failures reproduced with the earlier solver as well:

- tests/test_cli.py::test_cli_manifold_smoke[cg]
- tests/test_cli.py::test_cli_manifold_smoke[hybrid]
- tests/test_hybrid.py::test_manifold_hybrid_fit_matches_cg
- tests/test_statistics_optimization.py::test_fit_reference[manifold-1]
- tests/test_statistics_optimization.py::test_fit_reference[manifold-2]
- tests/test_statistics_optimization.py::test_fit_reference[manifold-3]
- tests/test_inverse_moment_init.py::test_recovery_gate_retains_failures_and_checks_each_pair

The first six use default local_quadratic with index_dim2, rejected by
ADP_Manifold at construction. The last imports multiv2 and raises the same
catalog KeyError as these collection errors:

- tests/test_experiment.py
- tests/test_multi_center_diagnostic.py
- tests/test_multiv2_quality.py

Catalog key: `mi-spokoini-m2-n`, experiments/multiv2.py:79. These paths do
not execute the modified Grassmann algorithm. They remain outside this slice.
Plain `uv run pytest` additionally lacked the benchmarks namespace; invoking
`python -m pytest` resolves that import without a code change.

Control evidence: `unrelated_baseline_check.txt`,7 failed/3 passed/3 collection
errors. The process-only pytest plugin replaces grassman.solve with the
saved prior public implementation, SHA256
8804e23a16c4cf2de65d9c7a0fddc20b03fe6156d19bc3f8888bff2f607d782a.
No source file was replaced. To reproduce that control, copy
`unrelated_baseline_plugin.txt` to /tmp/grassman_legacy_verification.py and use
`PYTHONPATH=/tmp uv run --no-sync python -m pytest -p grassman_legacy_verification`
with the named failing tests and `--continue-on-collection-errors --tb=short`.

Whole-repository tests are therefore **not all green**; Grassmann's focused
checks, source-equivalent public replay and the300 frozen benchmark fits pass.
