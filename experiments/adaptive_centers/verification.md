# Verification

Before numerical freeze (lead):

- New independent/reference/integration suite: `rtk proxy .venv/bin/python -m
  pytest tests/test_adaptive_centers.py -q`: **24 passed**. Kernel/boundary/
  zero-spectrum/offset support, alpha, direct covariance, sparse-vs-dense I/U,
  exact greedy marginals, PoU, validation, fixed/dynamic selection and common
  initialization/disabled-config identity. Log: verification/focused24.log.
- Shared relevant suite: `rtk proxy .venv/bin/python -m pytest
  tests/test_adaptive_centers.py tests/test_index_models.py
  tests/test_statistics_optimization.py -q -k 'not manifold'`: **83 passed,
  6 deselected**, including local/pilot/random, best/last, legacy/new and CLI
  compatibility. Log: verification/focused_shared.log.
- Scoped Ruff check/format on changed Python files: passed. Scoped Pyright on
  changed ADP modules: **0 errors, 0 warnings**; verification/types.log.
- Full `rtk proxy .venv/bin/python -m pytest -q`: collection blocked by three
  existing KeyError `mi-spokoini-m2-n` failures in tests/test_experiment.py,
  tests/test_multi_center_diagnostic.py and tests/test_multiv2_quality.py.
- Earlier unfiltered relevant suite (before four additional tests): **82 passed,
  3 failed**: tests/test_statistics_optimization.py::test_fit_reference[manifold-1],
  [manifold-2], [manifold-3]. CLI passes local_quadratic with m2; existing
  ADP/core/manifold/ADP_Manifold.py:71 rejects m!=1. None of that constructor,
  CLI or test files were changed in this task. Do not report a green full suite.

The initial red test failed on the absent AdaptiveCentersConfig API. Numerical
checks preceded benchmark freeze. Timing/RSS evidence belongs to frozen worker
rows and REPORT.md, not these test durations. GPU execution is outside the
experimental supported scope.

## Independent post-run audit

`verification/lead_audit.py` reads raw records without importing the benchmark
summary code. `verification/lead_audit.json` confirms 120 successful rows,
12 paired blocks, all 74 frozen source hashes plus protocol/snapshot/provenance,
one-thread workers, common input/init hashes and coefficient mapping. Maximum
orthogonality error is 1.80e-15. Every fit stopped at h_min; **zero inner
certificates converged**. This limits the recovery conclusions to this frozen
schedule and solver budget. Heldout seeds were not used.

`verification/trajectory_audit.py` replays multiplicative30/92802 with floor25
and floor25_search. Both initial/final basis hashes exactly reproduce the raw
records. At all 12 identical localization states: support and weights are
identical, alpha is bitwise equal, maximum I error is 1.34e-15, U error 1.12e-15,
mass error 2.85e-14. Yet final EDR differs by -0.01036. Small reduction-order
differences can enter the finite iterative solve and adaptive updates; the
specific solver branching decision was not established. Exact search does
not imply a bitwise identical full adaptive trajectory.

The diagnostic eta ratio can differ by 1 on singleton supports because a
roundoff numerator/denominator becomes 0/0 on another reduction path (defined
as 0). Eta is unused by the solver; it is recorded separately, without an
equality claim. No formula or numerical source was changed during timing.

The trajectory audit initially needed a basis orientation correction and then
excluded this diagnostic ratio from moment assertions; final assertions on
all working moments, support, alpha and replay hashes pass. See
`verification/trajectory_audit.json` for every state.

## Final compatibility patch and checks

After all timings, a regression was reproduced for accepted `seed=None`:
the new selection SeedSequence received `[None,92817]`. Two regression tests
failed before the fix. The selection stream now uses the integer entropy of
the existing root seed stream; for each frozen integer seed this equals the
original seed exactly. A dedicated stream comparison passes. Typing `cast`
has no runtime effect. This does not alter formulas or the integer benchmark
streams. Final current-source full-fit replays reproduce both recorded hashes.

Original manifests/snapshots remain immutable. The current source differs in
this expression and its typing import only; verification/seed_none.patch,
compatibility_overlay.json and current_compatibility_sources.zip preserve and
audit the change. The original 24-test source is also retained in that ZIP;
current tests append three cases. Original harness assert_frozen intentionally
rejects the patched tree; use the original snapshot to reproduce exact timings,
or verification/lead_audit.py to check the declared compatibility overlay.

Final current source: **27 feature tests** included in **86 passed,
6 deselected** shared checks (verification/final_shared86.log); scoped Ruff
check and format pass, Pyright **0 errors, 0 warnings** (final_types.log).
Unrelated full-suite collection/manifold failures above remain unresolved.

The lead independently verifies all 24 supplementary worker records, all 12
input/hash pairs, threads, source/protocol hashes, exact support/weights/alpha
and paired time ratios. Large-n median projected/baseline ratios .493 for
weights and .414 for alpha plus weights include geometry preparation and CSR
output, with common D construction excluded and reported separately. This is
localization-only evidence, not a full-fit recovery or streamed-baseline RAM
comparison. See supplement/REPORT.md and verification/lead_audit.json.
