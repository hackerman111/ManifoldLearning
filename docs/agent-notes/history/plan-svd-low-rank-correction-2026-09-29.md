task_id: svd-low-rank-correction-2026-09-29
status: done
change_class: ESTIMATOR (new opt-in rank constraint), EXACT shared kernels

# Implement `SVD_corr.tex`

Goal: add a deterministic solver for `B=P+Delta`, `rank(Delta)<=r`, minimizing
`sum_j mass_j ||I_j-U_j B.T g_j||² + lambda||Delta||²` at fixed local `g`.
Keep the existing `rank(B)<=r` solver available for comparisons and preserve
the public `solve` arguments; expose the correction as an explicit mode.

Non-goals: stochastic sampling, changed default HPAO, GPU support, global
optimality claims, or a large benchmark grid.

Known evidence: `SVD_corr.tex` derives the residual, rank-one gain, shifted
v-step, scale re-fit, QR output and diagnostics. Current `SVD.py` shares all
numerical primitives but initializes a rank-deficient B at zero. Its objective
and prior completion belong to the older matrix mode. The current tree has an
untracked user `SVD_corr.tex`; leave it untouched.

Mathematical invariants: correction starts at zero, `P P.T=I_m`, positive-mass
weighted SSE, `lambda||Delta||²`, rank at most r, accepted objective descent,
explicit v normal-residual certificate, float64 and bounded O(Jpd+Jpr+md)
working storage. QR of `P+Delta` returns an m-row basis, and refit uses that
basis. The old matrix mode remains behavior-compatible.

Exact read set: `SVD_corr.tex` relevant sections; `ADP/solver/SVD.py`,
`tests/test_svd_solver.py`, `_multi_operator.py`, `agent-notes/ADP/multi-index.md`,
and focused benchmark entry points only.

Work units:
- [x] P1: implement shared low-rank loop with mode-specific base residual and
  ridge target; expose correction mode, zero-rank case, diagnostics and QR.
- [x] P2: add small dense reference tests for objective, rank-one/scale
  stationarity, monotonicity, rank/QR, direct/LSMR and legacy behavior.
- [x] P3: run a bounded paired reference comparison (full fixed-g ridge,
  matrix rank-r, correction rank-r), record time/memory/passes/objective and
  correction spectrum, then run focused public-fit/lint checks.
- [x] P4: refresh the SVD route, decision and state; inspect final diff.

Stop conditions: do not accept a component that increases the exact fixed-g
objective; do not weaken finite/rank/certificate checks. If a reference or
public-fit check fails twice for the same mathematical reason, revise the
approach and checkpoint before further tuning. Do not infer broad recovery
or speed claims from the bounded comparison.

P1/P2 evidence: `tests/test_svd_solver.py` reports 9 passed. In correction
mode the factor objective matches direct `P+Delta` evaluation for zero and
positive ridge; rank 0 returns P; accepted rank steps descend; rank-one scale
and joint scale derivatives match dense references; unrestricted dense ridge
is a lower bound. Both explicit matrix and correction modes complete a public
multi-index fit. Existing matrix-mode tests still pass.

P3 evidence: `benchmarks/svd_correction.py` and
`experiments/svd_correction_2026-09-29/{results.json,README.md}` record three
seed-matched fixed-g comparisons at `(J,p,d,m)=(40,8,20,3)`, ridge 0.3 and
one BLAS thread. Rank-2 correction objectives are 120.718/117.213/231.584
versus matrix 213.893/198.687/202.140 and unrestricted dense reference
3.488/4.431/4.827. Traced memory, time, U passes and projector distances
are saved per run. Median times were 8.91 ms correction and 9.56 ms matrix;
these small-shape timings do not establish broad speed. Ruff check/format and
the 9 focused tests pass. The script is a fixed-g comparison, not an outer-fit
recovery claim.

P4 evidence: updated `agent-notes/ADP/{README,multi-index}.md`, appended the
mode decision to `agent-notes/DECISIONS.md`, and checked `git diff --check`.
The user-supplied untracked `SVD_corr.tex` was not edited.
