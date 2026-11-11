# Frozen study: metric profiled descent on hard Spokoiny

Task metric-spokoiny-2026-10-15. Freeze before running. This studies existing
optimizers, not a new estimator; no numerical library edits planned.

## Problem and arms

Use `benchmarks.grassman_benchmark._make_problem(seed)` unchanged. Its point is
n=800,d=50,m=2,J=800,P=10,N_lin=100,N_loc=10, local initialization, raw link
f(u1,u2)=(u1^3+u2)(u1-u2^3), X=2*Beta(1,1)-1, Gaussian noise sigma=.1.
Truth basis is experiments.data._spokoini_basis(50,2). Keep seed derivation,
kernel K(t)=max(1-t^2,0) with the existing squared-distance argument t
(no square root inserted), orthogonal tensor,
auto directions, redraw=True, all centers, center displacement0, select last,
a=1.010050167084168,h_min=1 and all other effective config unchanged. Serialize
the effective config and seed/init/data hashes; verify actual kernel/argument
and schedule from code in metadata. Dataset tau=1 is not solver local_ridge.

Shared solver: lambda_prox=.05 (or assert equal effective config), tol=1e-6,
local_ridge=0,max_angle=.5,workspace_bytes=16MiB; recycling0. Core retains its
existing internal fallbacks. Arms:

- core: production grassman.solve, existing energy_tol=.05.
- metric2: grassman_research.solve(method='metric',rank=2,metric_floor=1e-6).
- metric1: same, rank=1.
- isotropic2: frozen-only metric rank2,metric_floor=1. Scalar metric control.
- isotropic2_matched: frozen-only, same two-pass metric solver with an
  experiment-local metric_direction replacement returning isotropic truncated
  -G scaled to the spectral norm of the default metric proposal for that G.
  This preserves proposal magnitude on each call while removing orientation;
  it is an explicit APPROXIMATE ablation, never a production method.

For matched control compute Z=original_metric_direction(G,g,mass,q,1e-6),
E=-[G]_q, return E*||Z||2/||E||2 (zero when E=0). Both metric calls use this
rule. Shapes/rank/finite/FD checks precede benchmark. This control keeps each
call's spectral magnitude, not an identical trajectory or accepted step.

## Stages and fixed budgets

1. Verify code and small independent tests before timed workers. No concurrent
   timed work or tests. Each worker is a fresh CPU float64 subprocess pinned
   to one OPENBLAS/OMP/MKL thread; rotate arm order by seed/stage. Cold fits,
   excluding imports/data generation but including .fit setup. Fixed-snapshot
   solves may warm up separately; state the boundary and never mix timings.
2. Development full fits seeds0,1,2, arms core/metric2/metric1, inner budget5
   (9 fits). No hyperparameter selection. Wrapper may record only timer and
   existing diagnostics references; expensive scalar math/copies are deferred
   outside timed .fit. Report wrapper overhead boundary consistently.
3. Separate untimed core capture per seed; save inputs at outer call indices
   0,80,160 (zero-based) and final actually available call if160 absent. Record
   scalar per-call truth principal errors, R spectrum/condition, local ranks,
   stationarity, h/rho if exposed. Save U/I/B/mass only at these three points,
   not every step. Instrumented capture timings are excluded from speed claims.
   One metric2 instrumented fit on seed0 may collect corresponding curves and
   proposal norm/cap/backtracking counters; also excluded from timed results.
4. Frozen snapshots: each seed x each of three phases x 5 arms above x
   budgets5/100 (90 solves). Same actual arrays and initial index for all arms.
   Recompute full profile and ambient gradient independently after timing;
   certificate scale=max(1,sum mass*I^2). Report no convergence-speed comparison
   when either endpoint lacks the same certificate. Always keep finite budget
   objective, gradient, truth-projector error and rank/orthogonality results.
5. Fixed budget20 development full fits: seeds0,1,2 core/metric2 (6 fits), to
   distinguish a five-step budget effect from algorithmic performance.
6. Hold-out seeds3..7, core/metric2 at five steps only if development budget5
   gate: no per-seed projector regression >1e-4 AND (paired median fit speedup
   >=1.10 OR median relative projector reduction>=20% at paired median time
   ratio<=1.5). Failed development gate leaves validation seeds untouched.
   No default promotion in this task regardless of result. Any amendment must
   be written before affected runs, preserving old protocol hashes/artifacts.

## Measurements and audit

Save append-only JSONL, errors/timeouts, source/protocol/git-dirty hashes,
environment/NumPy/SciPy/BLAS/thread versions, exact settings, input/init/first
statistics hashes and full config. Timeout180s/fullfit,60s/frozen. Refuse
overwrite. All actual ranks may be below requested rank; expose that limit.

For full fits: fit wall, summed solver wall, process ru_maxrss KiB, outer stop/
iterations/selected iteration, final truth projector error and both principal
angle sines (via small overlap SVD, no d*d projector), orthogonality, inner
stop/certificate/rank-guard totals. Outer h_min does not imply convergence.
Use normalized projector error sqrt(mean(sin(theta)^2)). Summarize paired
median ratios and individual seeds, not ratio of aggregate medians alone.

For frozen solves: objective/initial-objective ratio, independent gradient
certificate, smooth/rank state, accepted steps, operator actions/evaluations,
wall/RSS and truth errors. Objective improvement need not improve recovery.
Instrumented seed0 curves add R eigenvalues, clipping frequency, chosen q,
cap and backtracking counts where implementable without library changes.
`metric_floor=1` gives -[G]_q/(sqrt(lambda_max(R))*||[G]_q||F), hence step scale
is a confound unless cap activates; never assert automatic clipping for q=2.
Use spectral norms and conditional bounds, not the unit-Frobenius bound on G.

Stop/revise on generator mismatch, failed formula/gauge/FD/certificate checks,
nonfinite/bad orthogonality, hidden regularization, uncontrolled threads or
overwriting raw evidence. Existing unrelated catalog/manifold test failures
are documented in the preceding experiment and are outside this task.
