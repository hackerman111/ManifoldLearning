task_id: svd-preconditioner-forced-lsmr-2026-09-30
status: done
change_class: NUMERICAL comparison only; no production changes

# Goal
Measure the opt-in diagonal preconditioner on the saved heavy Spokoiny full fit
(n=800,d=50,m=2), forcing LSMR in both paired variants. Compare runtime,
iterations, fit quality, and memory for precondition_v=False/True on seeds 0-2.

# Evidence and hypothesis
The previous direct-path comparison had zero LSMR iterations because d=50 is
below the default direct threshold 128. The current API supports forcing LSMR
with direct_max_dimension=None. Hypothesis: diagonal right scaling lowers
LSMR work on this task; it may add setup cost, so end-to-end time and quality
must be measured.

# Invariants and scope
Keep data/init seeds, fit settings, float64, one BLAS thread, solver tolerances,
and stopping schedule fixed. Only vary precondition_v. Do not change production
code or estimator. Retain errors; do not claim benefit from fewer iterations
unless fit quality and completion also remain comparable.

# Read set
experiments/svd_preconditioner_spokoini_2026-09-30/{benchmark.py,REPORT.md,manifest.json};
benchmarks/svd_vs_hybrid.py; ADP/solver/SVD.py; agent-notes/ADP/multi-index.md.

# Work units
- [x] R0: create a separate forced-LSMR runner and immutable manifest.
- [x] R1: run three paired seeds in fresh subprocesses; stop on configuration
  mismatch or repeated worker failure.
- [x] R2: analyze results and record limitations in a new report.
- [x] R3: checkpoint PLAN.md and agent-notes/STATE.md.

# Verification and stop conditions
Diagnostics must show direct_solves=0 and positive LSMR iteration counts in
both arms. Check identical paired data/init and compare projector distance,
outer steps, wall time, and process peak RSS. Three successful seed pairs are
the bounded experiment; do not tune settings based on these seeds.

# Result
All six fits used LSMR (`direct_solves=0`, positive LSMR iteration counts).
Median paired runtime decreased 5.44%; median paired LSMR iterations decreased
12.95%. Every fit exceeded the original 30 s budget and
`svd_inner_converged=false`; projector error changed by 0.00232 on seed 2.
Full report and raw evidence:
`experiments/svd_preconditioner_forced_lsmr_spokoini_2026-09-30/REPORT.md`.
