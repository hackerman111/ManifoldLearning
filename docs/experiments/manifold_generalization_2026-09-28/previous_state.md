# Current agent state

## Active task

single-multi-recovery-2026-09-27: initialization/solver recovery study using
Method.md. PLAN active, R0/R1/R2 complete. Working scope n=1000,d=10/100,
Gaussian tau=0.4 including symmetric links; user asked to continue.

## Established constraints

Current local already uses OPG (ridge gradients + SVD); local-cv weighted
PRESS; multi pilot tanh MLP. Prior HYBRID/reduced solver studies failed.
Lower inner objective alone is insufficient for recovery.
SIR+SAVE initialization is ESTIMATOR, isolated until proof/reference/
paired selection/untouched validation pass.

## Next action / routes

Recovery semantics were revised per user request: geometric quality alone
defines recovery; solver convergence/stationarity remain diagnostics. Manifold
also requires the worst local-projector error over all centers <=0.2.
The runner, phase reports, suite analyzer, paired recovery summaries, and
manifold validators now use this rule; schema version is 10. Existing tests
were updated to encode recovery despite nonconvergence. No test suite was run.
The existing `selection.jsonl` contains 105 rows for a 108-fit protocol, and
the selection process is no longer running. Preserve these provisional
artifacts; do not treat them as a completed phase or overwrite them.
Prototype `experiments/inverse_moment_init.py` passes dense/reference,
affine invariance, rank checks; stress n=10000,d=1000 takes 3.588 s,
RSS 392.24 MiB. Proof independently audited, limits in proof.md.
65 focused tests passed; Ruff/Pyright pass. Seeds 73000–73003 selection, 74000–74009 validation.
Earlier selection summaries/report use the old convergence-gated definition
and are provisional. Validation remains prohibited until a complete selection
under the revised definition is frozen. Do not reuse held-out seeds for tuning.
Artifacts: docs/experiments/index_recovery_2026-09-27/.
Read-only Luna scout has returned previous research routes.

## Preserved unrelated work

MyThink.txt, TODO.txt, experiments/README.md, Method.md,
experiments/analyze_suite.py, experiments/sparse_matrix_audit.py and sparse
audit artifacts preserved. Prior PLAN copied to sparse audit completed_plan.md.
Prior durable DECISIONS entries preserved.
