# Current agent state

Completed: `svd-preconditioner-forced-lsmr-2026-09-30`; PLAN.md is done.

On the saved heavy Spokoiny fit (`n=800,d=50,m=2`), forced-LSMR paired runs
with preconditioner off/on (seeds 0-2) showed median paired runtime change
`-5.44%` and LSMR iteration change `-12.95%`. All six runs had zero direct
solves and positive LSMR iterations. All retained 162 outer steps and stopped
at `h_min`, but `svd_inner_converged=false`; seed 2 projector distance shifted
by `-0.00232`. All six exceeded the original 30 s budget. No production code
changed and no default behavior was promoted.

Report, raw rows, summary, manifest, and reproduction scripts:
`experiments/svd_preconditioner_forced_lsmr_spokoini_2026-09-30/`.
This is a three-seed, task-specific result; process RSS includes imports/data.
