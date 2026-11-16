# Current state

Launch request large-fourteen-20261112 is **done**; three experiments remain
RUNNING IN BACKGROUND. User explicitly asked to launch and end the response;
do not wait or analyze unless user returns. PLAN.md describes launch-only scope.

Batch root: benchmarks/results/outputs/r_sdr/fourteen-large-20261112/
Active campaigns (each14 labels ×15 seeds ×2 methods=420 fits, total1260):
- d50-n4000-main, PID1388526, seeds0..14, N_J300.
- d100-n8000-main-nj400, PID1393943, seeds0..14, N_J400.
- d50-n4000-repeat, PID1388528, seeds15..29, N_J300.
All m3, Gaussian X, normalized signal/noise.3, public ADP-Grassmann-combined20
vs package MEANMAVE.14 labels retain duplicated f8 under its original names.
One thread/fit,7200s/4096MiB limits; three campaigns concurrent. Timing includes
host contention, so do not compare to isolated earlier timings without that limit.

Processes detached outside temporary sandbox, with stdin closed and independent
logfiles; they persist after the reply. launch.json has exact commands/PIDs and
old attempt; *.log files capture process output. launch-check.json records live
PIDs, manifests and started workers. status.py is read-only; run from host (or
with sandbox escalation) to see host PIDs. Each output has immutable source/env/
protocol and saved raw inputs/bases. completion.json appears at campaign end.

Important startup repair: first d100 attempt (old PID1388527/output
 d100-n8000-main) failed before fitting because N_J300 < ceil(8000/N_loc20)=400.
It was interrupted; its failed journal is preserved. Corrected campaign uses
explicit adp-d100-n8000.json {N_J:400}, a new output and unchanged solver. No
configuration was selected by EDR. All14 cases per size passed size preflight.

Implementation: two opt-in combined catalog profiles/CLI choices, tests in
 tests/test_r_sdr_large.py. Existing generators/defaults untouched.49 focused
tests passed, ruff/diff checks passed. Public solver code unchanged. Prior work
and frozen results retained; previous PLAN/STATE archived under history/
{plans,state}/large-fourteen-20261112-preexisting-*.md. No commit.

When user returns: inspect launch.json, all3 completion.json/runs.jsonl/logs,
report actual completeness/failures, then audit bases/pairing and analyze. Never
promote incomplete data. Resume only same commands with --resume; source/env
fingerprints must match. Do not restart duplicate processes or overwrite outputs.
