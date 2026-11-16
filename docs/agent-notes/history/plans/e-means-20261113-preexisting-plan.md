# Launch fourteen-function large benchmarks

Task large-fourteen-20261112; status **done**. User explicitly requests background
launch and ending the response without waiting for results. Scope completion is
successful persistent launch, not completion or scientific analysis of the runs.
Preserve all previous uncommitted work and frozen outputs; no commit.

**Goal:** Launch all14 prior case labels (seven-families + oscillation-seven)
for n4000,d50,m3 and n8000,d100,m3, 15 seeds0..14 and two methods. Launch a
separate n4000,d50,m3 repeat on independent seeds15..29. 420 fits per campaign,
1260 attempts total. The repeated f8 formula is retained under both original
labels, not silently deduplicated. Separate repeat seed choice was stated to user.
**Non-goals:** Wait for completion, analyze results, tune solvers, change formulas,
or change public defaults. User will return when results are available.
**Evidence:** Prior47 focused tests passed; generators, frozen workers and paired
protocol were verified. Host has32GiB RAM, about14GiB available at orientation.
**Change class:** Explicit experimental design profiles; no estimator change.
**Invariants:** Same14 links, Gaussian X, normalized signal/noise.3, true rank3;
public ADP-Grassmann-combined20 vs package MEANMAVE. Shared X/B/noise across
oscillation frequencies per seed, same data within method pairs. One BLAS thread,
4096MiB per-fit sampled RSS cap,7200s timeout because n/d grow. Three concurrent
independent campaigns; timing reflects concurrent execution. Save raw inputs,
bases, failures, full source/environment/protocol and cold fit/process time/RSS.

**Exact read set:** AGENTS.md; previous PLAN/STATE; agent-notes README/research/
WORKFLOW routes; r_sdr/{design.py,__main__.py,runner.py,protocol.py,README.md};
tests/test_r_sdr*.py. Source routes already identified; no broad scan needed.

**Work units and evidence:**
- [x] Add two opt-in14-case profiles without altering earlier cases. Failing
  tests first, then profile dimensions/link identity/CLI420 checks and existing
  focused suite/lint.
- [x] Prepare a self-contained launcher artifact and detached host processes,
  independent logs/PIDs/output directories; record the exact commands/config.
- [x] Verify all3 processes remain alive after launcher exits, all3 frozen
  manifests contain420 planned fits, and workers have started. Checkpoint routes,
  PLAN/STATE and stop the turn. Do not wait for first fit or overall completion.

**Stop conditions:** Occupied output, missing dependency, dead process or absent
manifest prevents claiming launch. Obtain runtime sandbox permission if needed
for persistence. Do not weaken safeguards to force fits through. Later failures
remain journaled; immutable resume uses the same options and --resume.

**Launch evidence:**49 focused tests passed; lint and diff checks passed. Active
PIDs1388526 (d50 main),1393943 (d100 NJ400),1388528 (d50 repeat). Exact commands,
logs/manifests and persistent PID check: benchmarks/results/outputs/r_sdr/
fourteen-large-20261112/{launch.json,launch-check.json,README.md,verification.log}.
Initial invalid d100 N_J300 attempt preserved and stopped; new output uses
explicit N_J400, minimum ceil(n/N_loc). All28 profile/center-budget combinations
passed direct size preflight. Launch objective fulfilled; computations remain
active and results have not been analyzed. User will return when ready.
