# Seven-family N_J sweep — 2026-11-16

Task `seven-families-nj-20261116`; status **done**.

**Goal:** Compare the standard ADP center budget with `N_J=0.8n` and `N_J=n`
on seven link families at `(n,d)=(1000,25)` and `(2000,50)`.

**Non-goals:** No public solver/default changes, tuning, promotion, or broad EDR
recovery claim. The earlier frozen campaigns and their outputs remain untouched.

**Known evidence:** The existing seven-family protocol uses `m=3`, Gaussian
`X`, noise `.3`, seeds `0..14`, `ADP-Grassmann-combined`, and
`MAVE-package`. In this runner the standard is `N_J=min(n,300)=300`. With
`N_loc=20`, the validity lower bounds are 50 and 100; all requested budgets
are valid. The seven link definitions are in `benchmarks/experiments/r_sdr/README.md`.

**Class and hypothesis:** `ESTIMATOR` comparison: changing `N_J` changes the
finite-center fit and its center sampling. More centers may reduce center
subsampling variability, but EDR gains are not guaranteed and the comparison
does not isolate center count from other `N_J`-dependent effects.

**Invariants:** Same seven functions, noise, `m`, seeds, generated datasets,
method settings, and one-thread protocol within each `(n,d)` block. Only ADP
`N_J` varies. Budgets are 300/800/1000 for `n=1000` and 300/1600/2000 for
`n=2000`. MEANMAVE runs once per size as a shared reference. Keep failures,
timeouts, certificates, time, and RSS visible. No retuning after results.

**Read set:** `benchmarks/experiments/r_sdr/{README.md,design.py,__main__.py,protocol.py,runner.py}`;
`ADP/engine/common/index_fit.py`; prior seven-family report and this checkout's
current `PLAN.md`/`STATE.md`. No other source changes planned.

**Bounded work units:**

- [x] Freeze an artifact-local campaign driver and six configs; validate the
  effective center budgets before long execution. Evidence: all six dry-runs
  accepted seven cases and passed `ADP_Config` plus `ceil(n/N_loc) <= N_J <= n`.
- [x] Run six serial campaigns: three `N_J` arms for each size, 15 seeds per
  family. Standard arm includes MEANMAVE; other arms run ADP only.
- [x] Audit completion, frozen configs, same-data hashes across arms, failures,
  and per-family quality/time/RSS/certificates; write a joined report.
- [x] Update `STATE.md`, mark this plan done, and preserve the prior completed
  plan and unrelated working-tree artifacts.

**Verification:** 840 planned fits (420 per size); 630 ADP fits plus 210 shared
MEANMAVE fits. Each ADP arm must have 105 rows per size and identical input and
truth arrays across the three `N_J` settings. Report EDR and recovery at `.1`,
fit/process time, RSS, failures, and all available solver certificates.

**Stop conditions:** Stop a configuration that violates size guards or reaches
the 300-second/4096-MiB per-fit limit; retain its rows and report incomplete
pairs. Do not change thresholds, seeds, or settings in response to observed
results.

**Output root:** `benchmarks/results/outputs/r_sdr/seven-families-nj-20261116/`.

**Checkpoint 1:** `(n,d,N_J)=(1000,25,300)` complete: 210/210 fits, zero
failed fits, process exit 0. Manifest, completion record, and per-fit rows are
saved under `n1000-d25/standard/`.

**Checkpoint 2:** `(n,d,N_J)=(1000,25,800)` complete: 105/105 ADP fits, zero
failed fits, process exit 0. Saved under `n1000-d25/nj800/`.

**Checkpoint 3:** All `(n,d)=(1000,25)` arms are complete: 420/420 total fits
(315 ADP and 105 MEANMAVE), zero failed fits. The 105 ADP input hashes agree
across all three budgets. Family-averaged mean EDR is about .391/.370/.351 for
`N_J=300/800/1000`; recovery is 0/105, 6/105, and 9/105. Median ADP fit time
grows from about .9s to 2.1s and 2.7s. Full per-family evidence is in
`PROGRESS-n1000.md`.

**Checkpoint 4:** `(n,d)=(2000,50,N_J=300)` complete: 210/210 fits, zero
failed fits. Mean EDR averaged over seven family means is about .497 for ADP
and .275 for MEANMAVE; recovery is 0/105 versus 61/105. ADP has a lower
family-mean EDR only on the oscillatory link. Full baseline results are in
`PROGRESS-n2000.md`.

**Checkpoint 5:** `(n,d,N_J)=(2000,50,1600)` complete: 105/105 ADP fits, zero
failures. All input hashes match the `N_J=300` baseline. Family-averaged mean
EDR is ~.383 versus .497 at `N_J=300`; recovery is 3/105 versus 0/105. Median
fit time is ~9.5s (about 4.9x baseline), median RSS ~165 MiB. MEANMAVE remains
at mean EDR ~.275 and 61/105 recovery. Results are in `PROGRESS-n2000.md`.

**Checkpoint 6:** `(n,d,N_J)=(2000,50,2000)` complete: 105/105 ADP fits, zero
failures; exit code 0. All 315 saved dataset-arm files for this sample-size
block have identical `X`, `y`, and truth arrays across the three budgets. The
full-size arm has mean EDR ~.362 and recovery 6/105, compared with ~.383 and
3/105 at 1600. Its median fit time is 11.808s and median RSS 184.7 MiB.

**Final checkpoint:** All six campaigns are complete (840/840 fits, zero
failures). The audit checked effective `N_J`, manifests, completion records,
data hashes, and direct array equality. Across `(1000,25)`, ADP mean
EDR/recoveries for `N_J=300/800/1000` are `.391/0`, `.370/6`, `.351/9`; at
`(2000,50)`, for `N_J=300/1600/2000`, they are `.497/0`, `.383/3`, `.362/6`.
MEANMAVE references are `.264` with 64/105 recovery and `.275` with 61/105.
No ADP fit reported outer convergence or all-inner certification; inner
certified solves were 0/1612 and 0/2205, respectively. Full results and limits
are in `benchmarks/results/outputs/r_sdr/seven-families-nj-20261116/REPORT.md`
and the linked detailed reports.
