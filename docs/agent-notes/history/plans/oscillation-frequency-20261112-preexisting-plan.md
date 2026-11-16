# Seven function families: ADP Grassmann versus MEANMAVE

Task `seven-families-20261112`; status **done**. The user approved the seven
formulas and paired protocol in chat. Execute in the current authorized
checkout; no commit or changes to public estimators/defaults.

**Goal:** Add an explicit `seven-families` experiment profile and finish
7 families × 15 seeds (0..14) × 2 methods = 210 fits, then audit and report.
**Non-goals:** Hyperparameter/rank selection, solver modifications,
additional n/d/noise grids, or rewriting historical results.
**Known evidence:** Runner records paired hashes, resources, failures,
certificates, frozen sources and resumable journals. MAVE 1.3.12 is installed.
Baseline tests/test_r_sdr.py: 27 passed, 2 failed because R paths still point
to the pre-reorganization layout. Frozen snapshots omit experiments.py.
**Change class:** New explicit experimental design (ESTIMATOR/data-generating
variants only); path repairs are EXACT. Public estimators remain unchanged.
**Hypothesis:** Recovery varies with link structure; no ranking is assumed.
Strong oscillations may defeat both methods at this sample size.

**Invariants:** X~N(0,I), n=2000,d=20,m=3; signal centered/scaled to sample std 1
(ddof=0), Gaussian noise std .3; independent B/X/noise RNG streams; same X/y/B
within each pair; float64; oracle m. Methods ADP-Grassmann-combined (public
combined20) and MAVE-package (MAVE::MEANMAVE, screen=d). One BLAS thread,
sequential cold workers, 300 s / 4096 MiB per fit; normalized projector EDR,
recovery EDR<=.1. Keep failures in denominators; inner certificates do not
certify outer convergence.

**Read set:** AGENTS.md; docs/agent-notes/{STATE,README,WORKFLOW}.md;
contracts/research.md; benchmarks/experiments/r_sdr/{README.md,design.py,
__main__.py,protocol.py,runner.py,worker.py,worker.R,report.py};
tests/{test_r_sdr.py,r_sdr_reference.R}; experiments.py; pyproject.toml.

**Bounded work units and acceptance evidence:**
- [x] Repair experiment snapshot/worker paths. Add a failing frozen-worker
  regression test, then pass it and existing R reference/resume tests.
- [x] Add seven-family profile and links, preserving legacy generation. Test
  exact formulas, deterministic generation, rank-3 gradient span, normalized
  signal and CLI count 210. Update experiment documentation.
- [x] Freeze and run all 210 fits with saved inputs and all outcomes. No tuning
  on these seeds. Source changes after freezing require a new campaign.
- [x] Independently audit inputs, pairs, bases/metrics and completeness. Produce
  a Russian report with EDR mean/std, recovery, angle, time, RSS, paired
  differences, failures and certification; checkpoint done.

**Stop conditions:** Degenerate/nonfinite generator, missing dependency,
snapshot mismatch, unpaired inputs or occupied output stops launch. Fit errors
and timeouts remain recorded. Infrastructure interruptions resume unchanged.
No aggregate claim from an incomplete campaign.

**Previous completed task:** four-directory-reorganization-20261107. Layout and
integrity caveat are retained in docs/PATH_MIGRATION.md and the committed prior
PLAN.md/STATE.md; do not modify archived result trees.

**Final evidence:** benchmarks/results/outputs/r_sdr/seven-families-20261112/
contains completion.json (210/210, zero failures), audit.json (105/105 successful
pairs, 122 source hashes, exact input regeneration, max independent EDR error
6.66e-16), REPORT.md, CSVs, raw bases/inputs and three PNG/two comparison formats.
verification.log records 39 passing focused tests; ruff check passed. Independent
read-only review found no issues and passed 16 selected tests. Full project suite
was not run because public solver code is unchanged. MEANMAVE has lower mean EDR
on six links; neither method recovers oscillations at threshold .1. Inner
certificates: ADP 0/525 solves. No tuning or solver/default changes.
