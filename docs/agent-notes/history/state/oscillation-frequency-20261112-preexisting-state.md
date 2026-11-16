# Current state

Task seven-families-20261112 is **done**. PLAN.md records acceptance evidence.
Route: benchmarks/experiments/r_sdr/README.md, seven-family profile and report link.

Completed 7 families × 15 seeds (0..14) × 2 methods = 210/210 fits, zero fit errors,
timeouts or memory-limit failures. Protocol: n=2000,d=20,m=3, Gaussian X, normalized
signal plus noise std .3; current public ADP-Grassmann-combined20 versus package
MEANMAVE, one BLAS thread, 300 s / 4096 MiB limits, cold sequential workers.

Artifacts: benchmarks/results/outputs/r_sdr/seven-families-20261112/
- REPORT.md: Russian results, EDR mean/std, recovery, angles, time/RSS, paired CIs,
  certificate status, identifiability and limitations.
- manifest.json / sources.zip / source/: immutable protocol and 122 source files.
- inputs/*.npz, truth/*.csv, fits/*/{basis.csv,meta.txt,worker.log}: raw evidence.
- runs.jsonl, runs.csv, summary.csv, paired.csv, completion.json: complete journal.
- audit.py / audit.json: independent source/input/pair/metric readback; pass.
- make_report.py: re-audits before building REPORT.md and comparison.png/.svg.
- verification.log: 39 focused tests passed. time-quality.png / recovery.png also saved.

EDR means ADP / MEANMAVE:
additive .4129/.1564; quadratic .1597/.0705; multiplicative .1133/.0467;
oscillatory .9107/.9261; saturating .1783/.0436; localized .0976/.0448;
nonsmooth .1344/.0470. EDR<=.1 recoveries: 16/105 versus 75/105. Neither recovers
oscillatory or additive links at this threshold. Median fit .850 s versus 4.425 s;
per-family median paired time ratio 5.13..5.28. These are fixed-protocol outcomes,
not a general ranking or a result over all members of each family.

All 105 pairs have identical within-pair payload/CSV hashes; all inputs regenerated
exactly from the snapshot. Independent projector EDR error <=6.66e-16; max-angle
readback error <=1.15e-11 degrees. All ADP inner solves lack a convergence
certificate (0/525); outer convergence remains unknown. Package stop is not exposed.

Code changes: explicit seven-families catalog and link_signal in design.py;
CLI profile, experiments.py included in snapshots, relocated frozen R-worker
and R-reference paths. ADP estimator/defaults unchanged. All 50 old full-profile
datasets match pre-change bytes at seed91. ruff check passed; independent review
found no issues, 16 selected tests passed. Full project suite not run. No commit.

Next action: none required. Recheck via python <campaign>/audit.py or regenerate
report via python <campaign>/make_report.py. Do not retune on these evaluation
seeds or overwrite this frozen output. Prior reorganization caveat (74 unexplained
archived-file hash differences) remains; no old artifacts were modified.
