# Current state

Task `seven-families-nj-20261116` is **done**. The six serial campaigns compared ADP-Grassmann-combined at standard `N_J=300`, `0.8n`, and `n` for seven link families at `(n,d)=(1000,25)` and `(2000,50)`. Protocol: `m=3`, Gaussian predictors, noise `.3`, seeds 0..14, with MEANMAVE (`MAVE-package`) as a shared per-size reference. This was an `ESTIMATOR` comparison; public code/defaults were unchanged.

All 840/840 planned fits completed with zero failed fits and exit code 0. The audit checked configs, manifests, completion records, hashes, and direct `X`, `y`, and truth array equality across budgets (315 dataset-arm files per size). The final summary is `benchmarks/results/outputs/r_sdr/seven-families-nj-20261116/REPORT.md`; full family and paired-seed tables are in `RESULTS-n1000.md` and `RESULTS-n2000.md` under the same output root.

At `(1000,25)`, ADP mean EDR/recovery for `N_J=300/800/1000` was `.391/0`, `.370/6`, `.351/9` (out of 105); MEANMAVE was `.264/64`. At `(2000,50)`, ADP mean EDR/recovery for `N_J=300/1600/2000` was `.497/0`, `.383/3`, `.362/6`; MEANMAVE was `.275/61`. Increasing `N_J` helped ADP on average, with link-dependent and non-monotone shifts, but MEANMAVE retained the lower aggregate EDR and higher recovery count. Median time, RSS, and per-family paired results are in the detailed reports.

No ADP fit reported outer convergence or all-inner certification. Inner certified solves were 0/1612 at `(1000,25)` and 0/2205 at `(2000,50)`, so EDR recovery is reported separately from solver certification. The observed `N_J` effects do not isolate center-sampling variance or establish a broad EDR recovery claim.

Plan of record: `PLAN.md` (done). The previous e-means plan was preserved at `docs/agent-notes/history/plans/e-means-all-centers-20261113-plan-at-20261116.md`.
