# Grassmann baseline-preserving intrinsic polish

Task `grassman-intrinsic-polish-2026-10-18`; status **closed at exploratory pilot**.
Outcome: no clear quality/time budget among7/8/10 on90800; EDR.971/.969/.947,median time1.069/1.118/1.123,d50budget10time1.50. No paired-selection gate claim;103800/104800 untouched.
Goal: improve recovery with <=10% median fit cost. Non-goals: estimator/ridge/statistics changes, tuning on consumed validation, dimension-specific rules.
Evidence: intrinsic GN independent100800 FAILED EDR.98546 with10xoutlier. Warmup101800 development90/90 FAILED EDR1.01325 and max28.75x, despite time.7551. One scalar first step does not prevent poor recovery. Both closed;102800 untouched. Production unchanged.
Hypothesis: preserve ALL original five warm-start steps before any intrinsicGN refinement; extra2/3/5 GN steps may improve high-dimensional recovery cheaply while retaining the complete baseline initial trajectory. APPROXIMATE optimizer, not a recovery/basin guarantee. Choose budget only on90800, then fresh103800/104800. No tuning on consumed validation or dimension-specific rules.
Invariants: original full minimum-norm objective and gradient, true all-center Armijo/final certificate, rank guards, no ridge, bounded4MiB rank-SVD and GN scratch, no d*d. Numerical rank classifier is heuristic; positive ridge bypasses it.
Read set: grassman.py:24-195,320-565; backtracking.py intrinsic helper; INTRINSIC_RANK/REPORT,WARMUP_REPORT; focused tests and R-SDR integration.

- [x] S0: reject warmup on complete90-fit development; archive, leave102800 untouched.
- [ ] S1: isolated original5 + intrinsicGN polish composition, independent first-phase exactness/fullfit references, history/certificate guards;90800 budget pilot2/3/5 before freeze.
- [ ] S2: freeze explicit method/budget; fresh103800..103809, d5/d20/d50,n500,m2, original5 and actual R MEANMAVE90fits. Same gates:30pairs/no failures, EDRgeo<=.95, case median delta<=.005, time median/p90<=1.10/1.25,RSS<=1.10,orth<=1e-10; individual EDRratio<=2. Report CIs/outliers/stops.
- [ ] S3: untouched104800..104809 only after development pass, unchanged settings/gates. No tuning on failures.
- [ ] S4: integrate validated explicit option, frozen reference/fullfit replay, focused checks; update notes/STATE/DECISIONS and mark done with evidence.

Stop on correctness/workspace/source-pairing mismatch or failed gate; close candidate, preserve artifacts. Tests/benchmarks authorized by user; preserve dirty work and snapshots.
