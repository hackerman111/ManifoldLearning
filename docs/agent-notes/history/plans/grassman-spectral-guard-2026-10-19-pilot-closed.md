# Grassmann spectral rank-guard direction

Task `grassman-spectral-guard-2026-10-19`; status **closed at exploratory pilot**.
Outcome:90800 budgets5/10/15 EDR1.007/.9775/.9415,time.545/.846/1.244. No clear joint winner;103800/104800 untouched. Five reference tests passed; prototype/artifacts retained.
Goal: improve EDR with <=10% median fit cost against original Grassmann5 and actual R MEANMAVE. Non-goals: estimator/ridge/statistics changes, rankcutoff weakening, tuning on consumed data.
Evidence: scalar10 too slow; constant5/Armijo/quadratic insufficient EDR; intrinsicGN failed heldout with10xoutlier; one-stepwarmup faileddev with28.75xoutlier. Original5+polish pilot7/8/10 has no clear joint winner (budget7 EDR.971/time1.069,10 EDR.947/time1.123 withd50time1.50). No103800 or104800 data consumed. Production unchanged.
Hypothesis: replace ONLY nonsmooth rank-one fallback with existing joint spectral negative-gradient geodesic, retain all nonzero singular directions (energy_tol0), smooth coreGN remains. No full-rank derivative model/intrinsic classifier. Standalone spectral already exists with mixed old evidence; this conditional/budgeted hybrid needs fresh evaluation.
Class APPROXIMATE optimization. Invariants: same minimum-norm all-center profile/gradient, full Armijo, original ridge/smoothness/finalcertificate semantics, boundedworkspace/no d*d. Spectral direction derivative at zero is -G_q/sigma1, slope=-||G_q||F^2/sigma1 on smooth strata; numerical rank boundaries still lack derivative certificates. Full trial values decide acceptance.
Read set: grassman.py profile/gradient and spectral solve branch; grassman_exp.py:153-179,306-340,500-515; rank_k/REPORT.md:108-175; backtracking.py; focused tests/R-SDR.

- [x] S0: close polish at exploratory pilot without consuming selection; preserve artifacts.
- [ ] S1: isolated spectral_guard flag/wrapper; independent geodesic/profile/slope/m1/smooth/workspace tests;90800 pilot5/10/15 then freeze one budget.
- [ ] S2: fresh103800..103809 d5/d20/d50,n500,m2,90fits. Gates:30pairs/no failures, EDRgeo<=.95, case median delta<=.005, time median/p90<=1.10/1.25,RSS<=1.10,orth<=1e-10, individualEDRratio<=2. Report pairedCIs/source/inputhashes/Rmedians/outliers/stops.
- [ ] S3: untouched104800..104809 only if S2passes, identical algorithm/settings/gates. No tuning on failures.
- [ ] S4: integrate validated explicit option, frozen/reference/fullfit replay, focusedchecks andnotes/STATE/DECISIONS; markdone only with evidence.

Stop on correctness/workspace/source-pairing mismatch or failedgate; preserve failedartifacts and untouchedvalidation. Tests/benchmarks authorized; preserve userdirty work and snapshots.
