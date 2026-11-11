# Grassmann safeguarded first direction

Task `grassman-intrinsic-warmup-2026-10-18`; status **closed: development rejected**.
Outcome:101800 complete90/90, EDR1.01325 and max28.7518x failed fixed gates;102800 untouched. No production promotion.
Goal: improve recovery with <=10% median fit cost. Non-goals: estimator/ridge/statistics changes, tuning on consumed validation, dimension-specific rules.
Evidence: intrinsic GN dev99800 passes EDR.89504/time.66773; untouched100800 validation FAILED EDR.98546 CI[.87116,1.18394], despite time.65760. One d5 error .05831->.58715. Close candidate; production unchanged. Prior four candidates also rejected, reports retained.
Hypothesis: a local GN model is unsafe as the first direction far from the warm-start solution; preserve one original full-profile bounded scalar rank-one search at the first nonsmooth step of each solve, then use regular-center GN. This is a new APPROXIMATE proposal, not a validation patch; use90800 only for pilot and fresh seeds for selection. No generic guarantee of correct basin is claimed.
Invariants: original full minimum-norm objective and gradient, true all-center Armijo/final certificate, rank guards, no ridge, bounded4MiB rank-SVD and GN scratch, no d*d. Numerical rank classifier is heuristic; positive ridge bypasses it.
Read set: grassman.py:24-195,320-565; backtracking.py intrinsic helper and rank-one branch; INTRINSIC_RANK/REPORT; focused tests and R-SDR integration.

- [x] S0: reject intrinsic after independent validation; archive old plan and record failure.
- [x] S1: isolated warmup,4 independent tests pass incl original first-step parity;21earlier tests and21R-SDR pass, scopedRuff/Pyrightclean.90800 pilot EDRgeo.9455,time median.715, dimensions EDRratios1.066/.941/.842; diagnostic-only, not paired acceptance. WARMUP_MATH/PROTOCOL frozen; no further tuning.
- [ ] S2: freeze new explicit method, budgets10nonsmooth/5smooth; fresh101800..101809, d5/d20/d50,n500,m2, original5 and actual R MEANMAVE90fits. Same gates:30pairs/no failures, EDRgeo<=.95, case median delta<=.005, time median/p90<=1.10/1.25,RSS<=1.10,orth<=1e-10. Additionally reject any >2x paired EDR regression. Report CIs/outliers/stops.
- [ ] S3: untouched102800..102809 only after development pass, unchanged settings/gates. No tuning on failures.
- [ ] S4: integrate validated explicit option, frozen reference/fullfit replay, focused checks; update notes/STATE/DECISIONS and mark done with evidence.

Stop on correctness/workspace/source-pairing mismatch or failed gate; close candidate, preserve artifacts. Tests/benchmarks authorized by user; preserve dirty work and snapshots.
