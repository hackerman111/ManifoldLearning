# Grassmann localized QR/SVD accuracy budget

Task `grassman-selective-profile-2026-10-18`; status **active**.
Goal: improve recovery with <=10% median full-fit cost by localizing QR/SVD fallbacks and spending savings on ten accurately searched rank-one steps.
Non-goals: alter estimator/statistics/ridge, drop centers, relax GN guards, tune on consumed development seeds.
Evidence: local scalar10 EDR.9093/time1.368 failed time; constant5 EDR.99973/time.8749 failed quality; Armijo10/smooth5 EDR.955496/time.95545 failed quality with one1.968x EDR outlier. All closed, heldout92800,94800,96800 ranges untouched. Production unchanged.
Hypothesis: _profile routes all centers to SVD when one center fails QR guard. QR for independently safe centers plus identical SVD for unsafe centers preserves coefficients/values/ranks/global nonsmooth guard. If unsafe centers are smooth, retain original whole-batch SVD to preserve normal_solve.
Class NUMERICAL profile, APPROXIMATE increased optimizer budget. Invariants: minimum-norm/ridge objective, cutoff/rank/smoothness/GN guards; nonsmooth normal_solve stays zero; same gradient/directions/geodesic/Armijo/live certificates; no d*d allocation. Independent augmented lstsq and production profile are references.
Read set: ADP/solver/grassman.py:24-195,320-565; experiments/grassman_accuracy_2026_10_18/{candidate,backtracking}.py and prior protocols/reports/MATH; focused tests and R-SDR integration.

- [x] S0: archive failed Armijo; preserve parser-failed and repaired outputs separately.
- [ ] S1: isolated selective profile plus accurate scalar10/smooth5; mathematical note, independent Luna references and focused checks; pilot90800 only, discard if savings insufficient.
- [ ] S2: freeze explicit method and same gates before fresh97800..97809 development, d-5,d-20,d-50, baseline5 and actual R MEANMAVE,90fits. Complete30pairs; EDR geometric ratio<=.95; case median delta<=.005; median/p90 fit ratio<=1.10/1.25; RSS<=1.10; orthogonality<=1e-10. Report failures/stops/certificates.
- [ ] S3: untouched98800..98809 only after S2 pass, identical algorithm/settings/gates.
- [ ] S4: integrate validated opt-in option; frozen replay, focused checks and affected notes/STATE/DECISIONS; done only with evidence. Failed development closes candidate, leaves heldout untouched.

Stop on objective/rank/gradient mismatch beyond roundoff, changed estimator, forbidden allocation, paired/source drift or failed frozen gate. No tuning on selection failures. Tests/benchmarks authorized by user. Preserve previous source snapshots and dirty work.
