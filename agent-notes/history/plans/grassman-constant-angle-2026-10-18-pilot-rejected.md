# Grassmann exact constant scalar curves

Task `grassman-constant-angle-2026-10-18`; status **active**.
Goal: improve recovery with <=10% median full-fit cost by recognizing exactly constant scalar curve contributions and spending savings on accurately searched rank-one steps.
Non-goals: alter estimator/statistics/ridge, drop centers, relax GN guards, tune on consumed development seeds.
Evidence: local scalar10 EDR.9093/time1.368 failed time; constant5 EDR.99973/time.8749 failed quality; Armijo10/smooth5 EDR.955496/time.95545 failed quality with one1.968x EDR outlier. All closed, heldout92800,94800,96800 ranges untouched. Production unchanged.
Rejected profile-only pilot:1.23-1.31x profile speedup but ten-step full fits1.26-1.68x slower; no selection seeds consumed. Hypothesis: repeated scalar refits of structural rank-one row support are avoidable. At tau0, if all changing/constant design columns occupy one row and a retained nonzero constant column remains, the fitted column space is fixed, so the profile equals target energy outside that row at every angle. Fully zero curve designs are also constant. Conservative constant-column/full-scale guard prevents cancellation/rank-cutoff issues. Other centers retain candidate.py guarded scalar/SVD path.
Class EXACT/NUMERICAL scalar cache, APPROXIMATE increased budget. Invariants: unchanged full profile/coefficients/rank/smoothness/GN guards, same directions/geodesic/Armijo and full true-profile trial acceptance/live certificates; no discarded centers/ridge/d*d. Independent augmented lstsq and production profile references.
Read set: ADP/solver/grassman.py:24-195,320-565; experiments/grassman_accuracy_2026_10_18/{candidate,backtracking}.py and prior protocols/reports/MATH; focused tests and R-SDR integration.

- [x] S0: archive failed Armijo and profile-only pilot; leave all heldout untouched.
- [ ] S1: isolated constant-angle prototype, proof, independent Luna scalar references/monotonicity checks. Pilot90800: compare budget8/10 and optional smooth5 cap; choose before new selection, discard if savings insufficient.
- [ ] S2: freeze explicit method and same gates before fresh97800..97809 development, d-5,d-20,d-50, baseline5 and actual R MEANMAVE,90fits. Complete30pairs; EDR geometric ratio<=.95; case median delta<=.005; median/p90 fit ratio<=1.10/1.25; RSS<=1.10; orthogonality<=1e-10. Report failures/stops/certificates.
- [ ] S3: untouched98800..98809 only after S2 pass, identical algorithm/settings/gates.
- [ ] S4: integrate validated opt-in option; frozen replay, focused checks and affected notes/STATE/DECISIONS; done only with evidence. Failed development closes candidate, leaves heldout untouched.

Stop on objective/rank/gradient mismatch beyond roundoff, changed estimator, forbidden allocation, paired/source drift or failed frozen gate. No tuning on selection failures. Tests/benchmarks authorized by user. Preserve previous source snapshots and dirty work.
