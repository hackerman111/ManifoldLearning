# Grassmann quadratic rank-one angle proposals

Task `grassman-quadratic-angle-2026-10-18`; status **active**.
Goal: improve recovery with <=10% median fit cost using up to three true-profile quadratic angle proposals, ten nonsmooth/five smooth steps.
Non-goals: alter estimator/statistics/ridge, drop centers, relax GN guards, tune on consumed development seeds.
Evidence: local scalar10 EDR.9093/time1.368 failed time; constant5 EDR.99973/time.8749 failed quality; Armijo10/smooth5 EDR.955496/time.95545 failed quality with one1.968x EDR outlier. All closed, heldout92800,94800,96800 ranges untouched. Production unchanged.
Rejected profile-only and constant-curve pilots: savings insufficient to fund ten steps; no97800 data consumed. Hypothesis: direct Armijo is fast but accepts the first adequate angle; a quadratic model using F(0), slope and F(t) proposes a better angle at modest cost. Evaluate at most three such proposals, retain the best actual Armijo-accepted value, then use original backtracking only if none accepted. No curvature/Hessian certificate claimed at rank boundaries.
Class APPROXIMATE optimizer, same objective. Invariants: original full profile, gradient, GN/rank/workspace guards, directions/geodesic, Armijo and live certificates; no Schur subtraction, discarded centers/ridge/d*d. Independent direct augmented lstsq and geometric trial references.
Read set: ADP/solver/grassman.py:24-195,320-565; experiments/grassman_accuracy_2026_10_18/{candidate,backtracking}.py and prior protocols/reports/MATH; focused tests and R-SDR integration.

- [x] S0: archive failed Armijo and profile/constant-curve pilots; leave all heldout untouched.
- [x] S1: quadratic3 references/audit8tests passed; original Armijo default array/diagnostics-exact to frozen snapshot. Pilot90800 done; fullsmooth10 not selected, freeze10nonsmooth/5smooth in QUADRATIC_PROTOCOL.md. Pending runner integration check before selection.
- [ ] S2: freeze explicit method and same gates before fresh97800..97809 development, d-5,d-20,d-50, baseline5 and actual R MEANMAVE,90fits. Complete30pairs; EDR geometric ratio<=.95; case median delta<=.005; median/p90 fit ratio<=1.10/1.25; RSS<=1.10; orthogonality<=1e-10. Report failures/stops/certificates.
- [ ] S3: untouched98800..98809 only after S2 pass, identical algorithm/settings/gates.
- [ ] S4: integrate validated opt-in option; frozen replay, focused checks and affected notes/STATE/DECISIONS; done only with evidence. Failed development closes candidate, leaves heldout untouched.

Stop on objective/rank/gradient mismatch beyond roundoff, changed estimator, forbidden allocation, paired/source drift or failed frozen gate. No tuning on selection failures. Tests/benchmarks authorized by user. Preserve previous source snapshots and dirty work.
