# Grassmann intrinsic-rank GN proposals

Task `grassman-intrinsic-gn-2026-10-18`; status **closed: rejected on validation**.
Outcome: unchanged100800 validation90/90 failed EDR.98546 versus.95, including10.069x individual regression. No production integration. INTRINSIC_REPORT.md preserves full evidence.
Goal: improve recovery with <=10% median fit cost using regular-center GN proposals; compare fixed5/10 budgets on pilot before freeze.
Non-goals: alter objective/statistics/ridge, discard centers from trials or certificates, assert full-rank derivatives at numerical rank boundaries, tune on consumed data.
Evidence: local scalar10 EDR.9093/time1.368 failed time; constant5 EDR.99973/time.8749 failed quality; Armijo10/smooth5 EDR.955496/time.95545 failed quality with one1.968x EDR outlier. All closed, heldout92800,94800,96800 ranges untouched. Production unchanged.
Quadratic development97800 completed90/90, EDR.96196 failed.95 despite time.8229; validation98800 untouched. Additional diagnostics: all43rank-deficient centers in14actual inputs have numerical rank(U)=rank(UY)<m. At exact intrinsic rank this profile is locally constant; model zero residual Jacobian there and original GN on regular centers. Finite-cutoff classification only supports an APPROXIMATE direction proposal. Require every other center passes original smooth guards; otherwise existing rank fallback. All true trials include every center, original full-gradient slope/certificate and rank-boundary semantics remain.
Class APPROXIMATE proposal, same objective. Invariants: full minimum-norm profile, gradient/directions, full Armijo/live certificates, workspace bound, no ridge/discarded centers/d*d. Positive localridge bypasses this new proposal. U-rank SVD only for deficient centers, chunk extra scratch4MiB. Independent exact constant-image reference and full direct profiles.
Read set: ADP/solver/grassman.py:24-195,320-565; experiments/grassman_accuracy_2026_10_18/{candidate,backtracking}.py and prior protocols/reports/MATH; focused tests and R-SDR integration.

- [x] S0: archive failed quadratic; leave98800 and prior heldout untouched.
- [x] S1: intrinsic GN references/audit4tests passed incl chunkbound. Pilot90800 GN10 improves allthree overGN5, runtime.50/.58/1.04x baseline and EDR.03293/.10204/.19848. Freeze10nonsmooth/5smooth before99800, no dimension-specific tuning; INTRINSIC_PROTOCOL.md.
- [x] S2: frozen99800..99809 development90/90,30pairs passes all gates: EDR geometric ratio.89504 CI[.84340,.94238], median/p90fit.66773/1.00834,RSS1.00485,orth3.70e-15.24wins6losses; worstEDRdelta+.00987 d20/99806. Complete hash audit passes; INTRINSIC_REPORT.md and raw analysis_summary.json.
- [ ] S3: launch untouched100800..100809 after S2 pass, identical algorithm/settings/gates; no tuning.
- [ ] S4: integrate validated opt-in option; frozen replay, focused checks and affected notes/STATE/DECISIONS; done only with evidence. Failed development closes candidate, leaves heldout untouched.

Stop on objective/rank/gradient mismatch beyond roundoff, changed estimator, forbidden allocation, paired/source drift or failed frozen gate. No tuning on selection failures. Tests/benchmarks authorized by user. Preserve previous source snapshots and dirty work.
