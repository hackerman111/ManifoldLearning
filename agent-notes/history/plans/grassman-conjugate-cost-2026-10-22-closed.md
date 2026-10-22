# Grassmann conjugate cost diagnosis

Task `grassman-conjugate-cost-2026-10-22`; status **closed at exploratory pilot**.
Goal: retain the promising conjugate10 accuracy with <=10% median extra fit cost
against original Grassmann5 and actual R MEANMAVE. No estimator/rank/ridge,
budget/direction tuning or promotion from consumed103800 data.
Evidence: frozen conjugate development90/90, EDRgeo.92707 (26/30 wins),
time1.13220/1.39751 failed. Report CONJUGATE_DEV_REPORT.md.104800 unused.
Hypothesis: reuse existing selective.profile per-center QR/SVD for conjugate10,
without changing direction/budget/acceptance. NUMERICAL QR/SVD roundoff only.
Measured profile fraction45.2%/29.6% d20/d50; gradient/projection each<1%,
so gradient contraction is rejected as a cost target. No estimator change.
Invariants: all-center objective, rank cutoff, full-gradient slope, live final
certificate, float64, thin bounded workspace/no d*d, deterministic API.
Read set: grassman._profile/_profile_svd, backtracking, selective.profile,
SELECTIVE_MATH.md, test_grassman_selective, conjugate tests, pilot_intrinsic.fit,
CONJUGATE_COST.md. Earlier selective+scalar10 failed; this changes CG profile only.

- [x] C0: low-overhead or cProfile diagnostic only on exploratory90800 d20/d50,
  original5/conjugate10; separate profiling overhead from fit timing. Luna
  summarizes hotspots/callcounts and theoretical removable cost. Stop if weak.
- [ ] C1: isolated per-center-profile flag/wrapper,33 focused tests/Ruff passed;
  independent audit clean. First90800 QR pilot is SLOWER than oldCG (1.164 median)
  because helper repeats safe QR work. Reuse the already computed QR factors,
  preserving guards/normal actions; reference/callcount tests then NEW saved
  unprofiled90800 pilot. No fresh data/default change. Fix intentional invalid-
  argument test typing (only Pyright diagnostic); no algorithm change for typing.
- [ ] C2: freeze one candidate before fresh105800 development90fits; same gates:
  EDRgeo<=.95, case median delta<=.005, median/p90 time<=1.10/1.25,
  RSS<=1.10,30completepairs/noerrors,orth<=1e-10,maxEDRratio<=2.
- [ ] C3: unchanged106800 heldout only after C2pass; integrate explicit validated
  experimental entry, focused replay/checks/notes. Failed gates close candidate.

Stop on correctness/pairing/workspace mismatch. Preserve old snapshots,
untouched validation and user dirty work. Tests/benchmarks authorized.

Outcome: original per-center QR pilot median time1.16391 vsCG; QR reuse
pilot1.1093..1.1283 vsCG across three cases, EDRgeo/base.94282,
time/base median/p901.05646/1.27888. No cost gain vsCG; no fresh data consumed.
42focused tests,Ruff/Pyright passed (then two stronger tangent tests added
and passed). Prototype/results retained. Next relative-GN hypothesis uses
still-unused105800/106800 data only after a separate pilot.

