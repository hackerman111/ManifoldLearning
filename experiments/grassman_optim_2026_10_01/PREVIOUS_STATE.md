# Current state
Completed grassman-solver-2026-10-01; PLAN status done.
Deliverable ADP/solver/grassman.py, CPU multi-index HPAOResult row-basis API.
Default core_gn/max_steps5; optional rank_one Schur/refit, adaptive spectral.
Implements full profiled eq:dg in all q*m core coordinates, polar Armijo,
cached U[Y,V]. tau0 same unpenalized finite-sketch profile as HPAO; lambda
GN damping only. tau>0 explicit estimator. SVD fixed-g objective distinct.
No existing solver defaults changed. Rank/conditioning guard value-based
rank-one; workspace spectral fallback; no boundary convergence certificate.
Final residual/gradient uses live U. API/routes ADP/{multi-index,solvers,README}.

Luna-owned reproducible benchmark: benchmarks/grassman_benchmark.py,
experiments/grassman_2026_10_01/REPORT.md + benchmark/ raw protocols/results.
Heavy Spokoiny n800 d50 m2 J800 p10,162 outer steps, common split seeds0–2.
Core5 times31.72/33.48/34.03s vs pairedSVD36.32/40.74/37.52s; pairedmedian
ratio .873574 (12.6% faster), projector .05417/.04728/.04494 vs
.11831/.08987/.12586. Median RSS +13332KiB. core20 ratio1.260 slower,
quality slightly worse than core5. ALL inner core calls hit cap/unconverged;
finalcore5 normalizedgrad .0188/.0282/.0335. No heldout validation claim.

ImprovementI: frozen rank_one Schur/fullrefit pairedmedian5.72x speedup,
lossdifference<=4.25e-8. Truth-distance scalar agrees; estimated projector
agreement only tested on small reference, not stored frozen outputs.
61-angle curve sharedprojection-inclusive totals3.038/70.423ms (23.2x),
maxabsdifference3.41e-13. ImprovementII: coreGN .0949s vsadaptive spectral
.2157s, similar profileloss65.0007/64.9999. No observedadaptive-rank advantage.
ExactQR compression cannot reduce p10<d+1. Six frozenmethods*3seeds pass.
Four initial harnesserrors preserved (one methodkw, three missingSVD rank),
corrected series no numericalfailures. Lead checked rawaggregation/sourcehashes.

Verification:18 independentnewtests + finalsharedsuite110pass. Ruff3newfiles
passes; Pyrightsolver0errors/warnings. Fullpytestcollection4existingerrors;
broadexcluding4:388pass28GPUskip19existingfailures (missingthreadpoolctl/sklearn,
multiv2catalogKeyError, defaultmanifoldlocal_quadraticrejectsm2). Reproduced19;
no failure invokes newsolver. Logs/repro: VERIFICATION.md, focused_tests.txt,
broader_failures.txt. No unrelated repairs mixed into task.

Usertimingaudit: oldSVDseed0 25.734s vsnow33.857/34.859s, same162outer,
1977inner/direct,103614Upasses and projectorerror to1e-15. OldPython3.13.12
Clang vsnow3.14.7GCC; NumPy/SciPy/platformsame, oldBLASunknown. Execution
cause notlocalized; contemporarypairedtimings used, no increasedwork claim.
Prior theory PLAN/STATE preserved in PREVIOUS_*; earlier unrelated unfinished
optimizer not resumed. Follow-up: heldoutseeds/standaloneconvergence if needed.
