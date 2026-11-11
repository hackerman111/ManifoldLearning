# Grassmann accuracy research, 2026-10-18

Status: user's2026-10-22 integration/budget request is implemented as combined20
public defaults: relative GN, quadratic3, exact zero-U exclusion, no smooth cap.
COMBINED_{MATH,PROTOCOL,REPORT}.md retain derivation/ablation/evidence.
Fresh development107800/validation108800 complete150+150; validation
EDRgeo.925311/time.860129 versus original5. Added components/budget do not
establish an accuracy gain versus accepted relative10 (EDR1.003418/time1.275697).
Public replay120/120 exactly matches saved bases/diagnostics/stops.

Current slice (2026-10-22): relative intrinsic-GN damping, PLAN.md and
RELATIVE_GN_MATH.md. Development EDRgeo.898376/time.650580 and independent
validation.923729/.653572 pass all8 frozen gates. Public port120/120 full fits
exactly reproduce saved bases/diagnostics; final direct public/R timing check
also passes90/90, EDR.923729,time.649100/.944153. Final RELATIVE_GN_REPORT.md;
RELATIVE_GN_{DEV,HELDOUT}_REPORT.md; public option is documented
in grassman.solve. Earlier rejected candidates are historical evidence below.
Conjugate10 development improved EDR (.92707) but failed
time (1.13220/1.39751), CONJUGATE_DEV_REPORT.md. Subsequent per-center QR reuse
remained slower than CG on exploration and is closed, CONJUGATE_QR_REUSE_PILOT.md.
105800/106800 are consumed. These earlier outcomes predate combined20 defaults.

## Diagnosis and invariants

Starting point: `SVD/glm.tex:621-638` and the live Grassmann/core GN source.
Fourteen actual outer solver calls were captured on exploratory seed90800,
cases d5/d20/d50. `diagnostics.json` and `frozen/` retain inputs and guards.
QR/SVD coefficient/objective/gradient differences were about1e-14, and
cached/live and orthogonality differences about1e-15. Rank-boundary flags
corresponded to actual local rank loss, not merely an overcautious threshold.
Weakening the global full-rank GN guard would apply an invalid derivative.

There is a separate scalar rank-cutoff defect: M=[e0,1e-15*e1], a=e1,I=e1
can yield scalar loss0 while direct lstsq drops the tiny direction and gives1.
The guarded per-center scalar prototype fixes this; full trial acceptance
already protects the original returned objective from unchecked scalar values.
References and conservative scale bound are in MATH.md and independent tests.

All candidate fits preserve the original tau0 finite-sketch objective,
statistics, local rank guards, gradient, full true-profile acceptance and live
certificates. More steps/different searches are APPROXIMATE optimizers. No
global optimum or statistical recovery theorem follows from lower objective.

## Frozen development comparisons

Each row below uses its own fresh30paired datasets (d5/d20/d50,n500,m2,
ten seeds each), original Grassmann5 and real R MAVE::MEANMAVE. One thread;
fit_seconds measures model.fit, RSS measures isolated process high-water.
Gates fixed before data: EDR geometric candidate/base ratio<=.95, per-case
median EDR delta<=.005; paired median/p90 time<=1.10/1.25; RSS<=1.10,
no failures, orthogonality<=1e-10. The10% time limit is the working assumption
for the user's "minimal time loss", pending an optional preference answer.

| Candidate | EDR geometric ratio | Median/p90 time ratio | Decision |
|---|---:|---:|---|
| Local scalar10, seeds91800..91809 | .9093 | 1.368/1.729 | time fails |
| Constant centers5, seeds93800..93809 | .99973 | .8749/1.1504 | quality fails |
| Direct Armijo10/smooth5, seeds95800..95809 | .95550 | .95545/1.12086 | quality fails |
| Quadratic3/10smooth5, seeds97800..97809 | .96196 | .82290/1.09843 | quality fails |
| Intrinsic GN10/smooth5, seeds99800..99809 | .89504 | .66773/1.00834 | development passes |
| Same intrinsic GN, validation100800..100809 | .98546 | .65760/.91733 | independent quality fails |
| One scalar warmup then GN, seeds101800..101809 | 1.01325 | .7551/.9099 | quality and28.75x outlier fail |

All four clean runs completed90/90 with matching paired hashes. Their
heldout seeds92800/94800/96800/98800 ranges were left untouched. Raw outputs:
`benchmarks/results/outputs/r_sdr/grassman-{accuracy,constant,armijo}-dev-2026-10-18*`;
see LOCAL10_REPORT.md, CONSTANT_REPORT.md, ARMIJO_REPORT.md for exact routes,
uncertainty, R medians, source/environment hashes, stops and limitations.
Armijo has21 wins/9 losses including a1.968x EDR regression at d20/95809.
Its initial parser-failed run is preserved separately; the clean repaired run
changes only metadata integration, not solver settings or data selection.

Two90800-only pilots were also rejected before consuming selection data:
per-center QR/SVD improves isolated profile time1.23-1.31x but fullfit10 is
1.26-1.68x slower; exact constant scalar curves cache only.11-1.5% centers and
remain too slow. Their proofs/reference tests and raw pilot JSONs are retained.

## Earlier direction hypotheses

`quadratic.py` uses up to3 true-profile quadratic angle proposals, chooses
the lowest actual Armijoaccepted value, and falls back to ordinary halving
if all proposals fail. It keeps the original10nonsmooth/5smooth budget of
the direct Armijo candidate. Derivation/limits: QUADRATIC_MATH.md.
Independent audit and90800 pilot preceded fresh97800 development. The EDR
gate failed;98800 validation remains untouched. See QUADRATIC_REPORT.md.

New intrinsic.py builds GN proposals from regular centers when deficient
centers have numerical rank(U)=rank(UY)<m. At exact intrinsic rank the latter
profiles are locally constant. At numerical cutoffs this remains an
APPROXIMATE model, not a full-rank derivative certificate. Every center stays
in the full gradient, true-profile trials and final certification. Source
diagnosis/limits: INTRINSIC_RANK.md. A90800-only budget pilot chooses10steps
for initial nonsmooth profiles/5smooth; no dimension-specific tuning. The
INTRINSIC_PROTOCOL was frozen before99800 development, which passed all
gates. Independent100800 validation FAILED EDR.98546, including one10.069x
regression despite finite/orthogonal output. No production promotion;
INTRINSIC_REPORT.md gives both complete90-fit runs and the saved diagnosis.

The next one-step warmup hypothesis also failed on complete101800 development
despite runtime savings;102800 validation was left untouched. WARMUP_REPORT.md
retains the28.75x outlier and all successful hashes/certificates.

The polish.py prototype preserves the ENTIRE original five-step solve before a
bounded intrinsicGN refinement. Skip certified/ridge/unavailable models;
preserve both phase histories and the reprofile boundary gap. Choose2/3/5
extra steps only on90800 pilot. Budgets7/8/10 yield EDR.971/.969/.947,
median time1.069/1.118/1.123; budget10 d50time1.50. No clear joint winner,
closed at exploratory pilot without consuming103800/104800; POLISH_MATH.md.

The spectral_guard.py pilot used the existing joint spectral gradient geodesic
only at a nonsmooth rank guard, keeping every available gradient singular
pair, smooth coreGN unchanged. No intrinsic classifier/full-rank derivative
there. No joint quality/time pilot winner; closed before selection.
SPECTRAL_GUARD_MATH.md. Later CG10 selection failed time and QR reuse pilot
did not improve full-fit cost; CONJUGATE_DEV_REPORT.md / CONJUGATE_QR_REUSE_PILOT.md.
