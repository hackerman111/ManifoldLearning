# Current state

Task oscillation-frequency-20261112 is **done**. PLAN.md records acceptance.
Route: benchmarks/experiments/r_sdr/README.md, seven oscillation frequencies.

Profile oscillation-seven: f_a=sin(a*z1)cos(1.25*a*z2)+sin(1.5*a*z3),
a=.5,1,2,3,4,6,8. n2000,d20,m3, Gaussian X, normalized signal plus noise.3.
15 seeds0..14, identical X/B/noise across frequencies and methods. Current
ADP-Grassmann-combined20 vs package MEANMAVE, one thread, cold sequential fits.
210/210 attempts completed without errors/timeouts/memory limits.

Artifacts: benchmarks/results/outputs/r_sdr/oscillation-frequency-20261112/
- REPORT.md: detailed Russian findings and limits; frequency-results.png/.svg,
  diagnostics.png/.svg, function-slices.png. Figures visually checked.
- manifest.json,sources.zip,source/: immutable protocol and123 source files.
- inputs/*.npz,truth/*.csv,fits/*/{basis.csv,meta.txt,worker.log}: raw evidence.
- audit.py/audit.json:105pair readback, cross-frequency sharing,123 source hashes,
  exact regenerated inputs, max EDR disagreement1.11e-15; max angle2.84e-10deg.
- high-frequency-reference.json: previous payload hashes/bases/provenance;
  a8 inputs and projectors match exactly (drift0), no older-directory dependency.
- analyze.py: audits first, then builds tables/report/plots from saved fits.
- population.csv,direction-alignment.csv,local-resolution.csv,
  paired-frequency-changes.csv,inner-certificates.csv,random-reference.json.
- verification.log:47 focused tests passed. ruff check/diff check passed.

EDR mean ADP / MEANMAVE:
a.5 .5537/.1994; a1 .3068/.0469; a2 .7523/.7517; a3 .9185/.9126;
a4 .9288/.9166; a6 .9197/.9259; a8 .9107/.9261.
Recovery EDR<=.1: MEANMAVE a1=15/15; all other cells and all ADP cells=0/15.
At a2 true-direction alignments for ADP/MAVE are z3=.9289/.9984, z1/z2 near.15.
At a>=3 all direction alignments are near random. Random reference EDR mean
.9214 (10000 draws), exact E[EDR²]=.85. No equivalence test was performed.

Mathematics: population gradient Gram remains positive definite, low a.5 has
weaker directions (condition23.7); at a>=2 condition4.5. Minimum eigenvalue grows
with frequency. Mean gradient decays exponentially; this is a separate diagnostic,
not the estimator objective. Gaussian formulas verified against quadrature.
Oracle true-Z20-neighbor coherence .9104 at a2, .8043 at a3, -.1191 at a8.
Thus density/coherence alone cannot establish the failure mechanism. ADP all525
inner solves hit max_steps; stationarity_applicable=False throughout, no inner
certificates and unknown outer convergence. Initialization/statistics/optimization
remain unseparated; no tuning or solver changes follow from these seeds.

Code: design.py has optional frequency/data_key fields and opt-in profile;
oscillation.py has exact population moments/signal_gradient; CLI and manifest
seed description updated.57 old datasets byte-identical at seed91. Previous
seven-family campaign untouched; its uncommitted changes retained and plan/state
archived under history/{plans,state}/oscillation-frequency-20261112-preexisting-*.md.
Independent source/math review passed. Final report review caught an aggregate
statement hiding per-seed exceptions at a2; corrected report gives A3>=.9 in
14/15 ADP and15/15 MAVE. ADP seed7 loses z3 (A3=.0791); MAVE seeds10/14 also
retain one mixed-plane direction (middle angles3.21/3.40deg), not the full plane.
All a2 fits still fail full EDR recovery. No unresolved review finding.

Next action: none required. Rebuild via python <campaign>/analyze.py. New settings
require new output and seeds for validation. Full project suite not run; no commit.
Prior archived-file integrity caveat (74 unexplained hash differences) persists;
no earlier artifact was changed.
