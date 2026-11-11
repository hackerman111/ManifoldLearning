# Scale-normalized intrinsic GN proposal

Task `grassman-relative-gn-2026-10-22`; status **done**.
Goal: improve Grassmann EDR with <=10% median extra fit cost vs original5 and
actual R MEANMAVE. No estimator/ridge/rank/certificate changes or tuning on
consumed data. Prior CG10 EDR.92707/time1.1322/1.3975 failed time; QR reuse
pilot stayed slower than CG (1.109..1.128), so cost slice closed at exploration.

Hypothesis: absolute lambda_prox=.05 is not invariant to response/mass units.
For eligible intrinsic regular-center GN proposals only, set damping to
lambda_prox*||J||F^2/(q*m). Under I->cI,mass->a mass, J,r scale c*sqrt(a),
damping scales a*c^2 and the augmented-LS minimizer/capped K is invariant.
Stopping-floor/rank/finite-precision caveats remain. APPROXIMATE optimizer,
same all-center true objective/slope/Armijo and live final certificate.
No convergence/recovery claim; independently audited algebra in prior Luna pass.
If scale is zero/nonfinite, skip this GN model and use existing spectral fallback.
Initial smooth GN and prior variants retain absolute damping unchanged.
Bounds: existing small augmented solve, O(rows*q*m), no new d*d/storage growth.
Read set: backtracking GN, grassman._core_jacobian, INTRINSIC_RANK,
intrinsic tests, frozen90800 calls, R-SDR dispatch/metadata.

- [x] R0: record damping units and proposed first steps on frozen90800 calls;
  derive invariant/dense augmented reference including degenerate-scale fallback.
  relative_gn_scales.json:12eligible/14calls, scale344..2797, damping17.2..139.9.
- [x] R1: explicit normalized_damping flag/wrapper, independent response/mass
  scaling, direct augmented solve, all-center/rank/smooth/default checks;
  Ruff/Pyright. Exploratory90800 budgets5/10/15, then freeze one or close.
  Implementation/reference/scaling/fallback checks:64 focused tests pass;
  Ruff/Pyright clean. Pilot5/10/15 EDRgeo1.06879/.95377/1.04254,
  time median.48757/.60728/.70264. Freeze10 before105800; pilot is exploratory,
  best quality with cost headroom, not proof of the .95 frozen gate.
- [x] R2: only if pilot justifies, fresh105800..105809 development90fits,
  d5/d20/d50,n500,m2, original5/candidate/actualRMEANMAVE,1thread.
  Fixed gates:30pairs/noerrors,EDRgeo<=.95,case median delta<=.005,
  time median/p90<=1.10/1.25,RSS<=1.10,orth<=1e-10,maxEDRratio<=2.
  Initial launch invalid: duplicate lambda_prox,63rows (21candidate errors).
  Raw run retained. Repair only runner argument forwarding; prototype hashes
  identical, inherited lambda still.05. Real-worker regression passes.
  Repaired output grassman-relative-dev-repaired-2026-10-22, same frozen data.
  90/90 successful,30matched pairs; all8 gates PASS. EDRgeo.898376
  CI[.844245,.948903],time.650580/.790378,RSS1.00524,worst1.07731x.
  RELATIVE_GN_DEV_REPORT.md. Start unchanged106800 validation.
- [x] R3: untouched106800..106809 after R2 passed;90/90/all8 gates PASS.
  EDRgeo.923729 CI[.886622,.956495],time.653572/.993297,RSS1.00211,
  worst1.03642x. Sources identical to development; RELATIVE_GN_HELDOUT_REPORT.md.
- [x] R4: port as explicit grassman.solve(relative_gn=True,max_steps=10),
  defaultFalse. Reuse existing profile/Jacobian/polar/Armijo; add only intrinsic
  model, relative damping and the prototype's direct rank-one fallback.
  TDD exact baseline/prototype comparisons on frozen/random/degenerate cases;
  full-fit replay all60 dev/heldout datasets for public option AND original
  baseline against saved bases/diagnostics. Update R-SDR explicit dispatch/config,
  docs, Ruff/Pyright/focused tests. No default promotion or universal claim.
  Port implemented;209 checks pass, Ruff clean. Pyright script Optional guard
  fixed/checked. Public-vs-frozen replay120/120 EXACT bases/all diagnostics,
  including60 default fits. Final direct public/R timing confirmation on the
  same consumed106800 data complete90/90/all8 gates PASS: EDR.923729,
  time.649100/.944153,RSS1.00170,worst1.03642x. Final Ruff/format/Pyright clean.
  RELATIVE_GN_REPORT.md, public-confirm raw source snapshots/analysis retained.

Stop on correctness/pairing/workspace failure or frozen failed gate; preserve
raw artifacts/user edits/unused validation. Tests/benchmarks authorized.
