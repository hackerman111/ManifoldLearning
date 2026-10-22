# Grassmann rank-one Armijo accuracy budget

Task `grassman-armijo-2026-10-18`; status **active**.
Goal: retain ten-step recovery improvement while meeting <=10% median full-fit cost by replacing repeated bounded scalar minimization with direct true-profile Armijo in the rank-one fallback.
Non-goals: change estimator/statistics/ridge or GN derivative guard, drop centers, tune on previously consumed development data, increase smooth GN budget.
Known evidence: original five-step rank-one fallback often spends13-30 full profile evaluations per step. Ten-step local scalar variant improved EDR9.1% but cost36.8%; constant-center five-step variant saved12.5% time but had no EDR gain. Both rejected; validation92800 and94800 ranges untouched. Pilot90800 of direct Armijo is faster at d5/d20, with improved d20/d50 EDR; no validation claim.
Hypothesis: direct true-profile Armijo eliminates scalar setup and dozens of evaluations, enabling ten rank-one steps; smooth profiles retain original five GN steps.
Class APPROXIMATE same objective. Invariants: original profile/gradient/SVD directions, rank/workspace guards, polar GN, exact rank-one geodesic, full QR/SVD every trial, original Armijo1e-4, maxangle.5, final live certificates; no Schur subtraction or approximate objective on rank-one trials; no dense d*d matrix. At nonsmooth rank boundaries only value-based acceptance asserted, no global convergence theorem.
Read set: ADP/solver/grassman.py; experiments/grassman_accuracy_2026_10_18/{backtracking,MATH,PROTOCOL,LOCAL10_REPORT,CONSTANT_PROTOCOL}.md/py and pilot_armijo*.json; R-SDR runner/worker/protocol/design and focused Grassmann tests.

- [x] R0: archive failed constant-center hypothesis; reuse existing baseline diagnostics and independent references.
- [x] R1/R2: independent Luna audit and direct augmented-lstsq/geodesic reference checks passed; 83 focused tests passed in6.37s. Direct Armijo10/smooth5 frozen in ARMIJO_PROTOCOL.md and explicit nondefault R-SDR method.
- [ ] R3: fresh95800..95809, existing d-5,d-20,d-50, baseline/candidate/R MEANMAVE,30paired datasets. Same gates: EDR geometric ratio<=.95, no case median delta>.005; median/p90 fit ratio<=1.10/1.25; RSS medianratio<=1.10; complete pairs, no failures, orthogonality<=1e-10.
- [ ] R4: untouched96800..96809 only after R3 pass with unchanged algorithm/settings/gates.
- [ ] R5: integrate validated opt-in solver, verify replay against frozen prototype, focused checks and affected notes/STATE/DECISIONS; mark done. Failure closes candidate and leaves validation untouched.

Stop conditions: objective/gradient mismatch, changed estimator, forbidden allocation, paired/source drift, failed frozen gate. No tuning on R3 failures. Tests/benchmarks requested by user. Preserve previous source snapshots and benchmark artifacts.
