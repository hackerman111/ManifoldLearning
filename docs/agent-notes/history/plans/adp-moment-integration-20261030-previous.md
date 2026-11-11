# Weak moments: derivation and paired verification

Task `weak-moment-20261030`; status **done**. Previous completed plan archived
at `agent-notes/history/plans/weak-moment-20261030-previous.md`.

Goal: develop and adversarially verify residual-metric, higher-moment and
conditional-mean ideas; report the strongest proved claim and exact gaps.
Non-goals: no universal recovery claim or production/default/API changes.
Class: opt-in ESTIMATOR; rectangular SVD vs Gram is NUMERICAL.

Evidence: historical WLS1/WLS2, shared-polynomial and failed Fourier studies.
Hypothesis: independent linear residualization reduces quadratic-moment variance;
rectangular perturbation avoids an unnecessarily squared gap. Degree-2 refinement
may improve recovery but lacks global convergence proof.
Invariants: X:(n,d), basis:(d,2), independent pilot/estimation samples, known
standard Gaussian theorem, float64, truth only for evaluation, retain failures.
Timing includes all initialization; no change to existing user edits.

Read set: experiments/local_polynomial.py:1-240, moment_quadratic_study.py:1-180,
shared_polynomial.py:1-305, shared_polynomial_study.py:20-100;
ADP/solver/grassman.py:405-455; ADP/engine/common/weights.py:88-160;
tex/multiindex.tex:952-1009; agent-notes/{WORKFLOW,contracts/research,contracts/numerics}.md.

Work units (artifacts: verification/weak-moment-20261030/):
- [x] R0 orientation and distinct-family comparison.
- [x] R1 scoped proofs and adversarial audit: THEORY.md, R1_AUDIT.md.
- [x] R2 isolated prototype; 7 exact Gaussian quadrature/reference/guard tests.
  ADP preflight on separate seed passed with N_lin=50,N_J=100,outer_steps=8.
  Verified actual kernel K(u)=max(1-u²,0), squared tensor-distance argument.
  Reference checks include WLS
  identities, invalid/rank guards, SVD/Gram reference.
- [x] R3 frozen 288/288 selection fits, no failures; residual/raw EDR ratio
  .51938, time ratio .77334, RSS ratio 1.0. Declared gate passed.
  Review found within-seed cross-case dependence; supplemental seed-cluster
  bootstrap also upper<1 (.57939). Original row bootstrap retained and labelled.
  Frozen 6 cases x 8 fresh seeds 3010100:3010108, n=2000,d=20,m=2.
  Arms: raw-full, raw-split, residual-split, residual+degree2, projected ADP,
  WLS1 ADP. N_lin=50,N_J=100,N_loc=20,N_phi=12,outer_steps=8.
  Measure EDR, weak coverage, wall time, process RSS, convergence.
  Causal residual gate on 2 quadratic Gaussian cases: zero failures;
  upper 95% paired-bootstrap geometric EDR ratio <1 vs raw-full;
  median time ratio <=3 and median RSS ratio <=1.5.
- [x] R4 288/288 validation fits; unchanged settings, no exceptions. Residual/raw
  EDR ratio .48820, time .77678, RSS .99890; declared gate passed.
  Seed-cluster bootstrap [.44962,.52472]. Refined quadratic recovery16/16,
  but no universal/high-order claim or promotion under the raw-moment time gate.
- [x] R5 independent final audit and AUDIT.json; moment fits rechecked, iterative
  fits not rerun. 34 focused tests passed; Ruff/format clean. Core Pyright clean;
  runner has one documented pre-existing-profiler-interface type diagnostic.
  REPORT.md/THEORY.md distinguish strict proofs, bootstrap evidence and all gaps.
  Durable notes updated. No production/default change or universal ADP theorem.

Stop: failed proof before study; failed R3 leaves validation untouched; no tuning
after selection. Stress-case failure forbids universal claims. Each phase bounded
by 15 minutes, 1 GiB/worker. Missing checker is reported. No full-suite claim.
