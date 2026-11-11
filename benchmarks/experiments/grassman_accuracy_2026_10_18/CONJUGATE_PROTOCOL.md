# Frozen conjugate rank-guard experiment (2026-10-22)

Explicit candidate: `ADP-Grassmann-conjugate` calls `conjugate_guard.solve`.
Budget chosen using only exploratory seed90800; no fresh selection data read.

- Same finite-sketch objective, all-center profile, rank cutoff, ridge and
  certificates. At a nonsmooth rank guard use PR+ with tangent-projection
  transport and all available singular directions; reset on rank changes or
  failed descent margin1e-3. Actual full-gradient slope and full-profile Armijo.
  Smooth core GN is unchanged. APPROXIMATE optimizer, no convergence theorem.
- Frozen budget:10 steps for initially nonsmooth profiles,5 initially smooth;
  `tol=1e-6`, `local_ridge=0`, original default angle/energy/workspace limits.
- Paired workload: d-5/d-20/d-50, n=500,m=2. Methods: original
  `ADP-Grassmann`5, candidate10, and actual R `MAVE-package` (MEANMAVE).
  Ten shared seeds per case, matched input/method-seed hashes, one BLAS thread,
  300-second and4096-MiB per-process limits; fit timer and process RSS separate.
- Development103800..103809; untouched validation104800..104809 only after
  development passes. Freeze algorithm/settings/gates across both stages.
- Reject on missing/failed pairs; candidate/base EDR geometric mean>.95;
  any case median EDR delta>.005; paired time median/p90>1.10/1.25;
  median RSS ratio>1.10; nonfinite output/orthogonality>1e-10;
  or any individual EDR ratio>2. Bootstrap10,000 resamples, seed908123.
- Save source/environment/configuration fingerprints, raw outcomes and all
  failures. Report actual R medians, stops/certificates, intervals and outliers.
  Failed selection closes this candidate and leaves validation unused.

Pilot90800 budgets5/10/15: EDRgeo1.38123/.94045/.96922;
median time.55456/.92183/1.28427. Ten chosen before development.
References: CONJUGATE_GUARD_MATH.md, CG_CHECKS.md, pilot_conjugate_guard.json.
Production SHA256:6d5aa753da9d9c8532545fa7519e63ba479c0f4082a8a6d52358cd4c66af72a0.
The immutable runner manifests pin every candidate source hash for each run.
No production/default or universal recovery claim follows from this protocol.
