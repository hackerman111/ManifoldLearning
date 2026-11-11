task_id: manifold-curvature-recovery-2026-09-28
status: done
change_class: ESTIMATOR only for explicit candidates; experiment criterion correction
---

# Diagnose and improve manifold recovery on curvature

Goal: explain the current curved failure mathematically and empirically, then
improve reproducible recovery for m=1,2,3 under a scalar response. Use full
local-subspace recovery only where identifiable (curved m=1 and flat m=1/2/3).
For curved m=2/3, test observable gradient containment and report generator
chart distances as descriptive diagnostics, never as identified recovery.

Non-goals: claim arbitrary curved rank-m charts are identified from one scalar
Y, change production defaults, weaken rank/residual checks, erase frozen
artifacts, or hide failed fits and bad centers.

Known evidence: `diagnosis.md` isolates spectrum feedback on flat m=3, broad
graph tangent mixing on curvature, and a nearest-chart query discretization
ceiling. `repair.md` proves gauge nonidentifiability of curved m>=2 generator
charts. Previous paired selection completed 360/360: unit spectrum passed the
flat broad m=3 gate 10/10 but did not repair curvature and had more support
failures. The 22-cell grid recovered only 2/40 identifiable attempts, both
flat m=2, with 21/82 rank failures. Existing flat validation artifacts need
completion audit.

Hypotheses: (H1) measured failure decomposes into nonidentifiability, tangent
mixing, gradient pilot bias/variance, spectrum feedback, mass/support boundary,
and query discretization; (H2) direct local gradient geometry or an auditable
small estimator variant can recover curved m=1 more reliably; (H3) curved
m=2/3 estimated spaces can stably contain the observable gradient even though
their remaining m-1 directions have no unique scalar-data target.

Invariants: float64; no truth in fit or parameter choice; seed pairing and
untouched held-out validation; all-center RMS and maximum principal sine;
separate fit failures, numerical residuals, wall time and memory; bounded
working arrays; explicit ESTIMATOR variants; old artifacts/default unchanged.

Read set: `agent-notes/ADP/manifold.md`; numerics/research contracts;
`docs/experiments/manifold_generalization_2026-09-28/{protocol,diagnosis,repair}.md`;
`experiments/{manifold_generalization,manifold_generalization_repair}.py`;
`ADP/engine/manifol_engine/{fit,weights,graphs,optimisation}.py`;
`ADP/core/manifold/ADP_Manifold.py`; relevant `tests/test_manifold.py`.

Work units:
- [x] Audit existing flat held-out run and publish the paired spectrum result,
  including regressions and numerical failures. All 120/120 rows complete and
  11 source hashes match; broad/unit recovers 30/30 flat fits (relative 14/30),
  while local/unit recovers only 19/30 with five numerical failures. See
  `docs/experiments/manifold_generalization_2026-09-28/repair.md`.
- [x] Derive the observable target and run small references: exact gradients,
  local linear/quadratic pilots, initialized and final projectors, exact-chart
  oracle; decompose curved error across m=1/2/3 without changing production.
  Complete 240/240 development diagnostic: broad/unit 60/60 fits but 0/60
  observable-gradient passes; exact-gradient graph also 0/60; local/unit
  21/60 rank failures. See `docs/experiments/manifold_curvature_2026-09-28/report.md`.
- [x] Choose at most one justified opt-in curved estimator candidate after the
  reference gate. Gaussian-Stein top-2m plus quartic scalar surrogate passed
  its derivative finite-difference check but failed the first frozen seed's
  12/12 all-center gradient gates. Its global active-subspace max sine was
  .321–.583. Reject before production implementation; no truth used in fit.
- [x] Apply the validation stop condition: no curved candidate passed
  preflight, so no untouched curved validation or positive recovery claim.
  Full curved m>=2 chart is mathematically unidentifiable from scalar f;
  m=1 and observable-gradient finite-sample limitations are quantified in
  `docs/experiments/manifold_curvature_2026-09-28/report.md`. The successful
  flat broad/unit candidate already passed 30/30 untouched validation.
- [x] Update focused tests, report, affected routing note, `STATE.md`, and
  durable decisions. Final manifests match live source hashes; the formatted
  diagnostic reproduced 240/240 semantic rows exactly, and the preflight
  completed 12/12. Focused pytest 20/20, Ruff check/format, Pyright 0 errors,
  and `git diff --check` passed. The curved recovery target was not achieved;
  the failed gate is the stopping result, not a positive validation claim.

Verification: analytic gauge example, finite-difference gradient check,
small-array reference, paired development and untouched validation with
manifest/source hashes, focused pytest, Ruff/Pyright on touched code and
`git diff --check`. Stop after repeated conceptual failure; never redefine
curved m>=2 chart recovery as a success metric.
