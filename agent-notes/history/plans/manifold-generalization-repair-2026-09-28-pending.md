task_id: manifold-generalization-repair-2026-09-28
status: pending
change_class: ESTIMATOR candidates and experiment metric/protocol correction
---

# Repair manifold generalization failure

Goal: correct the spectral localization failure and make the generalization
test's query criterion measure estimator error separately from unavoidable
nearest-chart discretization; test a defensible local-geometry repair for
curved cases. Keep numerical failures visible.

Non-goals: change solver certificates, rewrite frozen `main/` or `main_stop/`
results, relax rank checks, or assert universal recovery from one scalar Y.

Known evidence: see archived diagnosis plan and
`docs/experiments/manifold_generalization_2026-09-28/diagnosis.md`. Relative
eigenvalues fed back into the localization metric collapse flat m=3;
unit-spectrum counterfactual passes 10/10 paired flat fits. Broad curved graph
initialization remains biased even with exact gradients. Exact center bases
fail the current nearest-chart query gate on realized curved profiles.

Hypotheses: (H1) unit weighting on row directions prevents flat broad m=3
spectral collapse; (H2) local graph/pilot changes can reduce curved center
bias without rank or support regressions (rejected in preflight); (H3) direct
estimated-versus-exact center-chart distance separates estimator quality from
nearest-chart approximation. Curved m>=2 generator charts are not uniquely
identified by scalar Y without a structural restriction on the chart map.

Invariants: float64; original estimator and frozen outputs remain reproducible
via an explicit legacy option; no truth in fit; bounded working memory;
rank/residual checks; all center/query/failure rows retained; fixed seeds and
no post-hoc relaxation of the old recovery gate.

Read set: `agent-notes/ADP/manifold.md` and research/numerics contracts;
`docs/experiments/manifold_generalization_2026-09-28/{diagnosis,protocol}.md`;
`ADP/core/manifold/ADP_Manifold.py`;
`ADP/engine/manifol_engine/{fit,weights,graphs,optimisation}.py`;
`experiments/manifold_generalization.py`; `tests/test_manifold.py` and direct
callers of changed hooks.

Work units:
- [x] Derive localization invariant and test minimal spectral candidates on
  frozen paired seeds, including curved/local and flat controls. Stop any
  candidate that breaks them; keep a dense weight reference. Focused
  five-seed preflight: unit weights recover flat broad m=3 (5/5), while
  `max(lambda,alpha²)` remains 0/5; neither repairs curved center recovery.
- [x] Isolate a defensible curved-center estimator candidate. Compare against
  exact-gradient/oracle initialization before production changes; do not
  promote a candidate that fails its paired geometric/support gate. Tested
  k-nearest graph sizes 2–30 with exact/local-linear/local-quadratic gradients
  over five seeds: no curved high-candidate passed the center max gate, even
  with exact gradients. Existing local-quadratic m=1 also failed all curved
  profile gates. Stop promoting a graph/pilot tweak.
- [x] Implement an explicit unit-spectrum estimator option and corrected
  experimental query metric in a new protocol without altering frozen outputs.
  Dense-reference weights/graph, API/config checks, 16 manifold tests, Ruff,
  and Pyright pass. Full paired selection completed 360/360. Flat broad m=3
  passed 10/10 with unit and 0/10 with relative; no errors in either arm.
  Unit increased local-support numerical failures 31 to 35 across all
  profiles, so it remains opt-in and is not promoted as a general default.
- [ ] Run held-out flat validation, broader pytest/Ruff/Pyright, and checkpoint
  note/plan/state/decisions with measured limits.

Verification: exact small-array weight comparison; paired all-profile selection
on seeds 81000–81004 with each failed fit retained. Primary spectral gate:
unit mode must recover all ten flat broad m=3 fits (both noise levels) with
zero numerical failures; report any regressions in all other flat profiles.
Only if this gate passes, run untouched held-out flat seeds 82000–82004;
no curved held-out validation after the failed curved development gate and no
solver-status gate. Report
center RMS/max, query raw/oracle/estimation error, failures, wall time and RSS.
Stop a mechanism after repeated conceptual failure rather than silently
retuning the frozen test. If finite scalar-response data cannot support the
curved criterion, correct the test semantics and report the model limitation.
