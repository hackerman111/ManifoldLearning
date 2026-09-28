task_id: manifold-generalization-diagnosis-2026-09-28
status: done
change_class: experiment/diagnosis only; production estimator unchanged
---

# Diagnose manifold generalization failure

Goal: identify and verify the primary reason geometric recovery fails in the
new `manifold_generalization` tests, excluding solver recovery as an explanation.

Non-goals: change the solver/estimator, retune the frozen main protocol, or
declare an exploratory result to be untouched validation.

Known evidence: `main/` has 180/180 errors; `main_stop/` has 28/180 full
recovery and poor center geometry in many completed fits. Flat m=3 broad
support also fails frequently. The report notes high oracle query error on
curved cases, but that alone does not explain center errors.

Hypotheses: (H1) incorrect truth or metric; (H2) poor initial local gradient
geometry; (H3) ADP update/scale localization degrades a sound initial basis;
(H4) the target geometry is not identifiable from the specified scalar f or
cannot be represented accurately by sparse nearest charts.

Invariants: float64; frozen data/seeds and raw failure accounting; all center
and query metrics retained; compare actual geometry independent of solver
residual/stop status. No numerical-behavior change without a new plan.

Read set: `docs/experiments/manifold_generalization_2026-09-28/{protocol,report}.md`
and `main_stop/runs.jsonl`; `experiments/manifold_generalization.py`;
`ADP/engine/manifol_engine/{fit,weights,graphs,optimisation}.py` and exact
helpers they call; `agent-notes/ADP/manifold.md`.

Work units:
- [x] Audit completed fits: paired initial/partial/final center errors by
  profile and scale; separate geometric failures from numerical errors.
- [x] Run one small deterministic diagnostic that distinguishes initialization,
  statistic/update behavior, truth definition, and chart discretization.
- [x] State the root cause and scope with reproducible evidence; checkpoint
  `STATE.md` and, if warranted, a focused research note/decision.

Verification: independent truth/metric sanity check; cite run rows and exact
source lines; `git diff --check`. Stop when one causal account explains the
key flat and curved failures without treating solver diagnostics as recovery.

Outcome: `docs/experiments/manifold_generalization_2026-09-28/diagnosis.md`
records the frozen-row audit, five-seed exact-gradient initialization check,
and 10 paired flat `m=3` spectrum counterfactuals. Production source is unchanged.
