# Plan: promote verified Grassmann and isolate experiments

- Task: grassman-promotion-2026-10-03; status: done.
- Goal: promote verified optimized core into grassman.py; replace grassman_optim
  with grassman_exp containing failed-selection flat and full tangent GN.
- Non-goals: new optimization, estimator/tolerance/budget changes, new claims.
- Evidence: experiments/grassman_optim_2026_10_01/{REPORT,LEAD_AUDIT}.md:
  frozen ~2.52x speedup, fullfit median ratio0.942, close final projectors.
  Flat/full tangent selection failed; retain explicit experimental access.
- Classification: EXACT/NUMERICAL promotion; APPROXIMATE variants isolated.
  Preserve float64, minimum-norm profile, rank/conditioning/workspace guards,
  mass/ridge semantics, row-basis HPAOResult and capped convergence diagnostics.
- Read set: solver grassman{,_optim}.py; three Grassmann test modules;
  tests/test_glm_full_gn.py; relevant benchmark imports; full_gn prototype;
  pyproject.toml; solvers/multi-index notes and workflow/contracts.
- Units: (1) archive old reference, promote shared kernels/core, isolate flat
  and full GN; (2) update executable callers and regression tests;
  (3) focused tests, Ruff/format/type checks, paired relocation parity;
  (4) update affected routes, decision/state and mark done.
- Acceptance: original Grassmann tests + optimized independent/reference tests
  + flat/full GN tests pass; baseline and experimental outputs agree with
  pre-move frozen outputs; imports resolve without grassman_optim.
- Stop: any same-input quality/diagnostics regression requires correction;
  no benchmark retuning or heldout runs. Keep historical experiment rows intact.

- Evidence complete: 51 focused tests; 120 exact relocation pairs in
  experiments/grassman_promotion_2026_10_03/relocation_parity.json;
  Ruff/format pass; targeted Pyright0errors0warnings; benchmark import/routing
  checks pass. Historical series labels preserved; dependencies hashed.
  Report: experiments/grassman_promotion_2026_10_03/REPORT.md.
