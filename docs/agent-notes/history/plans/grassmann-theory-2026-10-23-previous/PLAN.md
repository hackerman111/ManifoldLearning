# Standalone multi-index / Grassmann export

Task `export-2026-10-23`; status **done**.

Goal: self-contained `export/` source distribution of live ADP multi-index,
current grassman.py, existing CLI/catalogs, paired comparisons, and tests.
Non-goals: numerical changes, solver promotion, historical results/logs,
new scientific claims, changes to root package APIs/defaults.
Known evidence: multi facade supports custom Grassmann; synthetic CLI only
supports CG/LSMR/HYBRID. Existing benchmark pairs SVD/current core_gn;
experiments.r_sdr provides actual R comparisons and source snapshots.
Class: packaging EXACT; export-only CLI adapter selects existing Grassmann.
Invariants: copied numerical sources byte-identical; all runtime files local;
current Grassmann defaults preserved; R resources shipped; no root imports
when directory is detached; errors/convergence/recovery stay distinct.
Read set: AGENTS/STATE/WORKFLOW/engineering; ADP cli-and-compat/multi-index/
solvers notes; pyproject; ADP cli.main and multi facade; experiments catalog/
runner/r_sdr dependency closure; benchmarks.grassman_benchmark; selected tests.

- [x] A Assemble source closure and install metadata; add failing export
  integrity/isolation checks first. Archive previous plan/state unchanged.
- [x] B Add export-only Grassmann CLI selection and focused CLI coverage;
  document install, single runs, paired runs and optional R/GPU dependencies.
- [x] C Verify detached imports, selected tests, actual single and paired CLI
  runs, wheel resources and source hashes; record evidence in export/VERIFY.md.
- [x] D Update routing, DECISIONS, PLAN and STATE; preserve prior work.

Acceptance: copy export to /tmp; no parent repository PYTHONPATH; import origins
inside detached tree (and built wheel); successful small multi Grassmann CLI;
existing SVD/Grassmann paired benchmark produces two successful rows;
selected CLI/experiment/Grassmann tests pass; worker.R in wheel; manifest hashes
verify; git diff --check clean. R/CUDA execution only claimed if actually run.
Scope correction: optional multiv2 is excluded because live original fails at
import on new Spokoiny grids; its coupled test_experiment.py is not shipped.
Main CLI/catalogs and required comparison paths are unaffected.

Stop: missing required dependency/resource, unexpected numerical divergence,
or failing isolation blocks completion until fixed or explicitly reported.
Previous plan/state: agent-notes/history/plans/export-2026-10-23-previous/.

Evidence: export/VERIFY.md and verification/ store detached139pass/2xfail,
2/2 SVD/core full fits,20/20 actual R MEANMAVE/core fits. Clean wheel+sdist
built; wheel locally installed in isolated venv and smoke-tested from /tmp.
Root export integrity/isolation regression passes; Ruff and diff checks pass.
Root full pytest blocked at3 pre-existing multiv2 collection errors; two
existing manifold CLI smoke failures reproduced and strict-xfailed in export.
No CUDA validation or numerical-source modifications.
