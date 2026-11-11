# Merge Vlad_EDR_M into EDR_code

Task `vlad-edr-m-merge-2026-10-26`; status **done**.

Goal: integrate the workspace copy of Vlad_EDR_M/Code, its experiments,
saved evidence and TeX into EDR_code. User ruling: EDR_code is more complete
and takes precedence. Work in the user's existing dirty checkout, preserving
all prior work. No commit, push, solver deletion or numerical default change.

Class: repository integration/documentation; numerical kernels stay authoritative
in EDR_code. New experimental modules remain opt-in. HPAOResult must keep
class identity with existing solver/engine contracts. Historical data, seeds,
source snapshots, hashes and failed/incomplete studies remain unchanged.

Read set: AGENTS.md, agent-notes/{STATE,README,WORKFLOW}.md, engineering/research
contracts, agent-notes/tex/README.md; source Code/{ADP,experiments,benchmarks,
tests,tex,verification,benchmark_outputs}; exact collision diffs.

- [x] A Inventory source/target, identify collisions and TeX duplicates; settle
  precedence with user. Read-only Luna scout completed.
- [x] B Copy non-conflicting code/results, archive conflicting source versions,
  record SHA-256/path mapping; provide only required compatibility additions.
- [x] C Catalog experiments and TeX with source/evidence/build links; preserve
  distinct document versions and historical provenance.
- [x] D Verify all imported bytes and prior target files, run focused and
  broader tests, CLI smoke/dry runs, compile standalone imported result TeX
  twice and inspect final logs. Update routing/state and mark done.

Acceptance: every relevant source file accounted for; zero unexplained changes
to pre-existing files; target solver/default compatibility preserved; imported
experiments import and pass their relevant tests; document links/builds checked.
No new research grids or historical benchmarks are run. Record pre-existing
failures separately; stop on data-loss risk or unresolved content collision.

Evidence directory: docs/integration/vlad_edr_m_20261026/.
Previous PLAN/STATE preserved in agent-notes/history/plans/
vlad-edr-m-merge-2026-10-26-previous/.

Final evidence: IMPORT_MANIFEST accounts for 23,326 source files (650,112,313
bytes), originals of all conflicts/adaptations retained, verifier reports no
source/preservation errors. 235 imported/integration tests pass; 41 modules
import; six CLI checks pass; four standalone TeX documents built 2+ passes
with clean final logs. Changed-code Ruff/format and git diff --check pass.
Independent review finding (single CLI shape) and two implicit M default
dependencies fixed with RED-to-GREEN fit regressions. Complete suite reports
1028 pass / 53 CUDA skips / 8 inherited failures / 3 inherited collection
errors; all failures/errors reproduced on the pre-merge SHA-verified baseline.
See evidence README for exact failing names and retained imported lint findings.
All task work complete; no commit/push or original-source deletion.
