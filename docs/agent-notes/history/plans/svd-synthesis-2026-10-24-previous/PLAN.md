# Current ADP / Grassmann theory document

Task `grassmann-theory-2026-10-23`; status **done**.

Goal: self-contained English TeX description of the entire current global
multi-index ADP algorithm with production `ADP/solver/grassman.py::solve`:
data protocol, initialization, localization, sketch moments, profiled solver,
canonical spectrum, bandwidth/anisotropy update, selection, diagnostics.
Also create an English, source-linked account of recent solver improvement
attempts derived from SVD/, with intent, evidence and promotion decisions.
Non-goals: numerical code/default/export changes, new variants, benchmarks,
or unproved recovery/global-convergence claims.
Class: documentation only; no estimator or optimizer changes.
Known evidence: normalized/new path uses mass-weighted finite sketches;
Grassmann combined20 is explicit; facade default remains LSMR.
Invariants: distinguish initializer ridge from solver local_ridge, damping
from statistical penalty, frozen profile from changing outer objective,
rank-smooth identities from numerical rank-boundary proposals.
Read set: AGENTS/WORKFLOW; ADP index-pipeline/multi-index/solvers notes;
tex routing; ADP_Config/ADP_Solver/multi facade; common index_fit, calculus,
weights, statistic, initialize; grassman.py; focused Grassmann tests.
History read set: recent solver report/math files routed from solvers and
DECISIONS; Oct18–22 accuracy/combined/lowrank studies, earlier glm/flat/rank-k
reports only for origins. Read-only Luna scout handles broad source routing.

- [x] A Verify live formulas and full outer-to-inner parameter flow.
- [x] B Write English TeX, derivations, complete pseudocode, source map.
- [x] C Check key formulas independently and against live primitives; run
  focused existing solver checks; build PDF twice and inspect warnings.
- [x] D Create SVD/GRASSMANN_IMPROVEMENT_HISTORY.md from verified report
  sources, distinguishing frozen inner results, full fits and failed gates.
- [x] E Add document routing and verified evidence; mark PLAN/STATE done.

Acceptance: complete outer algorithm plus current core/relative GN and
fallback paths; defaults match live code; formula checks pass; PDF builds
without unresolved references, missing glyphs, overfull boxes; task diff
check passes. No CUDA/recovery claim from CPU checks.
Evidence: tex/ADP_Grassmann_current.verification.md records35 passing existing
solver tests,4 independent covariance/kernel batches,3 full fits/9 outer
replays, Ruff/format/diff checks,15 SHA-256 hashes and a clean15-page PDF
build.27 historical report/TeX links exist. Representative PDF pages inspected.
SVD/GRASSMANN_IMPROVEMENT_HISTORY.md covers actual attempts/provenance,
quality/time/certificate evidence and failed gates; large benchmarks not rerun.
Corrected pipeline note's misleading default holdout wording and refreshed
current Grassmann locators. All numerical sources unchanged.
Stop: unresolved code/formula contradiction must be documented; missing TeX
dependency must be reported if PDF build cannot proceed.
Previous completed export plan/state preserved under
agent-notes/history/plans/grassmann-theory-2026-10-23-previous/.
