# Current state

Task grassmann-theory-2026-10-23 DONE. No commit/push.

User deliverables (English):
- tex/ADP_Grassmann_current.tex and .pdf:15-page self-contained theoretical
  description of the COMPLETE current global multi-index ADP algorithm with
  production grassman.py, including data protocol, initialization, kernel
  geometry, sketches, moments, profile, GN/fallbacks, canonical spectrum,
  bandwidth/alpha, score chronology, stopping and complete pseudocode.
- SVD/GRASSMANN_IMPROVEMENT_HISTORY.md: source-linked history of SVD-derived
  attempts, main focus18–22October with earlier full-GN/flat/Schur/metric/
  rank-k context. Differentiates original intrinsic failure, accepted
  relative10, requested combined20, pilot LoRA and failed projected10 gate.

Verification: tex/ADP_Grassmann_current.verification.md (commands, limits,
15 hashes).35 existing solver tests pass. Independent formula checker
tex/documentation/grassmann_current/check_formulas.py passes4moment/kernel
batches and3 full fits/9 outer replays. Ruff/format and git diff --check clean.
PDF built repeatedly:15pages, no warnings/bad boxes/missing glyphs/undefined
references; representative pages visually inspected.27history links checked.
Historical report numbers were re-read, not rebenchmarked; no new R/CUDA/
large-recovery run or full-suite claim. All numerical sources and export
snapshot unchanged. Existing dirty research/export changes preserved.

Important verified semantics: new/auto multi sketch is isotropic; kernel
receives squared distance and computes max(1-t^2,0); config.local_ridge is
initializer-only, solver ridge independently defaults0; config.lambda_penalty
is GN damping. Initial localization spectrum is ones. Default best-step
SSE uses incoming-geometry center means and labels the updated basis; default
all-training protocol is same-sample, not independent holdout. Corrected
misleading pipeline note wording and current Grassmann source locator.
Affected routes: ADP/{index-pipeline,multi-index,solvers}.md, tex/README.md.

Prior export task complete; prior PLAN/STATE preserved verbatim under
agent-notes/history/plans/grassmann-theory-2026-10-23-previous/.
No required work remains for either documentation request.
