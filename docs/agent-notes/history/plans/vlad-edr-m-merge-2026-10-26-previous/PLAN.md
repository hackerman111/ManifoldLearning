# SVD research synthesis

Task `svd-synthesis-2026-10-24`; status **done**.

Goal: a separate Russian scientific TeX document in SVD/ organizing the
AI-generated ideas and the results of their mathematical and empirical checks.
Class: documentation only. No numerical implementation/default changes, new
research candidates or new recovery/performance experiments.

Known evidence: SVD/ contains overlapping theoretical notes, archival proposals,
formula checks and a recent Grassmann study history. Original experiment reports
and saved rows, rather than the history alone, support numerical conclusions.
Invariants: distinguish fixed coefficients/profiled coefficients, basis rank/
correction rank/tangent rank, local loss/recovery, pilot/selection/validation,
solver damping/statistical ridge, microbenchmark/full-fit time.

Read set: SVD/ source corpus, the reports and artifacts it cites;
AGENTS.md, agent-notes/{STATE,README,WORKFLOW}.md, contracts/research.md,
tex/README.md. Read-only Luna scout routes the broad archival corpus.

- [x] A Inventory ideas, deduplicate sources, identify verified and missing evidence.
- [x] B Reopen primary reports and selected raw results; write a source-linked
  Russian synthesis with explicit mathematical domains and negative results.
- [x] C Verify quoted numbers/source paths, compile TeX at least twice, inspect
  final log and representative rendered pages; retain verification record.
- [x] D Update document routing and durable state, mark this plan done.

Acceptance: self-contained scientific exposition, conclusions tied to actual
checks, no pilot promoted to validation; source/evidence map; clean PDF build
without undefined references, missing glyphs or overfull boxes.
Stop/limits: missing data or conflicting reports are reported explicitly;
historical long benchmarks are not rerun. Existing dirty changes preserved.
Previous completed plan/state preserved verbatim in
agent-notes/history/plans/svd-synthesis-2026-10-24-previous/.

Evidence: SVD/AI_research_synthesis.tex and .pdf (20 pages),
SVD/synthesis_2026_10_24/VERIFICATION.md and evidence_audit.json.
Read-only corpus/draft review complete. Recomputed key aggregates for 640 saved
full-fit rows, checked 41 source links and 47 source hashes. Repeated 24 + 203
small formula checks in isolated copies. Final two XeLaTeX passes clean; all
pages inspected in contact sheets and selected final pages at larger scale.
No numerical implementation change or historical benchmark rerun.
