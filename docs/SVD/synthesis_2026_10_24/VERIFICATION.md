# SVD research synthesis: verification

Date: 2026-10-24. Deliverables:

- `SVD/AI_research_synthesis.tex` — standalone Russian scientific synthesis.
- `SVD/AI_research_synthesis.pdf` — 20-page compiled document.

Documentation only. No production numerical implementation, estimator, solver
default or historical experiment output was changed. Existing dirty research
work was preserved. Previous PLAN/STATE were copied verbatim to
`agent-notes/history/plans/svd-synthesis-2026-10-24-previous/`.

## Corpus and scientific review

The SVD source corpus, archive proposals, theory reports and linked primary
experiment reports were inspected. A read-only Luna scout routed the broad
corpus, checked missing provenance, and reviewed the draft. The lead reopened
the mathematical and numerical sources supporting the main conclusions.

The document distinguishes fixed-coefficient and profiled objectives, basis/
correction/tangent ranks, statistical ridge and optimizer damping, numerical
formula checks and full-fit recovery, pilots/selection/held-out data, kernel
timing/model.fit/process timing, and traced allocations/process RSS.

Material provenance findings retained in the text:

- kimi's 350-run figures have no linked raw artifact, seed/config list, exact
  error definition or timing boundary; they are reported as manuscript claims.
- The original five-seed adaptive-center tables have no available primary run
  provenance and contain internal inconsistencies. Their later report-inspired
  benchmark uses a different protocol and explicitly is not a reproduction.
- `edr_research_note.tex` names `edr_note_checks.py`, absent in this checkout.
- LoRA was not an arm of the failed projected10 development comparison.
- Combined20 improves the historical original5 control, without demonstrated
  incremental recovery improvement over accepted relative10.

## Evidence audit

Run from the repository root:

```sh
rtk proxy .venv/bin/python SVD/synthesis_2026_10_24/verify_evidence.py
```

Final result: passed. `evidence_audit.json` retains the computed values and
47 SHA-256 source fingerprints. All 41 bibliography file links exist.

The audit read 640 saved full-fit rows, checked complete unique method/case/seed
keys, successful status of every row, shared input fingerprints and (where
recorded) method seeds. It independently recomputed key point aggregates:

| Saved run | Rows | Paired datasets | Recomputed quantities |
|---|---:|---:|---|
| Relative GN held-out | 90 | 30 | EDR geometric ratio, median/p90 fit-time ratios |
| Combined held-out | 150 | 30 | All seven saved paired comparison point aggregates |
| Lowrank/projected development | 280 | 70 | Both ten-step controls and projected/budget comparison |
| Adaptive-center development | 120 | 12 | Selected median EDR differences, time ratios, recovery counts |

Intervals were retained from the saved analyses; bootstrap intervals were not
recomputed. No new statistical sample or performance measurement was created.
The SVD preconditioner validation summary (144 rows, successful gate) and
three saved angle-curve timing ratios were also checked. This is not a new
integrity audit of every historical source archive or every reported number.

The existing independent formula scripts were copied to a temporary directory
and run without production solver imports or changes to their original outputs:

- `SVD/documentation/check_math.py`: passed, 24 reported quantities.
- `SVD/theory_2026_10_01/check_formulas.py`: passed, 203 assertions in 35 groups.

Fresh outputs are `fresh_math_checks.json` and `fresh_formula_results.json`.
These checks concern small deterministic algebra/geometry examples, not EDR
recovery, timing, novelty, or convergence of the external adaptive procedure.

## PDF build and layout

The final source was built twice consecutively with XeLaTeX:

```sh
rtk proxy xelatex -halt-on-error -interaction=nonstopmode \
  -output-directory=SVD/synthesis_2026_10_24 SVD/AI_research_synthesis.tex
```

Final successful passes are recorded in `build_pass_7.txt` and
`build_pass_8.txt`. The final log contains no warnings, underfull/overfull
boxes, missing characters or undefined references. `build_audit.json` retains
the PDF metadata and hash. The final PDF is copied beside its TeX source, so
relative bibliography paths resolve from `SVD/`.

All 20 pages were inspected as rendered contact sheets. Final pages 1, 11, 16
and 18 were rendered at larger scale; numerical tables and the corpus/source
map were inspected. No clipping or broken layout was observed.

`rtk git diff --check` passed. `git diff --quiet -- ADP` returned zero.
No broad regression suite, R/CUDA run, large fit, or historical timing benchmark
was run for this documentation task. No convergence or new recovery claim is
made from successful document/formula verification.
