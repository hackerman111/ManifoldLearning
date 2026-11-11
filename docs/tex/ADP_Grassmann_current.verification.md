# Verification: current ADP with Grassmann

Verified 23 October 2026 against HEAD `4b06b98f4eb7c0d32735c7b7d208df6548b088d9`
and the live working tree. This task changes documentation and its formula
checker only; numerical source files and the existing export snapshot were
not modified. Previously dirty research and routing work was preserved.

## Deliverables

- [English TeX](ADP_Grassmann_current.tex) and [PDF](ADP_Grassmann_current.pdf):
  complete global multi-index ADP plus current `grassman.py::solve`.
- [Research history](../SVD/GRASSMANN_IMPROVEMENT_HISTORY.md): recent SVD-derived
  attempts, preceding ideas, saved full-fit results and promotion decisions.
  All relative source links were checked. Historic performance results were
  read from original reports, not rerun or newly validated in this task.

## Current-code checks

```
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_grassman.py tests/test_grassman_combined.py
```

Result: **35 passed in 0.41 s**. Existing tests independently verify local
least squares, full profiled Jacobian finite differences, the gradient/adjoint
identity, augmented GN, gauge invariance, orthogonality, rank guards, true
profile descent, combined defaults and exact constant-center restoration.

```
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tex.documentation.grassmann_current.check_formulas
```

Result: **4 kernel/moment batches, 3 full fits, 9 outer-update replays passed**.
The checker constructs covariance matrices directly for independent normalized
and legacy moments, checks both localization tensors and the squared-distance
kernel convention, verifies local-gradient initialization against augmented
least squares, and replays actual geometry, isotropic sketches, score
chronology, canonical spectrum, best-step selection and transform. Cases
include the forced first orthogonal step of a full-tensor fit, center exclusion,
and fixed directions. Inner updates reuse the public solver; independent
solver-derivative evidence comes from the existing finite-difference tests.
These small runs verify formulas and wiring, not recovery/generalization.
Ruff check and format check passed for the formula checker.

## PDF

Using installed pdfTeX 1.40.29 / TeX Live 2026 on this machine:

```
rtk proxy mkdir -p /tmp/adp-grassmann-tex
rtk proxy pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/adp-grassmann-tex tex/ADP_Grassmann_current.tex
# Repeat the pdflatex command to resolve the contents and references.
```

The final repeated build succeeded: **15 A4 pages**, with no TeX warnings,
unresolved references, overfull/underfull boxes, missing glyphs or errors.
Representative mathematical and source-map pages were rendered and visually
inspected. Only the PDF is copied into `tex/`; auxiliary files remain in `/tmp`.

## Provenance and limits

The TeX derives formulas from live code, rather than silently reconciling old
manuscript variants. The history distinguishes initial unnormalized intrinsic
GN from accepted relative GN, and reports combined20's absent incremental
recovery gain versus accepted10. The projected10 development gate failed;
LoRA has no development arm; the history preserves those boundaries.

A read-only Luna scout routed historical report sources under the repository's
AGENTS protocol. The lead reopened the selected original reports, proofs and
TeX locators before writing numerical conclusions. History links and task
whitespace checks passed. Source dates/headings can retain historical stages;
the history uses their completed result sections and live solver defaults.

No new large benchmark, GPU run, R comparison or full repository test suite
was performed. Historic speed/recovery results are workload-specific saved
evidence. The 35 focused tests and small replay are the fresh CPU checks.

## SHA-256 at completion

| File | SHA-256 |
|---|---|
| `ADP/core/ADP_Config.py` | `f241194edbed37994da9156a0fb8829dfc4dd7c9e80f14c728606f203eee4427` |
| `ADP/core/ADP_Solver.py` | `538f5cdb996cc41cf96de88416e722857c9d78f9aea836c9a1cb910f103b8a0f` |
| `ADP/core/multi/ADP_multi_index.py` | `d0e36b25e89a1ba8b9426a4669148f93ca2b540b2f7abf1675e665c236c59e0f` |
| `ADP/engine/common/index_fit.py` | `f7d7886697dd7ff4690642b6e168ba4068549ee39f909ebb515aee56aed9f4ea` |
| `ADP/engine/common/initialize.py` | `8146befd146847d4b88ac3b1f0c27c7eb68d00998ba44a3f334e114718b9b4bc` |
| `ADP/engine/common/calculus.py` | `5d50b69edbfa9363b33d791da24c50fa47a29ccb4e2e4a791ad7ee8e80615c5e` |
| `ADP/engine/common/weights.py` | `0cc5a04bbff3ae60e74da248ccd73b0128090072fdb1267c4334d78a7e7d006e` |
| `ADP/engine/common/statistic.py` | `32e3cd877044eeea0d05fdb37d2af80a474c9b2c72240a4e362b1b163122d977` |
| `ADP/solver/grassman.py` | `3a0807d0d31afded0705890a2478cdd539de1f1f553831dcf593861ddb3acda6` |
| `tests/test_grassman.py` | `2940f6b609389ac33e4c033a2e89078c934893762e6cb8eed89114fbfb4095f9` |
| `tests/test_grassman_combined.py` | `b7cd5beafedf706e9d3edc755104090c160bfc1322e7ffdafa391988793d187a` |
| `tex/ADP_Grassmann_current.tex` | `e3f2ed0a1300b13c2148d626c5c557d39f6b0e2ccc603c83b1c25b59efea2d47` |
| `tex/ADP_Grassmann_current.pdf` | `64d60b8c7e01586f4b3eb9da665c9afd077fbfe51f5754f514f21c6926bee5b2` |
| `tex/documentation/grassmann_current/check_formulas.py` | `9064acd7c1d5c145992df447578ba064733accbc19a6b0b514c85237a4dea72e` |
| `SVD/GRASSMANN_IMPROVEMENT_HISTORY.md` | `cc06e71f765371e63c39aaf2a27082740aff61dcc75f761adc188e36167aba95` |
