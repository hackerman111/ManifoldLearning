# Recovery criteria update

Date: 2026-09-27

The repository recovery rule now measures geometric recovery independently of
solver convergence. Single-index recovery uses `abs(cosine) >= 0.95`; multi-index
recovery uses `trace_score >= 0.95`. Solver convergence and stationarity remain
available in experiment diagnostics but do not gate recovery or candidate
admission. Numerical failures remain unrecovered and stay in the denominator.

Manifold experiments require both the full-center RMS local-projector error
`<= 0.2` and the maximum local-projector error over all centers `<= 0.2`. The
maximum is computed from the largest principal sine across every center and
direction. Current manifold experiment inputs use `index_dim=1`, so this is
exactly the worst-center local-projector distance.

The existing `protocol.json`, `report.md`, and `selection.jsonl` record the
previous protocol and are preserved as historical evidence. The selection file
has 105 rows for 108 planned fits, and no selection process is running. Its
summaries are incomplete and use the prior convergence-gated recovery rule.
They cannot authorize validation under the updated rule. A new selection must
use a fresh output directory because the source fingerprint has changed.
