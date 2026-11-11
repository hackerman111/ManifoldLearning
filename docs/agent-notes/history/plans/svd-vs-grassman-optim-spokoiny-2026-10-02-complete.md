task_id: svd-vs-grassman-optim-spokoiny-2026-10-02
status: done
change_class: EXPERIMENT; distinct solver methods/configurations

# Goal / non-goals
Measure SVD and grassman_optim full-fit wall time on heavy Spokoiny, paired
over seeds 0-2. Preserve each method's documented settings. This comparison is
descriptive across different solver methods; no solver or outer-fit changes.

# Protocol / invariants
Workload: n=800, d=50, m=2, J=800, p=10, float64. Fresh process per run,
alternating solver order, BLAS/OMP/MKL threads pinned to one. Record raw time,
RSS, final projector error, stopping state, source and input hashes. Preserve
failures. Do not infer estimator equivalence or general performance from three
seeds.

# Acceptance evidence
- [x] Six paired full fits completed; same input hashes within each seed.
- [x] Raw rows, protocol metadata, and concise report saved under
  `experiments/svd_vs_grassman_optim_spokoiny_2026-10-02/`.
- [x] Medians, per-seed times, RSS, quality, and stopping behavior summarized
  in `REPORT.md` and `agent-notes/STATE.md`.
