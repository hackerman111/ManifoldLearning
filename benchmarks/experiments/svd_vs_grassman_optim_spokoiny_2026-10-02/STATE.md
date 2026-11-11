# Current state

Current task `svd-vs-grassman-optim-spokoiny-2026-10-02` is complete; see
`PLAN.md` and `experiments/svd_vs_grassman_optim_spokoiny_2026-10-02/REPORT.md`.

On heavy Spokoiny (`n=800,d=50,m=2`, seeds 0-2), all six cold full fits
completed with paired input hashes matching and 162 outer iterations ending at
`h_min`. Median times: SVD 24.903 s, grassman_optim 23.484 s. Median paired
optimized/SVD ratio 0.967; seed 2 was slower for grassman_optim. Median peak RSS
was 104916/108704 KiB (SVD/grassman_optim). Projector error was smaller for
grassman_optim on each seed; the methods remain distinct algorithms.

Artifacts:
- `benchmarks/svd_grassman_optim_benchmark.py`
- `experiments/svd_vs_grassman_optim_spokoiny_2026-10-02/fullfit_runs.jsonl`
- `experiments/svd_vs_grassman_optim_spokoiny_2026-10-02/fullfit_protocol.json`
- `experiments/svd_vs_grassman_optim_spokoiny_2026-10-02/REPORT.md`

The prior grassman optimization work and its verification artifacts remain
under `experiments/grassman_optim_2026_10_01/`; do not infer solver
equivalence from the new cross-method timing run.
