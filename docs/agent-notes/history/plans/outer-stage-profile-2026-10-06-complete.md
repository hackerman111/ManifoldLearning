# Plan: outer ADP stage timing

- Task outer-stage-profile-2026-10-06; status done.
- Goal: identify largest stages outside Grassmann solve in same three Spokoiny fits.
- Class observational; no estimator, API or solver changes; same generator,
  float64, seed0/1/2, n800/d50/m2/J800/P10, one BLAS thread.
- Known evidence: solve10-22% fit; existing model.profile_ already times stages.
- Read set: index-pipeline note, index_fit.py:280-410 and update stage;
  multi.fit profiler contract, common/logger.py; prior experiment harness/results.
- Bounded units: Luna baseline fit with saved built-in profile_; separate
  top stages first; split weights/moments only if winner remains ambiguous;
  compare input/basis/diagnostics hashes to existing baselines; lead audit/report.
- Verification: all3 seeds exact output parity, nonoverlap timers, retain raw
  profile stage counts/time and explicit remainder plus RSS/environment.
- Stop: parity failure/nonfinite/error,900s per run. No optimization or tests.

- First seed: statistics14.65s vs update5.60s. Existing stages unambiguously
  identify main outside-solver cost; finish same three fits without adding
  nested instrumentation. Statistics remains weights plus moments combined.

- Evidence complete: all3 exact input/basis/diagnostics parity; raw builtin
  profiles cover fit with only .046s/fit remainder. Statistics56.4% pooled,
  update/alpha21.5%, solver15.4%. Report/protocol/raw under
  experiments/outer_stage_profile_2026_10_06/results/. No production edits.
