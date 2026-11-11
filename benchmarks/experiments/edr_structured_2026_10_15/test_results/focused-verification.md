# Focused verification

Commands ran in `/home/papayka/Documents/AMI/Vlad_EDR/EDR_code` with project
Python 3.13.12 (`uv run`). No timing benchmark ran concurrently.

| Command | Result |
|---|---|
| `rtk uv run pytest -q tests/test_edr_operators.py` | 10 passed in 0.36 s |
| `rtk uv run pytest -q tests/test_edr_schur.py` | 6 passed in 0.38 s after correcting the independent oracle to use spectral metric flooring and the documented metric-step normalization |
| `rtk uv run pytest -q tests/test_edr_research.py -k schur_derivative_guard_clears_core_before_metric_line_search` before the fallback fix | 2 failed as expected: both had `rank_guard_fallbacks > 0` and `accepted_steps == 0` |
| Same two-case regression after the fallback fix | 2 passed in 0.35 s |
| `rtk uv run pytest -q tests/test_edr_operators.py tests/test_edr_schur.py tests/test_edr_research.py tests/test_grassman.py tests/test_grassman_flat.py` | 73 passed in 0.64 s |
| `rtk proxy ruff check ADP/solver/statistic_operator.py ADP/solver/_grassman_schur.py ADP/solver/grassman_research.py tests/test_edr_operators.py tests/test_edr_schur.py tests/test_edr_research.py` | All checks passed |
| `rtk proxy ruff format --check ADP/solver/statistic_operator.py ADP/solver/_grassman_schur.py ADP/solver/grassman_research.py tests/test_edr_operators.py tests/test_edr_schur.py tests/test_edr_research.py` | 6 files already formatted |
| Scoped `uv run pyright` on the three solver modules and three new tests, with `UV_CACHE_DIR=/tmp/edr-structured-2026-10-15-uv-cache` | 0 errors, 0 warnings, 0 informations |

The focused suite includes independent direct augmented-lstsq values and
ambient gradients, dense/factorized operator parity, both normalizations,
rectangular Schur cores, tau=0 guards, metric whitening, trust-region negative
curvature, recycling certification, ridge 0/0.1, and the public multi-index
callback smoke. Completed full-suite verification and its existing failures
are recorded in [full-suite-status.md](full-suite-status.md).
