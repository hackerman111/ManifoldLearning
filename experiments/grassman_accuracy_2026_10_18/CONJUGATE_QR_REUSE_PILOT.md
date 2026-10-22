# QR factor reuse: exploratory outcome

Raw immutable artifact: pilot_conjugate_qr_reuse.json. Seed90800, n500,m2,
d5/d20/d50, identical data/method seeds, one BLAS thread, original5/CG10/QR-CG10.
The per-center helper reuses the guarded QR factors rather than repeating QR.
All-center true-profile acceptance, rank thresholds, steps and direction stay fixed.

| Case | QR/base EDR ratio | QR/base fit time | QR/CG fit time |
|---|---:|---:|---:|
| d5 | 1.06898 | .95133 | 1.12830 |
| d20 | .91697 | 1.05646 | 1.09298 |
| d50 | .85499 | 1.33449 | 1.10931 |

EDR geometric ratio .94282; base time median/p90 1.05646/1.27888.
It remains slower than CG on all three exploratory cases. Close this cost
candidate before fresh development. These three reused datasets are exploratory
evidence only; neither a full paired rejection nor an independent validation.
Original first QR pilot and profile/timing artifacts remain unchanged.

Verification: 42 focused tests passed, including independent augmented-lstsq
finite-difference gradients on mixed rank centers; Ruff and Pyright passed
before the final two gradient tests were added. Production grassman.py unchanged.
