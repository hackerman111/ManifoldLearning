# Conjugate QR exploratory pilot

One unprofiled, single-thread run: seed 90800, method seed 1173550236; cases d-5/d-20/d-50, n=500, m=2. Budgets were original 5, old conjugate 10, and conjugate QR 10. This is a three-case exploratory pilot, not held-out evidence. Full environment, settings, source/input hashes, fit metrics, and inner diagnostics are in `pilot_conjugate_qr.json`.

## QR relative to original and old conjugate

| Case | QR/original EDR ratio | QR/original time ratio | QR/old EDR ratio | QR/old time ratio |
|---|---:|---:|---:|---:|
| d-5 | 1.06898 | 0.99004 | 1.00000 | 1.19540 |
| d-20 | 0.91697 | 1.10137 | 1.00000 | 1.16391 |
| d-50 | 0.85352 | 1.38465 | 1.00585 | 1.16213 |

Across the three cases, QR/original EDR geometric mean was 0.94228; time ratio median/p90 was 1.10137/1.32799; maximum case EDR ratio was 1.06898. QR/old-conjugate EDR geometric mean was 1.00195; time ratio median/p90 was 1.16391/1.18910.

## Per-case solver details

| Case | Method | Fit s | EDR | Outer stop | Inner calls / accepted steps | Inner stop counts | Max inner orthogonality |
|---|---|---:|---:|---|---:|---|---:|
| d-5 | original5 | 0.38159 | 0.030890 | h_min | 2 / 10 | max_steps 2 | 4.60e-16 |
| d-5 | old conjugate10 | 0.31604 | 0.033021 | h_min | 2 / 20 | max_steps 2 | 4.74e-16 |
| d-5 | conjugate QR10 | 0.37779 | 0.033021 | h_min | 2 / 20 | max_steps 2 | 4.01e-16 |
| d-20 | original5 | 0.64126 | 0.108376 | h_min | 5 / 25 | max_steps 5 | 6.03e-16 |
| d-20 | old conjugate10 | 0.60681 | 0.099377 | h_min | 5 / 50 | max_steps 5 | 8.29e-16 |
| d-20 | conjugate QR10 | 0.70627 | 0.099377 | h_min | 5 / 50 | max_steps 5 | 1.20e-15 |
| d-50 | original5 | 0.67154 | 0.230428 | h_min | 7 / 35 | max_steps 7 | 1.41e-15 |
| d-50 | old conjugate10 | 0.80012 | 0.195532 | h_min | 7 / 60 | max_steps 7 | 1.24e-15 |
| d-50 | conjugate QR10 | 0.92984 | 0.196675 | h_min | 7 / 65 | max_steps 7 | 1.37e-15 |

Maximum absolute old/QR inner `profile_objective` difference by case was 8.88e-16 (d-5), 2.73e-12 (d-20), and 14.96594 (d-50); inner calls are compared by position. Maximum output-basis orthogonality error for QR was 3.17e-16, 3.17e-16, and 1.13e-16, respectively.
