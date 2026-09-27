# C1: результат пилотной диагностики

Протокол и gate зафиксированы до запуска в `c1_protocol.md`. Raw:
`benchmark_outputs/diagnostic/multi_center_c1_20260925/{manifest.json,seeds.csv,centers.csv}`.
Код `experiments/multi_center_diagnostic.py`, revision
`84eb9bf323b09b2a64e13f2d47dce0241b4dbb59`, dirty fingerprint
`4f2480ecf5588ba46c6b682e247b71a46b97ef6408626918122e360e28222ada`.

| Точка | Полные seed | Информативные по gate | Медиана q90/q10 `n_eff_outer` | Медиана rank-deficient fraction | Медиана q90/q10 condition ratio | Медиана spread nearest weight cosine |
|---|---:|---:|---:|---:|---:|---:|
| d10 | 10/10 | 10/10 | 22.86 | 0.022 | 1.915 | 0.403 |
| n1000/d100 | 10/10 | 10/10 | 15.74 | 0.020 | 1.602 | 0.330 |

Каждый из 20 seed имеет 500 строк центров; failures нет. У d10 медианный
support первой outer-окрестности 40, минимум 1; у d100 — 104, минимум 1.
Медиана ближайшего cosine весов — 0.607 и 0.527 соответственно. На каждой
точке минимум 7/10 seed требовалось для перехода, получено 10/10.

Суммарное wall 8.636 с, process peak RSS 97.40 МиБ при одном потоке
OpenBLAS/OMP/MKL, Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1. `ru_maxrss`
фиксирует пик всего процесса, не отдельную аллокацию и не затраты будущего
full fit. Порог 30 минут / 2 GiB выдержан.

Решение: перейти к C2. Диагностика показывает неоднородность и отдельные
почти пустые окрестности, но не причинную связь с EDR recovery. H2, экономия
full-fit wall и безопасность сокращения J ещё не проверены.
