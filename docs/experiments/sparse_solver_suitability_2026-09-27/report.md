# Разреженность функционалов и применимость sparse solver-ов

Аудит измеряет структурные нули точных матриц текущих функционалов. Он не
изменяет estimator и не сравнивает wall-clock solver-ов.

- Задач: 20/20;
  каждая задача включает single, multi и manifold.
- Вещественная точность: float64; near-zero считается как
  `0 < |a| <= 1e-12 max(|A|)` и не приравнивается к структурному нулю.
- Git: `6a540eedd4b5c98f1fae2698960e5aa89b931a36`; dirty status сохранён в JSON.

## Набор задач и матрицы

На каждом tier проверено по 10 задач для каждой ветки: всего 60
захватов live objective/system — первый штатный оператор для данного
seed-а. Полная внешняя сходимость не измерялась. Отклик задавался как
`Y = sin(X[:,0]) + 0.8 sin(X[:,1]) + N(0, 0.05^2)`. Профили включали
dense iid/offset, корреляции 0.30/0.70, zero-inflation 40%/70% вне двух
сигнальных координат, блочную корреляцию и признак масштаба `1e-4`.

| Tier | n | d | J | P | N_loc | N_lin | N_manifold |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 240 | 12 | 24 | 16 | 32 | 80 | 6 |
| medium | 1200 | 100 | 100 | 32 | 200 | 700 | 16 |

Single/multi: измерен weighted design `A[j,p,a,k] = sqrt(mass[j]) *
c[j,a] * U[j,p,k]`, развёрнутый в `(J*P) x (m*d)`, затем `A.T @ A`
и ridge-вариант с `lambda_prox=0.05`. Manifold: для каждого target
измерен локальный design `sqrt(mass[j] * graph_weight[j]) * slope[j,a] *
U[j,p,k]` и B-system normal matrix с projector penalty. В JSON сохранены
параметры и метрики каждого task-а. Single/multi normal имеет размер
`(m*d) x (m*d)`; manifold B-system — `(r*d) x (r*d)`. Matrix-free
действия сверены с dense reference на 3 случайных векторах для
single/multi и 3 векторах на каждый manifold target.

## Сводка по семейству и размеру

| Tier | Семейство | Задач | Размер design A | Плотность A, median [min] | Плотность normal, median [min] | Near-zero A, median [max] | CSR graph | Ошибка action |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | single | 10/10 | `384x12` | 100% | 100% | 0% [1.04167%] | — | 7.141e-16 |
| small | multi | 10/10 | `384x24` | 100% | 100% | 0% [1.04167%] | — | 8.018e-16 |
| small | manifold | 10/10 | `160x12 per target` | 100% | 100% | 0% [16.6667%] | 51.56% | 3.631e-15 |
| medium | single | 10/10 | `3200x100` | 100% [99.9991%] | 100% | 0% [0.0190625%] | — | 1.218e-15 |
| medium | multi | 10/10 | `3200x200` | 100% [99.9988%] | 100% | 0% [0.01875%] | — | 1.239e-15 |
| medium | manifold | 10/10 | `1888x100 per target` | 100% [99.9375%] | 100% | 0% [0.9375%] | 64.23% | 7.747e-16 |

В manifold размер design A показывает первый target; плотности
design и B-system считаются по всем target-матрицам
всех задач tier-а. CSR graph — медиана плотности по задачам; граф не
переносит свою структуру в feature-space матрицы B-system.

## Вывод

Минимальная точная плотность design матриц — 99.9375%, normal matrices — 100%. Редкие точные нули есть только в отдельных medium zero-inflated-70 design-матрицах; массового zero pattern для sparse storage/factorization нет. CSR-граф manifold имеет собственную измеренную плотность, но не разреживает feature-space B-system. Вывод ограничен этими synthetic профилями.

Аудит оценивает структурную применимость, но не преимущество по времени
или памяти. Для такого вывода нужен отдельный парный benchmark с теми
же stopping certificates и качеством.

Подробные per-task / per-target квантили, failures, near-zero доли и
provenance находятся в `audit.json`.
