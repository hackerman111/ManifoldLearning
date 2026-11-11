# A-норма в SVD: реализация и парный пилот

Добавлены opt-in `metric_power=0/.5/1`, `metric_floor` и передача текущего
alpha/спектра/типа тензора в публичном multi-index fit. Default p=0 сохранён.
Численная реализация проверена; устойчивый выигрыш по EDR не установлен.

## Протокол

Selection: seeds 11/23/37, 48 fit + 48 fixed-g строк; validation:
seeds 101/103/107, 24 fit строки. Все 120 запусков завершились без исключений;
повторяющихся строк нет, fingerprint данных внутри пар совпадает.
Fit: small n500,d20,m3,J40 и medium n900,d60,m3,J64; N_phi12,
N_lin60/100, 12 outer steps, a=1.12, lambda=.05, last selection,
full kernel после первого orthogonal шага, локальная инициализация.
Оба ограничения rank=2: matrix и correction. Четыре варианта:
Frobenius, p=.5, p=1, p=1+rho=.1. Точные настройки и hashes — в manifests.
Float64, один BLAS thread, свежий subprocess на измерение, perf_counter
вокруг fit/solve; память — process peak RSS вместе с импортами и данными.
Все selection seeds нечётные: порядок floor,tensor,sqrt,Frobenius одинаков.
Поэтому различия времени порядка процентов нельзя считать надёжным speedup.

Заранее заданный selection gate: отсутствие failures, max ухудшение
projector error <=.02, медианное улучшение >0 по шести парам данного rank mode.
Это агрегированный пилот по двум размерам; ниже приведены отдельные группы.
p=1 прошёл gate в обоих режимах и выбран для held-out проверки.
p=.5/floor провалили matrix gate (max ухудшение .03749/.03065);
в correction они прошли, но выбран p=1 с наибольшим selection улучшением.

## Full fit: p=1 против Frobenius

Δ error <0 означает улучшение. В каждой строке три парных seed.

| Фаза | Rank mode | Размер | median Δ error | max ухудшение | median time ratio | median Δ RSS KiB |
|---|---|---|---:|---:|---:|---:|
| selection | matrix | medium | 0.00155974 | 0.0196094 | 1.0046 | 148 |
| selection | matrix | small | -0.000785309 | -0.000727317 | 1.0144 | 28 |
| selection | correction | medium | -0.00263285 | -0.00130517 | 1.0179 | 228 |
| selection | correction | small | -0.0115631 | 0.00294195 | 1.0245 | 56 |
| validation | matrix | medium | -0.00751081 | -0.00642017 | 1.0069 | -40 |
| validation | matrix | small | -3.33732e-05 | 0.00407318 | 0.9933 | 80 |
| validation | correction | medium | 0.000156452 | 0.0106115 | 1.0214 | 84 |
| validation | correction | small | -0.0010081 | 0.00230052 | 1.0272 | -16 |

Correction: selection median Δ=-.00574, validation median Δ=+.0000703;
validation gate FAILED (нет медианного улучшения). Matrix: validation
median Δ=-.00443 и max ухудшение .00407; пилотный gate PASSED, но часть
baseline ошибок .4–.8, лишь три seeds на размер и нет гарантии inner convergence.
Продвижения default нет. Значения loss разных metric вариантов нельзя
сравнивать как одинаковый функционал.

В selection inner_converged=true у 26/96 строк (fit+fixed); в validation
только у 1/24. У всех p=1 validation fits inner_converged=false.
Все fits остановлены outer_steps=12; это ограниченные fit-запуски, а не
полный bandwidth schedule до h_min. Сходимость внешнего алгоритма не доказана.

## Fixed-g d=100/300

Три seeds, m3,J32,p6, alpha=.1,Lambda=(1,.4,.1), lambda=.05.
d100 использует direct, d300 — LSMR. Коэффициентная ошибка относительно
синтетического full-rank truth не является projector recovery.

| d | Mode | Вариант | median time ratio | median LSMR iteration ratio | median Δ coefficient error |
|---|---|---|---:|---:|---:|
| 100 | matrix | sqrt | 0.9462 | direct | 0.022393 |
| 100 | matrix | tensor | 0.9203 | direct | 0.023515 |
| 100 | matrix | floor | 0.9629 | direct | 0.022606 |
| 300 | matrix | sqrt | 0.8760 | 0.8713127277000823 | 1.1781 |
| 300 | matrix | tensor | 0.8422 | 0.8514905786030478 | 5.0172 |
| 300 | matrix | floor | 0.7497 | 0.8040897872840522 | 0.99578 |
| 100 | correction | sqrt | 0.9581 | direct | -0.1529 |
| 100 | correction | tensor | 0.9878 | direct | -0.14309 |
| 100 | correction | floor | 1.0355 | direct | 0.024082 |
| 300 | correction | sqrt | 0.7778 | 0.7728705412054121 | 0.94729 |
| 300 | correction | tensor | 0.2550 | 0.29274292742927427 | 0.99418 |
| 300 | correction | floor | 0.2020 | 0.19130535055350553 | 0.95207 |

На d300 анизотропный штраф допускает существенно большие коэффициентные
ошибки, хотя собственный regularized loss ниже. Это ожидаемая опасность
ослабленного штрафа вне prior; малый loss не доказывает восстановление.
У p=1 alpha=.1 condition(A)=101; floor=.1 ограничивает её значением 9.18.
Дополнительный U буфер: 75/360 KiB в small/medium fit и 150/450 KiB
в fixed d100/d300. Малые RSS дельты шумны и не отменяют фактический буфер.

## Включение

```python
from ADP import ADP_Config, ADP_multi_index, ADP_solver
from ADP.solver.SVD import solve

model = ADP_multi_index(
    3,
    ADP_Config(multi_tensor="full"),
    ADP_solver(
        solve, rank=2, low_rank_target="correction", metric_power=1.0, metric_floor=0.1
    ),
).fit(X, y)
```

Для прямого fixed-g API передаются metric_alpha, metric_eigenvalues,
metric_tensor="full"/"orthogonal". В model API они берутся из текущего
шага; фиксированные настройки тех же имён конфликтуют и отклоняются.
Метрика использует текущий фактический tensor; full соответствует формуле
из предложения, orthogonal — существующему альтернативному ядру.

## Проверка и воспроизведение

91 tests passed: metric, SVD, model/CLI regressions.
Ruff и `git diff --check` проходят. Шесть численных replay-строк совпали с сохранёнными
по objective, normal residual и projector/coefficient error; hashes — в
`verification.json`. Standard Pyright не завершился за 30 s и остаётся непроверенным.
Математика и границы сертификатов: [AUDIT.md](AUDIT.md).

```sh
rtk proxy cp -a experiments/svd_a_metric_2026-09-30 \
  experiments/svd_a_metric_2026-09-30-rerun
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python \
  experiments/svd_a_metric_2026-09-30-rerun/benchmark.py
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python \
  experiments/svd_a_metric_2026-09-30-rerun/benchmark.py \
  --phase validation --candidates tensor
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache uv run --no-sync python \
  experiments/svd_a_metric_2026-09-30-rerun/analyze.py
```

Benchmark отказывается перезаписывать существующие runs/manifests. Команды выше
копируют каталог в соседний путь эксперимента; их нужно выполнять последовательно.
Raw rows и manifests исходного прогона сохранены без изменений.
После runs добавлены строгие guards complex/nonorthogonal входов, уменьшен
chunk budget и помечены координаты rank_scales; арифметика измеренных случаев
не изменилась. Финальные hashes и численный replay — в verification.json.
Manifest hashes относятся к измеренной версии; финальные hashes записаны отдельно.
