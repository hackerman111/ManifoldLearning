# Тестовый SVD-поиск по уменьшению вдоль градиента

Реализован явный `rank_one_search="gradient"` в `ADP/solver/SVD.py`.
Default `"alternating"` сохранён. Класс APPROXIMATE: тот же fixed-g
функционал, другой конечный поиск rank-one добавки.

## Математический контракт

В matrix-режиме low_rank=B, prior=P, base_residual=I;
в correction-режиме low_rank=Delta, prior=0,
base_residual=I-forward(U,P,g). В обоих:
F(X)=sum mass ||base_residual-U X.T g||²+lambda||X-prior||².
Q=sum mass g e.T U+lambda(prior-X)=-grad(F)/2.

Для каждой ненулевой SVD-пары Q=left diag(s) right.T берутся единичные
(a,v), R=a.T Q v и D=sum mass (g.T a)² ||U v||²+lambda.
Из точного тождества F(X+sigma*a*v.T)-F(X)=D*sigma²-2R*sigma
получаем sigma=R/D, gamma=R²/D. Выбирается наибольший gamma среди
этого конечного набора; Q не проецируется на ортогональное дополнение V.
Попеременные условные a/v solves пропускаются. Общие QR/SVD сжатие,
совместный refit масштабов и проверка невозрастания цели сохраняются.
Метрика использует существующий whitening; поиск идёт в его координатах.

Это не поиск глобально лучшего rank-one шага и не совместный спуск по L,R.
Ведущая сингулярная компонента может иметь меньший gamma из-за кривизны.
Если выбранная добавка не увеличивает ранг, текущий greedy loop отвергает
её и возвращает `no_rank_growth`: это эвристический stop, не стационарность.
`rank_tolerance` также не сертификат. В matrix-режиме сравнение целей
начинается с B=0; уменьшение относительно F(P) не обещается.
`v_normal_residual_applicable=False`, пустые inner/LSMR tuples и нулевые
счётчики означают отсутствие v-подзадачи, а не успешный residual certificate.

Кандидаты: thin SVD размера m*d, до m проходов Uv на добавку.
Scratch O(md+Jp), включая сохранённый лучший Uv и текущий буфер;
Hessian размера d*d или (m*d)² не строится. Постоянные массивы и
whitening сохраняют прежний контракт памяти.

## Проверки

`tests/test_svd_gradient.py`: 11 случаев — независимый плотный дизайн,
производная через конечные разности, точные sigma/gamma и матрица после
одного шага, коллинеарность, неравные массы, lambda=0/.7/1e4, fixture
Q=diag(10,2) с выигрышами 1 и 4, несколько рангов, deterministic replay,
rank/gain stops, нулевой градиент/ранг, invalid/nonfinite/zero-curvature,
публичное multi-index обучение и запрет условных v solves.

`tests/test_svd_metric.py` расширен на оба поиска: dense whitening reference
для matrix/correction, p=.5/1, full/orthogonal и floor.
Итог: **124 tests pass**, включая SVD, metric, index models и multi-solver
certificate/derivation; focused Ruff и git diff --check проходят.
Pyright ADP/solver/SVD.py не завершился за 30 секунд (exit 124),
результат статического type check не подтверждён.

## Ограниченный парный fixed-g пилот

24/24 строк без ошибок, seeds 11/23/37, d100/300, J60,p8,m4, rank2,
lambda=.7, Gaussian U с geomspace(.2,5,d) масштабами столбцов,
positive mass geomspace(.2,3,J), float64, один BLAS thread.
Один прогрев и один timed solve на конфигурацию; отдельный tracemalloc
replay совпадает по B точно. Peak excludes заранее созданные input arrays
и может не учитывать все native allocations. Это малый синтетический
пилот, не полное обучение ADP и не доказательство восстановления.
В таблице A=alternating, G=gradient; все ratios парные G/A, затем медиана.

| d | target | A time, s | G time, s | time G/A | peak A, MiB | peak G, MiB | objective G/A |
|---|---|---:|---:|---:|---:|---:|---:|
| 100 | matrix | 0.01086 | 0.00099 | 0.09619 | 0.632 | 0.190 | 2.379 |
| 100 | correction | 0.00856 | 0.00086 | 0.10160 | 0.639 | 0.197 | 2.464 |
| 300 | matrix | 1.38130 | 0.00129 | 0.00093 | 0.433 | 0.412 | 8.128 |
| 300 | correction | 1.22639 | 0.00121 | 0.00097 | 0.446 | 0.425 | 8.750 |

Все outputs достигли rank2 и остановились по rank_limit. Невозрастание
истории цели проверено отдельно; inner convergence обычного метода
сохранён в raw rows, gradient не выдаёт его за своё свойство.
Относительная ошибка сырой B против синтетической полной Btrue записана
в CSV отдельно; это не projector recovery metric и цели при обоих
rank targets могут иметь разные лучшие допустимые матрицы.

Gradient уменьшает время этой ограниченной задачи ценой существенно
худшей цели (около 2.4x при d100, 8–9x при d300). Пилот не поддерживает
замену default. Не проводились подбор параметров, held-out recovery study
или сравнение с HPAO-LSMR. Три seeds и один timed solve на конфигурацию
не устанавливают устойчивую оценку производительности.

## Включение и воспроизведение

```python
from ADP import ADP_solver
from ADP.solver.SVD import solve as solve_svd

solver = ADP_solver(solve_svd, rank=2,
                    low_rank_target="correction", rank_one_search="gradient")
```

Запуск из корня репозитория (перезаписывает только этот пилот):

```sh
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run --no-sync python experiments/svd_gradient_2026-09-30/pilot.py
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest -q tests/test_svd_gradient.py tests/test_svd_solver.py tests/test_svd_metric.py tests/test_index_models.py tests/test_multi_solver_derivation.py tests/test_multi_solver_certificate.py
```

Raw results: runs.csv; shapes/options/data/versions/BLAS/source SHA and dirty
checkout: metadata.json. Final source SHA matches the completed pilot.
Source formulas: SVD/SVD_solver.tex eq:Q, eq:gain and subsection
«Выбор направления по уменьшению, а не только по градиенту».
