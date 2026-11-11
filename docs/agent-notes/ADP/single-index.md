# Single-index: алгоритм и API

## Публичный путь

`ADP.core.single.ADP_single_index` — estimator-фасад. `fit(X,Y)` передает задачу общему `engine.common.index_fit.fit_index(mode="single", index_dim=1)`, а solver по умолчанию — current HPAO-LSMR с `max_steps=3`, `tol=1e-6`. Пользовательский solver должен удовлетворять текущему контракту `HPAOResult`; старый `ADP_Solver.fit` — отдельный совместимый путь. Источники: **SRC-SI-FIT**, **SRC-IFIT-ENTRY**, **SRC-SOLVER-ADAPTER**.

Финальный индекс имеет форму `(d,)`, нормируется и ориентируется относительно предыдущего индекса, чтобы знак не скакал между шагами. Знак не идентифицирует направление модели: result использует абсолютный cosine. Model state также сохраняет trace, коэффициенты по центрам, диагностики solver, effective parameters и профиль фаз. Источники: **SRC-IFIT-INIT**, **SRC-SI-RESULT**, **SRC-SI-FIT**.

## Single-index локализация

Для каждого наблюдения и центра код делит смещение на скалярную проекцию вдоль `beta` и ортогональный остаток. Текущая ветка `estimator="new"` подает в kernel аргумент

    (projection2 + rho**2 * orthogonal2) / h**2

Ветка `legacy` использует иной живой аргумент:

    (rho**2 * distance2 + projection2) / h**2

Они не алгебраически взаимозаменяемы и в конспекте намеренно разведены. `rho` ограничен `[0,1]`: при текущем масштабе бисекцией ищется наибольшее значение, сохраняющее среднюю kernel mass не меньше `N_loc`; если даже при `rho=0` цель недостижима, следующий масштаб не выполняется. Источники: **SRC-SI-LOCALIZE**, **SRC-SI-RHO**.

Если направления локализованные, случайный вектор строится из независимого Gaussian `z` и скаляра `xi` как `rho*z + xi*beta`, затем нормируется. Изотропный режим использует ту же функцию с нулевым beta и `rho=1`. Направления либо redraw-ятся на каждой внешней итерации, либо переиспользуются согласно `redraw_directions`.

## Полный цикл

1. `fit_index` выбирает центры; seed потоков для центров, init и направлений независим.
2. Стартовый индекс получается из local weighted gradients, local-cv, pilot MLP либо QR-normalized random draw.
3. По матрице квадратов расстояний подбирается исходный `h`, с целевой средней mass `N_loc` и нижней границей `h_min`.
4. По `beta,h,rho` строятся веса; CPU/GPU ветка извлекает локальные `I/U` и массу.
5. HPAO решает локальные slope coefficients и proximal global update индекса; подробности — [solvers.md](solvers.md).
6. Outer-loop делит `h` на `a`, обновляет `rho` и повторяет. Он останавливается при `h_min`, заданном `outer_steps` или infeasible `rho`.
7. `select_step="best"` выбирает минимальную SSE на значениях ответа в центрах; `last` возвращает последний масштаб.

## Что не следует предполагать

- `ADP_single_index.fit` сам не поддерживает solver GPU-ветку, если используется произвольный solver. GPU solver разрешен только при built-in `solve_lsmr`; GPU statistics без device solver передают вычисления solver-у на CPU.
- `smart_weights` сохраняется как config/metadata, но поиском по текущему `ADP/` не найдено подключения этого флага к `fit_index` или `calculate_weight`. Не выводите из имени параметра, что текущая модель использует `NeighborhoodEngine`.
- `ADP_single_index_engine.py` содержит небольшой orientation helper, но основной численный цикл не находится там.

## Локаторы

| ID | Фрагмент | Команда sed |
|---|---|---|
| SRC-SI-FIT | публичный fit, solver default, state publication | `rtk proxy sed -n '19,93p' ADP/core/single/ADP_single_index.py` |
| SRC-SI-RESULT | sign-invariant result metric | `rtk proxy sed -n '11,43p' ADP/core/single/ADP_single_index_result.py` |
| SRC-SI-VALIDATION | input/index validation, fitted-state checks | `rtk proxy sed -n '14,60p' ADP/core/single/ADP_single_index_utils.py` |
| SRC-SI-ORIENT | line sign orientation helper | `rtk proxy sed -n '1,22p' ADP/engine/single_index/ADP_single_index_engine.py` |
| SRC-SI-LOCALIZE | new/legacy kernel argument for single weights | `rtk proxy sed -n '13,73p' ADP/engine/common/weights.py` |
| SRC-SI-RHO | bisection and infeasible mass handling | `rtk proxy sed -n '136,207p' ADP/engine/common/calculus.py` |
| SRC-SI-DIRECTIONS | localized and isotropic sketch construction | `rtk proxy sed -n '209,258p' ADP/engine/common/calculus.py` |
| SRC-SI-DRIVER | outer-loop selection/stop details | `rtk proxy sed -n '280,540p' ADP/engine/common/index_fit.py` |

Для каждого ID полный файл и общие источники есть в [README.md](README.md#каталог-исходников); shared pipeline и исходные формы I/U — в [index-pipeline.md](index-pipeline.md).
