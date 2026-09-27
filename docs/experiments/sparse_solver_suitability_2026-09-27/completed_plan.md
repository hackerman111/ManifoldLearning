task_id: sparse-objective-matrix-audit-2026-09-27
status: done
change_class: EXACT
---

# Разреженность матриц минимизируемых функционалов ADP

## Цель и границы

Проверить структуру точных матриц функционалов single, multi и основного
manifold estimator на 10 малых и 10 средних синтетических задачах для каждой
ветки. Оценить, даёт ли их разреженность основание использовать sparse
solver-ы. Не менять estimator, функционалы, solver-ы или defaults; не заявлять
выигрыш по времени/памяти без отдельного парного benchmark.

## Протокол и инварианты

- 20 seed-профилей, каждый даёт по одному захвату текущего первого
  objective/operator на каждой ветке (60 захватов); полная outer convergence
  для проверки структурных нулей не требуется и не измеряется.
- Small `(n,d,J,P)=(240,12,24,16)`, `N_loc=32`, `N_lin=80`,
  `N_manifold=6`; medium `(1200,100,100,32)`, `N_loc=200`, `N_lin=700`,
  `N_manifold=16`. Float64, Epanechnikov; single `m=1`, multi `m=2`,
  manifold `m=1`, `estimator="manifold"`.
- Покрыть dense, коррелированные, zero-inflated и near-constant признаки;
  не ослаблять live rank/solver guards. Различать exact zero и near-zero
  (`0 < |a| <= 1e-12 max(|A|)`).
- Материализовать точные weighted single/multi design и `A.T @ A` (также
  ridge-вариант); для manifold — per-target design и B-system normal matrix с
  projector penalty. Сверить действия с matrix-free кодом на нескольких
  случайных векторах.

## Источники и артефакты

- Источники: `agent-notes/ADP/{index-pipeline,solvers,manifold}.md`;
  live `ADP/solver/LSMR.py`, `ADP/engine/manifol_engine/optimisation.py` и
  соответствующие fit/config источники, перечисленные в audit runner.
- Повторяемый runner: `experiments/sparse_matrix_audit.py`.
- Итоговые данные и отчёт: `docs/experiments/sparse_solver_suitability_2026-09-27/`.
- Предварительные rank-failure проходы сохранены в
  `docs/experiments/sparse_solver_suitability_2026-09-27/pilot_rank_failures/`;
  финальная сетка прошла без отказов.

## Итог и проверка

- [x] Все 60 objective/system captures завершены: 10 small и 10 medium для
  каждой ветки.
- [x] Плотность design: median 100%; minimum по всем матрицам 99.9375%.
  Точные нули встречаются только в отдельных medium `zero_inflated_070`
  design-матрицах; плотность всех normal matrices — 100%.
- [x] Matrix-free action совпадает с dense reference; максимальная ошибка
  на контрольных векторах `3.631e-15`. Near-zero доли сохранены отдельно.
- [x] Вывод ограничен этими профилями: структурных оснований для sparse
  storage/factorization не найдено; скорость/память sparse solver-ов не
  измерялись.
- [x] Ruff format/check и `git diff --check` пройдены; полный audit runner
  завершился с 20/20 задачами и сохранил JSON/Markdown отчёты.
- [x] `agent-notes/STATE.md` обновлён; вывод по текущим профилям добавлен в
  `agent-notes/DECISIONS.md`.

## Сохранённая посторонняя работа

Изменения для прежнего suite-digest/loss-quantile задания в
`experiments/README.md` и `experiments/analyze_suite.py` остаются вне scope;
также сохранены существовавшие пользовательские изменения `MyThink.txt` и
`TODO.txt`. В финальном status также обнаружен untracked `Method.md`; этот файл
не читался и не менялся. Эти файлы остаются вне scope аудита.
