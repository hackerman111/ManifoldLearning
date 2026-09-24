# S2 — проверка reduced-objective derivation и proof gate

Математический вывод: [reduced_lbfgs.md](proofs/reduced_lbfgs.md). Независимая таблица предпосылок, guard-ов и тестов: [audit_reduced_lbfgs.md](proofs/audit_reduced_lbfgs.md). Они доказывают корректность gradient, геометрии и условную сходимость к стационарности **усечённого текущего objective** внутри rank-stable области. Для точного joint least squares при отброшенных положительных сингулярных значениях утверждается лишь явно ограниченная ошибка; глобальный минимум и статистическое восстановление не доказаны.

Малый reference `experiments.multi_solver_derivation.evaluate_reduced` реализует формулу (3) напрямую по локальным SVD и не является рабочим solver. `tests/test_multi_solver_derivation.py`: 5 тестов проходят; проверены tangent finite differences при m=2/5, rank0/rank1/full, basis rotation и контрпример положительной отброшенной сингулярной компоненты.

Frozen-проверка:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.multi_solver_derivation \
  --frozen-dir benchmark_outputs/diagnostic/multi_solver_search_s1_20260924 \
  --output benchmark_outputs/diagnostic/multi_solver_search_s2_derivation_20260924.json
```

Результат: [JSON](../../../benchmark_outputs/diagnostic/multi_solver_search_s2_derivation_20260924.json). У всех 12 frozen задач rank pattern сохранился для `±1e-3, ±1e-4, ±1e-5` нормированного horizontal шага, и относительная ошибка центральной конечной разности уменьшилась у всех 12 при переходе `1e-3 → 1e-4`. Наибольшая ошибка: `1.27e-4`, `1.27e-6`, `4.31e-8` соответственно. Максимальные value/normalized-gradient defects от усечения: `1.56e-31` и `2.21e-18`, существенно ниже заранее записанных guard-порогов `1e-10 max(1,fτ)` и `1e-8`. Эти проверки подтверждают применимость выведенной формулы **локально на selection**, а не заменяют доказательство или гарантию вдоль всей будущей траектории.

Proof gate S2 закрыт для одного guarded reduced L-BFGS прототипа. Guard обязан явно отказать при смене rank/cutoff, нарушении error bound, nonfinite значениях и провале line search; без них теорема неприменима. Существующий HYBRID отклонён по [парному S2 сравнению](s2_hybrid.md). Второй новый прототип пока не выбран.
