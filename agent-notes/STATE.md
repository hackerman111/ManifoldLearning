# Current agent state

## Task

- `multi-index-solver-retry-2026-09-24` завершён отрицательным результатом.
- `PLAN.md` done, R1–R3/R5 done, R4 skipped by frozen gate.
- Старый завершённый план архивирован в
  `agent-notes/history/plans/multi-index-solver-search-2026-09-24-complete.md`.
- Полный итог `docs/experiments/multi_solver_retry_2026-09-24/completion.md`;
  связанные `r1.md`, `h6.md`, `h7.md`, `hypotheses.md`, proofs и raw paths там.

## Проверенные факты

- Frozen selection: 12 hash-verified float64 задач d10/d100, seed1000–1002,
  outer0/2, CPU/1 BLAS thread. Held-out seed3000–3019 не использованы.
- HPAO normal rejects на четырёх сохранённых системах действительно устранимы
  ужесточением LSMR atol при прежней lambda; backward-error и исходный normal
  certificate имеют разные критерии. Это не объяснило медленные AO шаги.
- Три худших reduced-L-BFGS endpoints имели положительную проверенную
  касательную кривизну; их иной basin вероятен, но глобальная оптимальность
  не доказана. Медленный SVD-reference стоил8–10× batched envelope evaluation.
- H6 уменьшил d100 normal rejects206→103, но AO и certificates3/6 не изменил;
  median time ratio1.264. Его production patch отменён, архивирован как
  `docs/experiments/multi_solver_retry_2026-09-24/h6_rejected.patch`.
- H7 (`experiments/reduced_gauss_newton.py`) остался только явным CPU
  APPROXIMATE прототипом. Полный Jacobian/adjoint и плотный ridge reference
  проверены; d100 4/6 certificates, median spent time3.883×,
  peakRSS1.748×, objective+0.160 на сертифицированной задаче и одна явная
  line-search failure у cutoff. В публичный ADP/CLI не подключён.
- Ни один кандидат не прошёл frozen gate; full fits и recovery не проверялись.
  Прежние dirty changes/артефакты сохранены.

## Проверка и текущие границы

- 74 профильных теста, целевой Ruff/Pyright и форматирование прошли;
  `uv lock --check`, `git apply --check` архивного H6 patch прошли.
- Общий pytest: 321 passed, 28 GPU skipped, 14 failed в manifold контуре.
  Изолированный повтор тех же tests: те же14; там есть конфликт
  `local_quadratic`/`index_dim=2`, три пороговых recovery assertion и другие
  manifold assertions. Их исходники в этой solver-попытке не менялись.
  `validation_manifold.txt` в папке отчёта.
- Общий Pyright: 12 ошибок в других experiment-файлах, 1 warning legacy;
  три затронутых solver experiment-файла:0 errors. Общий Ruff затрагивает
  только отдельный каталог `test/` (10 замечаний); целевой
  `ADP/ experiments/ tests/` прошёл. Staged diff check видит CRLF в ранее
  подготовленных manifold CSV; unstaged diff check прошёл. JSON диагностики
  и CPU/1thread окружение сохранены в папке отчёта.
- Маршрут источников `agent-notes/ADP/solvers.md` обновлён. Дальнейший
  solver-поиск требует новой гипотезы/плана, не retune на этих frozen данных.
