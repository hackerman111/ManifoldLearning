# Повторная попытка оптимизации multi-index solver: отрицательный результат

Причины старых отказов установлены раздельно. На четырёх одинаковых
линейных системах HPAO исходная точность LSMR (`atol=1e-10`) давала
normal-residual ratio 0.119–0.240 при требовании≤0.1; при `atol=1e-12`
ratio 0.00129–0.00258 **при прежнем lambda**. Критерий остановки LSMR для
несовместной LS-задачи не эквивалентен нормальному residual certificate HPAO.
Значит, часть старых повышений lambda была следствием недостаточной точности
линейной поправки. [Диагностика R1](r1.md),
[документация SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html).

Однако это не было главной причиной медленной внешней оптимизации: H6
уменьшил normal rejects на d100 с206 до103, но число AO шагов и итоговые
сертификаты сохранились, а median wall вырос на26%. На трёх прежних худших
точках L-BFGS проверенная касательная кривизна положительна; численно они
выглядят как отдельные локальные области, поэтому один warm HPAO шаг не
позволил избежать худшего objective. Старый прототип дополнительно платил
примерно8–10× за медленный SVD-reference на каждой evaluation. Эти факты
различают цену реализации и проблему basin.

|Кандидат|d100 certificates|Median spent wall к HPAO|Критический провал|
|---|---:|---:|---|
|H6, более точная прежняя correction|3/6 (baseline3/6)|1.264|Скорость и сходимость не улучшились|
|H7, полный reduced Gauss–Newton|4/6|3.883|Objective +0.160 на сертифицированной задаче; один явный line-search failure; RSS1.748×|

[H6](h6.md) и [H7](h7.md) имеют отдельные raw файлы, proofs и reference
проверки. H6 production-поправка после провала gate отменена; её код и
регистрирующий экспериментальный путь сохранены как
[`h6_rejected.patch`](h6_rejected.patch), который проходит `git apply --check`.
Patch нужен только для воспроизведения H6, не применяется к текущему solver.
H7 сохранён как явный экспериментальный прототип; изменение estimator,
публичного API/default и подмена certificate не проводились. Старое рабочее
дерево и ранее полученные артефакты сохранены.

Плановый gate ни одним кандидатом не пройден. Поэтому full-fit selection,
оценка trace_score/recovery на новых fits и held-out seed3000–3019 не
выполнялись. Нельзя заявлять ни ускорение production solver, ни улучшение
статистического восстановления. Следующий поиск требует новой математически
обоснованной и заранее ограниченной гипотезы; подбор ещё одного tolerance
или числа warm шагов на тех же frozen задачах не даёт независимого evidence.

Проверка после завершения: 74 профильных теста прошли; Ruff и Pyright для
затронутых Python-файлов, форматирование, `uv lock --check` и проверка patch
прошли. Окружение CPU/1thread и BLAS записано в
[`validation_environment.json`](validation_environment.json). Общий pytest:
321 passed, 28 GPU skipped, 14 failed в manifold
контуре; отдельный повтор дал те же 14, см.
[`validation_manifold.txt`](validation_manifold.txt). Среди причин —
`local_quadratic estimator requires index_dim=1` при старых m=2 сценариях,
а также три пороговых recovery assertion. Глобальный Pyright дал12 ошибок в
других experiment-файлах и одно предупреждение legacy, см.
[`validation_pyright.json`](validation_pyright.json). Целевой Ruff для
`ADP/`, `experiments/`, `tests/` прошёл; общий Ruff сообщил10 замечаний только
в отдельном каталоге `test/`, см.
[`validation_ruff.json`](validation_ruff.json). `git diff --check` для
рабочих изменений прошёл; проверка staged diff обнаружила CRLF CSV в ранее
подготовленных manifold-артефактах, не относящихся к этому solver-поиску.
