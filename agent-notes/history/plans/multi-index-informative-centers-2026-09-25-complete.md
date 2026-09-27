---
task_id: multi-index-informative-centers-2026-09-24
status: done
created_utc: 2026-09-24
updated_utc: 2026-09-25
change_class: ESTIMATOR
---

# PLAN — информативные центры multi-index

## Цель и границы

Найти воспроизводимое правило выбора центров и их числа для multi-index:
использовать **меньше** центров, чем случайный baseline `J₀`, не ухудшая
восстановление EDR-подпространства на независимых задачах. «Вершина» здесь —
центр `X[center_indices]`, а не направление `N_phi` или переменная solver.
Солвер, его tolerances, kernel, `N_loc`, `N_phi` и objective фиксированы.
Менять центры/`J` — **ESTIMATOR**, поэтому кандидат сначала существует только
как явный экспериментальный режим; default не менять. Предыдущий план:
`agent-notes/history/plans/multi-index-solver-retry-2026-09-24-complete.md`.

## Известное и гипотезы

- Live `fit_index` выбирает `J` индексов без replacement; центры влияют на
  инициализацию, `h`/`alpha`, local `I/U/mass` и SSE выбора outer-шагa.
  Одинаковый solver loss на разных центрах не является мерой качества.
- Предыдущий поиск solver не прошёл frozen gate; улучшение выбора центров пока
  **не проверено**. `trace_score=tr(P_hat P_true)/m` выше при лучшем
  восстановлении; recovery дополнительно требует технической сходимости.
- H1: часть случайных центров имеет малую эффективную локальную выборку,
  плохую обусловленность или дублирующую информацию. Отбор по допустимости,
  покрытию и предельному приросту **обучающей** информации может сократить `J`
  без потери recovery. Это эмпирическая гипотеза: surrogate не гарантирует EDR.
- H2: информативный отбор должен обойти случайный выбор **при том же J**;
  иначе эффект объясняется только числом центров.

## Инварианты и сравнение

Сохранить `float64`, формы `(J,d)/(J,P,d)`, rank/residual/convergence checks,
ортонормальность basis, отдельные seed-потоки и bounded memory: без `(J,n,d)`,
`(JP,md)` или плотных больших матриц. Candidate score вычислять только из
обучающих `X,Y` и пилотных статистик; истинный projector, validation/held-out
ответы и `trace_score` при выборе центров недоступны. Включать стоимость
пилота/отбора в полный wall/RSS. Направления сопрягать по ID центра, чтобы
смена `J` не меняла направления общих центров; проверить одинаковое
маргинальное распределение. Инициализацию проверить отдельно с общим пилотом
и в реальном full fit. При `training_set=exclude_centers` нужен заранее
фиксированный общий train/evaluation split. Основной протокол:
`training_set=all`, `center_displacement=0`.

На одинаковых `(n,d,m,N_loc,N_phi)` и solver budget сравнить `random(J₀)`,
`random(J)` и отбор при том же `J`. Случайные `J` делать вложенными префиксами
одной перестановки обучающих строк. Внутренняя `select_step=best` использует
SSE на выбранных центрах; отдельно проверить `select_step=last`, затем
production `best`. Качество оценивать по истинному projector синтетической
задачи, скрытому от отбора центров. Не сравнивать raw objective или center-SSE
между разными `J` как доказательство улучшения.

## Точный маршрут чтения

`AGENTS.md -> PLAN.md -> agent-notes/STATE.md -> agent-notes/README.md ->
agent-notes/ADP/{index-pipeline,multi-index,solvers}.md ->
agent-notes/contracts/{numerics,research,engineering}.md ->
agent-notes/WORKFLOW.md`. Live: `ADP/engine/common/index_fit.py:187–540`,
`calculus.py:16–109,259–422`, `initialize.py:70–184`,
`statistic.py:29–229`, `ADP/core/ADP_Config.py:23–112`,
`experiments/{multiv2_quality,diagnostic,runner}.py`, `experiments/README.md`.
Перед реализацией проверить соответствующие тесты.

## Ограниченные этапы и evidence

| Шаг | Работа | Критерий завершения |
|---|---|---|
| C1 | На 2 точках `d=10/100, n=1000, m=2`, с новыми **selection** seeds `4000–4009` измерить у random `J₀` распределения `mass`, `n_eff`, support, rank/conditioning, дублирования и цену pilot. | Таблица всех центров и failures; предсказания H1/H2 и бюджет записаны до выбора правила. Старые solver-frozen задачи не служат selection data. |
| C2 | Вывести одно простое правило: отсев недопустимых окрестностей и greedy прирост информации при ограничении покрытия; score/порог/остановку по `J` формализовать **до** full-fit. Проверять `⌊J₀/4⌋`, `⌊J₀/2⌋`, `⌊3J₀/4⌋`, максимум `J₀`; выбирать минимальный допустимый `J` без validation labels. | Вывод surrogate и его границ, оценка time/memory, детерминизм и обработка вырождения; малый dense/reference пример подтверждает score и greedy. |
| C3 | Изолированный прототип и paired full fits: random `J₀`, random same `J`, выбранные центры. Заморозить остальные настройки; выбрать не более одного варианта score/остановки по selection seeds `4000–4009`. | Raw `trace_score`, recovery, convergence, failures, `J`, суммарные wall/RSS; контроль seeds, reference-тест и раздельная диагностика init/solver. Selection gate: кандидат не хуже random `J₀` по критериям C4 и не хуже random same `J` по recovery и median `trace_score`. |
| C4 | Только прошедший selection gate алгоритм один раз проверить на нетронутых held-out seeds `5000–5019` и обеих точках; заранее заморозить код/пороги/конфигурацию. | Для **каждого** seed `J<J₀`; нет новых numerical failures; каждый fit с успешным baseline также converged/recovered у кандидата; `trace_score` не ниже baseline с допуском только `1e-10` на округление. Показать paired разницы и полную стоимость отбора. |
| C5 | При успехе оставить явную опцию, добавить focused tests и обновить affected notes; при провале зафиксировать отрицательный результат и не менять default. | Соответствующие pytest/Ruff/type checks, `git diff --check`, план/STATE/DECISIONS и воспроизводимые артефакты. |

## Итог этапов

- C1 done: precommitted protocol и 20/20 пилотов; gate 10/10 на обеих точках,
  8.636 с, peak RSS 97.40 МиБ. `docs/experiments/multi_center_selection_2026-09-25/c1_result.md`.
- C2 done, отрицательно: малый dense reference прошёл; 20/20 selection seed
  без numerical failures, но `J<500` найден для 0/10 d10 и 9/10 d100.
  Абсолютное покрытие 0.99 недостижимо для допустимого пула в 11 случаях.
  `docs/experiments/multi_center_selection_2026-09-25/c2_result.md`.
- C3/C4 skipped по зафиксированной C2 границе; full fits и held-out
  `5000–5019` не запускались. H2 и recovery этого правила не проверены.
- C5 done: отрицательный результат и raw сохранены; только изолированные
  модули `experiments/`, тест и документация, public default не менялся.
  Целевые Ruff, Pyright, pytest (60 passed без manifold), lock/import и
  `git diff --check` прошли. Отдельный существующий manifold-тест падает
  на ожидании `recovered=False` при фактическом `True`.

## Stop conditions

Остановить ветку до C4, если C1 не подтверждает информативные различия,
surrogate требует недопустимой памяти/цены, C2 reference не совпадает, либо
C3 ухудшает recovery/качество или даёт новые failures. Не подбирать пороги на
held-out и не исключать неуспешные fits из знаменателя. Если C4 не прошёл,
цель не достигнута: сообщить это прямо, сохранить random baseline и не
продвигать новый estimator. Утверждение «никогда не хуже» для любых данных
невозможно получить из surrogate; заявлять только проверенный диапазон.
Зафиксированный до C2 запуска `c2_rule.md` добавил явную остановку, если
хотя бы на одном selection seed нет `J<J₀`; она сработала.
