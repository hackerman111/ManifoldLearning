---
task_id: adp-proof-first-shared-idea-2026-09-26
status: done
created_utc: 2026-09-26
updated_utc: 2026-09-26
change_class: ESTIMATOR
---

# PLAN — новая идея для multi-index и manifold ADP

## Цель и границы

Найти и проверить **один** математически обоснованный способ существенно
улучшить восстановление EDR-подпространства, полное время fit или пиковую
память в multi-index и/или manifold ADP. Приоритет — механизм, применимый к
обеим веткам; результат в одной ветке достаточен при честно указанной
границе применимости. Порядок: **формула и доказательство → независимый
аудит доказательства → малый reference → ограниченный эксперимент →
нетронутая validation**. До proof gate не менять алгоритм и не подбирать
параметры по full fits. Публичные defaults/API и прежние отрицательные
результаты этим планом не пересматриваются. Предыдущий план сохранён в
`agent-notes/history/plans/multi-index-informative-centers-2026-09-25-complete.md`.

## Известное и гипотезы

- Multi оценивает один глобальный projector; manifold — projector каждого
  центра с графовой связью. Обе ветки строят локальные направленные моменты
  `I=(J,P)` и `U=(J,P,d)`, но законы направлений и solver-цели различаются.
  При **фиксированных** весах `I_j=Φ_j c_j`, `U_j=Φ_j C_j` для локальных
  cross-moment `c_j` и covariance action `C_j`. Это исходное алгебраическое
  наблюдение; его нужно вывести с точной live-нормировкой для обеих веток.
- На трудной multi-точке `d=100` увеличение inner budget и две solver-попытки
  не дали приемлемого общего выигрыша. Coverage/logdet отбор центров
  остановился на C2 до full fits: его score не доказывает EDR quality.
  Manifold `local_quadratic` уже улучшил радиальный `m=1`, но остаётся
  отдельным estimator и не решает общий `m>1` случай.
- **H1, направленный скетч:** заменить независимые направления блоками
  ортогональных случайных направлений с теми же одномерными маргиналами.
  Для изотропной manifold-статистики это может уменьшить дисперсию
  квадратичной ошибки скетча при том же `P` или позволить меньший `P`.
  Для multi после анизотропного преобразования и нормировки требуется
  **отдельное** доказательство или контрпример. Совместный закон направлений
  меняется, значит это `ESTIMATOR`, даже если каждая маргиналь сохранена.
- **H2, резерв следующего цикла:** если H1/H3 не проходят proof/cost gate,
  построить stochastic
  gradient по центрам/направлениям для **фиксированной конечной ADP-цели**:
  control variate, периодический полный градиент, окончательный полный
  сертификат. До прототипа доказать несмещённость, границу variance и
  сходимость к стационарной точке на нужной геометрии. Это `APPROXIMATE`
  solver только при сохранении цели и моментов. H1 и H2 не смешивать в
  одном сравнении; возможное ускорение и recovery пока гипотезы.
- **H3, разделимая нейросетевая аппроксимация:** по мотивам
  [Separable Physics-Informed Neural Networks (SPINN)](https://papers.nips.cc/paper_files/paper/2023/hash/4af827e7d0b7bdae6097d44977e87534-Abstract-Conference.html)
  проверить низкоранговую модель отклика
  `f_θ(x)=Σ_{r=1}^R a_r ∏_{k=1}^m q_{rk}((B x)_k)`, `BBᵀ=I_m`.
  **Обучать `B` совместно** с одномерными факторами: при замороженном
  pilot-basis градиенты всегда лежат в его старом span и не могут исправить
  EDR-ошибку. Для multi извлекать projector из градиентной covariance
  обученной функции; для manifold рассмотреть локальные chart-факторы и
  взвешенные gradient covariances лишь после отдельной оценки ошибки и цены
  на каждый центр. Это новый `ESTIMATOR`, а не существующий MLP-pilot
  инициализации. Физического PDE в задаче нет, поэтому PDE-residual из
  SPINN сюда не переносится. Его выигрыш от декартовой сетки также нельзя
  предполагать для нерегулярных `(X,Y)`; нужна явная проверка числа
  уникальных координат, операций, training wall и peak RSS.

Первичный литературный аудит сравнит конкретные ADP-моменты, H3 и условия
теорем со [smoothed gradient outer products для multi-index](https://arxiv.org/abs/2312.15469),
[Riemannian SVRG](https://arxiv.org/abs/1605.07147) и
[orthogonal random features](https://research.google/pubs/orthogonal-random-features/).
Последняя работа относится к kernel features: её вывод о variance не
переносится на ADP без вывода. SPINN даёт архитектурный мотив, а не теорему
об EDR recovery или скорости на ADP-данных. Эквивалентную известную идею
не называть новой.

## Математический gate

Для **каждой** ветки, предлагаемой к тесту, отдельный proof dossier:

1. Точная конечновыборочная цель, формы, закон направлений, нормировка mass,
   rank/cutoff и graph penalty; сверить live-код и релевантный TeX, не
   отождествляя несовпадающие формулы.
2. Теорема с явными предпосылками и полным выводом: что сохраняется точно;
   для стохастических H1/H2 — несмещённость и variance/concentration;
   для каждого кандидата — **условная** граница projector error через
   eigengap либо stationarity solver. Теорема о скетче/solver сама по себе
   не гарантирует recovery полного outer-loop.
3. Контрпример или область отсутствия гарантии; `P<d`/`P≥d`, multi
   anisotropy, зависимость от pilot, вырождение mass/rank и float64 rounding.
4. Оценка операций и peak bytes при `n=10_000,d=1_000`: без `(J,n,d)`,
   `(JP,md)`, плотных `d×d` на центр и `(md)^2`. Для H2 — tangent,
   adjoint/retraction и полный certificate исходной цели.
5. Для H3 — теорема восстановления из `Γ=E[∇f(X)∇f(X)ᵀ]` при положительном
   eigengap: разложить ошибку градиента на ограниченные approximation,
   statistical и optimization части, затем вывести perturbation bound для
   projector. Для manifold добавить bias от изменения локального
   подпространства внутри окрестности. Проверить, что один лишь малый MSE
   отклика не гарантирует малую ошибку градиента; указать условия гладкости,
   ограниченного separation rank `R`, identifiability и cross-fitting.
   Без контролируемой derivative error H3 не проходит proof gate.

Независимый проход воспроизводит вывод, проверяет предпосылки на live-коде
и ищет контрпример. Провал любого пункта закрывает кандидата либо сужает
его до доказанной ветки **до** реализации. Численный reference не заменяет
математическое доказательство.

## Точный маршрут чтения

`AGENTS.md -> PLAN.md -> agent-notes/STATE.md -> agent-notes/README.md ->
agent-notes/ADP/{index-pipeline,multi-index,manifold,solvers}.md ->
agent-notes/tex/{multi-index-adp,manifold-adp}.md ->
agent-notes/contracts/{numerics,research}.md -> agent-notes/WORKFLOW.md`.
Для R0/R1: `SRC-STAT` (`ADP/engine/common/statistic.py:38–229`),
`SRC-MI-DIRECTIONS` из multi note, `SRC-MAN-WEIGHTS`
(`ADP/engine/manifol_engine/weights.py:311–360`), `SRC-HPAO-ENTRY/OP`,
`SRC-MAN-STEP/B-SYSTEM`, `SRC-INIT-OTHER`
(`ADP/engine/common/initialize.py:337–387`), `X-MI-PRE`, `A-OBJECTIVE`
и профильные tests. Для H3 также `agent-notes/contracts/engineering.md`
перед выбором NN-зависимости и backend.
Расширять чтение только при конкретном пробеле в доказательстве.

## Ограниченные этапы и evidence

| Шаг | Работа | Критерий перехода |
|---|---|---|
| R0 | Зафиксировать baseline, формулы и bottleneck обеих веток; проверить литературу H1/H2/H3. Для H3 аналитически оценить separation rank известных benchmark-функций в EDR-координатах и возможность повторного использования координат на нерегулярных данных. | Карта общего/различного, baseline-протокол и диагностический time/RSS; выбрать не более двух идей для полного доказательства по математической осуществимости и ожидаемой полной цене, до результатов кандидатов. |
| R1 | Доказать выбранные H1/H2/H3, каждую как отдельный вариант; не более двух proof dossiers и одного прототипа за цикл. | Доказательство и отдельный аудит, контрпримеры/условия, классификация изменения, оценка памяти. Для H3 обязательно ограничить derivative error и projector error; без этого R2 запрещён. |
| R2 | Один изолированный CPU-прототип, малый dense/reference и стресс на вырожденных окрестностях; заморозить код, конфигурацию, seed-схему и **численные пороги** paired gate до full fits. | Проверены тождества, adjoint/certificate где нужны, finite/rank guards, bounded memory; float64 и целевые pytest/Ruff/type checks. Для H3 — ещё gradient reference и полная стоимость обучения. |
| R3 | Paired selection full fits на **новых** seed-потоках: multi `d=10/100,m=2`, manifold `m=1/2` на задачах с запасом для улучшения baseline. Сравнить baseline, кандидат и подходящий контроль: при меньшем `P` тот же `P`, для H3 обычную MLP сопоставимой ёмкости и существующий MLP-pilot; учитывать обучение/подбор/QR/certification. | Raw per-fit quality, recovery, convergence/completion, failures, wall/RSS; заранее замороженный gate пройден хотя бы в одной ветке без новых numerical failures. При провале остановиться. |
| R4 | Один раз проверить прошедший кандидат на нетронутых held-out seed той же ветки, задач и порогов. | Paired разницы по каждому seed, uncertainty, полная стоимость и failure accounting; никаких подборов по validation. Успех одной ветки не переносить на другую. |
| R5 | Лишь после R4 success решить вопрос об явной опции; обновить affected routes и научный отчёт. | Нужные проверки, `git diff --check`, воспроизводимые артефакты, итог в `PLAN.md`/`STATE.md`/`DECISIONS.md`. |

Итоговый checkpoint: **R0/R1/R2/R3 выполнены; R4/R5 остановлены по gate**.
Формулы, разница
live/TeX по manifold penalty, baseline-диагностика и выбор только H1
зафиксированы в `docs/experiments/proof_first_shared_2026-09-26/r0.md`.
H3 не допущен к R1 из-за отсутствия проверяемого derivative-error bound
и ценового основания для нерегулярных данных; H2 остаётся резервом
следующего цикла. H1 proof и независимый аудит:
`docs/experiments/proof_first_shared_2026-09-26/{h1_proof,r1_audit}.md`.
R2 reference и стресс прошли; численные paired thresholds, конфигурации,
seed-потоки и fingerprint заморожены **до full fits** в
`docs/experiments/proof_first_shared_2026-09-26/r2_protocol.md`.
R0 manifold CLI-диагностика оказалась `local_quadratic` из-за live default;
она исключена из H1 baseline. Оба R3 manifold случая явно используют
`estimator="manifold"`; m=2 остаётся заведомо хрупким rank-стрессом.
R3: 66/66 fit на 61000–61005, ни одна из четырёх точек не прошла
строгий per-fit quality/full-cost gate. Причины, сырые строки и same-row
контроль: `docs/experiments/proof_first_shared_2026-09-26/r3_result.md`,
`selection/{runs.jsonl,summary.json,manifest.json}`. H1 остаётся
доказанным для фиксированной цели, но полного выигрыша не показал.
Production ADP/default/API не менялись.

Selection seed 61000–61005 использованы только здесь; held-out
62000–62019 остались нетронутыми. Прежние `4000–4009`, `5000–5019`
center-selection и `3000–3019` solver не использовались для выбора H1.
Multi: `trace_score` выше лучше, recovery только при convergence. Manifold:
RMS principal sine/projector error ниже лучше, все центры; completion
отдельно от inner certificate. Нецертифицированное качество не считать
recovery. Выигрыш по времени/памяти считать по полному fit с сохранённым
качеством, не по одному kernel.

## Stop conditions

Остановиться после R0 без отличимого механизма; после R1 без корректной
теоремы или применимых предпосылок (для H3 также без derivative-error
контроля или при взрывном separation rank); после R2 при провале reference,
certificate или бюджета памяти; после R3 при провале frozen gate. Не снижать
rank/residual/quality критерии, не исключать failures, не подбирать на
held-out и не заявлять универсального выигрыша по конечному числу seed.
Отрицательный результат сохранять как результат исследования.
