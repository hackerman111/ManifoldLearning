# Начальные приближения и solver для single/multi index

Дата: 2026-09-27. Статус: selection выполняется; промежуточные результаты
не являются заключением. Артефакты ниже будут дополнены после полного прогона.

## Что из Method.md относится к текущей задаче

| Идея | Решение и основание |
|---|---|
| OPG, gradient PCA | Уже реализовано в `index_init=local`: локальные ridge-gradients, затем SVD; для normalized estimator учитывается mass. Это baseline, а не новая оптимизация. |
| SIR + SAVE | Изолированный initializer: SIR для inverse means, SAVE для conditional covariance и симметричных зависимостей. Whitening и обратное преобразование обязательны для коррелированных X. Проверяется полный ADP fit. |
| local-cv, neural pilot | Уже доступные альтернативы; сравниваются на тех же данных. Pilot доступен только multi. |
| PHD и DGSM | PHD — отдельная гипотеза второго момента, не включена в ограниченное сравнение. DGSM ранжирует координаты, но вращённое плотное подпространство нельзя безопасно восстановить отбором координат без дополнительных assumptions. |
| Sparse MAVE / Riemannian stochastic optimization | Перспективный отдельный estimator: меняет локальную регрессию, локализацию и objective. Теорема stationarity упрощённого SMAVE не доказывает recovery текущего ADP. |
| Leverage / Nyström, sensitivity / CRAIG coresets | Изменяют систему/веса/наблюдения; требуют отдельного estimator proof и paired recovery gates. Уменьшение ошибки аппроксимации kernel/gradient само по себе не доказывает сохранение ADP recovery. |
| H2O, Quest, SparseK, hierarchical attention | Полезные механизмы отбора для других задач. Для ADP нужны собственные bounds на статистики и ошибки projector; готовой гарантии в Method.md нет. Exact kernel support pruning уже есть в текущем pipeline. |
| PARDISO, CHOLMOD, AMG, GMRES | Выбор определяется структурой оператора. Предыдущий sparse audit: design практически плотный, normal matrices плотные; PDE hierarchy и sparse factorization здесь не обоснованы. Формирование normal matrix повышает требования к памяти и может ухудшить conditioning. |
| CG | Уже реализован; сравнивается с LSMR при одинаковом cap=80 и неизменном ADP objective. Native convergence flags имеют разные условия; общий gradient/orthogonality criterion записывается отдельно. |

Указатели `:chatgpt-content-reference{...}` в Method.md не являются доступными
библиографическими ссылками. Для использованных научных идей проверены первичные
источники:

- [Glaws, Constantine, Cook, Inverse regression for ridge recovery](https://arxiv.org/html/1702.02227): inverse moments, coverage и ограничения распределения.
- [Pautrel, Portier, Riemannian Stochastic Optimization for Sufficient Dimension Reduction](https://arxiv.org/html/2606.00413v1): sparse MAVE как другой estimator.

## Математический и численный допуск

[proof.md](proof.md) содержит Gaussian population argument, обязательный
coverage assumption, QR whitening/back-transform и matrix-free PSD action.
Независимый read-only аудит проверил алгебру и границы утверждений.
Population argument не является гарантией конечновыборочного или adaptive ADP
recovery. Число slices=10, веса SIR/SAVE=1 зафиксированы до selection.

Прототип: `experiments/inverse_moment_init.py`. Проверки включают независимый
SVD/dense reference, forward/adjoint, PSD, affine equivariance, большие offsets,
ранг и invalid inputs; summary gate проверяется на failures и paired regressions.
65 профильных tests прошли; Ruff/Pyright/format пройдены до фиксации manifest.

`reference_stress.json`: n=10000,d=1000,m=2, float64; initializer 3.588 s,
process RSS 392.24 MiB, eigen residual 1.36e-11, orthogonality error 4.44e-16.
Это проверка ресурсов и численной реализации, не качества полного fit.
Память O(nd+d²), thin QR время O(nd²); n>d и полный численный ранг обязательны.

## Фиксированный протокол

`protocol.json` фиксирует исходники, commit/dirty state, библиотечное окружение,
BLAS, конфигурации и seeds. Gaussian X, tau=0.4, n=1000, d=10/100,
noise=0.2, J=500,P=40, outer=3, cap=80. Всего 108 selection fits:
три single профиля с четырьмя вариантами и три multi с пятью, по четыре seeds
73000–73003. Validation 74000–74009 допускается только для прошедшего профиля.
Все варианты используют одинаковые X/Y/truth/model seed; SHA256 X/Y записан.
Initializer не получает truth. Все failures остаются в знаменателе.

Quality: abs cosine single; normalized projector trace multi, выше лучше;
порог 0.95. Recovery требует quality pass и native solver convergence.
Stationarity отдельно: max(Riemannian gradient,local gradient,orthogonality)<1e-6.
Для CG native certificate требует один certified step, HPAO — два.

Selection на каждом профиле требует: все paired quality deltas>=-1e-10,
не меньше convergence/recovery counts, без failures, median quality gain>=0.01
либо recovery rate gain>=0.25, median paired time ratio<=2 и RSS ratio<=1.5.
Validation повторяет nonregression/resource gates и требует положительного
median quality gain>1e-10 либо recovery gain. Gates после результатов не меняются.
Fresh process RSS включает imports/data; fit wall time включает initializer.
Timeout 180 s/fit, budget 2400 s/phase. Измерения выполнены с BLAS=1. Короткие verification suites выполнялись
параллельно с частью selection; время не является изолированным performance
benchmark. Validation выполняется без параллельных проверок.

## Воспроизведение

Фазовые файлы не перезаписываются. Для нового прогона используйте новый каталог:

```sh
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m experiments.index_recovery selection --output /tmp/index-recovery-repeat
rtk proxy env UV_CACHE_DIR=/tmp/adp-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python -m experiments.index_recovery validation --output /tmp/index-recovery-repeat
```

Вторая команда завершится отказом, если нет прошедшего selection кандидата.
`selection.jsonl` / `validation.jsonl` — полные trace, diagnostics и failures;
`*_summary.json` — paired deltas и каждый gate. Не смешивать результаты
повторного прогона с зафиксированными здесь данными.

## Границы интерпретации и дальнейшие гипотезы

Разрешённый sample eigengap и маленький eigen residual удостоверяют решение
выборочной spectral задачи, а не близость к истинному подпространству. При
n=1000, десяти slices и d=100 каждый slice содержит лишь 100 наблюдений;
оценка conditional covariance шумна и после центрирования имеет rank<=99.
Это возможная причина слабого multi старта; причинная гарантия не установлена.
Распределение и coverage также ограничивают inverse moments.

Для current multi fit нужно различать две проблемы: выбранный внешний шаг
может иметь хорошее качество, но не получить полный solver certificate;
при d=100 качество может быть низким при корректно решённой внутренней
задаче. Эти проблемы требуют разных гипотез. Отбор только certified outer
steps изменит правило выбора и может снизить geometric quality. Он не
проверен в этом исследовании. Изменение estimator/localization по SMAVE
требует нового proof/reference/selection протокола; уже использованные seeds
нельзя объявить новой независимой проверкой.

## Что означает сходимость

Здесь различаются три уровня. Линейный solve LSMR/CG вычисляет одну поправку;
его residual certificate относится только к этой линейной задаче. Внутренний
ADP solver на фиксированной статистике оптимизирует подпространство и local
coefficients; native convergence проверяет gradients, feasibility, изменение
objective и aligned step (для HPAO два последовательных certified steps).
Outer loop меняет bandwidth/statistics; завершение outer_steps не является
доказательством его математической сходимости.

Geometric quality сравнивает оценку с truth только в synthetic diagnostics.
Сходимость не использует truth. Поэтому stationary решение может иметь плохое
качество, а нестационарный iterate — хорошее. Recovery в этом протоколе требует
quality>=0.95 и native convergence выбранного шага. Численная сходимость означает
выполнение проверок остановки с конечной точностью, не статистическую
состоятельность и не доказательство достижения global optimum.
