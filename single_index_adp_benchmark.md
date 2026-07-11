# Спецификация бенчмарка single-index ADP

## Область действия

Документ задаёт воспроизводимый протокол проверки текущей реализации **single-index Average Derivative Procedure (ADP)** со случайными направленными измерениями, анизотропными локальными весами, внешней структурной адаптацией и внутренней чередующейся минимизацией ALS с matrix-free CG.

В документ не входят:

- multi-index модели;
- локальные подпространства и manifold-обобщения;
- сравнение ALS с variable projection, L-BFGS, Adam, римановыми методами и другими оптимизаторами;
- сравнение CG с LSQR, LSMR и другими рабочими solver как самостоятельная исследовательская задача;
- разработка новой регуляризации или новой параметризации внутренней задачи.

Плотное решение малой линейной системы применяется только как **численный эталон** для проверки текущего matrix-free CG. Оно не считается baseline-методом и не участвует в сравнении алгоритмов.

## Обозначения статуса утверждений

| Метка | Значение |
|---|---|
| **[Л]** | результат или стандартная постановка из научной литературы |
| **[М]** | вывод из формул и математических свойств рассматриваемого алгоритма |
| **[Г]** | экспериментальная гипотеза, которую требуется проверить |
| **[П]** | предложение по организации бенчмарка |

## Проектный контракт реализации и хранения

Бенчмарк расширяет существующий контур проекта, а не создаёт независимый
runner. Общие измерители и CSV-примитивы остаются в `adp/common`, базовый
benchmark API — в `adp/evaluation`, а масштабный single-index протокол получает
собственный слой `adp/evaluation/single_index`. Публичной точкой входа остаётся
`run_benchmarks.py`.

### Существующие точки расширения

| Ответственность | Текущий файл | Правило расширения |
|---|---|---|
| RSS и wall-clock | `adp/common/resource_monitor.py` | использовать `ResourceMonitor`, не добавлять второй sampler |
| стабильная CSV-схема | `adp/common/experiment_log.py` | использовать `CSVTable`, `flatten_mapping`, `stable_run_id`, `merge_csv_shards` |
| базовые сценарии | `adp/evaluation/scenarios.py` | сохранить совместимость `BenchmarkScenario`; большие реестры вынести в `adp/evaluation/single_index/scenarios.py` |
| запуск одного метода | `adp/evaluation/runner.py` | переиспользовать генерацию, baseline и ресурсные поля через узкий адаптер |
| сводки и графики | `adp/evaluation/reports.py` | общие сводки не дублировать; протокольные отчёты вынести в `adp/evaluation/single_index/reports.py` |
| CLI | `adp/evaluation/cli.py`, `run_benchmarks.py` | добавить профиль single-index без нарушения `--quick` и `--grid` |
| эталон серии | `experiments/adp_confirmatory_common.py` | переиспользовать структуру `series/runs/iterations/initial_parameters/artifacts`, но не импортировать семантику экспериментов 4–6 |

Новые модули `adp/evaluation/single_index` отвечают только за этот научный
протокол. Универсальные операции записи, измерения ресурсов и идентификации
запусков не переносятся туда из `adp/common`.

### Каталог одной серии

Каждая серия хранится отдельно:

```text
benchmark_outputs/
└── single_index/
    └── <series_id>/
        ├── single_index_series.csv
        ├── single_index_runs.csv
        ├── single_index_iterations.csv
        ├── single_index_solver_iterations.csv
        ├── single_index_initial_parameters.csv
        ├── single_index_summary.csv
        ├── single_index_failures.csv
        ├── single_index_artifacts.csv
        └── plots/
            └── *.png
```

`single_index_solver_iterations.csv` создаётся только для C08–C11 и T07, где
нужны траектории ALS/CG. Для остальных сценариев достаточно агрегатов внешнего
шага в `single_index_iterations.csv`. Это ограничивает объём логов без потери
диагностической информации.

JSON-файлы и JSON-объекты внутри CSV-ячеек не используются. Вложенные настройки
разворачиваются в скалярные столбцы через `flatten_mapping`; короткие скалярные
последовательности кодируются существующим разделителем `|`. Большие векторы,
матрицы, временные ряды RSS и сами выборки в CSV не записываются.

### Нормализованные CSV-таблицы

Все таблицы содержат `schema_version` и `series_id`. Таблицы с результатами
конкретного запуска дополнительно содержат `run_id`. `run_id` детерминированно
строится из `scenario_id`, метода, номера повтора, seed и fingerprint полной
конфигурации; поэтому одинаковый job можно обнаружить при возобновлении серии.
`series_id` состоит из UTC-времени старта и короткого fingerprint конфигурации:
это позволяет хранить несколько независимых повторов одной конфигурации, а
возобновлять именно выбранный каталог через `--resume`.

| Таблица | Гранулярность | Обязательное содержимое |
|---|---|---|
| `series.csv` | одна строка на серию | версия кода, dirty-флаг, профиль, время старта/завершения, requested/completed/failed jobs, окружение, параллелизм, fingerprint конфигурации |
| `runs.csv` | один method fit | scenario/method/seeds, final metrics, status/error, stop reason, resource fields, число iteration rows, время сохранения |
| `iterations.csv` | один outer step | `outer_k`, `h_k`, `rho_k`, local-mass aggregates, objective, cosine, beta delta, component timings |
| `solver_iterations.csv` | один ALS/CG diagnostic step | `outer_k`, `inner_k`, `cg_k`, relative objective/residual, projective delta, CG info |
| `initial_parameters.csv` | один запланированный job | полная развёрнутая конфигурация данных, алгоритма, solver, seeds и инициализации до первого fit |
| `summary.csv` | scenario × method | count, median/mean/IQR/q05/q95, CI, success/failure rates, time/RSS aggregates |
| `failures.csv` | один failed run | `run_id`, категория, exception type/message, stage, last completed outer/inner step |
| `artifacts.csv` | один артефакт | имя, тип, относительный путь, размер файла и статус создания |

`runs.csv` является первичной таблицей результата. `summary.csv`,
`failures.csv` и графики восстанавливаются из первичных CSV и не считаются
единственным источником данных.

### Границы измерения времени и памяти

Для ADP поля `algorithm_*` берутся из `ADPResult.resource_usage`; для baseline
аналогичное окно оборачивает только вызов метода. Поля `full_run_*` измеряются
от начала генерации или загрузки данных до вычисления метрик и записи основной
строки/итераций worker-а.

Каждое окно содержит:

- `*_time_sec`;
- `*_rss_start_mib`;
- `*_rss_min_mib`;
- `*_rss_mean_mib`;
- `*_rss_max_mib`;
- `*_rss_peak_delta_mib`;
- `*_memory_samples`;
- `*_memory_source`.

`result_persist_time_sec` записывается отдельно. Минимум, среднее и максимум —
абсолютный RSS процесса внутри окна; `rss_peak_delta_mib` — прирост максимума
относительно RSS в начале окна. История отдельных RSS samples не хранится.
Полное окно включает сохранение payload результата. Финальная CSV-строка,
которая сообщает сами `full_run_*`, по определению записывается после закрытия
измерителя; её время входит в `result_persist_time_sec`, но не в
`full_run_time_sec`. Эта граница одинакова для ADP и baseline.

### Запись, параллелизм и возобновление

До запуска первого job атомарно создаются `series.csv` со статусом `running` и
`initial_parameters.csv`. Статус серии меняется только атомарной заменой строки:
`running -> complete`, `running -> partial` или `running -> failed`.

Каждый worker пишет только в свои PID-шарды скрытого каталога
`.<series_id>_shards/`. Строка run пишется последней и служит commit-marker
завершённого job; iteration rows без соответствующей run-строки считаются
остатком прерванного job. Родительский процесс потоково объединяет финальный CSV
и новые шарды, проверяет заголовки и уникальность ключей и только после успешной
атомарной публикации удаляет временные файлы.

При `--resume <series_dir>` runner читает только ключевые столбцы `run_id` и
`status` из финального `runs.csv` и сохранившихся run-шардов. Он пропускает
`success` и по явному флагу повторяет `failed`. Строки прерванного job удаляются
из iteration/solver shards по отсутствующему commit-marker до повторного
dispatch. Несовпадение `schema_version` или fingerprint конфигурации завершает
запуск ошибкой до создания worker. Повторная строка с тем же `run_id` или
повторный iteration key в финальной таблице запрещены.

При исключении строка failed run всё равно сохраняется вместе с последними
доступными `algorithm_*`/`full_run_*`. Прогресс выводится в stderr с `flush=True`
через заданное число jobs, чтобы `tail -f` работал при запуске с перенаправлением
вывода. При process-level parallelism BLAS/OpenMP threads фиксируются равными 1.

### Обязательный паспорт серии

До запуска экспериментов создаётся неизменяемая часть строки
`single_index_series.csv`. Без неё результаты нельзя считать воспроизводимыми.

Паспорт серии должен содержать:

| Поле | Что записать |
|---|---|
| Версия кода | commit hash, branch и состояние рабочей директории |
| Формула ядра | точное выражение `K(q)`, включая то, является ли `q` расстоянием или квадратом расстояния |
| Локальное среднее | подтверждение формулы с делением на локальную массу |
| Правило выбора \(h_0\) | mean, minimum или фиксированный квантиль локальной массы |
| Правило выбора \(\rho_k\) | минимальное или максимальное допустимое значение, границы поиска и tolerance |
| Центры | способ выбора, наличие perturbation, масштаб perturbation |
| Направления | распределение на шаге 0 и на шагах \(k\geq1\), частота обновления |
| ALS | порядок полу-шагов, нормировка, критерий остановки |
| CG | tolerance, `maxiter`, preconditioner, обработка `info` |
| Ridge | точная формула и масштаб ridge-поправки |
| Тип данных | float32 или float64 |
| Параллелизм | число процессов, BLAS-потоков и CPU-потоков |
| Окружение | Python, NumPy, SciPy, BLAS, ОС, CPU, RAM |
| Случайность | алгоритм генератора и схема разбиения seed |
| Контракт данных | `schema_version`, `series_id`, fingerprint конфигурации |
| Хранение | относительные пути первичных таблиц и политика resume |

### Зафиксированные допущения этой спецификации

В предоставленных материалах есть несовпадения между версиями формул. Поэтому основной протокол использует следующие допущения.

1. Локальное среднее нормировано:

   \[
   \bar X_j
   =
   \frac{\sum_{i=1}^{n}w_{ij}X_i}
   {\sum_{i=1}^{n}w_{ij}}.
   \]

   Без деления на локальную массу константная часть локальной линейной аппроксимации не сокращается.

2. Аргумент ядра представляет квадрат локального расстояния:

   \[
   q_{ij}=\|T(X_i-x_j)\|_2^2.
   \]

3. Референсная запись Epanechnikov-ядра:

   \[
   K(q)=(1-q)_+.
   \]

   Если код использует \((1-q^2)_+\), это записывается в паспорте серии и не исправляется скрыто. Математические unit-тесты выполняются с фактической формулой кода, а статистические результаты маркируются формулой ядра.

4. Основная реализация использует несколько направлений на центр:

   \[
   |\Phi_j|=P=n_\Phi.
   \]

5. Основная анизотропная метрика имеет вид:

   \[
   T_k^2
   =
   h_k^{-2}\bigl(\rho_k^2I_d+\beta_{k-1}\beta_{k-1}^{\mathsf T}\bigr).
   \]

6. \(\rho_k\) выбирается как наибольшее значение в \([0,1]\), при котором выполняется установленное ограничение на локальную массу. При монотонно убывающем ядре рост \(\rho\) уменьшает массу, поэтому такой выбор даёт наименее агрессивную анизотропию среди допустимых значений.

7. Основное правило массы использует нижний квантиль \(Q_{0.05}\). Условие по средней массе из рукописи рассматривается как отдельная ablation, поскольку среднее может скрывать пустые окрестности отдельных центров.

---

# 1. Формальная постановка задачи

## 1.1. Входные данные и статистическая модель

Наблюдаются независимые пары

\[
(X_i,Y_i),
\qquad
X_i\in\mathbb R^d,
\quad
Y_i\in\mathbb R,
\qquad
i=1,\ldots,n.
\]

Одноиндексная модель условного среднего задаётся равенством

\[
\mathbb E[Y\mid X=x]
=
g(x)
=
f(\beta^{*\mathsf T}x),
\]

где

- \(\beta^*\in\mathbb R^d\) — неизвестное индексное направление;
- \(f:\mathbb R\to\mathbb R\) — неизвестная функция связи;
- \(\varepsilon=Y-f(\beta^{*\mathsf T}X)\) удовлетворяет
  \(\mathbb E[\varepsilon\mid X]=0\).

При дифференцируемой \(f\)

\[
\nabla g(x)
=
f'(\beta^{*\mathsf T}x)\beta^*.
\]

**[Л]** Все ненулевые градиенты условного среднего лежат на прямой, натянутой на \(\beta^*\). Average derivative estimators используют это свойство для восстановления индексного направления без глобальной параметризации \(f\).

## 1.2. Целевой объект и идентифицируемость

Масштаб \(\beta^*\) не идентифицируется, поскольку для любого \(c\neq0\)

\[
f(\beta^{*\mathsf T}x)
=
\widetilde f((c\beta^*)^{\mathsf T}x),
\qquad
\widetilde f(t)=f(t/c).
\]

Фиксируется нормировка

\[
\|\beta^*\|_2=1.
\]

Знак также не идентифицируется:

\[
\beta^*\sim-\beta^*.
\]

Целевой математический объект представляет прямую

\[
[\beta^*]=\{\beta^*,-\beta^*\}.
\]

Основная метрика восстановления должна учитывать эту инвариантность.

## 1.3. Необходимые условия идентифицируемости

Для восстановления направления требуются следующие условия.

1. Ненулевой производный сигнал:

   \[
   \mathbb E\bigl[f'(\beta^{*\mathsf T}X)^2\bigr]>0.
   \]

2. Ненулевая дисперсия индекса:

   \[
   \operatorname{Var}(\beta^{*\mathsf T}X)>0.
   \]

3. Отсутствие другого неколлинеарного направления \(b\), для которого существует функция \(\widetilde f\) такая, что

   \[
   f(\beta^{*\mathsf T}X)
   =
   \widetilde f(b^{\mathsf T}X)
   \quad\text{почти наверное}.
   \]

4. Достаточная локальная изменчивость признаков. Агрегированная матрица внутренней задачи должна иметь ненулевую кривизну в направлениях, ортогональных истинной прямой.

5. Для выбранных центров должно существовать достаточно наблюдений с ненулевыми весами.

## 1.4. Желательные, но не обязательные свойства

К желательным свойствам относятся:

- гладкость \(f\) на области концентрации индекса;
- ограниченная или субгауссовская плотность признаков;
- конечные моменты шума и признаков;
- умеренное число обусловленности \(\operatorname{Cov}(X)\);
- отсутствие сильных выбросов;
- однородная плотность данных около центров;
- сопоставимые масштабы координат;
- гомоскедастичность.

Бенчмарк должен последовательно нарушать каждое из этих свойств.

## 1.5. Локальные веса

Для центра \(x_j\), \(j=1,\ldots,J\), задаётся

\[
w_{ij}^{(k)}
=
K(q_{ij}^{(k)}).
\]

На нулевом шаге:

\[
q_{ij}^{(0)}
=
\frac{\|X_i-x_j\|_2^2}{h_0^2}.
\]

На внешнем шаге \(k\geq1\):

\[
q_{ij}^{(k)}
=
\frac{
\rho_k^2\|X_i-x_j\|_2^2
+
\langle X_i-x_j,\beta_{k-1}\rangle^2
}{h_k^2}.
\]

Эквивалентно:

\[
q_{ij}^{(k)}
=
\|T_k(X_i-x_j)\|_2^2,
\]

\[
T_k^2
=
h_k^{-2}
\left(
\rho_k^2I_d+
\beta_{k-1}\beta_{k-1}^{\mathsf T}
\right).
\]

При уменьшении \(\rho_k\) локальная окрестность расширяется в направлениях, ортогональных \(\beta_{k-1}\), относительно направления \(\beta_{k-1}\).

## 1.6. Локальное среднее и направленные статистики

Локальная масса:

\[
N_j
=
\sum_{i=1}^{n}w_{ij}.
\]

Нормированное локальное среднее:

\[
\bar X_j
=
N_j^{-1}
\sum_{i=1}^{n}w_{ij}X_i.
\]

Для каждого центра генерируется набор единичных направлений

\[
\Phi_j
=
\{\phi_{j1},\ldots,\phi_{jP}\},
\qquad
\|\phi_{jp}\|_2=1.
\]

Для \(p=1,\ldots,P\):

\[
I_{jp}
=
\sum_{i=1}^{n}
Y_i
\langle X_i-\bar X_j,\phi_{jp}\rangle
w_{ij},
\]

\[
U_{jp}
=
\sum_{i=1}^{n}
(X_i-\bar X_j)
\langle X_i-\bar X_j,\phi_{jp}\rangle
w_{ij}
\in\mathbb R^d.
\]

Обозначим

\[
I_j=(I_{j1},\ldots,I_{jP})^{\mathsf T}
\in\mathbb R^P,
\]

\[
U_j=
\begin{pmatrix}
U_{j1}^{\mathsf T}\\
\vdots\\
U_{jP}^{\mathsf T}
\end{pmatrix}
\in\mathbb R^{P\times d}.
\]

При локальной линейной аппроксимации

\[
f(\beta^{\mathsf T}X_i)
\approx
c_j+\ell_j\beta^{\mathsf T}(X_i-x_j)
\]

центрирование удаляет \(c_j\), и получается

\[
\mathbb E[I_j]
\approx
\ell_jU_j\beta.
\]

## 1.7. Внутренняя задача ALS

При фиксированных весах, центрах и направлениях решается

\[
F(\beta,\ell)
=
\sum_{j=1}^{J}
\|I_j-\ell_jU_j\beta\|_2^2
+
\lambda\|\beta-\beta_{\mathrm{prior}}\|_2^2.
\]

При фиксированном \(\beta\):

\[
\widehat\ell_j
=
\frac{
\langle I_j,U_j\beta\rangle
}{
\|U_j\beta\|_2^2+\eta_\ell
},
\]

где \(\eta_\ell\geq0\) — локальная ridge-поправка.

При фиксированных \(\ell_j\):

\[
A\beta=b,
\]

\[
A
=
\sum_{j=1}^{J}
\ell_j^2U_j^{\mathsf T}U_j
+
(\lambda+\eta_\beta)I_d,
\]

\[
b
=
\sum_{j=1}^{J}
\ell_jU_j^{\mathsf T}I_j
+
\lambda\beta_{\mathrm{prior}}.
\]

Система решается текущим matrix-free CG.

После обновления выполняется нормировка:

\[
s=\|\beta\|_2,
\qquad
\beta\leftarrow\beta/s,
\qquad
\ell_j\leftarrow s\ell_j.
\]

Она сохраняет произведение \(\ell_j\beta\).

## 1.8. Внешняя адаптация

Шаг 0:

1. выбираются центры;
2. определяется \(h_0\) по условию локальной массы;
3. направления генерируются равномерно на сфере;
4. вычисляются \(I_j,U_j\);
5. внутренняя задача решается ALS+CG;
6. результат нормируется.

Шаг \(k\geq1\):

1. обновляется bandwidth

   \[
   h_k=h_{k-1}/a,
   \qquad a>1;
   \]

2. выбирается \(\rho_k\in[0,1]\), сохраняющее требуемую локальную массу;
3. строится \(T_k\);
4. при предусмотренном расписании обновляются направления;
5. пересчитываются веса и локальные статистики;
6. внутренняя задача решается с prior \(\beta_{k-1}\);
7. проверяется внешний критерий остановки.

## 1.9. Настраиваемые параметры

К настраиваемым параметрам относятся:

- \(J\) — число центров;
- \(P=n_\Phi\) — число направлений на центр;
- \(n_{\min}\) — требуемая локальная масса;
- \(K\) или минимальный bandwidth — число внешних шагов;
- \(a\) — скорость уменьшения \(h_k\);
- \(\lambda\) — штраф к предыдущему направлению;
- частота обновления направлений;
- формула и параметры ядра;
- способ выбора центров.

Параметры оптимизатора:

- \(M_{\max}\) — максимальное число ALS-шагов;
- \(\tau_{\mathrm{ALS}}\) — tolerance ALS;
- \(R_{\mathrm{CG,max}}\) — лимит CG;
- \(\tau_{\mathrm{CG}}\) — tolerance CG;
- \(\eta_\ell,\eta_\beta\) — ridge-поправки;
- preconditioner текущего CG, если он уже входит в реализацию.

## 1.10. Вычислительная сложность

Обозначим:

- \(J\) — число центров;
- \(P\) — число направлений;
- \(K\) — число внешних шагов;
- \(M\) — среднее число ALS-шагов;
- \(R_{\mathrm{CG}}\) — среднее число CG-итераций.

Прямое вычисление локальных статистик имеет порядок

\[
O(JnPd).
\]

Одно matrix-free умножение CG:

\[
O(JPd).
\]

Полная оценка времени:

\[
O\left(
KJnPd
+
KMR_{\mathrm{CG}}JPd
\right).
\]

Память при хранении всех \(U_j\):

\[
O(JPd).
\]

При хранении матрицы расстояний или весов добавляется

\[
O(Jn).
\]

Эти оценки требуется проверить эмпирически, поскольку фактическая стоимость зависит от chunking, BLAS и числа CG-итераций.

## 1.11. Теоретические гарантии и их граница

**[Л]** Для классического single-index ADE и structure-adaptive direct estimator известны результаты о состоятельности и \(\sqrt n\)-скорости при регулярных условиях.

**[М]** Эти гарантии не переносятся автоматически на текущую реализацию, поскольку она использует:

- конечный случайный sketch из \(P\) направлений;
- конкретное правило выбора \(h_k\) и \(\rho_k\);
- пересэмплирование направлений;
- приближённый CG;
- конечное число ALS- и внешних итераций;
- конкретную регуляризацию к предыдущей оценке.

**[Г]** Для текущей реализации состоятельность и скорость улучшения по \(n\) должны быть проверены экспериментально.

---

# 2. Анализ научной области

## 2.1. Решаемая задача

Алгоритм решает semiparametric single-index regression и одномерное sufficient mean dimension reduction:

\[
\mathbb E[Y\mid X]
=
f(\beta^{*\mathsf T}X).
\]

В отличие от полной непараметрической регрессии в \(\mathbb R^d\), оценивается конечномерное направление и одномерная функция связи.

## 2.2. Известные методы той же или близкой задачи

| Класс | Методы | Используемая информация | Основной режим отказа |
|---|---|---|---|
| Линейные | OLS | \(\operatorname{Cov}(X,Y)\) | симметричная связь даёт нулевой первый момент |
| Латентный score | PLS1 | covariance с откликом | нелинейность и выбросы |
| Average derivative | ADE, density-weighted ADE | средний градиент регрессии | многомерное сглаживание и слабая производная |
| Локальные градиенты | OPG | внешние произведения локальных градиентов | стоимость и нестабильные локальные fit |
| Локальная single-index регрессия | MAVE, RMAVE | локальная линейная ошибка | bandwidth и начальное направление |
| Обратная регрессия | SIR | \(\mathbb E[X\mid Y]\) | симметричные зависимости и linearity condition |
| Вторые обратные моменты | SAVE | \(\operatorname{Cov}(X\mid Y)\) | чувствительность к оценке ковариаций |
| Вторые производные | PHD | усреднённая Hessian-структура | линейная связь и тяжёлые хвосты |
| Semiparametric least squares | Ichimura SLS | одномерный nonparametric fit для каждого направления | высокая стоимость tuning |
| Structure-adaptive ADP | текущий метод | локальные направленные производные | local mass, sketch, инициализация, адаптация |

Методы, отличающиеся от текущего ADP только способом оптимизации одной и той же целевой функции, в этот бенчмарк не включаются.

## 2.3. Типичные практические задачи

Single-index модель применима, когда отклик в основном зависит от одного линейного score, но форма зависимости неизвестна. Практические классы задач:

- физические характеристики материалов;
- риск-score и индексы состояния;
- спрос и экономические показатели;
- технологические параметры производства;
- интерпретируемое снижение размерности перед одномерным smoother.

На реальных данных истинная \(\beta^*\) неизвестна. Поэтому проверяются prediction, bootstrap-стабильность и переносимость на отложенные данные.

## 2.4. Простые, средние и трудные режимы

Эти определения задают операционную шкалу бенчмарка.

| Режим | Условия |
|---|---|
| Простой | \(d\leq20\), \(n/d\geq50\), SNR \(\geq20\), \(\kappa(\Sigma)\leq3\), гладкая монотонная \(f\), нет выбросов |
| Средний | \(20<d\leq200\), \(10\leq n/d<50\), SNR от 5 до 20, \(\kappa(\Sigma)\leq100\), гладкая немонотонная \(f\) |
| Трудный | \(d>200\), \(n/d<10\), SNR \(<5\), \(\kappa(\Sigma)>100\), тяжёлые хвосты, выбросы или негладкая \(f\) |

## 2.5. Типичные ошибки реализации

1. Отсутствие деления локального среднего на \(N_j\).
2. Перепутанные \(h\) и \(h^2\), \(\rho\) и \(\rho^2\).
3. Неверная монотонность binary search по \(h\) или \(\rho\).
4. Неправильный выбор минимального или максимального допустимого \(\rho\).
5. Потеря веса в \(I_{jp}\) или \(U_{jp}\).
6. Перепутанный порядок осей \(J,P,d\).
7. Ошибка транспонирования в \(U_j^{\mathsf T}U_j\).
8. Неправильный перенос масштаба между \(\beta\) и \(\ell_j\).
9. Сравнение направлений без модуля скалярного произведения.
10. Абсолютный stopping criterion, зависящий от масштаба \(Y\).
11. Игнорирование `CG info > 0`.
12. Ненулевой или отрицательный denominator без обработки.
13. Подмена корреляции координат общим случайным сдвигом наблюдений.
14. Использование одного seed для данных, центров, направлений и инициализации.
15. Выбор гиперпараметров по тестовой выборке.
16. Исключение failed runs из итоговых таблиц.

## 2.6. Вопросы, которые нельзя проверить одним экспериментом

- достаточность \(P\) при росте \(d\);
- вклад числа центров отдельно от числа наблюдений;
- область притяжения при разных функциях связи;
- статистический предел отдельно от sketch-ошибки;
- влияние локальной массы отдельно от bandwidth schedule;
- переносимость synthetic tuning на реальные данные;
- соответствие empirical scaling теоретической сложности.

---

# 3. Карта параметров

## 3.1. Разделение параметров

Параметры не смешиваются между группами.

- **Параметры данных** описывают наблюдаемое распределение и загрязнения.
- **Параметры модели** описывают \(\beta^*\), \(f\) и отклонение от single-index модели.
- **Параметры алгоритма** определяют локализацию, sketch и внешний цикл.
- **Параметры оптимизатора** относятся только к текущим ALS и CG.
- **Параметры реализации** определяют вычислительный backend.

## 3.2. Базовый сценарий B0

Если в конкретном эксперименте параметр не указан, используется B0.

### Данные

\[
n=1000,
\qquad
d=20,
\qquad
X\sim N(0,I_d).
\]

### Истинное направление

Плотное направление:

\[
\widetilde\beta_r\sim N(0,1),
\qquad
\beta^*=\widetilde\beta/\|\widetilde\beta\|_2.
\]

Одно \(\beta^*\) фиксируется для всех повторов B0, а в отдельном тесте оценивается вариабельность по направлениям.

### Функция связи

\[
f(z)=\tanh(1.5z).
\]

Перед добавлением шума signal центрируется и нормируется:

\[
\mu_i=f(\beta^{*\mathsf T}X_i),
\]

\[
\widetilde\mu_i
=
\frac{\mu_i-\bar\mu}{s_\mu}.
\]

Это позволяет задавать шум через SNR независимо от масштаба функции.

### Шум

\[
Y_i=\widetilde\mu_i+\sigma_\varepsilon\varepsilon_i,
\qquad
\varepsilon_i\sim N(0,1),
\]

\[
\mathrm{SNR}
=
\frac{\operatorname{sd}(\widetilde\mu)}{\sigma_\varepsilon}
=10.
\]

Следовательно, \(\sigma_\varepsilon=0.1\).

### Параметры алгоритма

\[
J=200,
\qquad
P=32,
\qquad
n_{\min}=64.
\]

\[
a=\sqrt2,
\qquad
K_{\max}=8.
\]

\[
\lambda=0.01\,n_{\min}=0.64.
\]

Центры выбираются без возвращения из training-выборки. Perturbation центров в B0 отсутствует.

Шаг 0 использует равномерные направления на сфере:

\[
\phi=g/\|g\|_2,
\qquad
g\sim N(0,I_d).
\]

На следующих шагах применяется текущая формула адаптивного распределения направлений из реализации. Если код обновляет направления не на каждом шаге, это расписание сохраняется.

### Параметры оптимизатора

\[
M_{\max}=20,
\qquad
\tau_{\mathrm{ALS}}=10^{-6}.
\]

\[
\tau_{\mathrm{CG}}=10^{-8},
\qquad
R_{\mathrm{CG,max}}=\max(100,5d).
\]

\[
\eta_\ell=10^{-12},
\qquad
\eta_\beta=10^{-10}.
\]

Основные запуски выполняются в float64.

## 3.3. Полная карта

| Параметр | Тип | Смысл | Рабочий диапазон | Риск малого значения | Риск большого значения |
|---|---|---|---|---|---|
| \(n\) | параметр данных | размер выборки | 250–8000 | высокая sampling variance | время и память |
| \(d\) | параметр данных | размерность признаков | 5–1000 | слишком простой тест | недостаток данных и sketch |
| \(n/d\) | параметр данных | обеспеченность наблюдениями | 2–100 | статистическая деградация | высокая стоимость |
| SNR | параметр данных | signal-to-noise ratio | 0.5–\(\infty\) | шум доминирует | сценарий становится тривиальным |
| \(\nu\) | параметр данных | степени свободы Student noise | 2.5–\(\infty\) | тяжёлые хвосты | Gaussian предел |
| \(\epsilon_Y\) | параметр данных | доля выбросов отклика | 0–0.20 | нет загрязнения | модель загрязнения доминирует |
| \(\epsilon_X\) | параметр данных | доля leverage points | 0–0.20 | нет загрязнения | геометрия локальных окрестностей разрушается |
| \(\rho_X\) | параметр данных | корреляция координат | 0–0.98 | независимый дизайн | почти вырожденная covariance |
| \(\kappa(\Sigma)\) | параметр данных | число обусловленности | 1–\(10^4\) | изотропный дизайн | численная и статистическая нестабильность |
| \(r_{\mathrm{scale}}\) | параметр данных | отношение масштабов координат | 1–1000 | одинаковые масштабы | dominance крупных координат |
| \(\alpha_{\mathrm{het}}\) | параметр данных | гетероскедастичность | 0–2 | гомоскедастичность | локальная variance доминирует |
| \(\beta^*\) | параметр модели | истинное направление | support 1–\(d\) | осевая простота | плотное направление |
| \(f\) | параметр модели | функция связи | linear, tanh, sine, quadratic, kink | слабая нелинейность | oscillation или негладкость |
| \(\omega\) | параметр модели | частота oscillating link | 0.5–8 | почти линейный участок | слишком малый масштаб локальной линейности |
| \(\delta_{\mathrm{mis}}\) | параметр модели | нарушение single-index | 0–1 | корректная модель | один индекс перестаёт быть достаточным |
| \(J\) | параметр алгоритма | число центров | 0.05\(n\)–\(n\) | нестабильное усреднение | линейный рост стоимости |
| \(P\) | параметр алгоритма | направления на центр | 2–128 | sketch не покрывает пространство | линейный рост времени и памяти |
| \(n_{\min}\) | параметр алгоритма | минимальная локальная масса | 8–256 | singular и шумные локальные задачи | oversmoothing |
| \(h_0\) | параметр алгоритма | начальная ширина | определяется массой | пустые окрестности | высокий bias |
| \(a\) | параметр алгоритма | скорость уменьшения \(h_k\) | 1.05–2 | адаптация медленная | резкая потеря массы |
| \(\rho_k\) | параметр алгоритма | анизотропия | 0–1 | сильная зависимость от текущей \(\beta\) | почти изотропные веса |
| \(\lambda\) | параметр алгоритма | штраф к prior | 0–10 в relative scale | seed sensitivity | прилипание к prior |
| renewal period | параметр алгоритма | частота новых направлений | never–every step | фиксированная sketch error | лишняя stochastic variance |
| center perturbation | параметр алгоритма | шум центров | 0–1 local scale | центры совпадают с observations | центры вне плотных областей |
| \(M_{\max}\) | параметр оптимизатора | ALS-итерации | 1–32 | недорешённая задача | лишнее время |
| \(\tau_{\mathrm{ALS}}\) | параметр оптимизатора | stopping tolerance | \(10^{-3}\)–\(10^{-9}\) | ранний stop | лишние итерации |
| \(\tau_{\mathrm{CG}}\) | параметр оптимизатора | CG tolerance | \(10^{-3}\)–\(10^{-10}\) | неточный beta-step | лишние matvec |
| \(R_{\mathrm{CG,max}}\) | параметр оптимизатора | лимит CG | 25–10\(d\) | `info > 0` | лишнее время |
| \(\eta_\ell\) | параметр оптимизатора | ridge slope-step | 0–\(10^{-4}\) relative | division by small value | bias slopes |
| \(\eta_\beta\) | параметр оптимизатора | ridge beta-step | 0–\(10^{-4}\) relative | ill-conditioning | bias direction |
| dtype | параметр реализации | числовая точность | float32/float64 | rounding | удвоенная память |
| chunk size | параметр реализации | блок центров | 8–512 | overhead | OOM |
| processes | параметр реализации | process parallelism | 1–physical cores | CPU простаивает | oversubscription |
| BLAS threads | параметр реализации | внутренние потоки BLAS | 1–physical cores | недоиспользование | конкуренция с process pool |
| data seed | параметр реализации | генерация данных | независимый поток | смешение источников variance | отсутствует |
| fit seed | параметр реализации | центры, directions, init | независимый поток | смешение источников variance | отсутствует |

---

# 4. Матрица экспериментов

В таблице приведены основные сценарии. Подробные спецификации находятся в разделе 5.

| ID | Гипотеза | \(n\) | \(d\) | Изменяемый фактор | Уровни | Повторы | Primary metric | Критерий успеха |
|---|---|---:|---:|---|---|---:|---|---|
| C01 | локальные статистики совпадают с прямыми суммами | 101 | 3 | backend формул | direct/current | 1 | relative error | \(<10^{-12}\) |
| C02 | weighted centering удаляет константу | 401 | 1 | shift \(Y\) | 7 уровней | 1 | residual/gap | \(<10^{-12}\) |
| C03 | точная linear model восстанавливается | 729 | 3 | отсутствует | fixed | 1 | \(1-|\cos|\) | \(<10^{-10}\) |
| C04 | результат эквивариантен к вращению | 729 | 3 | orthogonal transform | 20 | 20 | projective gap | \(<10^{-9}\) |
| C05 | корректно масштабирование \(X,h\) | 729 | 3 | scale \(X\) | \(10^{-3}\)–\(10^3\) | 7 | weight/beta gap | tolerance |
| C06 | shift и scale \(Y\) не меняют прямую при \(\lambda=0\) | 729 | 3 | \(a+bY\) | 15 | 15 | projective gap | \(<10^{-8}\) |
| C07 | массы монотонны по \(h,\rho\) | 300 | 5 | random pairs | 1000 | 1 | violations | 0 |
| C08 | ALS восстанавливает noiseless synthetic solution | synthetic | 10 | initialization | 20 | 20 | \(1-|\cos|\) | \(<10^{-10}\) |
| C09 | CG совпадает с dense reference | synthetic | 5–50 | system | 4 sizes | 50 | residual/gap | \(<10^{-9}\) |
| C10 | objective не растёт после полного ALS cycle | synthetic | 10 | iteration | 20 | 20 | increase count | 0 above tolerance |
| C11 | stopping не зависит от масштаба objective | 1000 | 20 | scale \(Y\) | \(10^{-4}\)–\(10^4\) | 20 | final direction gap | \(<10^{-5}\) |
| C12 | dtype/chunk/threads не меняют математику | 1000 | 20 | implementation | grid | 20 | projective gap | predefined |
| S01 | качество зависит от link, но сохраняется при smooth links | 1000 | 20 | \(f\) | 6 links | 50 | angular loss | expected ordering |
| S02 | ошибка уменьшается с \(n\) при фиксированном алгоритме | 250–4000 | 20 | \(n\) | 5 | 50 | median loss | negative slope |
| S03 | практический pipeline улучшается с \(n\) при \(J\propto n\) | 250–4000 | 20 | \(n\) | 5 | 50 | median loss/time | negative slope |
| S04 | ошибка растёт при снижении SNR | 1000 | 20 | SNR | 8 | 50 | success rate | monotone degradation |
| S05 | при фиксированном \(n/d\) есть dimension boundary | variable | 10–1000 | \(d,n/d\) | 7×5 | 50 | success rate | boundary measured |
| S06 | nuisance dimensions отличаются от роста signal support | variable | 10–1000 | \(d,s\) | focused grid | 50 | angular loss | mechanisms separated |
| T01 | существует plateau по \(P\) | 1000 | 20 | \(P\) | 2–128 | 50 | loss/time | smallest plateau |
| T02 | требуемое \(P\) растёт с \(d\) | variable | 20–500 | \(P,d\) | focused grid | 50 | success rate | interaction measured |
| T03 | существует plateau по \(J\) | 1000 | 20 | \(J\) | 50–1000 | 50 | loss/time | smallest plateau |
| T04 | \(n_{\min}\) даёт bias-variance trade-off | 1000 | 20 | \(n_{\min}\) | 8–256 | 50 | loss/failure | stable interval |
| T05 | \(\lambda\) стабилизирует, но создаёт bias к prior | 1000 | 20 | \(\lambda\), init cosine | focused grid | 50 | loss/init sensitivity | Pareto interval |
| T06 | слишком быстрое сжатие bandwidth разрушает массу | 1000 | 20 | \(a\) | 1.05–2 | 50 | loss/q05 mass | safe interval |
| T07 | существует достаточная точность ALS+CG | 1000 | 20 | tolerances/iterations | grids | 50 | gap to strict run | \(\leq0.01\) |
| T08 | renewal directions полезен только в части режимов | 1000 | 20 | renewal period | 5 | 50 | paired loss | effect localized |
| T09 | правило local mass меняет weak-center failures | 1000 | 20 | mean/quantiles | 5 | 50 | failure/loss | preferred policy |
| T10 | perturbation центров имеет ограниченный рабочий диапазон | 1000 | 20 | center noise | 7 | 50 | loss/mass | interval found |
| R01 | Gaussian noise degradation закономерна | 1000 | 20 | SNR | 8 | 100 | success rate | breakdown point |
| R02 | Student noise увеличивает tails of error | 1000 | 20 | \(\nu\) | 5 | 100 | median/IQR | degradation point |
| R03 | response outliers вызывают breakdown | 1000 | 20 | contamination | 6 | 100 | success rate | breakdown point |
| R04 | leverage points сильнее response outliers | 1000 | 20 | \(X\)-outliers | 3 types×6 | 100 | paired loss | mechanisms separated |
| R05 | heteroskedasticity увеличивает variance | 1000 | 20 | \(\alpha_{het}\) | 5 | 100 | loss/IQR | degradation point |
| R06 | AR correlation ухудшает conditioning | 1000 | 20 | \(\rho_X\) | 8 | 100 | loss/CG | degradation point |
| R07 | near-singular covariance имеет статистический и численный эффекты | 1000 | 20 | \(\kappa\) | 8 | 100 | loss/residual | causes separated |
| R08 | coordinate scales проверяют preprocessing sensitivity | 1000 | 20 | scale ratio | 6 | 100 | loss | raw/standardized split |
| R09 | nonlinear feature dependence нарушает Gaussian geometry | 1000 | 20 | dependence strength | 6 | 100 | loss/mass | boundary |
| R10 | sparse и dense \(\beta^*\) ведут себя различно | 1000 | 100 | support | 6 | 100 | loss | support effect |
| R11 | oscillating link уменьшает local linearity scale | 1000 | 20 | \(\omega\) | 5 | 100 | loss | frequency boundary |
| R12 | loss of smoothness ухудшает local derivative fit | 1000 | 20 | smoothness | 6 | 100 | loss | smoothness boundary |
| R13 | nonuniform density создаёт weak centers | 1000 | 20 | mixture separation | 5 | 100 | loss/q05 mass | mass-linked decline |
| R14 | малое число точек в окрестностях вызывает local failure | 1000 | 20 | forced local sparsity | 6 | 100 | invalid centers | threshold |
| R15 | misspecified two-index signal создаёт irreducible rank-one bias | 2000 | 20 | \(\delta_{mis}\) | 7 | 100 | dominant direction loss | bias curve |
| M01 | runtime scaling по \(n\) соответствует модели | 250–8000 | 20 | \(n\) | 6 | 20 | time | exponent CI |
| M02 | runtime scaling по \(d\) соответствует модели | 4000 | 10–1000 | \(d\) | 7 | 20 | time | exponent CI |
| M03 | runtime линейно зависит от \(J\) вне overhead | 2000 | 50 | \(J\) | 6 | 20 | time | exponent near 1 |
| M04 | runtime линейно зависит от \(P\) вне overhead | 2000 | 50 | \(P\) | 6 | 20 | time | exponent near 1 |
| M05 | outer/inner iterations имеют отдельную стоимость | 2000 | 50 | \(K,M\) | grids | 20 | component time | exponents |
| M06 | memory соответствует \(Jn+JPd\) | grids | grids | size | grid | 20 | peak RSS | fitted model |
| M07 | parallel speedup насыщается | 4000 | 100 | workers | 1–cores | 20 | speedup/efficiency | saturation found |
| M08 | chunking меняет память, но не результат | 4000 | 100 | chunk | 8–512 | 20 | memory/gap | gap tolerance |
| I01 | существует область притяжения | 1000 | 20 | initial cosine | 0–0.9 | 100 | success rate | basin boundary |
| I02 | область притяжения зависит от link | 1000 | 20 | cosine/link | 6×5 | 100 | success rate | boundaries |
| I03 | practical initializers отличаются по basin entry | 1000 | 20 | initializer | 7 | 100 | final loss/iterations | paired comparison |
| I04 | seed variance разделяется на data и algorithm variance | 1000 | 20 | seed component | nested | 100 | variance components | decomposition |
| B01 | ADP сравнивается с statistical baselines | representative | variable | method | 8–9 | 50–100 | loss/time | paired CI |
| A01 | outer adaptation даёт причинный вклад | representative | variable | full/step0 | 2 | 100 | paired delta | CI excludes 0 |
| A02 | anisotropy даёт причинный вклад | representative | variable | \(\rho=1\)/full | 2 | 100 | paired delta | CI excludes 0 |
| A03 | bandwidth decay даёт причинный вклад | representative | variable | \(a=1\)/full | 2 | 100 | paired delta | CI excludes 0 |
| A04 | random sketch имеет измеримую цену | small/medium | variable | random/full basis | 2 | 100 | paired delta | sketch gap |
| A05 | renewal directions изменяет sketch variance | representative | variable | fixed/renewed | 2 | 100 | paired delta | scenario-specific |
| A06 | regularization current component stabilizes path | representative | variable | \(\lambda=0\)/full | 2 | 100 | paired delta | stability effect |
| A07 | mass quantile защищает weak centers | nonuniform | 20 | mean/q05 | 2 | 100 | failure/loss | lower failures |
| A08 | center perturbation полезна или вредна | representative | variable | 0/current | 2 | 100 | paired delta | effect localized |
| A09 | negative controls обнаруживаются | exact tasks | small | broken formula | controls | 1–20 | unit metrics | must fail |
| D01 | Airfoil допускает стабильный one-index score | real | 5 | method | baselines | nested CV | RMSE/stability | predefined |
| D02 | Concrete допускает стабильный one-index score | real | 8 | method | baselines | nested CV | RMSE/stability | predefined |
| D03 | Wine Quality допускает переносимый score | real | 11 | method | baselines | nested CV | RMSE/MAE | predefined |
| D04 | Superconductivity проверяет medium-d scaling | real | 81 | method | baselines | nested CV | RMSE/time | predefined |

---

# 5. Подробное описание сценариев

## 5.1. Общий статистический протокол

### Разделение случайности

Для каждого запуска используются независимые seed:

\[
s_{\mathrm{data}},
\quad
s_{\beta},
\quad
s_{\mathrm{centers}},
\quad
s_{\mathrm{directions}},
\quad
s_{\mathrm{init}}.
\]

При paired comparison все компоненты seed, кроме специально изменяемого, совпадают.

### Разбиение данных

Synthetic recovery не требует test set для метрики направления. Если одновременно оценивается prediction:

- 60% training;
- 20% validation;
- 20% test.

Все параметры ADP и baseline выбираются только по validation. Test открывается один раз после фиксации конфигурации.

### Primary metric

\[
c_\beta
=
|\widehat\beta^{\mathsf T}\beta^*|.
\]

\[
L_\beta
=
1-c_\beta.
\]

Дополнительные метрики:

\[
\theta_\beta
=
\arccos(c_\beta),
\]

\[
d_{\pm}
=
\min\{\|\widehat\beta-\beta^*\|_2,
\|\widehat\beta+\beta^*\|_2\}
=
\sqrt{2(1-c_\beta)}.
\]

### Категории результата

| Категория | Критерий |
|---|---|
| Точное восстановление | \(c_\beta\geq0.99\) |
| Успешное восстановление | \(c_\beta\geq0.90\) |
| Частичное восстановление | \(0.80\leq c_\beta<0.90\) |
| Деградация | median \(c_\beta<0.80\) |
| Breakdown | \(P(c_\beta\geq0.80)<0.80\) |
| Failed run | exception, NaN, zero norm, CG nonconvergence, invalid local mass или нарушение CSV-контракта серии |

Порог 0.80 представляет практически значимый порог бенчмарка, а не теоретическую константу.

### Число повторов

Pilot:

\[
R=20.
\]

Стандартное сравнение:

\[
R=50.
\]

Robustness и initialization:

\[
R=100.
\]

Редкие отказы:

\[
R\geq200.
\]

После pilot число повторов уточняется по формуле

\[
R
\geq
\left(
\frac{1.96s}{\epsilon}
\right)^2,
\]

где \(s\) — pilot standard deviation метрики, а \(\epsilon\) — требуемая половина 95% CI.

### Обязательная статистика отчёта

Для каждого сценария выводятся:

- median;
- mean как вторичная статистика;
- IQR;
- q05 и q95;
- bootstrap 95% CI median;
- success rate;
- failure rate;
- худшие пять запусков;
- полное распределение primary metric;
- причины всех failed runs.

## 5.2. Уровень 1. Проверка математической корректности

### C01. Прямое вычисление \(I_j,U_j\)

| Поле | Спецификация |
|---|---|
| Гипотеза | **[М]** Векторизованная реализация совпадает с определяющими суммами |
| Генерация | \(n=101,d=3,J=7,P=5\); фиксированные float64 \(X,Y,w,\phi\) |
| Изменяемый фактор | direct loops против текущего backend |
| Фиксируемые факторы | все массивы и порядок сумм |
| Повторы | 1 deterministic dataset |
| Primary metric | relative Frobenius error \(I,U\) |
| Вторичные | ошибка \(N_j,\bar X_j\), identity \(U_{jp}=C_j\phi_{jp}\) |
| Ожидаемый результат | ошибки порядка machine precision |
| Критерий успеха | relative error \(<10^{-12}\) в float64 |
| Критерий неудачи | любое превышение или несовпадение осей |
| Альтернативное объяснение | различный порядок summation; повторить с long double/reference summation |
| Разрешённый вывод | формулы локальных статистик реализованы корректно |
| Неразрешённый вывод | алгоритм статистически корректен |

### C02. Удаление константной части

| Поле | Спецификация |
|---|---|
| Гипотеза | Нормированное weighted centering делает статистики инвариантными к добавлению константы к \(Y\) при \(\lambda=0\) |
| Генерация | \(d=1,n=401,X_i\) равномерная сетка на \([-2,2]\), \(Y=2.5X\) |
| Фактор | shift \(a\in\{-100,-10,-1,0,1,10,100\}\) в \(Y'=Y+a\) |
| Фиксируемые | центры, веса, направления, solver |
| Повторы | 1 |
| Primary | projective direction gap и residual |
| Вторичные | разность \(I_j(Y+a)-I_j(Y)\) |
| Ожидание | направление и \(I_j\) не меняются |
| Успех | gap \(<10^{-12}\), relative residual \(<10^{-12}\) |
| Неудача | систематическая зависимость от \(a\) |
| Альтернативное объяснение | boundary asymmetry; повторить только на внутренних центрах |
| Разрешённый вывод | локальное среднее и центрирование корректны |
| Неразрешённый вывод | корректность анизотропной части |

### C03. Точная трёхмерная linear model

| Поле | Спецификация |
|---|---|
| Гипотеза | В отсутствии шума ADP восстанавливает точное направление |
| Генерация | сетка \(9^3=729\), \(\beta^*=(2,-1,2)/3\), \(Y=-0.4+1.8\beta^{*T}X\) |
| Фактор | отсутствует |
| Фиксируемые | \(J=27\) внутренних центров, deterministic directions, \(\lambda=0\) |
| Повторы | 1 |
| Primary | \(1-|\widehat\beta^T\beta^*|\) |
| Вторичные | local slope error, objective, rank/condition |
| Ожидание | точное направление, slopes 1.8 |
| Успех | direction loss \(<10^{-10}\), relative objective \(<10^{-18}\) |
| Неудача | любое существенное отклонение при full rank |
| Альтернативное объяснение | rank deficiency; исключается eigenvalue check |
| Разрешённый вывод | базовая формула ALS и статистик согласована |
| Неразрешённый вывод | устойчивость к шуму |

### C04. Ортогональная эквивариантность

| Поле | Спецификация |
|---|---|
| Гипотеза | Поворот признаков и всех геометрических объектов поворачивает оценку тем же образом |
| Генерация | данные C03 |
| Фактор | 20 Haar-like orthogonal matrices \(Q\) |
| Фиксируемые | transformed centers и directions: \(X'=XQ^T,\phi'=Q\phi\) |
| Повторы | 20 |
| Primary | \(1-|\widehat\beta'^TQ\widehat\beta|\) |
| Вторичные | equality weights/objective |
| Ожидание | machine-level agreement |
| Успех | gap \(<10^{-9}\) |
| Неудача | coordinate-dependent result |
| Альтернативное объяснение | non-deterministic reduction; повторить single-thread |
| Разрешённый вывод | нет ошибки осей или транспонирования |
| Неразрешённый вывод | устойчивость при correlated design |

### C05. Масштабирование признаков

| Поле | Спецификация |
|---|---|
| Гипотеза | При \(X'=sX,x'=sx,h'=sh\) веса и направление неизменны |
| Генерация | C03 |
| Фактор | \(s\in\{10^{-3},10^{-2},10^{-1},1,10,10^2,10^3\}\) |
| Фиксируемые | rescaled centers и bandwidth |
| Повторы | 7 |
| Primary | max weight error и projective gap |
| Вторичные | \(h_0'/h_0\) |
| Ожидание | \(h_0'=sh_0\), одинаковые веса |
| Успех | weight error \(<10^{-12}\), beta gap \(<10^{-8}\) |
| Неудача | масштабная зависимость сверх rounding |
| Альтернативное объяснение | binary-search tolerance absolute; проверить relative tolerance |
| Разрешённый вывод | scale handling формул корректен |
| Неразрешённый вывод | standardization не нужна на реальных данных |

### C06. Сдвиг и масштаб отклика

| Поле | Спецификация |
|---|---|
| Гипотеза | При \(Y'=a+cY,c\neq0\) и \(\lambda=0\) оценивается та же прямая |
| Генерация | C03 |
| Фактор | \(a\in\{-10,0,10\},c\in\{0.01,0.1,1,10,100\}\) |
| Фиксируемые | data geometry, directions, relative stopping |
| Повторы | 15 |
| Primary | projective gap |
| Вторичные | slope scale ratio, iterations |
| Ожидание | same direction; slopes scale by \(c\) |
| Успех | gap \(<10^{-8}\) |
| Неудача | dependence on \(a\) or \(|c|\) |
| Альтернативное объяснение | absolute tolerance; сравнить strict solve |
| Разрешённый вывод | корректно центрирование и relative stopping |
| Неразрешённый вывод | regularized path invariant при \(\lambda>0\) |

### C07. Монотонность массы и binary search

| Поле | Спецификация |
|---|---|
| Гипотеза | Для убывающего \(K\) масса не убывает по \(h\) и не возрастает по \(\rho\) |
| Генерация | \(n=300,d=5,J=30\), Gaussian data |
| Фактор | 1000 random pairs \(h_1<h_2\), \(\rho_1<\rho_2\) |
| Фиксируемые | \(X,centers,\beta\) |
| Повторы | 1 dataset |
| Primary | число monotonicity violations |
| Вторичные | feasibility gap выбранных \(h_0,\rho_k\) |
| Ожидание | ноль нарушений |
| Успех | no violation \(>10^{-12}\); search constraint выполнен |
| Неудача | противоположная монотонность или infeasible result |
| Альтернативное объяснение | non-monotone custom kernel; тогда это свойство не применимо и kernel не подходит для binary search |
| Разрешённый вывод | search direction и формулы \(q\) корректны |
| Неразрешённый вывод | выбранное mass rule статистически лучше |

### C08. Изолированный ALS на noiseless statistics

| Поле | Спецификация |
|---|---|
| Гипотеза | Текущий ALS восстанавливает exact factorization \(I_j=\ell_j^*U_j\beta^*\) |
| Генерация | \(d=10,J=100,P=20\); Gaussian \(U_j,\ell_j^*,\beta^*\) |
| Фактор | 20 initial directions с cosine от 0.1 до 0.9 |
| Фиксируемые | \(I_j\), \(\lambda=0\), strict CG |
| Повторы | 20 |
| Primary | direction loss |
| Вторичные | objective, slope-product error, ALS iterations |
| Ожидание | exact solution при full rank и подходящей basin |
| Успех | loss \(<10^{-10}\) для starts в basin; objective near zero |
| Неудача | systematic nonzero residual |
| Альтернативное объяснение | nonconvex basin для orthogonal start; отдельно от formula error |
| Разрешённый вывод | текущие ALS update и нормировка корректны |
| Неразрешённый вывод | global convergence ALS |

### C09. Matrix-free CG против dense reference

| Поле | Спецификация |
|---|---|
| Гипотеза | Matrix-free оператор решает ту же линейную систему, что и явная матрица |
| Генерация | synthetic \(U_j,\ell_j,I_j\); \(d\in\{5,10,20,50\}\) |
| Фактор | system size и condition number \(1,10,10^2,10^4\) |
| Фиксируемые | одна и та же \(A,b\) |
| Повторы | 50 на размер |
| Primary | relative residual и solution gap |
| Вторичные | CG iterations, `info` |
| Ожидание | agreement до tolerance |
| Успех | residual \(<10^{-9}\), relative gap \(<10^{-8}\), `info=0` |
| Неудача | `info>0`, wrong matvec, large gap |
| Альтернативное объяснение | dense reference ill-conditioned; проверяется backward error |
| Разрешённый вывод | текущий matrix-free CG реализован верно |
| Неразрешённый вывод | CG лучше другого solver |

### C10. Монотонность objective ALS

| Поле | Спецификация |
|---|---|
| Гипотеза | При точных или достаточно точных полу-шагах objective не возрастает после полного ALS cycle |
| Генерация | C08 и B0 fixed statistics |
| Фактор | ALS iteration 1–20 |
| Фиксируемые | \(I,U,\lambda,prior\) |
| Повторы | 20 |
| Primary | число превышений \(F_{m+1}>F_m+\epsilon\) |
| Вторичные | величина increase, CG residual |
| Ожидание | nonincreasing sequence |
| Успех | ноль increases выше \(10^{-10}\max(1,F_0)\) |
| Неудача | reproducible increases при small residual |
| Альтернативное объяснение | renormalization и regularization not scale-invariant; objective измеряется после согласованной rescaling |
| Разрешённый вывод | ALS steps согласованы с objective |
| Неразрешённый вывод | найден global minimum |

### C11. Независимость stopping от масштаба objective

| Поле | Спецификация |
|---|---|
| Гипотеза | Relative stopping даёт одно направление при масштабировании \(Y\) |
| Генерация | B0 без внешнего обновления weights |
| Фактор | \(Y'=10^rY,r=-4,\ldots,4\) |
| Фиксируемые | centers, directions, normalized init, \(\lambda=0\) |
| Повторы | 20 datasets |
| Primary | gap к result при \(r=0\) |
| Вторичные | iterations, stop reason |
| Ожидание | direction invariant |
| Успех | median gap \(<10^{-5}\), max \(<10^{-3}\) |
| Неудача | systematic early stop at small scale |
| Альтернативное объяснение | ridge not scaled; повторить \(\eta=0\) и relative ridge |
| Разрешённый вывод | stopping criterion масштабно устойчив |
| Неразрешённый вывод | оптимальный tolerance выбран |

### C12. Реализационная воспроизводимость

| Поле | Спецификация |
|---|---|
| Гипотеза | dtype, chunking и thread count не меняют результат сверх ожидаемой rounding error |
| Генерация | один simple и один medium B0 dataset |
| Фактор | float32/64, chunk 8–512, threads 1–8 |
| Фиксируемые | seed и порядок данных |
| Повторы | 20 |
| Primary | projective gap к float64 single-thread reference |
| Вторичные | objective, failures, runtime |
| Ожидание | chunk/thread equivalence; controlled float32 difference |
| Успех | float64 gap \(<10^{-8}\), float32 gap \(<10^{-3}\) |
| Неудача | nondeterministic large changes |
| Альтернативное объяснение | non-associative reductions; если gap влияет на success category, backend считается нестабильным |
| Разрешённый вывод | вычислительная реализация воспроизводима |
| Неразрешённый вывод | float32 достаточен во всех hard regimes |

## 5.3. Уровень 2. Базовое статистическое качество

### S01. Семейство функций связи

Используются функции:

\[
f_1(z)=z,
\]

\[
f_2(z)=\tanh(1.5z),
\]

\[
f_3(z)=\sin(z),
\]

\[
f_4(z)=\frac{z^2-1}{\sqrt2},
\]

\[
f_5(z)=z\sin(\sqrt5z),
\]

\[
f_6(z)=|z|.
\]

Каждый signal центрируется и нормируется до variance 1. SNR фиксируется равным 10.

| Поле | Спецификация |
|---|---|
| Гипотеза | Smooth links восстанавливаются; symmetric links сильнее зависят от initialization |
| Фактор | тип \(f\) |
| Фиксируемые | B0 data/algorithm parameters |
| Повторы | 50 |
| Primary | median \(L_\beta\) |
| Вторичные | success rate, initial cosine, iterations |
| Ожидание | linear/tanh проще; quadratic и absolute труднее для OLS init |
| Успех | при oracle-cosine 0.5 smooth links дают median cosine \(\geq0.9\) |
| Неудача | linear noiseless-like regime не восстанавливается |
| Альтернативное объяснение | failure init; проверяется I02 |
| Разрешённый вывод | link-specific working regimes |
| Неразрешённый вывод | универсальная устойчивость к любой \(f\) |

### S02. Рост \(n\) при фиксированном алгоритме

\[
n\in\{250,500,1000,2000,4000\},
\qquad d=20.
\]

Фиксируются:

\[
J=200,
\quad P=32,
\quad n_{\min}=64.
\]

Для \(n=250\) используется \(n_{\min}=32\), чтобы условие было feasible; это значение фиксируется заранее и отмечается как исключение.

Проверяется slope:

\[
\log L_\beta=a+b\log n.
\]

Критерий успеха: upper 95% CI для \(b\) меньше нуля либо loss уже достиг floor \(<0.01\).

### S03. Рост \(n\) в практическом режиме

Те же \(n\), но:

\[
J=\min(n,0.25n),
\qquad
n_{\min}=\min(64,0.1n).
\]

S02 отделяет статистический эффект \(n\) при фиксированной вычислительной конфигурации. S03 оценивает pipeline, в котором число центров растёт с данными.

### S04. Аддитивный Gaussian noise

\[
\mathrm{SNR}
\in
\{\infty,40,20,10,5,2,1,0.5\}.
\]

Момент деградации:

\[
\mathrm{SNR}_{break}
=
\max\{s:P(c_\beta\geq0.8)<0.8\}.
\]

Проверяется ordered trend median loss с помощью isotonic summary и paired differences соседних уровней.

### S05. Размерность и \(n/d\)

\[
d\in\{10,25,50,100,200,500,1000\},
\]

\[
n/d\in\{2,5,10,20,50\}.
\]

Primary result:

\[
d_{stat}(r)
=
\max\{d:P(c_\beta\geq0.8)\geq0.8\},
\qquad r=n/d.
\]

Для различения sample и sketch limitations сценарий повторяется на пограничных точках с \(P=32,64,128\).

### S06. Nuisance dimensions и signal support

Режим A:

\[
s=10
\]

фиксируется, а \(d\) растёт.

Режим B:

\[
s/d=0.25
\]

фиксируется.

Если режим A деградирует при постоянном signal support, измеряется цена неинформативных координат. Это не доказывает способность ADP к variable selection.

## 5.4. Уровень 3. Чувствительность к параметрам алгоритма

### T01. Число направлений \(P\)

\[
P\in\{2,4,8,16,32,64,128\}.
\]

Основная метрика: loss. Вторичные: time, memory, variance по direction seed.

Рабочее значение:

\[
P_{work}
=
\min\left\{
P:
\operatorname{median}L(P)
\leq
\min_{P'}\operatorname{median}L(P')+0.01
\right\}.
\]

Слишком малое \(P\): большая variance между direction seed и gap к full-basis oracle.

Слишком большое \(P\): качество не улучшается, время и память растут.

### T02. Взаимодействие \(P\times d\)

\[
d\in\{20,50,100,200,500\},
\]

\[
P\in\{8,16,32,64,128\}.
\]

\(n/d=20\) фиксируется. Это взаимодействие анализируется отдельно, поскольку требуемое число направлений может расти с размерностью.

### T03. Число центров \(J\)

\[
J\in\{50,100,200,400,700,1000\}.
\]

Слишком малое \(J\): высокая variance между center seed.

Слишком большое \(J\): plateau качества и линейный рост стоимости.

Рабочее значение выбирается по тому же правилу \(+0.01\) к лучшему median loss.

### T04. Локальная масса \(n_{\min}\)

\[
n_{\min}\in\{8,16,32,48,64,96,128,256\}.
\]

Признаки слишком малого значения:

- высокий q95 \(|\ell_j|\);
- малые denominator \(\|U_j\beta\|^2\);
- CG nonconvergence;
- высокая failure rate.

Признаки слишком большого значения:

- рост \(h_0\);
- слабая локальность;
- увеличение bias.

Рабочий диапазон должен обеспечивать:

\[
P(c_\beta\geq0.9)\geq0.8,
\]

\[
P(\text{failed})\leq0.01,
\]

и loss не более чем на 0.02 выше минимума.

### T05. Штраф \(\lambda\)

Чтобы сравнение не зависело от масштаба задачи, используется relative scale:

\[
\lambda_{rel}
=
\lambda/\bar a,
\]

где

\[
\bar a
=
\operatorname{median}_{r}
\left[
\sum_j\ell_j^2\sum_pU_{jpr}^2
\right]
\]

оценивается после первого slope-step pilot.

Сетка:

\[
\lambda_{rel}
\in
\{0,10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},0.1,0.3,1,3,10\}.
\]

Начальные cosine:

\[
c_0\in\{0,0.5,0.9\}.
\]

Слишком большое значение определяется по отсутствию движения от плохого prior и росту data-fit residual.

### T06. Скорость уменьшения bandwidth

\[
a\in\{1.05,1.1,1.2,\sqrt2,1.6,1.8,2\}.
\]

Фиксируется \(K_{\max}=8\).

Признаки слишком большого \(a\):

- q05 mass падает ниже target;
- \(\rho_k\) достигает нижней границы;
- quality растёт на шаге 1 и затем падает.

Признаки слишком малого \(a\):

- \(h_k\) меняется мало;
- full result совпадает со step0;
- дополнительные шаги не улучшают loss.

### T07. Точность текущего ALS+CG

Изменяются только параметры текущей реализации:

\[
M_{\max}\in\{1,2,4,8,16,32\},
\]

\[
\tau_{ALS}\in\{10^{-3},10^{-4},10^{-5},10^{-6},10^{-7},10^{-8}\},
\]

\[
\tau_{CG}\in\{10^{-3},10^{-4},10^{-5},10^{-6},10^{-7},10^{-8},10^{-9}\},
\]

\[
R_{CG,max}\in\{25,50,100,2d,5d,10d\}.
\]

Strict reference использует тот же ALS+CG:

\[
M_{\max}=64,
\quad
\tau_{ALS}=10^{-10},
\quad
\tau_{CG}=10^{-12},
\]

с достаточным `maxiter`.

Рабочая конфигурация:

\[
1-
|\widehat\beta_{candidate}^{\mathsf T}
\widehat\beta_{strict}|
\leq0.01,
\]

\[
P(\text{CG nonconvergence})\leq0.01.
\]

Этот сценарий не сравнивает ALS с другими оптимизаторами.

### T08. Обновление направлений

Уровни:

- направления фиксированы на всех шагах;
- обновление каждые 4 шага;
- каждые 2 шага;
- на каждом шаге;
- фактическое расписание текущего кода.

Фиксируются данные, центры и initial directions. Изменяется только расписание subsequent directions.

### T09. Правило локальной массы

Сравниваются:

- mean;
- minimum;
- q01;
- q05;
- q10.

Основной сценарий B0 и nonuniform-density R13.

Mean может дать хорошую среднюю массу при пустых окрестностях отдельных центров. Minimum может сделать bandwidth слишком большим из-за одного outlier-center.

### T10. Perturbation центров

Центры задаются:

\[
x_j=X_{i_j}+\tau_c s_Xv_j,
\qquad
v_j\sim N(0,I_d),
\]

\[
\tau_c\in\{0,0.01,0.05,0.1,0.25,0.5,1\}.
\]

Здесь \(s_X\) — median coordinate standard deviation после preprocessing.

## 5.5. Уровень 4. Устойчивость

### R01. Gaussian noise

Совпадает с S04, но использует 100 повторов и локальное уточнение около breakdown SNR.

### R02. Тяжёлые хвосты шума

\[
\varepsilon
=
\frac{t_\nu}{\sqrt{\nu/(\nu-2)}},
\qquad
\nu\in\{\infty,10,5,3,2.5\}.
\]

Variance фиксирована равной единице. Изменяется только tail behavior.

Момент деградации: первый \(\nu\) при движении от \(\infty\) вниз, где median cosine падает на 0.10 либо IQR удваивается.

### R03. Выбросы в отклике

\[
Y_i^{cont}
=
Y_i+B_iA_is_Y,
\]

\[
B_i\sim\operatorname{Bernoulli}(\epsilon_Y),
\]

\[
A_i\in\{-10,+10\}
\]

с равными вероятностями.

\[
\epsilon_Y\in\{0,0.01,0.025,0.05,0.10,0.20\}.
\]

### R04. Выбросы в признаках

С вероятностью \(\epsilon_X\):

\[
X_i^{cont}=X_i+10s_Xv_i.
\]

Типы \(v_i\):

- \(v_i=\beta^*\);
- \(v_i\perp\beta^*\);
- random sphere direction.

Уровни contamination совпадают с R03.

Сравнение R03/R04 paired по исходным чистым данным.

### R05. Гетероскедастичность

\[
Y_i
=
\widetilde\mu_i
+
\sigma
\frac{1+\alpha|Z_i|}{c_\alpha}\eta_i,
\]

\[
Z_i=\beta^{*T}X_i,
\qquad
\eta_i\sim N(0,1),
\]

где \(c_\alpha\) нормирует среднюю variance.

\[
\alpha\in\{0,0.25,0.5,1,2\}.
\]

### R06. Коррелированные признаки

Корректный AR(1) дизайн:

\[
\Sigma_{rs}=\rho_X^{|r-s|},
\]

\[
\rho_X\in\{0,0.2,0.4,0.6,0.8,0.9,0.95,0.98\}.
\]

\[
X_i\sim N(0,\Sigma).
\]

Общий случайный сдвиг всех наблюдений не используется как генератор feature correlation.

### R07. Почти вырожденная covariance

Пусть

\[
\lambda_r
=
\kappa^{-(r-1)/(d-1)},
\]

\[
\Sigma=Q\operatorname{diag}(\lambda_1,\ldots,\lambda_d)Q^T,
\]

где \(Q\) фиксированная random orthogonal matrix.

\[
\kappa\in\{1,3,10,30,100,300,1000,10^4\}.
\]

Сравниваются preprocessing:

- raw;
- coordinate standardization;
- training-estimated whitening;
- oracle whitening как диагностический режим.

Oracle whitening не является практическим baseline.

### R08. Разные масштабы координат

\[
s_r
=
r_{scale}^{(r-1)/(d-1)},
\]

\[
X'=X\operatorname{diag}(s_1,\ldots,s_d),
\]

\[
r_{scale}\in\{1,3,10,30,100,1000\}.
\]

Сравниваются raw и training standardization. Изменение covariance orientation не вводится.

### R09. Нелинейная зависимость признаков

Генерируются latent variables \(Z\sim N(0,I_d)\). Первые пары координат задаются:

\[
X_1=Z_1,
\]

\[
X_2=\sqrt{1-\gamma^2}Z_2+
\gamma\frac{Z_1^2-1}{\sqrt2},
\]

\[
\gamma\in\{0,0.2,0.4,0.6,0.8,0.95\}.
\]

Остальные координаты равны соответствующим \(Z_r\). Linear covariance не полностью описывает эту зависимость.

### R10. Sparse и dense direction

При \(d=100\):

\[
s\in\{1,2,5,10,25,100\}.
\]

На support генерируются Gaussian coefficients и нормируются.

Вывод ограничивается влиянием sparsity истинного направления. Текущий ADP не объявляется методом feature selection.

### R11. Осциллирующая связь

\[
f_\omega(z)=\sin(\omega z),
\]

\[
\omega\in\{0.5,1,2,4,8\}.
\]

Signal variance нормируется. Рост \(\omega\) изменяет локальный масштаб функции, а не SNR.

### R12. Нарушение гладкости

Smooth approximation ReLU:

\[
f_\tau(z)
=
\tau\log(1+e^{z/\tau}),
\]

\[
\tau\in\{1,0.5,0.2,0.1,0.05\}.
\]

Предельный уровень:

\[
f_0(z)=\max(0,z).
\]

Отдельно проверяется \(f(z)=|z|\).

### R13. Неоднородная плотность

\[
X
\sim
\frac12N(-\mu v,I_d)
+
\frac12N(\mu v,I_d),
\]

\[
\mu\in\{0,1,2,3,4\}.
\]

Положения \(v\):

- \(v=\beta^*\);
- \(v\perp\beta^*\);
- random angle.

Отслеживаются q01, q05, median и minimum локальной массы.

### R14. Принудительно разреженные окрестности

Чтобы отделить local sparsity от общей плотности, центры делятся на две группы. Доля \(p_{tail}\) центров выбирается из tail region \(\|X\|\) выше заданного квантиля:

\[
q\in\{0.8,0.9,0.95,0.975,0.99\}.
\]

Изменяется только положение центров, данные сохраняются B0.

Момент degradation задаётся долей центров с

\[
N_j<n_{\min}.
\]

### R15. Ошибочная спецификация single-index

\[
Y
=
f(\beta_1^TX)
+
\delta_{mis}g(\beta_2^TX)
+
\varepsilon,
\]

\[
\beta_1^T\beta_2=0,
\]

\[
\delta_{mis}\in\{0,0.05,0.1,0.2,0.4,0.8,1\}.
\]

Алгоритм по-прежнему оценивает одно направление. Primary target — \(\beta_1\) только при \(\delta_{mis}<1\). Дополнительно вычисляется главный eigenvector Monte Carlo approximation матрицы

\[
C_\nabla
=
\mathbb E[\nabla g(X)\nabla g(X)^T].
\]

Разрешённый вывод: качество лучшего rank-one derivative representation при misspecification.

Неразрешённый вывод: способность восстанавливать multi-index подпространство.

## 5.6. Уровень 5. Масштабирование

### M01. Масштабирование по \(n\)

\[
d=20,
\qquad
n\in\{250,500,1000,2000,4000,8000\}.
\]

Фиксируются \(J=200,P=32,K=8\), чтобы измерить отдельную зависимость от \(n\).

### M02. Масштабирование по \(d\)

\[
n=4000,
\qquad
d\in\{10,25,50,100,200,500,1000\}.
\]

Для больших \(d\) качество может падать. Runtime измеряется независимо от того, проходит ли quality threshold.

### M03. Масштабирование по \(J\)

\[
J\in\{50,100,200,400,800,1600\},
\]

\[
n=2000,d=50,P=32.
\]

### M04. Масштабирование по \(P\)

\[
P\in\{4,8,16,32,64,128\}.
\]

\[
n=2000,d=50,J=400.
\]

### M05. Внешние и внутренние итерации

Отдельные серии:

\[
K\in\{1,2,4,6,8,12\},
\]

\[
M_{max}\in\{1,2,4,8,16,32\}.
\]

Нельзя менять \(K\) и \(M\) одновременно в one-factor оценке.

### M06. Модель памяти

Для grid \((n,d,J,P)\) подгоняется

\[
M_{peak}
=
\alpha Jn
+
\beta JPd
+
\gamma nd
+
\delta.
\]

Измеряется RSS процесса, а не только Python allocator. Для окна алгоритма и
полного запуска сохраняются минимум, среднее, максимум и прирост пика
относительно начала окна. Модель (M_{peak}) подгоняется по
`algorithm_rss_peak_delta_mib` и отдельно проверяется по
`full_run_rss_peak_delta_mib`; смешивать эти две границы в одной регрессии
нельзя.

### M07. Параллельное масштабирование

\[
w\in\{1,2,4,8,16\}
\]

до числа физических ядер.

При process-level parallelism число BLAS threads фиксируется равным 1.

\[
S(w)=T(1)/T(w),
\]

\[
E(w)=S(w)/w.
\]

### M08. Chunking

\[
B\in\{8,16,32,64,128,256,512\}.
\]

Критерий математической эквивалентности:

\[
1-|\widehat\beta_B^T\widehat\beta_{reference}|
<10^{-8}
\]

в float64.

## 5.7. Уровень 6. Инициализация и сходимость

### I01. Начальный cosine

Генерация направления:

\[
\beta_0
=
c_0\beta^*
+
\sqrt{1-c_0^2}v,
\]

\[
v^T\beta^*=0,
\qquad
\|v\|_2=1.
\]

\[
c_0\in\{0,0.1,0.25,0.5,0.75,0.9\}.
\]

Область притяжения:

\[
c_{basin}
=
\min\left\{
c_0:
\operatorname{LCB}_{95\%}
P(c_{final}\geq0.9)
\geq0.8
\right\}.
\]

### I02. Link-specific basin

I01 повторяется для:

- linear;
- tanh;
- quadratic;
- absolute value;
- \(\sin(4z)\).

Это различает неинформативную инициализацию от отсутствия сигнала для метода.

### I03. Практические инициализации

Сравниваются:

- random sphere;
- OLS;
- PLS1;
- SIR;
- SAVE;
- PHD;
- isotropic ADP step0.

Это сравнение методов получения начального направления, а не сравнение оптимизаторов внутренней задачи.

Метрики:

- initial cosine;
- final cosine;
- improvement;
- outer/inner iterations;
- failure rate;
- final objective.

### I04. Разложение variance

Nested protocol:

- 50 data seeds;
- для каждого data seed 10 direction/init seeds;
- на подмножестве 10 data seeds дополнительно 10 center seeds.

Variance decomposes descriptively into:

- between-data variance;
- center variance;
- direction/init variance.

Это не требует предположения normality; сообщаются empirical variance components и bootstrap CI.

## 5.8. Уровень 7. Сравнение с baseline

### B01. Набор baseline

Обязательные методы:

1. random unit direction;
2. OLS;
3. PLS1;
4. SIR;
5. SAVE;
6. PHD;
7. OPG или классический ADE;
8. MAVE или RMAVE;
9. ADP step0;
10. full structure-adaptive ADP.

Не включаются методы, отличающиеся от full ADP только заменой ALS или CG.

### Representative scenarios

Baseline запускаются минимум на следующих сценариях:

- linear, SNR 20, \(d=20\);
- tanh, SNR 10, \(d=20\);
- quadratic, SNR 10, \(d=20\);
- sine frequency 4;
- \(d=100,n/d=20\);
- \(d=200,n/d=5\);
- AR correlation 0.8;
- \(\kappa=300\);
- Student \(t_3\) noise;
- 5% response outliers;
- sparse \(s=10,d=100\);
- nonuniform mixture \(\mu=3\).

### Равенство условий

- одинаковые данные;
- одинаковые train/validation/test splits;
- одинаковые preprocessing;
- одинаковые data seed;
- одинаковый tuning budget;
- отсутствие tuning по test;
- одинаковый одномерный smoother после найденного направления для prediction comparison.

### Prediction comparison

После оценки \(\widehat\beta\) для каждого метода обучается один и тот же одномерный smoother:

\[
Y\sim s(\widehat\beta^TX).
\]

Метрики:

- test RMSE;
- MAE;
- \(R^2\);
- direction recovery на synthetic data;
- runtime;
- memory;
- failure rate.

## 5.9. Уровень 8. Ablation study

### Общий причинный критерий

Для каждого seed:

\[
\Delta_r
=
L_{ablated,r}-L_{full,r}.
\]

Компонент считается полезным в сценарии, если:

- median \(\Delta_r\geq0.02\);
- paired bootstrap 95% CI не содержит 0;
- эффект повторяется минимум на двух соседних уровнях сложности.

### A01. Step0 против полного внешнего цикла

Отключается вся внешняя адаптация. Проверяется общий вклад refinement.

### A02. Без анизотропии

Фиксируется

\[
\rho_k=1.
\]

Остальная внешняя схема сохраняется.

### A03. Без уменьшения bandwidth

Фиксируется

\[
h_k=h_0.
\]

Анизотропия и direction renewal сохраняются.

### A04. Full directional basis

На малых и средних \(d\) случайные направления заменяются детерминированным базисом или orthogonal frame с \(P=d\). Это oracle вычислительной аппроксимации, а не практический baseline.

### A05. Fixed directions

Направления шага 0 используются на всех внешних шагах.

### A06. Без текущей regularization

Фиксируется \(\lambda=0\). Другие формы штрафа не исследуются.

### A07. Mean mass против q05 mass

Проверяется защита weak centers при неоднородной плотности.

### A08. Центры без perturbation

Сравнивается фактическое правило current implementation и \(\tau_c=0\).

### A09. Negative controls

Искусственно вводятся:

- ненормированное локальное среднее;
- удаление одного множителя веса;
- замена \(\rho^2\) на \(\rho\);
- отсутствие нормировки \(\beta\);
- перестановка осей \(P,d\) в тестовом контуре.

Negative controls должны провалить C01–C07. Они не включаются в статистические выводы.

## 5.10. Уровень 9. Реальные данные

### D01. Airfoil Self-Noise

| Свойство | Значение |
|---|---|
| \(n\) | 1503 |
| \(d\) | 5 |
| Признаки | непрерывные физические параметры |
| Target | sound pressure, непрерывный |
| Пропуски | отсутствуют в стандартной версии |
| Риски | разные масштабы, взаимодействия, possible misspecification |
| Метрика | RMSE, MAE |

### D02. Concrete Compressive Strength

| Свойство | Значение |
|---|---|
| \(n\) | 1030 |
| \(d\) | 8 |
| Признаки | количественные состав и возраст |
| Target | compressive strength |
| Пропуски | отсутствуют в стандартной версии |
| Риски | корреляция компонентов, нелинейный age effect |
| Метрика | RMSE, MAE |

### D03. Wine Quality, white

| Свойство | Значение |
|---|---|
| \(n\) | 4898 |
| \(d\) | 11 |
| Признаки | physicochemical measurements |
| Target | ordinal quality, рассматривается как regression target |
| Пропуски | отсутствуют в стандартной версии |
| Риски | target imbalance, discreteness, correlation |
| Метрика | RMSE, MAE, Spearman correlation |

### D04. Superconductivity

| Свойство | Значение |
|---|---|
| \(n\) | 21263 |
| \(d\) | 81 |
| Признаки | composition-derived quantitative features |
| Target | critical temperature |
| Пропуски | проверяются перед запуском |
| Риски | высокая корреляция, неоднородная плотность, misspecification |
| Метрика | RMSE, MAE, runtime, memory |

### Протокол real data

Для D01–D03:

- 10 повторов 5-fold outer CV;
- 3-fold inner CV;
- одинаковые folds для всех методов.

Для D04:

- 3 повтора 5-fold outer CV;
- 3-fold inner CV.

Preprocessing обучается только на соответствующем training fold.

### Стабильность направления

Для bootstrap/fold estimates \(\widehat\beta_b\):

\[
S_{pair}
=
\operatorname{median}_{b<b'}
|\widehat\beta_b^T\widehat\beta_{b'}|.
\]

Дополнительно:

- coordinate bootstrap CI после согласования знака;
- sensitivity к удалению 1% и 5% observations;
- stability рангов \(|\widehat\beta_r|\);
- prediction на outer test.

Интерпретация коэффициентов разрешена только при \(S_{pair}\geq0.9\) и устойчивости prediction.

---

# 6. Baseline и ablation

## 6.1. Обоснование baseline

| Метод | Причина включения |
|---|---|
| Random direction | нижняя граница информативности |
| OLS | минимальный статистический baseline и информированная инициализация |
| PLS1 | простой latent-score baseline |
| SIR | классический first-moment SDR |
| SAVE | baseline для симметричных зависимостей |
| PHD | second-order SDR baseline |
| ADE/OPG | ближайший производный класс |
| MAVE/RMAVE | локальная single-index оценка без random directional sketch |
| ADP step0 | отделяет начальную изотропную оценку от адаптации |
| Full ADP | объект исследования |

## 6.2. Что не считается baseline

Не считаются baseline:

- dense solve внутренней системы;
- strict tolerance текущего CG;
- alternative ALS solver;
- variable projection;
- новая regularization;
- oracle whitening;
- true direction initialization;
- full directional basis.

Они применяются только как диагностические oracle или unit-test references.

## 6.3. Обязательные ablation

| Компонент | Ablation | Проверяемый вклад |
|---|---|---|
| Внешняя адаптация | step0 only | refinement целиком |
| Анизотропия | \(\rho=1\) | shape локальных окрестностей |
| Bandwidth decay | \(h_k=h_0\) | уменьшение масштаба |
| Renewal directions | fixed directions | обновление sketch |
| Random sketch | full basis при малом \(d\) | sketch approximation |
| Current regularization | \(\lambda=0\) | устойчивость path |
| Mass policy | mean вместо q05 | защита weak centers |
| Center perturbation | \(\tau_c=0\) | влияние jitter |

---

# 7. План усложнения

Последовательность запусков:

\[
\text{unit tests}
\rightarrow
\text{noiseless linear model}
\rightarrow
\text{smooth correct single-index}
\rightarrow
\text{one-factor tuning}
\]

\[
\rightarrow
\text{single assumption violation}
\rightarrow
\text{focused interactions}
\rightarrow
\text{combined violations}
\rightarrow
\text{scaling}
\rightarrow
\text{real data}.
\]

## 7.1. Условия перехода

1. К уровню 2 переходят только после прохождения C01–C12.
2. К robustness переходят после выбора рабочих диапазонов \(P,J,n_{\min},a,\lambda\).
3. Комбинированные нарушения запускаются только после одиночных.
4. Scaling выполняется для математически корректной конфигурации.
5. Baseline и real data используют параметры, выбранные без test leakage.

## 7.2. Комбинированные нарушения

| ID | \(d\) | \(n/d\) | SNR | Дополнительные факторы |
|---|---:|---:|---:|---|
| CR01 | 100 | 10 | 5 | AR correlation 0.8 |
| CR02 | 100 | 10 | 5 | 5% response outliers |
| CR03 | 200 | 5 | 2 | \(\kappa=300\) |
| CR04 | 200 | 10 | 5 | Student \(t_3\) noise |
| CR05 | 500 | 5 | 2 | sparse support \(s=10\) |
| CR06 | 200 | 5 | 2 | correlation 0.8 и 5% response outliers |
| CR07 | 200 | 5 | 2 | mixture separation 3 и \(\kappa=100\) |

Комбинированные сценарии не используются для оценки индивидуального эффекта фактора.

---

# 8. Обязательные графики и таблицы

## 8.1. Графики решений

| ID | Ось \(x\) | Ось \(y\) | Группировка | CI | Scale | Какой вывод разрешён |
|---|---|---|---|---|---|---|
| G01 | \(n\) | median \(L_\beta\) | link/method | bootstrap 95% | log-log | улучшается ли recovery с \(n\) |
| G02 | SNR | success rate | method | Wilson 95% | log x | noise breakdown |
| G03 | \(d\) | success rate | \(n/d\) | Wilson 95% | log x | dimension boundary |
| G04 | \(P\) | median loss | \(d\) | bootstrap 95% | log x | minimal directional budget |
| G05 | \(P\) | time и memory | \(d\) | bootstrap 95% | log-log | cost of sketch |
| G06 | \(J\) | loss и time | scenario | bootstrap 95% | log x | center plateau |
| G07 | \(n_{\min}\) | loss | link | bootstrap 95% | log x | bias-variance interval |
| G08 | \(\lambda_{rel}\) | loss | initial cosine | bootstrap 95% | log x | stability-bias trade-off |
| G09 | initial cosine | success rate | link | Wilson 95% | linear | basin boundary |
| G10 | robustness level | median cosine | method | bootstrap 95% | appropriate | degradation point |
| G11 | wall time | loss | method | bootstrap 95% | log-log | quality-cost frontier |
| G12 | fitted time | observed time | scaling family | CI model | log-log | adequacy complexity model |

## 8.2. Внутренняя диагностика

| ID | Ось \(x\) | Ось \(y\) | Группировка | Вывод |
|---|---|---|---|---|
| G13 | outer step \(k\) | median \(|\beta_k^T\beta^*|\) | scenario | улучшает ли refinement |
| G14 | outer step \(k\) | \(h_k\) | seed | корректен ли bandwidth path |
| G15 | outer step \(k\) | \(\rho_k\) | seed | когда возникает анизотропия |
| G16 | outer step \(k\) | q01/q05/median \(N_j\) | scenario | сохраняется ли local mass |
| G17 | ALS iteration | relative objective | outer step | корректна ли внутренняя сходимость |
| G18 | ALS iteration | projective delta | outer step | стабилизируется ли направление |
| G19 | CG iteration | relative residual | condition level | достигается ли tolerance |
| G20 | sorted center | \(\|U_j\beta\|^2\) | scenario | доля слабых local slopes |
| G21 | scenario | failure category share | method | механизм failed runs |

Projective delta:

\[
\delta_{dir}(\beta_m,\beta_{m-1})
=
\sqrt{2\left(1-|\beta_m^T\beta_{m-1}|\right)}.
\]

Он заменяет обычную норму разности, чувствительную к смене знака.

## 8.3. Таблицы

Обязательные таблицы:

1. `single_index_series.csv` с паспортом серии;
2. `single_index_runs.csv` с полной строкой каждого run;
3. `single_index_iterations.csv` с outer-step diagnostics;
4. `single_index_solver_iterations.csv` для сценариев C08–C11 и T07;
5. `single_index_initial_parameters.csv` с параметрами всех jobs до запуска;
6. scenario-level median, mean, IQR, q05, q95 и CI;
7. success и failure rates;
8. отдельная таблица причин failed runs и остановки ALS/CG;
9. component time, `algorithm_time_sec` и `full_run_time_sec`;
10. min/mean/max/peak-delta RSS для algorithm и full run;
11. empirical scaling exponents;
12. selected hyperparameters;
13. paired baseline differences;
14. ablation effects;
15. real-data nested-CV results;
16. bootstrap direction stability;
17. `single_index_artifacts.csv` со всеми созданными файлами.

Не строятся графики, которые дублируют одну и ту же статистику без нового решения. Например, одновременно line plot mean, barplot mean и table mean для одного параметра не требуются.

---

# 9. Правила интерпретации

| Наблюдение | Возможная причина | Дополнительная проверка | Допустимый вывод |
|---|---|---|---|
| Падение с \(d\) | мало наблюдений | увеличить \(n/d\) | sample limitation, если помогает |
| Падение с \(d\) | мало направлений | увеличить \(P\), full-basis oracle | sketch limitation, если oracle стабилен |
| Full basis тоже падает | local statistics нестабильны | exact/statistical reference на малом \(d\) | не только sketch |
| Strict CG лучше standard | недорешённый beta-step | residual, `info`, T07 | limitation текущих tolerances |
| Strict и standard совпадают | внутренний solve не причина | robustness/oracle tests | искать statistical/algorithmic cause |
| Objective падает, cosine не растёт | неправильный basin | I01/I02 | optimization path попал в другое stationary solution |
| Quality падает после внешнего шага | слишком агрессивный \(h_k\) | T06 и path mass | schedule problem |
| \(\rho_k\) на границе 0 | масса недостаточна | увеличить \(h\), проверить feasibility | anisotropy constraint infeasible |
| Mean mass достаточна, q05 мала | слабые центры скрыты средним | T09/A07 | mean rule недостаточно защищает centers |
| OLS init не работает на \(z^2\) | нулевой первый момент | SAVE/step0/oracle cosine | limitation инициализации |
| Все init не работают | слабый derivative signal или method bias | увеличивать \(n\), SNR, full basis | не только initialization |
| Fixed-data seed variance высокая | directions/init randomness | I04 | algorithmic variance |
| Between-data variance высокая | sampling uncertainty | увеличить \(n\) | statistical variance |
| Prediction хорошее, cosine плохой | non-identifiability или alternative direction | population/oracle check | нельзя объявлять recovery \(\beta^*\) |
| Cosine хороший, prediction плохой | misspecified smoother или test shift | common smoother/oracle link | direction alone not enough |
| Standardization резко помогает | coordinate scale sensitivity | C05/R08 | preprocessing обязателен |
| Whitening помогает при high \(\kappa\) | conditioning | oracle whitening | covariance-driven failure |
| Float32 меняет category success | numerical instability | C09/C12 | float32 недопустим в этом режиме |
| Chunk или threads меняют float64 result | reduction/race issue | single-thread reference | implementation instability |
| Runtime растёт быстрее theoretical | CG iterations или hidden copies | component timings | конкретное bottleneck найдено |
| Full ADP не лучше step0 | внешняя адаптация бесполезна в режиме | A01–A03 | refinement не нужен в этом режиме |
| Full-basis oracle лучше random directions | finite sketch error | T01/T02/A04 | увеличить или изменить directional budget |
| Oracle и full ADP одинаково падают | предел данных/модели | увеличить \(n\), SNR, проверить misspecification | вероятный method/data limit |
| Хороший synthetic, нестабильный real direction | single-index misspecification | bootstrap/fold stability | нельзя интерпретировать коэффициенты |

## 9.1. Различение метода и реализации

Используется диагностическая лестница:

1. direct sums против backend;
2. dense reference против current CG на малой системе;
3. strict ALS+CG против standard settings;
4. full directional basis против random sketch;
5. step0 против внешней адаптации;
6. рост \(n\) при фиксированном режиме;
7. misspecification и real-data stability.

Первый уровень, на котором исчезает gap, локализует причину.

Пример:

- dense и CG совпадают;
- strict и standard совпадают;
- full basis значительно лучше random sketch.

Допустимый вывод: ограничивает directional sketch, а не CG или tolerance.

Другой пример:

- full basis и random sketch одинаковы;
- увеличение \(n\) улучшает качество;
- при фиксированном \(n/d\) degradation сдвигается вправо.

Допустимый вывод: наблюдается статистическая граница, а не явная ошибка реализации.

---

# 10. Минимальный, полный и исследовательский набор запусков

## 10.1. Минимальная проверка корректности

Состав:

- C01–C12;
- S01: linear, tanh, quadratic;
- S02: \(n=500,1000,2000\);
- S04: SNR 20, 5, 1;
- T01: \(P=8,32,128\);
- T04: \(n_{\min}=16,64,256\);
- T07: loose, standard, strict;
- I01: cosine 0, 0.5, 0.9;
- baseline: random, OLS, SAVE, OPG/ADE, MAVE/RMAVE, step0, full ADP;
- ablation: step0, no anisotropy, fixed directions.

Повторы:

- deterministic tests: 1–20;
- stochastic pilot: 20;
- initialization points: 50.

Приблизительное число отдельных fits:

\[
900\text{–}1300.
\]

Цель: определить, можно ли доверять реализации и начинается ли ожидаемое статистическое улучшение.

## 10.2. Полный бенчмарк

Состав:

- все C-сценарии;
- S01–S06;
- T01–T10;
- R01–R15;
- M01–M08;
- I01–I04;
- B01 на 12 representative scenarios;
- A01–A09;
- D01–D04.

Повторы:

- 50 для стандартных сравнений;
- 100 для robustness и initialization;
- 200 на пограничных failure scenarios;
- nested CV на real data.

Оценка числа fits:

\[
18000\text{–}28000.
\]

Точная величина зависит от числа baseline и refinement-grid после screening.

## 10.3. Усиленный запуск для публикации

Этап 1. Screening:

- 250–300 configurations;
- 20 repeats;
- one-factor grids и space-filling design только для вторичных взаимодействий.

Этап 2. Confirmatory:

- 100–150 заранее выбранных configurations;
- 100 repeats;
- зафиксированные hypotheses и primary metrics.

Этап 3. Rare failures:

- 20–30 boundary configurations;
- 500 repeats.

Этап 4. Baseline и ablation:

- 40–60 representative configurations;
- 8–10 statistical methods;
- 100 paired repeats.

Этап 5. Real data:

- nested CV;
- bootstrap stability;
- повтор на другой машине или backend.

Оценка:

\[
60000\text{–}90000
\]

отдельных fits.

Список confirmatory scenarios, seeds, exclusions и критерии успеха фиксируются до запуска confirmatory-этапа.

---

# 11. Итоговый диагностический чек-лист

| Вопрос | Метрика или эксперимент | Критерий ответа |
|---|---|---|
| Реализация корректна? | C01–C12 | все обязательные tolerances пройдены |
| Локальные статистики корректны? | C01–C02 | relative error и shift invariance |
| Нормировка направления корректна? | C03, C08 | \(\|\beta\|=1\), product preserved |
| Геометрические инвариантности соблюдены? | C04–C06 | projective gaps ниже tolerance |
| \(h_0\) и \(\rho_k\) выбираются корректно? | C07 | monotonicity и mass constraint |
| ALS корректно решает noiseless task? | C08, C10 | near-zero loss, nonincreasing objective |
| CG корректен? | C09 | residual, solution gap, `info=0` |
| Stopping масштабно устойчив? | C11 | invariant direction under scale \(Y\) |
| Результат вычислительно воспроизводим? | C12 | chunk/thread/dtype criteria |
| Метод сходится? | G17–G19 | objective, projective delta и CG residual стабилизируются |
| Результат статистически воспроизводим? | I04 | small within-data algorithm variance в working regime |
| Есть улучшение с \(n\)? | S02–S03 | negative loss slope с CI |
| Где начинается деградация по шуму? | S04/R01 | first level success rate below 0.8 |
| Где начинается degradation по \(d\)? | S05 | \(d_{stat}(n/d)\) |
| Какие параметры критичны? | T01–T10 | effect \(\geq0.02\), CI excludes 0 |
| Достаточно ли направлений? | T01/T02/A04 | plateau и gap к full basis |
| Достаточно ли центров? | T03 | quality plateau и seed variance |
| Корректен ли local mass regime? | T04/T09/R13/R14 | low failure и controlled bias |
| Чувствителен ли метод к prior? | T05/I01 | basin и regularization trade-off |
| Достаточна ли точность текущего ALS+CG? | T07 | gap к strict run \(\leq0.01\) |
| Устойчив ли метод к heavy tails? | R02 | degradation threshold |
| Устойчив ли метод к выбросам? | R03–R04 | contamination breakdown |
| Устойчив ли метод к correlation? | R06–R07 | correlation/condition boundary |
| Нужна ли standardization? | R08 | raw vs standardized paired effect |
| Как влияет sparsity \(\beta^*\)? | R10/S06 | support-specific curves |
| Как влияет гладкость \(f\)? | R11–R12 | frequency/smoothness boundaries |
| Как влияет неоднородная плотность? | R13–R14 | quality linked to local mass |
| Что происходит при misspecification? | R15 | rank-one bias curve |
| Какова стоимость вычислений? | M01–M08 | time/memory models и exponents |
| Масштабирование соответствует формулам? | M01–M06 | fitted exponents и residuals |
| Параллелизм полезен? | M07 | speedup и efficiency |
| Метод превосходит baseline? | B01 | paired recovery/prediction и cost frontier |
| Внешняя адаптация нужна? | A01–A03 | repeated paired benefit |
| Random sketch ограничивает качество? | A04–A05 | gap и variance |
| Current regularization полезна? | A06 | lower variance without excessive bias |
| Mass quantile нужна? | A07 | fewer weak-center failures |
| Center perturbation нужна? | A08 | scenario-specific paired effect |
| Negative controls обнаружены? | A09 | все broken variants провалили unit tests |
| Направление стабильно на реальных данных? | D01–D04 | pairwise cosine и bootstrap stability |
| Есть переносимость на test? | D01–D04 | nested-CV metrics |
| Метод или реализация ограничивают качество? | диагностическая лестница | первый oracle, устраняющий gap |
| Где практическая граница применимости? | S05, R, M, D | одновременно пройдены quality, failure, time и memory thresholds |
| Что исследовать дальше? | failure attribution | компонент, связанный с первым устойчивым gap |

## 11.1. Правило финального ответа по реализации

Реализация признаётся математически корректной, если одновременно:

1. пройдены C01–C10;
2. C11 не обнаруживает масштабно-зависимого stopping;
3. C12 не обнаруживает существенной зависимости от chunk/thread;
4. failed runs не скрываются;
5. linear и tanh regimes показывают улучшение с \(n\);
6. strict ALS+CG не даёт систематически лучшего направления, чем standard settings, более чем на 0.01 cosine loss.

## 11.2. Правило практической границы

Для фиксированного класса данных practical boundary представляет максимальную сложность, при которой одновременно выполняются:

\[
P(c_\beta\geq0.8)\geq0.8,
\]

\[
P(\text{failed})\leq0.05,
\]

\[
T\leq T_{budget},
\]

\[
M_{peak}\leq M_{budget}.
\]

Бюджеты времени и памяти записываются до запуска scaling study.

## 11.3. Решения после бенчмарка

- Если C01–C07 не проходят, статистические эксперименты не интерпретируются.
- Если C09 не проходит, исправляется matrix-free оператор или обработка CG.
- Если T07 показывает gap к strict run, уточняются параметры текущего ALS+CG.
- Если full-basis oracle лучше random sketch, исследуется directional budget.
- Если oracle initialization лучше practical initializers, исследуется начальная оценка.
- Если full ADP не лучше step0, пересматриваются внешние \(h_k,\rho_k\), mass rule и renewal directions.
- Если качество растёт с \(n\), но падает при фиксированном низком \(n/d\), фиксируется статистическая граница.
- Если oracle и practical варианты одинаково деградируют, дальнейшая численная оптимизация текущей реализации не устранит этот предел.
- Если real-data prediction приемлемо, но направления нестабильны, коэффициенты не интерпретируются как устойчивый scientific score.

---

# 12. Первичные научные источники

1. Härdle, W.; Stoker, T. M. *Investigating Smooth Multiple Regression by the Method of Average Derivatives*. Journal of the American Statistical Association, 1989.
2. Powell, J. L.; Stock, J. H.; Stoker, T. M. *Semiparametric Estimation of Index Coefficients*. Econometrica, 1989.
3. Hristache, M.; Juditsky, A.; Spokoiny, V. *Direct Estimation of the Index Coefficient in a Single-Index Model*. Annals of Statistics, 29(3), 595–623, 2001.
4. Hristache, M.; Juditsky, A.; Polzehl, J.; Spokoiny, V. *Structure Adaptive Approach for Dimension Reduction*. Annals of Statistics, 2001.
5. Xia, Y.; Tong, H.; Li, W. K.; Zhu, L.-X. *An Adaptive Estimation of Dimension Reduction Space*. Journal of the Royal Statistical Society: Series B, 64(3), 363–410, 2002.
6. Li, K.-C. *Sliced Inverse Regression for Dimension Reduction*. Journal of the American Statistical Association, 86, 316–342, 1991.
7. Li, K.-C. *On Principal Hessian Directions for Data Visualization and Dimension Reduction*. Journal of the American Statistical Association, 87, 1025–1039, 1992.
8. Cook, R. D.; Weisberg, S. *Sliced Inverse Regression for Dimension Reduction: Comment*. Journal of the American Statistical Association, 1991. SAVE используется как стандартная inverse-regression линия второго момента.
9. Ichimura, H. *Semiparametric Least Squares and Weighted SLS Estimation of Single-Index Models*. Journal of Econometrics, 58, 71–120, 1993.
10. Huber, P. J. *Robust Estimation of a Location Parameter*. Annals of Mathematical Statistics, 35, 73–101, 1964. Используется как основа contamination protocol, а не как источник гарантий ADP.

# 13. Проверка полноты отчёта

Финальный отчёт считается полным, если он содержит:

- паспорт серии в `single_index_series.csv`;
- все failed runs;
- unit-test results;
- scenario definitions и seed list;
- primary metric до просмотра результатов;
- distributions, а не только means;
- baseline tuning protocol;
- ablation paired differences;
- empirical time и memory models;
- practical boundaries по noise, dimension, correlation и initialization;
- отдельное заключение о методе и отдельное заключение о реализации;
- список выводов, которые результаты не позволяют сделать.

---

# 14. Проектный план внедрения

> **Для agentic workers:** выполнять задачи последовательно с
> `superpowers:test-driven-development` и отмечать checkbox после проверки.
> Для исполнения всего документа использовать
> `superpowers:executing-plans`; существующие публичные API и CSV-файлы нельзя
> менять без отдельного compatibility-теста.

**Цель:** реализовать научный протокол single-index ADP как возобновляемую серию
экспериментов поверх существующих модулей проекта, сохраняя первичные данные и
метаданные в нормализованных CSV.

**Архитектура:** `adp/common` предоставляет только универсальные операции
идентификации, ресурсов и CSV. Пакет `adp/evaluation/single_index` описывает
сценарии, хранение серии, orchestration и отчёты этого протокола. Текущие
`adp/evaluation/runner.py` и `reports.py` остаются совместимыми и предоставляют
общие операции запуска метода и расчёта базовых метрик.

**Стек:** Python, NumPy, pandas, `csv`, `concurrent.futures`, matplotlib, pytest.

## 14.1. Карта файлов

| Файл | Действие | Ответственность |
|---|---|---|
| `adp/common/experiment_log.py` | изменить | fingerprint конфигурации, расширенный стабильный `run_id`, атомарная замена single-row CSV |
| `adp/evaluation/runner.py` | изменить | выделить переиспользуемый запуск одного метода без изменения `run_benchmark_suite` |
| `adp/evaluation/single_index/__init__.py` | создать | узкий экспорт single-index benchmark API |
| `adp/evaluation/single_index/types.py` | создать | dataclass типов `SingleIndexScenario`, `SingleIndexJob`, `SingleIndexSeriesConfig` |
| `adp/evaluation/single_index/schema.py` | создать | единственный источник порядка CSV-столбцов и `SCHEMA_VERSION` |
| `adp/evaluation/single_index/scenarios.py` | создать | реестр C/S/T/R/M/I/B/A/D и profile selection |
| `adp/evaluation/single_index/datasets.py` | создать | synthetic factories, real-data cache, checksum и split metadata |
| `adp/evaluation/single_index/baselines.py` | создать | единый adapter contract для random/OLS/SIR/SAVE/PHD/PLS/OPG/ADE/MAVE |
| `adp/evaluation/single_index/correctness.py` | создать | специализированные executors C01–C12 и dense references малой размерности |
| `adp/evaluation/single_index/executors.py` | создать | dispatch correctness/recovery/scaling/real-data jobs по типу сценария |
| `adp/evaluation/single_index/storage.py` | создать | lifecycle серии, worker shards, resume и финальная публикация |
| `adp/evaluation/single_index/runner.py` | создать | dispatch jobs, progress, failure capture, resource boundaries |
| `adp/evaluation/single_index/reports.py` | создать | summaries, failure attribution, protocol plots |
| `adp/evaluation/cli.py` | изменить | подкоманда `single-index` и её аргументы |
| `adp/evaluation/__init__.py`, `adp/benchmarks.py` | изменить | совместимые re-export новых entrypoints |
| `README.md` | изменить | команды запуска, структура серии, resume и семантика ресурсов |

Тесты распределяются по ответственности:

- `tests/test_single_index_benchmark_schema.py`;
- `tests/test_single_index_benchmark_scenarios.py`;
- `tests/test_single_index_benchmark_executors.py`;
- `tests/test_single_index_benchmark_storage.py`;
- `tests/test_single_index_benchmark_runner.py`;
- `tests/test_single_index_benchmark_reports.py`;
- существующие `tests/test_benchmarks.py`, `tests/test_cli.py`,
  `tests/test_experiment_csv_log.py` остаются regression-набором.

## 14.2. Task 1. Идентификаторы и атомарные CSV-операции

**Файлы:** `adp/common/experiment_log.py`,
`tests/test_experiment_csv_log.py`.

- [ ] Добавить RED-тест на одинаковый fingerprint для mapping с разным порядком
  ключей и разный fingerprint при изменении вложенного параметра.
- [ ] Добавить RED-тест на обратную совместимость текущего четырёхаргументного
  `stable_run_id(...)` и на чувствительность к новому `config_fingerprint`.
- [ ] Добавить RED-тест: атомарная замена single-row CSV не оставляет `.tmp` и
  не повреждает старый файл при ошибке записи.
- [ ] Реализовать интерфейсы:

```python
def configuration_fingerprint(values: Mapping[str, Any]) -> str: ...

def stable_run_id(
    experiment: str,
    scenario_id: str,
    method: str,
    seed: int,
    *,
    config_fingerprint: str = "",
) -> str: ...

def replace_single_row_csv(path: str | Path, row: Mapping[str, Any]) -> Path: ...
```

- [ ] Канонизировать mapping рекурсивно; `Path`, NumPy scalars и tuples должны
  давать стабильное скалярное представление без JSON-ячейки в итоговой CSV.
- [ ] Выполнить
  `python -m pytest tests/test_experiment_csv_log.py -q`; ожидается PASS.
- [ ] Commit: `feat: extend stable experiment CSV identities`.

## 14.3. Task 2. Типы, схемы и реестр сценариев

**Файлы:** `adp/evaluation/single_index/types.py`,
`adp/evaluation/single_index/schema.py`,
`adp/evaluation/single_index/scenarios.py`,
`tests/test_single_index_benchmark_schema.py`,
`tests/test_single_index_benchmark_scenarios.py`.

- [ ] Написать RED-тесты на уникальность scenario ID, положительные размеры,
  конечные числовые параметры и разрешённые уровни `C/S/T/R/M/I/B/A/D`.
- [ ] Написать RED-тесты, что `smoke` является подмножеством `minimal`, а
  `minimal` — подмножеством `full`; smoke содержит хотя бы C01, S01, M01 и один
  baseline. `publication` является отдельным зафиксированным confirmatory
  профилем из раздела 10.3 и не обязан быть надмножеством screening-сетки.
- [ ] Зафиксировать dataclass-контракт:

```python
@dataclass(frozen=True, slots=True)
class SingleIndexScenario:
    scenario_id: str
    family: str
    executor: Literal["correctness", "recovery", "scaling", "real_data"]
    hypothesis: str
    data: Mapping[str, Scalar]
    algorithm: Mapping[str, Scalar]
    solver: Mapping[str, Scalar]
    repeats: int
    methods: tuple[str, ...]
    record_solver_trace: bool = False

@dataclass(frozen=True, slots=True)
class SeedBundle:
    data: int
    beta: int
    centers: int
    directions: int
    init: int

@dataclass(frozen=True, slots=True)
class SingleIndexJob:
    scenario: SingleIndexScenario
    method: str
    repeat: int
    seeds: SeedBundle
    run_id: str

@dataclass(frozen=True, slots=True)
class SingleIndexSeriesConfig:
    profile: str
    base_seed: int
    jobs: int
    statistics_workers: int
    retry_failed: bool = False
```

- [ ] В `schema.py` определить стабильные tuples `SERIES_COLUMNS`,
  `RUN_COLUMNS`, `ITERATION_COLUMNS`, `SOLVER_ITERATION_COLUMNS`,
  `INITIAL_PARAMETER_COLUMNS`, `FAILURE_COLUMNS`, `ARTIFACT_COLUMNS`.
- [ ] Включить в `RUN_COLUMNS` все поля `algorithm_*`, `full_run_*`,
  `result_persist_time_sec`, status/error/stage, final quality metrics и seeds.
- [ ] В реестре выразить значения из разделов 5 и 10 этого документа, не
  копируя формулы генерации в runner. C01–C12 получают `executor="correctness"`,
  D01–D04 — `executor="real_data"`, остальные сценарии используют recovery или
  scaling executor.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_schema.py tests/test_single_index_benchmark_scenarios.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: define single-index benchmark registry`.

## 14.4. Task 3. Хранилище серии и resume

**Файлы:** `adp/evaluation/single_index/storage.py`,
`tests/test_single_index_benchmark_storage.py`.

- [ ] Написать RED-тест, что новая серия создаёт каталог и первыми публикует
  `single_index_series.csv` со статусом `running` и
  `single_index_initial_parameters.csv` со всеми jobs.
- [ ] Написать RED-тест на PID-шарды: объединение потоковое, порядок заголовков
  проверяется, финальный файл публикуется атомарно, успешно слитые шарды удаляются.
- [ ] Написать RED-тест resume: успешные `run_id` пропускаются, failed
  повторяются только при `retry_failed=True`, дубликаты в финальном `runs.csv`
  запрещены.
- [ ] Написать RED-тест восстановления после прерывания: run-shard считается
  commit-marker, orphan iteration rows удаляются, а уже завершённый job из
  неслитого shard не запускается повторно.
- [ ] Написать RED-тест, что несовпадение schema/fingerprint останавливает resume
  до создания worker-а.
- [ ] Реализовать интерфейс:

```python
class SingleIndexSeriesStore:
    @classmethod
    def create(cls, root: Path, config: SingleIndexSeriesConfig, jobs: Sequence[SingleIndexJob]) -> Self: ...

    @classmethod
    def resume(cls, series_dir: Path, config: SingleIndexSeriesConfig) -> Self: ...

    def pending_jobs(self, jobs: Sequence[SingleIndexJob]) -> Iterator[SingleIndexJob]: ...
    def append_worker_rows(self, table: str, rows: Iterable[Mapping[str, Scalar]]) -> int: ...
    def finalize(self, *, status: str) -> Mapping[str, Path]: ...
```

- [ ] Записывать пути в `artifacts.csv` относительно каталога серии и добавлять
  размер файла после успешного создания.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_storage.py tests/test_experiment_csv_log.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: add resumable single-index CSV series store`.

## 14.5. Task 4. Исполнители, данные и baseline adapters

**Файлы:** `adp/evaluation/single_index/datasets.py`,
`adp/evaluation/single_index/baselines.py`,
`adp/evaluation/single_index/correctness.py`,
`adp/evaluation/single_index/executors.py`,
`tests/test_single_index_benchmark_executors.py`.

- [ ] Написать RED-тесты synthetic factories: одинаковый seed воспроизводит
  данные, разные seed-компоненты меняют только соответствующий объект, а
  `corr > 0` создаёт ненулевые внедиагональные ковариации координат, а не только
  случайный сдвиг и общий scale.
- [ ] Написать RED-тесты, что NaN/inf, неверная beta-размерность, неположительные
  `n_centers`/`n_directions` и недопустимые covariance parameters отклоняются до
  создания job rows с нечисловыми данными.
- [ ] Реализовать correctness dispatcher C01–C12. Каждый executor возвращает
  плоский `RunOutcome` с primary/secondary metrics и diagnostic rows; dense
  матрицы разрешены только в явно малых reference-сценариях C01/C08/C09.
- [ ] Выделить общий baseline adapter:

```python
@dataclass(frozen=True, slots=True)
class RunOutcome:
    metrics: Mapping[str, Scalar]
    iterations: tuple[Mapping[str, Scalar], ...]
    solver_iterations: tuple[Mapping[str, Scalar], ...]
    stop_reason: str

def execute_job(job: SingleIndexJob, config: SingleIndexSeriesConfig) -> RunOutcome: ...
```

- [ ] Переиспользовать текущие adapters SIR/SAVE/PHD/PLS из
  `adp/evaluation/runner.py`. Random direction, OLS, step0 и full ADP реализовать
  без внешних зависимостей. OPG/ADE/MAVE/RMAVE подключать через явные adapters;
  отсутствие optional dependency записывать как `status="unavailable"`, а не
  молча исключать метод из сравнения.
- [ ] Для D01–D04 использовать локальный пакет `adp_D1_data`: читать
  `dataset_manifest.csv`, разрешать файлы только внутри `prepared/`, проверять
  target, `rows`, `features` и SHA-256. Исходные CSV хранить вне каталога
  результатов; в таблицах серии записывать источник, относительный локальный
  путь, размер и checksum. Основной D-путь не выполняет сетевой fetch даже при
  `allow_download=True`.
- [ ] Для real data сохранять только индексы split/fold и seeds, но не копировать
  полные датасеты в CSV серии. Preprocessing fit выполняется только на train
  части соответствующего fold.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_executors.py tests/test_adp.py tests/test_benchmarks.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: add single-index benchmark executors`.

## 14.6. Task 5. Потоковый runner и ресурсные границы

**Файлы:** `adp/evaluation/runner.py`,
`adp/evaluation/single_index/runner.py`,
`tests/test_single_index_benchmark_runner.py`, `tests/test_benchmarks.py`.

- [ ] Сначала добавить regression-тест текущего `run_benchmark_suite`, затем
  выделить из него helper запуска одного метода так, чтобы старые столбцы и
  значения остались совместимыми.
- [ ] Написать RED-тест, что full-run окно включает data generation, создание
  модели, fit/baseline, метрики и запись worker rows; проверить
  `full_run_time_sec >= algorithm_time_sec`.
- [ ] Написать RED-тест failed job: строка run и строка failure сохраняются,
  traceback не прерывает серию, доступные ресурсные поля не теряются.
- [ ] Написать RED-тест process pool fallback и progress-строк вида
  `completed/total scenario=... method=... seed=...` с немедленным flush.
- [ ] Реализовать публичный интерфейс:

```python
def run_single_index_benchmark(
    config: SingleIndexSeriesConfig,
    output_root: str | Path,
    *,
    resume: str | Path | None = None,
) -> Mapping[str, Path]: ...
```

- [ ] Seed разделить детерминированно на data, beta, centers, directions и init;
  paired jobs получают одинаковые неизменяемые компоненты.
- [ ] При process-level jobs установить `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; `statistics_workers=1` оставить
  безопасным default, значение 2 и выше включать только явно.
- [ ] Не накапливать iteration rows всей серии в родительском процессе. Worker
  пишет shards, родитель хранит только progress/status и выполняет merge.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_runner.py tests/test_benchmarks.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: run single-index benchmark series`.

## 14.7. Task 6. Агрегаты, failure attribution и графики

**Файлы:** `adp/evaluation/single_index/reports.py`,
`tests/test_single_index_benchmark_reports.py`.

- [ ] Написать RED-тесты для median/mean/IQR/q05/q95, bootstrap CI, Wilson CI,
  success/failure rate и worst-five selection на маленькой фиксированной таблице.
- [ ] Написать RED-тест, что failed rows входят в denominator success rate, но
  NaN quality не попадает в quantile calculation.
- [ ] Написать RED-тесты scaling fit для M01–M06 и paired difference для
  baseline/ablation с совпадающими data seeds.
- [ ] Реализовать чтение первичных CSV с явными `usecols`/`dtype`; отчёты не
  должны зависеть от сохранённого pandas index.
- [ ] Реализовать графики G01–G21 только из CSV. Подписи пользовательских PNG —
  на русском; имена столбцов и machine-readable значения остаются английскими.
- [ ] Если построение отдельного PNG падает, численные CSV остаются опубликованы,
  а `artifacts.csv` получает status/error этого артефакта.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_reports.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: report single-index benchmark series`.

## 14.7.1. Task 6.5. Manifest-driven источник D01–D04

**Файлы:** `adp/evaluation/single_index/datasets.py`,
`tests/test_single_index_benchmark_executors.py`.

- [ ] Добавить RED-тест с временным `dataset_manifest.csv` и
  `prepared/D01_airfoil_self_noise.csv`. Желаемый вызов остаётся
  `load_cached_real_dataset("D01", data_dir, allow_download=False)`, а результат
  обязан вернуть target из manifest, `(n, d)`, checksum, официальный URL и путь
  prepared-файла.
- [ ] Добавить RED-тесты несовпадения checksum, размерности, повторяющегося ID,
  отсутствующего файла и пути с `..`; ожидаются явные `ValueError` или
  `DatasetUnavailable` до вызова executor.
- [ ] Заменить `_OPENML_NAMES` на manifest parser. Parser требует столбцы
  `id,file,rows,features,target,sha256,official_page`, проверяет единственность
  ID и разрешает `file` относительно `<data_dir>/prepared` без выхода из этого
  каталога.
- [ ] Не вызывать `_download_openml`: параметр `allow_download` сохранить ради
  совместимости сигнатуры, но отсутствие локального пакета всегда считать
  `DatasetUnavailable`.
- [ ] Выполнить
  `python -m pytest tests/test_single_index_benchmark_executors.py -q`;
  ожидается PASS.
- [ ] Commit: `feat: load D benchmarks from adp D1 data package`.

## 14.8. Task 7. CLI, smoke-проверка и документация

**Файлы:** `adp/evaluation/cli.py`, `adp/evaluation/__init__.py`,
`adp/benchmarks.py`, `tests/test_cli.py`, `README.md`.

- [ ] Добавить RED-тесты CLI для `single-index --help`, `--profile`, `--jobs`,
  `--statistics-workers`, `--resume`, `--retry-failed`, `--max-scenarios`,
  `--data-dir`, `--allow-download` и некорректных значений.
- [ ] Добавить подкоманду без изменения текущих форм
  `python run_benchmarks.py --quick`, `--grid` и `stress`.
- [ ] Задокументировать команды:

```bash
python run_benchmarks.py single-index \
  --profile smoke \
  --jobs 1 \
  --statistics-workers 1 \
  --data-dir adp_D1_data \
  --output benchmark_outputs/single_index

python run_benchmarks.py single-index \
  --resume benchmark_outputs/single_index/<series_id>
```

- [ ] Выполнить CLI smoke; ожидается exit code 0, отсутствие JSON и наличие
  `series/runs/iterations/initial_parameters/summary/artifacts` CSV.
- [ ] Выполнить `python -m pytest -q`; ожидается PASS всего набора.
- [ ] Выполнить `git diff --check`; ожидается отсутствие whitespace errors.
- [ ] Commit: `docs: document single-index benchmark workflow`.

## 14.9. Условия завершения реализации

План выполнен только если одновременно:

1. smoke-профиль запускается из `run_benchmarks.py` и создаёт отдельный каталог серии;
2. primary rows пишутся потоково и не требуют хранения всех iterations в RAM;
3. series можно возобновить без повторения успешных jobs и без дубликатов `run_id`;
4. failed runs, ошибки и последние доступные ресурсы сохраняются;
5. algorithm и full-run time/RSS имеют разные явно проверенные границы;
6. все агрегаты и графики воспроизводятся из CSV без исходных Python objects;
7. старые benchmark, confirmatory и stress entrypoints проходят regression-тесты;
8. JSON-артефакты для новой серии отсутствуют;
9. обязательный baseline либо имеет строки результатов, либо явно помечен
   `unavailable`; публикационный вывод о сравнении не строится при отсутствии
   заранее обязательного метода;
10. полный pytest и `git diff --check` проходят.
