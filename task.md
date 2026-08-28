# Разбор новой версии ADP и реестр экспериментов

## Исходные документы

Основной источник новой спецификации single-index и multi-index —
`multiindex.tex`. Файл `manifold.tex` почти полностью совпадает со старым
`tex/manifold_v2.tex`, кроме случайной буквы `a` в начале файла.
`manifold-ade.tex` описывает отдельное manifold-расширение.

При реализации нельзя молча смешивать формулы из этих версий. Для single- и
multi-index ниже за основу принят `multiindex.tex`; старые требования из
`manifold.tex` сохранены как отдельные варианты или эксперименты.

## Необходимые изменения алгоритма

### Общие изменения для single-index и multi-index

1. Нормировать статистики `I_j` и `U_j` на локальную массу и решать задачу

   \[
   \sum_j \operatorname{mass}_j
   \left\|I_j-U_j(\ldots)\right\|^2.
   \]

   Сейчас `ADP/engine/statistic.py` возвращает статистики, умноженные на
   `mass`, а `ADP/cli.py` не передаёт `statistics.mass` в `solve()`. В
   результате текущий функционал использует другое взвешивание, близкое к
   `mass^2`.

   Классификация относительно текущей реализации: `ESTIMATOR`.

2. Развести две величины:

   - `mass = sum(weights)` — в TeX именно она обозначена как `n_eff`;
   - настоящий effective sample size
     \((\sum w)^2 / \sum w^2\), уже существующий в коде как `n_eff`.

3. Добавить в статистики локальное среднее ответа

   \[
   S_j = \frac{\sum_i Y_i w_{ij}}{\sum_i w_{ij}}.
   \]

   Оно требуется для step-wise data fit и stopping rule:

   \[
   \operatorname{err}_k
   = \sum_j \left(Y_{i(j)}-S_j^{(k)}\right)^2.
   \]

4. Хранить не только последний индекс, но и:

   - лучший шаг по `err_k`;
   - полный trace `h_k`, `rho_k`/`alpha_k`, `err_k`;
   - solver status, число итераций и residual certificate;
   - метрики качества индекса или подпространства.

### Single-index

Новая версия задаёт локализующий тензор

\[
T_k^2 = h_k^{-2}
\left[
\rho_k^2(I-\beta\beta^\top)+\beta\beta^\top
\right].
\]

Поэтому аргумент ядра должен иметь вид

\[
\frac{
\rho_k^2\|x_i-x_j\|^2
+(1-\rho_k^2)\langle x_i-x_j,\beta\rangle^2
}{h_k^2}.
\]

В `ADP/engine/weights.py:42` сейчас отсутствует множитель
`1 - rho**2`. Из-за этого даже `rho=1` не даёт изотропный первый шаг.
Ту же формулу нужно использовать в `calculate_rho_k()` при поиске максимально
допустимого `rho`.

Дополнительно требуется:

- передавать в HPAO/LSMR `mass`, `lambda_penalty` и TeX-значение `k_max`;
- использовать default `k_max = 3` для single-index вместо текущих `10`;
- строить локальную PCA-инициализацию по

  \[
  J_{\mathrm{EDR}}
  = \sum_j \operatorname{mass}_j g_jg_j^\top;
  \]

- не использовать явный `inverse` из TeX: оставить `lstsq`, QR или SVD;
- оставить `local_ridge` явной опцией, поскольку ridge меняет estimator;
- обновлять `beta_init = beta_k` перед построением статистик следующего шага;
- нормализовать результат и выравнивать знак относительно предыдущей оценки.

### Multi-index

Второй вариант тензора из TeX уже соответствует текущему
`calculate_multi_weight()`:

\[
T_k^2 = h_k^{-2}
\left[
\alpha_k^2(I-P^\top P)+P^\top\Lambda P
\right].
\]

Его можно сохранить как основной вариант. Нельзя строить плотную матрицу
`d x d`: действие тензора уже выражается через ортогональную компоненту и
малую матрицу размера `m x m`.

После AO требуется восстановить структурные собственные значения:

\[
G = \sum_j \operatorname{mass}_j l_jl_j^\top,
\qquad
\mathcal J = B^\top G B.
\]

`P` и `Lambda` нужно получать низкоранговым SVD без материализации
`mathcal J`. Текущий `_solver_index()` использует только
`coefficients.T @ coefficients` и не учитывает локальные массы.

Новая main procedure явно требует изотропные направления

\[
\phi = z/\|z\|, \qquad z\sim N(0,I),
\]

тогда как текущий CLI генерирует направления из локализующего тензора. Закон
направлений меняет конечный random-sketch objective, поэтому это изменение
классифицируется как `ESTIMATOR`. Старый вариант следует оставить отдельной
экспериментальной опцией.

## Противоречия и неоднозначности в TeX

Перед изменением кода необходимо явно выбрать интерпретацию следующих мест.

1. Для multi-index написано «наименьшее `alpha_k`, обеспечивающее массу».
   Поскольку масса убывает при росте `alpha`, это почти всегда даст `alpha=0`.
   По single-index разделу и математической монотонности должно использоваться
   наибольшее допустимое значение.

2. В начале `multiindex.tex` встречается `T = h^{-2} I`, хотя в основной
   процедуре корректно `T = h^{-1} I`.

3. `P` одновременно описан как отображение `R^d -> R^m` и как проектор
   `R^d -> R^d`. В коде следует сохранить контракт `P.shape == (m, d)` и
   проектор `P.T @ P` формы `(d, d)`.

4. Для multi-index закон направлений явно изотропный, а в новой single-index
   процедуре сказано только «renew directions». Это нужно закрепить отдельным
   параметром эксперимента, а не выбирать молча.

5. Формула коррелированного дизайна

   \[
   X_i = \sigma_X\{\tau z_0 + (1-\tau)z_i\}
   \]

   не делает `tau` непосредственной корреляцией и не сохраняет дисперсию
   `sigma_X^2` для промежуточных значений. В метаданных эксперимента следует
   записывать точную формулу, а не только имя параметра.

6. `manifold.tex` начинается с `a% !TEX ...`; лишнюю букву следует удалить,
   если файл будет компилироваться.

## Полный реестр экспериментов

В реестр включены активные синие задания, неокрашенные экспериментальные
требования и отдельно отмеченные закомментированные синие пункты.

### Single-index

#### SI-1. Генерация данных

Источник: `multiindex.tex:477`.

- `n in {800, 1000, 1200, 2000}`, default `1000`;
- `d in {10, 20, ..., 90, 100}`, default `100`;
- `sigma_eps in {0, 0.1, 0.2, 0.4, 0.6, 0.8, 1}`, default `0.2`;
- `tau in {0, 0.2, 0.4, 0.8}`, default `0.4`;
- `f(x) = sin(s x)` и `f(x) = x sin(s x)`;
- `s in {1, 2, 3, 4}`, default `3` для `x sin(s x)`;
- проверять условие `n / sigma_eps**2 >= 20 d`;
- строить `Y` против `X.T @ beta_true`;
- предусмотреть перегенерацию всей реализации данных.

#### SI-2. Инициализация

Источник: `multiindex.tex:635`.

- `N_lin = d + s N_loc`, `s in {1, 2, 3}`, default `3`;
- `N_J = s n / N_loc`, включая исследование зависимости от `N_J`;
- displacement `nu in {0, 0.1, 0.5, 1}`, default `0.1`;
- сравнить обучение на всей выборке и с исключением testing indices;
- измерять `abs(beta_true @ beta_init)`;
- выводить первые две собственные величины `J_EDR`;
- подбирать `N_lin` по качеству инициализации;
- фиксировать частоту вырожденной или неидентифицируемой инициализации.

#### SI-3. Внутренний AO

- измерить one-step improvement;
- записывать качество после каждого AO-шага;
- сравнить `k_max in {3, 5, 7}`, default `3`;
- исследовать влияние `lambda`;
- проверить, возникает ли деградация после лишних AO-шагов.

#### SI-4. Structural adaptation и tuning

Источник: `multiindex.tex:717`.

- `N_loc in {7, 10, 15, 20}`, default `20`;
- `N_phi in {N_loc, 2 N_loc, 3 N_loc}`, default `N_loc`;
- `lambda in {0, 0.05, 0.1, 0.5, 1}`, default `0.05`;
- `a in {2**(1/4), 2**(1/2), 2}`, default `sqrt(2)`;
- `k_max in {3, 5, 7}`, default `3`;
- `h_min = s sigma_X / sqrt(n)`, `s in {1, 2, 3}`, default `3`;
- разрешить ручной выбор всех параметров, кроме ядра;
- старый документ дополнительно требует исследовать влияние `sigma_X` и
  `sigma_eps` при `d=10`.

#### SI-5. Диагностика траектории

- на каждом шаге записывать `h_k`;
- записывать `h_k / rho_k`;
- записывать `err_k`;
- записывать `abs(beta_k @ beta_true)`;
- строить траекторию `rho_k`;
- строить или выводить `h_0`;
- строить `Y` против `X.T @ beta_k` и `X.T @ beta_true`.

#### SI-6. Stopping rule

Источник: `multiindex.tex:895`.

- сравнить последний допустимый шаг со шагом минимального test fit;
- сравнить initialization, full procedure и stopping-selected result;
- проверить предположение об унимодальности `err_k`;
- не удалять неуспешные запуски из итоговой таблицы.

#### SI-7. Сравнение с другими методами

Источник: `multiindex.tex:907`.

- ADE;
- SIR;
- MAVE;
- все методы запускать на одной реализации данных;
- method-specific randomness разделять, если common random numbers не являются
  частью протокола.

#### SI-8. Breaking dimension

Источник: `multiindex.tex:921`.

- по 100 симуляций для каждого `d in {10, 20, ..., 100}`;
- повторять для каждой комбинации `n`, `sigma_eps`, `tau` и link function;
- box plots для initialization, full procedure, stopping rule, ADE, SIR, MAVE;
- агрегировать не только качество, но также wall-clock, peak memory и failures.

### Multi-index

#### MI-1. Генерация данных

- генерировать случайный ортонормальный истинный базис;
- additive link `f(x1, x2) = x1**2 + sin(s x2)`;
- multiplicative link `f(x1, x2) = x1 sin(s x2)`;
- `s in {1, 2, 3, 4}`;
- для `m > 2` явно определить продолжение link function, а не оставлять
  неуказанные координаты неиспользуемыми.

#### MI-2. Инициализация

Источник: `multiindex.tex:1368`.

- сетка по `N_lin` и `N_J`;
- вывод первых `m + 1` собственных значений `J_EDR`;
- projector/subspace error `epsilon(P_init, P_true)`;
- частота rank-deficient initialization;
- сравнение local PCA и других уже существующих инициализаций должно быть
  отдельным экспериментом, а не скрытой заменой default.

#### MI-3. Внутренний AO

- one-step improvement;
- итоговое качество после каждого AO-шага;
- локальные rank losses;
- LSMR status, iterations, residual и stationarity diagnostics;
- влияние `lambda` и `k_max`.

#### MI-4. Варианты тензора

Старый `manifold.tex` требует реализовать и сравнить:

\[
T^2 = h^{-2}\{\alpha^2 I + P^\top\Lambda P\},
\]

и

\[
T^2 = h^{-2}\{\alpha^2(I-P^\top P)+P^\top\Lambda P\}.
\]

Второй вариант указан как default и совпадает с новой main procedure. Первый
вариант должен быть отдельным estimator-вариантом.

#### MI-5. Закон и обновление направлений

- сравнить fixed directions и redraw на каждом outer step;
- сравнить изотропные направления с направлениями из локализующего тензора;
- исследовать `N_phi`, особенно режимы около `N_phi = m`;
- не смешивать варианты в одном random stream.

#### MI-6. Structural trace и stopping

Источник: `multiindex.tex:1437`.

- `h_k`;
- `h_k / alpha_k`;
- projector error;
- test fit `err_k`;
- initialization/full/stopping comparison;
- отдельные причины остановки: `h_min`, `outer_steps`, local-mass limit,
  numerical failure.

#### MI-7. Сравнение методов

- SIR и MAVE на той же реализации данных;
- старый документ дополнительно требует dimension-breakdown для multi-index;
- качество сравнивать по главным углам или проекторной метрике, а не по
  элементам произвольного базиса.

### Manifold learning

Основной синий блок: `manifold-ade.tex:797`.

#### MAN-1. Сложность

- memory requirement как функция `n`, `d`, `m`, `N_J`, `N_phi`;
- computational cost по тем же параметрам;
- отдельно учитывать рабочую и постоянную память;
- не материализовать массивы форм `(J,n,d)` или плотные семейства проекторов
  `J x d x d`.

#### MAN-2. Локальные градиенты

- точность `grad g(x_j)`;
- зависимость от `N_lin`, manifold bandwidth и dimension;
- частота вырожденных локальных систем.

#### MAN-3. Инициализация и синхронизация

- качество после первого initialization run;
- качество после каждого synchronization repetition;
- частота failure;
- сравнение warm start из глобального multi-index подпространства с локальной
  инициализацией.

#### MAN-4. Structural adaptation

- one-step improvement;
- качество локальных подпространств до и после шага;
- влияние числа synchronization repetitions.

#### MAN-5. Регуляризация

- выбор `lambda_M`/`lambda_manifold`;
- сетка `N_loc`, `N_J`, `N_phi`, `lambda_M`, `k_max`, `a`, `h_min`;
- в артефактах хранить requested и effective значения.

#### MAN-6. Breaking dimension

- деградация с ростом `d`;
- одновременная регистрация качества, wall-clock, memory и failure rate.

### Синие задания, не являющиеся экспериментами

Они также должны быть сохранены в backlog:

- проверить elimination lemma для вспомогательных observables;
- проверить single-index closed form для локальных коэффициентов;
- проверить single-index closed form для обновления `beta`;
- вывести closed form для multi-index локальных коэффициентов;
- вывести closed form для обновления `B`;
- проверить lemma для manifold penalty;
- реализовать оба multi-index тензора;
- один синий фрагмент лишь окрашивает ссылку на уравнение и не задаёт работу.

Закомментированные синие пункты, которые всё ещё выражают потенциальные
требования:

- поддержать оба варианта train/test split;
- исследовать влияние `N_J`;
- проверить multi-index one-step и final quality;
- проверить положительность начального cosine в single-index.

## Минимальная последовательность реализации

1. Зафиксировать противоречия TeX: максимальное `alpha`, форму `P` и закон
   направлений.
2. Исправить single-index weight и поиск `rho`.
3. Нормировать `I/U`, передать `mass` в solver и добавить `S_j`.
4. Исправить mass-weighted local PCA initialization.
5. Восстановить multi-index `P, Lambda` низкоранговым SVD.
6. Добавить test-fit trace и выбор лучшего шага.
7. Оставить alternative tensors/direction laws отдельными вариантами.
8. Добавить малые dense reference tests и только после них запускать полные
   экспериментальные сетки.

## Текущее состояние проверки

На 29 августа 2026 года реализованы пункты 1–8 минимальной последовательности:

- новый single-index тензор и согласованный поиск максимального `rho`;
- нормированные `I/U`, внешний вес `mass` в HPAO-LSMR и локальное среднее `S`;
- mass-weighted gradient PCA для single- и multi-index initialization;
- mass-weighted low-rank SVD для `P, Lambda` без матрицы `d x d`;
- изотропные направления для нового multi-index режима;
- отдельные `new/legacy`, `isotropic/localized`, `orthogonal/full` варианты;
- trace `h`, `rho/alpha`, `h/(rho/alpha)`, `err`, quality, eigenvalues и solver
  diagnostics, а также выбор `best/last` шага;
- сохранение trace и выбранного шага в CSV экспериментальных запусков.
- полные однофакторные SI/MI-сетки и отдельные breaking-dimension сетки;
- train на всей выборке или без test indices, displacement центров и
  fixed/redraw directions как явные estimator-варианты;
- первые `m + 1` собственных значения локальной gradient PCA, initial/last/
  selected quality, wall-clock, traced peak memory и failures в артефактах;
- компактные CSV/Markdown-таблицы и читаемые PNG-графики без ADE/SIR/MAVE.

Изменения оценивателя изолированы явными параметрами. `new` является новым
default, а прежние формулы веса и ненормированных статистик доступны через
`estimator=legacy`. Для single-index default внутреннего AO равен `3`, для
multi-index — `5`; `lambda_penalty=0` также поддержан явно.

Проверка текущего checkout:

- `ruff format --check .` — 24 файла отформатированы;
- `ruff check .` — без ошибок;
- `pyright` — 0 ошибок и предупреждений;
- `pytest -q` — 21 тест проходит;
- `uv lock --check` — 69 пакетов разрешены;
- `import ADP` и двухшаговые smoke-run single/multi завершаются.

Reference-тесты покрывают обе single-index формулы, поиск `rho`, оба
multi-index тензора, dense и sparse статистики, устойчивость к большому
сдвигу `X`, mass-weighted PCA/SVD, rank-deficient failure, изотропные
направления, `lambda=0`, trace/stopping и публичный single-index solver.

Замер `calculate_statistics` для `(n,d,J,P)=(800,40,100,20)`, `float64`, пять
повторов: retained legacy median `0.022028 s`, new median `0.018570 s`;
`tracemalloc` peak соответственно `13.135 MiB` и `13.133 MiB`. `tracemalloc`
не учитывает все нативные выделения NumPy/BLAS.

## Реализованный каталог экспериментов и отчётов

Для single-index доступны селекторы `si-n`, `si-d`, `si-noise`, `si-tau`,
`si-link`, обе частотные сетки, `si-scale`, `si-nlin`, `si-centers`,
`si-displacement`, `si-training`, `si-kmax`, `si-nloc`, `si-nphi`,
`si-lambda`, `si-a`, `si-hmin`. Для multi-index добавлены соответствующие
сетки и отдельные `mi-init`, `mi-tensor`, `mi-direction-law`,
`mi-direction-refresh`.

`si-breaking` и `mi-breaking` содержат полный декартов продукт

```text
n(4) x d(10) x sigma_eps(7) x tau(4) x link/frequency(8) = 8960 points
```

и используют `100` повторов по умолчанию. Они намеренно не входят в короткие
алиасы `si`, `mi` и `report`, чтобы случайно не запустить по `896000` fit для
каждого режима. Сравнений с ADE, SIR, MAVE и другими внешними методами в
runner нет.

Каждая серия сохраняет полный `runs.csv`, воспроизводимый `series.json`,
агрегаты `summary.csv`, `trace_summary.csv`, отдельный `failures.csv` и
неперегруженный `summary.md`. Графики находятся в `plots/<selector>/`:
`quality.png`, `stages.png`, `runtime.png`, `memory.png`, `failures.png`,
`trajectory.png`. На ось выводится только один фактор; для полного факторного
эксперимента отчёт показывает компактные main-effect панели, а все сочетания
остаются в `runs.csv`.

Команды запуска:

```bash
uv run --group bench python -m ADP.experiment --list
uv run --group bench python -m ADP.experiment \
  --experiment si --profile full --runs 20
uv run --group bench python -m ADP.experiment \
  --experiment mi --profile full --runs 20
uv run --group bench python -m ADP.experiment \
  --experiment si-breaking --profile full
uv run --group bench python -m ADP.experiment \
  --experiment mi-breaking --profile full
```

Полные breaking-серии в текущем checkout определены, но целиком не
запускались: это отдельный длительный вычислительный этап. Проверены реальные
малые end-to-end серии `si-link` и `mi-tensor`, включая построение всех таблиц
и PNG; во втором случае trace подтвердил переключение `orthogonal -> full` на
втором outer-шаге.

Не выполнены и остаются отдельным этапом: manifold-сетки и
manifold-реализация. Сравнение с ADE/SIR/MAVE исключено по текущему требованию.
