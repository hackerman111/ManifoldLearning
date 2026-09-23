# ADP: карта алгоритмов из TeX

Эти заметки сжимают математическое содержание трёх исходников `tex/`. Они описывают рукописные алгоритмические спецификации, а не гарантированное поведение текущего Python-кода. Для реализации читайте тематическую заметку и сверяйте спорные детали по локаторам ниже.

Все исходные фрагменты трёх файлов учтены в каталоге: каждый код задаёт непересекающийся диапазон строк, а команда `sed` извлекает его из корня репозитория. Пометка `source-only` означает, что фрагмент намеренно не пересказан как часть ADP-алгоритма, но на него есть точный локатор; остальные фрагменты кратко пересказаны в указанной заметке. Это явные границы конспекта, а не молчаливые пропуски. Номера строк нужно обновить, если исходный TeX изменится.

## Какую заметку читать

| Задача | Заметка | Основные локаторы исходника |
|---|---|---|
| Один индекс, ADP | [single-index-adp.md](single-index-adp.md) | `[M-ADP-SI-*]`, `[X-SI-*]` |
| Общее EDR-подпространство с несколькими индексами | [multi-index-adp.md](multi-index-adp.md) | `[X-MI-*]`, `[M-MI-*]`, `[M-ADP-MI-*]` |
| Локально меняющееся EDR-подпространство (manifold) | [manifold-adp.md](manifold-adp.md) | `[A-*]`, `[M-MAN-*]` |

## Общая нотация

- `n` — число наблюдений, `d` — размерность `X_i`, `m` — размерность EDR-подпространства, `J` — число центров, `P` — число направлений на центр.
- `X_i ∈ R^d`, `Y_i ∈ R`; центр `x_j ∈ R^d`; вес `w_ij = K(||T_j (X_i - x_j)||²)`.
- `N_j = Σ_i w_ij`, `Xbar_j = N_j⁻¹ Σ_i w_ij X_i` — локальная масса и взвешенный центр.
- `U_j ∈ R^(P×d)` хранит в строках векторы `u_{j,s}ᵀ`; `I_j ∈ R^P` — направленные отклики.
- Для одного индекса `β ∈ R^d`, `||β||=1`; знак не идентифицируется.
- Для `m` индексов `P ∈ R^(m×d)`, `P Pᵀ=I_m`; EDR-подпространство задаётся проектором `PᵀP`, а не конкретным базисом.

## Существенные различия между исходниками

Не склеивайте варианты без отдельного математического решения; диапазоны здесь — `[X-MI-*]`, `[A-*]` и соответствующие `[M-*]`.

| Вопрос | Нормированная версия `[X-MI-PRE]`, `[A-OBJECTIVE]` | Дубликаты в `manifold.tex` `[M-MAN-OBJ]`, `[M-ADP-SI-PRE]` |
|---|---|---|
| Масштаб `I_j` и `U_j` | Оба момента делятся на `N_j`; в целевой функции затем появляется вес `N_j` `[X-MI-PRE]`, `[A-OBJECTIVE]`. | В ADP single-index моменты записаны как ненормированные суммы, а целевая функция — без `N_j` `[M-ADP-SI-PRE]`; manifold-дубликат местами добавляет `N_j` при другой нормировке `[M-MAN-OBJ]`. |
| Константа локальной аппроксимации | В ADP исчезает после центрирования вокруг `Xbar_j` `[A-OBJECTIVE]`, `[X-MI-PRE]`. | В общих single/multi-index разделах оставлены `f0_j` и `S_j` `[M-SI-PRE]`, `[M-MI-PRE]`; это более широкая модель, а не та же упрощённая ADP-цель. |
| Начальный локализатор | В multi-index ADP старт — `T_0=h_0⁻¹I` `[X-MI-PRE]`, `[X-MI-STRUCT]`. | В общем multi-index изложении встречается `T_0=h_0⁻²I`, хотя веса раскрыты как `K(h_0⁻²||Δx||²)` `[M-MI-IMPL]`. |
| Масштаб локализатора и веса | — | В общих single-index preliminaries одновременно указаны `T=h⁻²I` и вес `K(h⁻²||Δx||²)`, что при буквальном `K(||TΔx||²)` даёт несовпадение степеней полосы `[M-SI-PRE]`, `[X-SI-PRE]`; ADP-инициализация отдельно пишет `T_0=h_0⁻¹I` `[M-ADP-SI-INIT]`. |
| Направления `s` | Теоретическое описание предлагает `N(0,T²)`, а main procedure может пересэмплировать из `N(0,I)` `[X-MI-STRUCT]`, `[X-MI-MAIN]`. | Закон обновления указан не везде или различается между версиями `[M-ADP-SI-MAIN]`, `[M-ADP-MI-MAIN]`. |
| Анизотропия | Multi-index ADP даёт два варианта: регуляризация всего пространства или только ортогонального дополнения `[X-MI-STRUCT]`. | Single-index ADP содержит `α²I + ββᵀ`, не совпадающее с `α²(I−ββᵀ)+ββᵀ` `[M-ADP-SI-MAIN]`, `[M-SI-ANISO]`. |

## Полный каталог исходных фрагментов

Команды рассчитаны на запуск из корня репозитория. Идентификатор в тематических заметках однозначно указывает на строку этой таблицы; столбец «Извлечь» содержит готовый код для копирования.

| Код | Исходный диапазон | Что сверять / где отражено | Извлечь оригинал |
|---|---|---|---|
| `M-SETUP` | `tex/manifold.tex:1–38` | source-only: макросы и заголовок TeX, не алгоритм. | `rtk proxy sed -n '1,38p' tex/manifold.tex` |
| `M-SI-INTRO` | `tex/manifold.tex:39–68` | Вводная single-index модель; single-index note, целевая величина. | `rtk proxy sed -n '39,68p' tex/manifold.tex` |
| `M-SI-PRE` | `tex/manifold.tex:69–256` | Общие single-index предварительные определения и локальные статистики; single-index note, сравнение с ADP. | `rtk proxy sed -n '69,256p' tex/manifold.tex` |
| `M-SI-AO` | `tex/manifold.tex:257–380` | source-only: общий single-index AO, не ADP AO. | `rtk proxy sed -n '257,380p' tex/manifold.tex` |
| `M-SI-ANISO` | `tex/manifold.tex:381–458` | Общая анизотропия и structural adaptation; single-index note, сравнение формул. | `rtk proxy sed -n '381,458p' tex/manifold.tex` |
| `M-SI-INIT` | `tex/manifold.tex:459–514` | Инициализация локальной линейной регрессией; single-index note. | `rtk proxy sed -n '459,514p' tex/manifold.tex` |
| `M-SI-DOE` | `tex/manifold.tex:515–532` | source-only: план design-of-experiments. | `rtk proxy sed -n '515,532p' tex/manifold.tex` |
| `M-SI-TUNE` | `tex/manifold.tex:533–594` | Tuning parameters; single-index note, ориентиры параметров. | `rtk proxy sed -n '533,594p' tex/manifold.tex` |
| `M-SI-PROC` | `tex/manifold.tex:595–774` | source-only: общий single-index procedure, не подмена ADP. | `rtk proxy sed -n '595,774p' tex/manifold.tex` |
| `M-MI-INTRO` | `tex/manifold.tex:775–831` | Вводная multi-index модель; multi-index note. | `rtk proxy sed -n '775,831p' tex/manifold.tex` |
| `M-MI-PRE` | `tex/manifold.tex:832–962` | source-only: общие multi-index предварительные определения; используется для сравнения с центрированной ADP-целью. | `rtk proxy sed -n '832,962p' tex/manifold.tex` |
| `M-MI-AO` | `tex/manifold.tex:963–1045` | source-only: общий multi-index AO, не ADP AO. | `rtk proxy sed -n '963,1045p' tex/manifold.tex` |
| `M-MI-STRUCT` | `tex/manifold.tex:1046–1092` | source-only: общий multi-index structural adaptation. | `rtk proxy sed -n '1046,1092p' tex/manifold.tex` |
| `M-MI-INIT` | `tex/manifold.tex:1093–1106` | source-only: общая инициализация multi-index через локальные градиенты. | `rtk proxy sed -n '1093,1106p' tex/manifold.tex` |
| `M-MI-IMPL` | `tex/manifold.tex:1107–1273` | source-only: общий implementation/settings-блок; не считать ADP-спецификацией. | `rtk proxy sed -n '1107,1273p' tex/manifold.tex` |
| `M-MAN-INTRO` | `tex/manifold.tex:1274–1297` | Постановка manifold learning; manifold note. | `rtk proxy sed -n '1274,1297p' tex/manifold.tex` |
| `M-MAN-SCALE` | `tex/manifold.tex:1298–1333` | Две шкалы manifold-модели; manifold note. | `rtk proxy sed -n '1298,1333p' tex/manifold.tex` |
| `M-MAN-OBJ` | `tex/manifold.tex:1334–1432` | Общая manifold-цель и penalty; сравнение с ADP-спецификацией. | `rtk proxy sed -n '1334,1432p' tex/manifold.tex` |
| `M-MAN-ONESTEP` | `tex/manifold.tex:1433–1632` | Общий manifold one-step; manifold note, с оговоркой о нормировке/intercept. | `rtk proxy sed -n '1433,1632p' tex/manifold.tex` |
| `M-MAN-INIT` | `tex/manifold.tex:1633–1698` | Общая manifold-инициализация. | `rtk proxy sed -n '1633,1698p' tex/manifold.tex` |
| `M-MAN-MAIN` | `tex/manifold.tex:1699–2065` | Общий manifold structure-adaptive цикл; manifold note, дублирующее изложение. | `rtk proxy sed -n '1699,2065p' tex/manifold.tex` |
| `M-ADP-INTRO` | `tex/manifold.tex:2066–2076` | Краткая вводная к ADP-разделу; контекст к single/multi-index заметкам. | `rtk proxy sed -n '2066,2076p' tex/manifold.tex` |
| `M-ADP-SI-PRE` | `tex/manifold.tex:2077–2174` | ADP single-index moments и обозначения; single-index note. | `rtk proxy sed -n '2077,2174p' tex/manifold.tex` |
| `M-ADP-SI-AO` | `tex/manifold.tex:2175–2225` | ADP single-index AO; single-index note. | `rtk proxy sed -n '2175,2225p' tex/manifold.tex` |
| `M-ADP-SI-INIT` | `tex/manifold.tex:2226–2272` | ADP single-index инициализация. | `rtk proxy sed -n '2226,2272p' tex/manifold.tex` |
| `M-ADP-SI-MAIN` | `tex/manifold.tex:2273–2427` | ADP single-index structure-adaptive процедура. | `rtk proxy sed -n '2273,2427p' tex/manifold.tex` |
| `M-ADP-MI-INTRO` | `tex/manifold.tex:2428–2484` | Вводная часть ADP multi-index процедуры; multi-index note. | `rtk proxy sed -n '2428,2484p' tex/manifold.tex` |
| `M-ADP-MI-AO` | `tex/manifold.tex:2485–2565` | ADP multi-index one-step alternating; multi-index note. | `rtk proxy sed -n '2485,2565p' tex/manifold.tex` |
| `M-ADP-MI-MAIN` | `tex/manifold.tex:2566–2694` | ADP multi-index outer procedure; здесь есть формула, похожая на оставшийся single-index фрагмент. | `rtk proxy sed -n '2566,2694p' tex/manifold.tex` |
| `X-SETUP` | `tex/multiindex.tex:1–49` | source-only: макросы и технический заголовок. | `rtk proxy sed -n '1,49p' tex/multiindex.tex` |
| `X-SI-INTRO` | `tex/multiindex.tex:50–79` | Вводная single-index модель. | `rtk proxy sed -n '50,79p' tex/multiindex.tex` |
| `X-SI-PRE` | `tex/multiindex.tex:80–263` | Моменты кратко пересказаны; prediction/generalization — source-only. | `rtk proxy sed -n '80,263p' tex/multiindex.tex` |
| `X-SI-DATA` | `tex/multiindex.tex:264–308` | source-only: single-index data-fit и разделение train/test. | `rtk proxy sed -n '264,308p' tex/multiindex.tex` |
| `X-SI-AO` | `tex/multiindex.tex:309–392` | Single-index alternating optimization. | `rtk proxy sed -n '309,392p' tex/multiindex.tex` |
| `X-SI-ANISO` | `tex/multiindex.tex:393–476` | Single-index anisotropy и structural adaptation. | `rtk proxy sed -n '393,476p' tex/multiindex.tex` |
| `X-SI-DOE` | `tex/multiindex.tex:477–509` | source-only: single-index design-of-experiments план. | `rtk proxy sed -n '477,509p' tex/multiindex.tex` |
| `X-SI-INIT` | `tex/multiindex.tex:510–625` | Single-index инициализация локальной регрессией. | `rtk proxy sed -n '510,625p' tex/multiindex.tex` |
| `X-SI-MAIN` | `tex/multiindex.tex:626–894` | Single-index main procedure, включая initialization и structural adaptation. | `rtk proxy sed -n '626,894p' tex/multiindex.tex` |
| `X-SI-STOP` | `tex/multiindex.tex:895–906` | Single-index stopping rule. | `rtk proxy sed -n '895,906p' tex/multiindex.tex` |
| `X-SI-RESULTS` | `tex/multiindex.tex:907–920` | source-only: список сравнений и экспериментальный план. | `rtk proxy sed -n '907,920p' tex/multiindex.tex` |
| `X-SI-BREAKDOWN` | `tex/multiindex.tex:921–951` | source-only: breakdown dimension и план сравнительного исследования. | `rtk proxy sed -n '921,951p' tex/multiindex.tex` |
| `X-MI-INTRO` | `tex/multiindex.tex:952–1009` | Вводная multi-index модель. | `rtk proxy sed -n '952,1009p' tex/multiindex.tex` |
| `X-MI-PRE` | `tex/multiindex.tex:1010–1148` | Multi-index моменты и предварительные определения. | `rtk proxy sed -n '1010,1148p' tex/multiindex.tex` |
| `X-MI-AO` | `tex/multiindex.tex:1149–1236` | Multi-index alternating optimization. | `rtk proxy sed -n '1149,1236p' tex/multiindex.tex` |
| `X-MI-QUALITY` | `tex/multiindex.tex:1237–1259` | Метрика качества оценки projector/subspace. | `rtk proxy sed -n '1237,1259p' tex/multiindex.tex` |
| `X-MI-DATA` | `tex/multiindex.tex:1260–1272` | Multi-index data fit. | `rtk proxy sed -n '1260,1272p' tex/multiindex.tex` |
| `X-MI-STRUCT` | `tex/multiindex.tex:1273–1348` | Multi-index structural adaptation и локализатор. | `rtk proxy sed -n '1273,1348p' tex/multiindex.tex` |
| `X-MI-PROC` | `tex/multiindex.tex:1349–1402` | Описание multi-index процедуры и её инициализации. | `rtk proxy sed -n '1349,1402p' tex/multiindex.tex` |
| `X-MI-MAIN` | `tex/multiindex.tex:1403–1579` | Multi-index main procedure. | `rtk proxy sed -n '1403,1579p' tex/multiindex.tex` |
| `A-OVERVIEW` | `tex/manifold-ade.tex:1–24` | Постановка manifold learning; manifold note. | `rtk proxy sed -n '1,24p' tex/manifold-ade.tex` |
| `A-SCALES` | `tex/manifold-ade.tex:25–60` | Multiscale modeling. | `rtk proxy sed -n '25,60p' tex/manifold-ade.tex` |
| `A-OBJECTIVE` | `tex/manifold-ade.tex:61–160` | Objective и manifold penalty. | `rtk proxy sed -n '61,160p' tex/manifold-ade.tex` |
| `A-ONESTEP` | `tex/manifold-ade.tex:161–362` | Basic one-step procedure. | `rtk proxy sed -n '161,362p' tex/manifold-ade.tex` |
| `A-INIT` | `tex/manifold-ade.tex:363–426` | Инициализация локальной линейной регрессией. | `rtk proxy sed -n '363,426p' tex/manifold-ade.tex` |
| `A-MAIN` | `tex/manifold-ade.tex:427–817` | Structure-adaptive manifold learning, настройки и внешний цикл. | `rtk proxy sed -n '427,817p' tex/manifold-ade.tex` |

## Черновые места, важные для дальнейшей работы

1. В исходниках веса определены через `K(||TΔx||²)`, а в таблицах Epanechnikov записан как `K(t)=(1−t²)_+`; сверяйте определения в `[M-SI-PRE]`, `[X-SI-PRE]` и `[X-MI-PRE]`.
2. Условие выбора полосы обычно ограничивает **среднюю по центрам** массу, а не массу каждого центра; оно не гарантирует `N_j>0` для всех `j`.
3. Встречаются редакторские пометки `please check`, незаполненный closed form для multi-index обновления `[X-MI-AO]` и несовпадающие дублирующие формулы `[M-ADP-MI-MAIN]`. Пометки не считать доказательством.
4. Нормировка `Λ`, веса локальной ошибки, закон направлений и индексирование локальных наклонов влияют на метод; сверяйте их по соответствующим кодам, а не по одиночной формуле.
5. Здесь нет проверки формул по текущей реализации и нет заявлений о качестве или сходимости алгоритма.
