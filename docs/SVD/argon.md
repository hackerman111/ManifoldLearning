Gemini
New chat
Library
Images
Scheduled
Customize
More
New notebook
Untitled notebook
Untitled notebook
Соотнесение билетов и конспектов
Реши следующую задачу с максимально возможной глубиной анализа. Приоритет: корректность, полнота исследования и качество решения
Реши следующую задачу с максимально возможной глубиной анализа. Приоритет: корректность, полнота исследования и качество решения
Реши задачи 2-6. Для второй а задачи используй мои записки
Ветвь • Оптимизация производительности и устранение ботлнеков
Замкнутый вид дзета-функции Римана от 3
Оптимизация производительности и устранение ботлнеков
Анализ производительности исходного кода
Знакомство с моделью Gemini
Новости о новых моделях Google
Спектральная кластеризация на лекции
Выход Gemini 4 Argon
Удлинение стрелки рукава
Редактирование размерной сетки куртки
Разбор OpenAI и математических агентов
Разбор функций обратного прохода нейросети
YouTube-каналы по теории ML и DL
Каналы по ML и кластеризация
Таймкоды лекции по PyTorch
Начало работы
Тестирование ИИ в математике
Тест ИИ на математических задачах
Обзор ролика про язык Rust
Разбор видео «Фасолька//Финал Эпопеи»
Tsoding's Opinion on Neovim
Фотожаба "иди и сосси" в клипе Lida
Обзор китайского красного чая
Невозможный стакан: как пить?
Как пить из стакана без дна
Загадка: Как пить из стакана без дна
Как сделать стакан из трубки
Стакан без дна: загадка и решения
Замена персонажа на постере
Gemini Spark Setup
Обзор моделей транскрипции OpenAI
Minimalist Violin Focus Music Generation
Activity
Dronten, Netherlands From your IP address
Conversation with Gemini

Реши следующую задачу с максимально возможной глубиной анализа.

Приоритет: корректность, полнота исследования и качество решения. Скорость ответа и экономия вычислений не имеют значения.

Инструкции:

    Think very hard before answering. Используй максимально доступную глубину внутреннего рассуждения. Не торопись переходить к окончательному ответу.
    Исследуй задачу с нескольких независимых сторон. Рассмотри альтернативные подходы, включая нетривиальные и менее очевидные.
    Не останавливайся на первом правдоподобном решении. Проверь, существуют ли более сильные результаты, более общие формулировки или существенные контрпримеры.
    Для каждого перспективного подхода исследуй его предпосылки, ограничения и потенциальные причины неудачи.
    Проверяй критические утверждения независимо. Для математических результатов используй доказательства, для алгоритмов — анализ сложности, инварианты и тесты, где это применимо.
    Попытайся опровергнуть собственные основные выводы. Если обнаружишь ошибку или существенное ограничение, пересмотри решение.
    Не подменяй трудные части задачи общими рассуждениями. Если проблему нельзя решить полностью, установи точную границу полученного результата.
    Продолжай исследование, пока остаются содержательные непроверенные направления в пределах доступных ресурсов.

Формат ответа:

Представь окончательные результаты, строгие обоснования, наиболее сильные альтернативы, ограничения и нерешённые вопросы. Отделяй доказанные утверждения от гипотез.

Не увеличивай длину ответа искусственно. Основной вычислительный бюджет направь на внутренний анализ и проверку, а не на многословное изложение.

Задача:

Итак. Тебе даны мои разрезнненые записи касающиеся оптимизации солвера, часть из них делала ии и они содержат ошибки/слишком смелые заявления, и в целом алгоритма для multi index задачи EDR + ade + анизотропный тензор. У меня есть интуиция, что есть метод связанный с разложением подпросттранства B во что-то иное, возможно модифицированное (как вариант с экспонентой на многообразии) и в общем случае разложением возможно измененного многообрзаия в что-то иное, но более простое, что можно легко и быстро считать. Ты должен:

    внедрить идею SVD солвера в эту теорию, как частный случай этой идеи
    провести ресерч на поиск похожих идей/методов в науке
    формализовать, обобщить, улчушить эту версию

Под формализацией я понимаю создание единого tex документа, который бы объяснял, раскрывал и рассортировывал идеи теории. Tex документ должен быть достаточно краток, чтобы избежать воды, хорошо сверстан, и при этом содержать в нужной степени подробности все что необходимо

Под обобщением я понимаю создание теории, которая бы смогла взять суть из всего что было и сжать эту суть в математически корректную теорию, чтобы каждая часть записки была частным случаем этой теории

Под улучшением я понимаю использование более эффективных идей основанных на обобщенной теории, под улучшением я понимаю увеличение скорости, качества восстановления или уменьшения затрат на памяь

Теория для задачи:

% !TEX program = lualatex

% Самодостаточная сборка без предупреждений Overfull \hbox (выполнить дважды):

% lualatex -halt-on-error -interaction=nonstopmode EDR_master_theory_v3.tex

\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}

\setmainfont{Noto Serif}

\setsansfont{Noto Sans}

\setmonofont{DejaVu Sans Mono}[Scale=MatchLowercase]

\usepackage[russian]{babel}

\usepackage[margin=20mm]{geometry}

\usepackage{amsmath,amssymb,amsthm,mathtools,booktabs,tabularx,array}

\usepackage{microtype,enumitem,xcolor}

\usepackage[colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black]{hyperref}

\setlength{\parindent}{0pt}

\setlength{\parskip}{4pt}

\setlength{\emergencystretch}{4em}

\tolerance=2500

\hbadness=10000

\hfuzz=1.5pt

\sloppy

\setlist{nosep,leftmargin=*}

\setcounter{tocdepth}{2}

\numberwithin{equation}{section}

% Макросы

\newcommand{\R}{\mathbb R}

\newcommand{\Id}{\mathrm I}

\newcommand{\T}{^{\mathsf T}}

\newcommand{\F}{_{\mathrm F}}

\newcommand{\ip}[2]{\langle #1,#2\rangle_{\mathrm F}}

\newcommand{\Gr}{\mathrm{Gr}}

\newcommand{\St}{\mathrm{St}}

\newcommand{\Exp}{\operatorname{Exp}}

\newcommand{\polar}{\operatorname{polar}}

\newcommand{\grad}{\operatorname{grad}}

\newcommand{\qf}{\operatorname{qf}}

\newcommand{\rot}{\operatorname{rot}}

\DeclareMathOperator{\rank}{rank}

\DeclareMathOperator{\tr}{tr}

\DeclareMathOperator{\diag}{diag}

\DeclareMathOperator{\spanop}{span}

\DeclareMathOperator{\range}{range}

\DeclareMathOperator{\vecop}{vec}

\DeclareMathOperator*{\argmin}{arg\,min}

\DeclareMathOperator*{\argmax}{arg\,max}

\newtheorem{theorem}{Теорема}[section]

\newtheorem{proposition}[theorem]{Утверждение}

\newtheorem{lemma}[theorem]{Лемма}

\newtheorem{corollary}[theorem]{Следствие}

\theoremstyle{definition}

\newtheorem{definition}[theorem]{Определение}

\theoremstyle{remark}

\newtheorem{remark}[theorem]{Замечание}

\newcolumntype{L}{>{\raggedright\arraybackslash\hspace{0pt}}X}

\newcolumntype{P}[1]{>{\raggedright\arraybackslash\hspace{0pt}}p{#1}}

\title{\textbf{Разделимые модели, малоразмерные карты и плоские вращения}\\[4pt]

\large Математическая теория SVD-решателя, оптимизации на многообразии Грассмана,\\

касательных полей Такера и анизотропной адаптации в задачах EDR/ADE}

\author{Сводная исследовательская записка (редакция 3)}

\date{Октябрь 2026 г.}

\begin{document}

\maketitle

\vspace{-7mm}

\begin{abstract}

Различные алгоритмические подходы к внутреннему шагу многоиндексного оценивания эффективного подпространства снижения размерности (EDR) методом средних производных (ADE/ADP) со структурной адаптацией объединяются вокруг сепарабельного функционала $\Phi(\theta)=\min_c\{\|b-\mathcal A(\theta)c\|_2^2+\|\mathcal R(\theta)c\|_2^2\}+\Omega(\theta)$ и локального обновления в малоразмерной карте $\theta_+=\chi_\theta(\mathcal E(\alpha))$.%[cite: 2, 3, 5]

Определены условия, при которых выбор словаря $\mathcal A$, регуляризатора $\mathcal R$, штрафа $\Omega$, карты $\chi$ и подпространства направлений $\mathcal E$ порождает: (1) классическое альтернирование (AO), (2) матричный и поправочный режимы SVD-решателя при фиксированных коэффициентах, (3) геодезические и полярные вращения на многообразии Грассмана $\Gr(m,d)$, (4) пространственно-переменное поле подпространств в оболочке Такера и (5) спектральные методы типа OPG/MAVE.%[cite: 2, 3, 5, 6]

Выведены теорема Эккарта--Янга в сепарабельной метрике, оценка сохранения доли $(1-\theta)/(1+\theta)$ максимального ранк-1 выигрыша при возмущении оператора, формула вложения SVD-поправки в геодезическую окружность $\Gr(m,d)$, рациональный и тригонометрический профили Шура со стоимостью $O(J)$ на пробный угол после подготовки и условия сходимости для backtracking-поиска Армихо с контролем длины шага.%[cite: 1, 2, 3, 5, 9]

Отдельно исследованы задача оптимального по затратам приближения нормального оператора и условия совместной наблюдаемости при неполных локальных измерениях ($p < d$).%[cite: 2, 9]

\end{abstract}

\tableofcontents

\medskip

\textbf{Статус результатов и открытые вопросы.} В документе строго разделяются три уровня утверждений:%[cite: 2, 3, 9]

\begin{enumerate}

\item \textbf{Доказанные алгебраические и геометрические результаты:} формулы профилирования VARPRO, белёная теорема Эккарта--Янга, гарантия доли $(1-\theta)/(1+\theta)$ для ранк-1 шага, вложение полярной поправки в $\Gr(m,d)$, скалярные профили Шура и внутренняя сходимость при контролируемых шагах.%[cite: 1, 2, 3, 5, 9]

\item \textbf{Условные оценки возмущения:} теорема об ошибке восстановления подпространства (Раздел~\ref{sec:extensions}) опирается на явное условие ограниченной совместной наблюдаемости $\mu_{\rm obs}>0$; при $p < d$ оно не вытекает из поузловой невырожденности $U_j$ и требует отдельной верификации.%[cite: 2, 9]

\item \textbf{Алгоритмические гипотезы и открытые проблемы:} преимущество $\kappa$-гомотопии, ускорение сквозного цикла EDR и безусловная статистическая сходимость внешней итерации структурной адаптации ядра не доказаны и требуют экспериментальной и теоретической проверки.%[cite: 2, 3, 9]

\end{enumerate}

\section{Наблюдаемая модель, статистики ADP и четыре роли метрик}\label{sec:model}

Рассматривается многоиндексная регрессионная модель наблюдений $(x_i,y_i)\in\R^d\times\R$, $i=1,\ldots,n$:%[cite: 2, 5, 6, 9]

\begin{equation}\label{eq:mim}

y_i = f(P_* x_i) + \varepsilon_i = \psi(Y_*\T x_i) + \varepsilon_i,\qquad

Y_* = P_*\T \in \St(d,m)=\{Y\in\R^{d\times m}: Y\T Y = \Id_m\},\quad m\ll d.

\end{equation}

Поскольку $\nabla f(x) = Y_*\nabla\psi(Y_*\T x)$, при невырожденности матрицы вторых моментов градиента $\mathbb E[\nabla\psi(Y_*\T x)\nabla\psi(Y_*\T x)\T]\succ 0$ линейная оболочка градиентов совпадает с целевым EDR-подпространством $[Y_*]=\spanop(Y_*)\in\Gr(m,d)$ \cite{hjps,mave}.%[cite: 2, 3, 6]

Качество любой ортонормированной оценки $Y\in\St(d,m)$ ($P=Y\T$) инвариантно измеряется квадратом хордового расстояния:%[cite: 1, 3, 4, 6]

\begin{equation}\label{eq:chordal}

\epsilon(Y,Y__) = d\_{\rm ch}^2(Y,Y__) = \tfrac12\|YY\T - Y__Y__\T\|\F^2 = m - \|Y_*\T Y\|\F^2 = \|Y\T(\Id_d - Y_*Y_*\T)\|\F^2.

\end{equation}

Для центров локализации $\{x_j\}_{j=1}^J$, симметричного положительно определённого (СПО) оператора локализации $D_j\succ 0$ и ядра $K(\cdot)$, принимающего на вход \emph{квадрат} расстояния, определим локальные веса и моменты:%[cite: 2, 3, 6]

\begin{align}

w_{ij} &= K\bigl(h^{-2}(x_i-x_j)\T D_j(x_i-x_j)\bigr), &

N_j &= \sum_{i=1}^n w_{ij} > 0, &

\bar x_j &= N_j^{-1}\sum_{i=1}^n w_{ij}x_i,\label{eq:weights}\\

z_{ij} &= x_i - \bar x_j, &

\Sigma_j &= N_j^{-1}\sum_{i=1}^n w_{ij} z_{ij}z_{ij}\T, &

c_j &= N_j^{-1}\sum_{i=1}^n w_{ij} z_{ij} y_i.\label{eq:moments}

\end{align}

При заданной матрице пробных направлений (скетче) $S_j\in\R^{p\times d}$ направленные ADP-статистики равны $U_j = S_j\Sigma_j\in\R^{p\times d}$ и $I_j = S_j c_j\in\R^p$.%[cite: 2, 3, 5]

Локальная линейная аппроксимация функции связи порождает модель:%[cite: 2, 6]

\begin{equation}\label{eq:local-model}

I_j \simeq U_j Y_j g_j,\qquad g_j\in\R^m,\quad j=1,\ldots,J,

\end{equation}

где в глобальной многоиндексной постановке $Y_1=\cdots=Y_J=Y$.%[cite: 2, 10]

Нормировка на локальную массу $N_j$ уже выполнена внутри $U_j$ и $I_j$, поэтому в целевом функционале вес $N_j$ применяется один раз.%[cite: 2, 7, 9]

\begin{remark}[Разделение четырёх метрик и сохранение оценивателя]\label{rem:four-metrics}

В алгоритме действуют четыре различных метрических объекта:%[cite: 2, 3]

\begin{enumerate}

\item \textbf{Оператор локализации $D_j\succ 0$:} входит в ядро \eqref{eq:weights}, меняет статистики $(N_j,U_j,I_j)$ и задаёт внешнюю итерацию структурной адаптации \cite{hjps}.%[cite: 2, 3, 6]

\item \textbf{Метрика невязки $W_j\succeq 0$:} задаёт взвешивание $\|I_j - U_j Y g_j\|_{W_j}^2$ (сводится к евклидовой норме заменой $U_j\leftarrow W_j^{1/2}U_j$, $I_j\leftarrow W_j^{1/2}I_j$; в базовом ADP $W_j=\Id_p$).%[cite: 2, 3]

\item \textbf{Метрика регуляризатора базиса $H\succ 0$:} входит в штраф близости $\|(B-P)H^{1/2}\|\F^2$ при фиксированных коэффициентах.%[cite: 2, 3, 8]

\item \textbf{Метрика касательного шага / предобусловливатель $\mathcal M_\theta\succ 0$:} определяет выбор направления поиска при фиксированной целевой функции.%[cite: 2, 3, 9]

\end{enumerate}

Изменение $D_j$, $W_j$, $H$ или переход от фиксированных $g_j$ к профилированию меняет статистический оцениватель; изменение $\mathcal M_\theta$ или точное алгебраическое сжатие данных сохраняет оцениватель.%[cite: 2, 9]

\end{remark}

\section{Единый математический каркас: VARPRO и малоразмерные карты}\label{sec:framework}

\subsection{Сепарабельный мастер-функционал и корректность профилирования}

Пусть $\theta\in\mathcal X$ обозначает искомую структуру (матрицу $B\in\R^{m\times d}$, репер $Y\in\St(d,m)$, поле $\{Y_j\}_{j=1}^J$ или пару $(Y,\Lambda)$), а $\Psi_j(\theta)\in\R^{p\times k}$ — локальный матричный словарь (в линейной модели первого порядка $\Psi_j(\theta)=U_j Y$, $k=m$).%[cite: 1, 2, 3, 5]

Сформируем блочно-диагональный оператор дизайна $\mathcal A(\theta)\colon \R^{Jk}\to\R^{Jp}$ с блоками $(\mathcal A(\theta)c)_j = \sqrt{N_j}\Psi_j(\theta)c_j$ и вектор откликов $b_j = \sqrt{N_j}I_j$.%[cite: 2, 9]

\begin{definition}[Сепарабельная мастер-задача и схема малоразмерной редукции]\label{def:master}

Общая задача внутренней оптимизации определяется сепарабельным функционалом типа VARPRO \cite{varpro} и локальной параметризацией шага:%[cite: 2, 3, 5]

\begin{equation}\label{eq:master}

\Phi(\theta) = \min_{c\in\R^{Jk}}\left\{

\|b - \mathcal A(\theta)c\|_2^2 + \|\mathcal R(\theta)c\|_2^2

\right\} + \Omega(\theta),\qquad

\theta_+(\alpha) = \chi_\theta\bigl(\mathcal E(\alpha)\bigr),

\end{equation}

где $\mathcal R(\theta)$ — линейный оператор регуляризации коэффициентов, $\Omega(\theta)$ — структурный штраф на $\theta$, $\mathcal E(\alpha)=\sum_{\ell=1}^s \alpha_\ell E_\ell$ — линейное подпространство малой размерности $s$, а $\chi_\theta$ — гладкая локальная карта или ретракция ($\chi_\theta(0)=\theta$).%[cite: 2, 3, 5]

При фиксированных коэффициентах $c$ операция $\min_c$ опускается, давая условный квадратичный функционал $F(\theta)=\Phi(\theta;c)$.%[cite: 2, 9]

\end{definition}

\begin{proposition}[Регулярность профиля VARPRO и калибровочная инвариантность]\label{prop:profile}

Пусть на открытом множестве $\mathcal U\subset\mathcal X$ отображения $\theta\mapsto\mathcal A(\theta)$, $\theta\mapsto\mathcal R(\theta)$ и $\theta\mapsto\Omega(\theta)$ принадлежат классу гладкости $C^s$ ($s\ge 1$) либо вещественно-аналитичны ($C^\omega$), а расширенный оператор $\widetilde{\mathcal A}(\theta) = [\mathcal A(\theta); \mathcal R(\theta)]$ имеет на $\mathcal U$ локально постоянный ранг \cite{varpro}.%[cite: 2, 3]

Тогда:

\begin{enumerate}

\item Если $\widetilde{\mathcal A}(\theta)$ имеет на $\mathcal U$ полный столбцовый ранг $Jk$, то оператор Грама $\mathcal G(\theta) = \mathcal A(\theta)\T\mathcal A(\theta) + \mathcal R(\theta)\T\mathcal R(\theta)\succ 0$ обратим и принадлежит тому же классу $C^s$ (соответственно $C^\omega$), единственный минимум по $c$ равен $c^\star(\theta) = \mathcal G(\theta)^{-1}\mathcal A(\theta)\T b$, а профилированный функционал принадлежит классу $C^s$ (соответственно $C^\omega$) на $\mathcal U$ и имеет блочный вид:%[cite: 2, 3]

\begin{equation}\label{eq:profile-block}

\Phi(\theta) = \|b\|_2^2 - b\T\mathcal A(\theta)\bigl(\mathcal A(\theta)\T\mathcal A(\theta) + \mathcal R(\theta)\T\mathcal R(\theta)\bigr)^{-1}\mathcal A(\theta)\T b + \Omega(\theta).

\end{equation}

\item \textbf{Поузловая специализация:} Если регуляризатор сепарабелен по центрам, $\|\mathcal R(\theta)c\|_2^2 = \tau\sum_{j=1}^J N_j\|c_j\|_2^2$ с $\tau\ge 0$, а отображения $\theta\mapsto\Psi_j(\theta)$ и $\Omega(\theta)$ вещественно-аналитичны, то при $\tau>0$ условие полного столбцового ранга $\widetilde{\mathcal A}(\theta)$ выполнено всюду автоматически (включая точки вырождения $\Psi_j(\theta)$), а \eqref{eq:profile-block} распадается на $J$ независимых $k\times k$ систем и задаёт вещественно-аналитическую функцию на всей области определения:%[cite: 1, 2, 3, 4]

\begin{equation}\label{eq:profile-node}

\Phi(\theta) = \sum_{j=1}^J N_j\bigl(\|I_j\|_2^2 - b_j(\theta)\T A_j(\theta)^{-1} b_j(\theta)\bigr) + \Omega(\theta),\quad

A_j = \Psi_j\T\Psi_j + \tau\Id_k,\quad b_j = \Psi_j\T I_j.

\end{equation}

При $\tau=0$ без условий постоянного ранга функция $\Phi(\theta)$ в общем случае лишь полунепрерывна сверху.%[cite: 1, 3]

\item \textbf{Калибровочная инвариантность:} Если под действием группы преобразований $Q\in\mathcal G$ словарь эквивариантен: $\Psi_j(\theta\cdot Q) = \Psi_j(\theta)\rho(Q)$ с обратимым представлением $\rho(Q)$, а $\|\mathcal R(\theta\cdot Q)\tilde c\|_2 = \|\mathcal R(\theta)c\|_2$ и $\Omega(\theta\cdot Q)=\Omega(\theta)$ при $c_j = \rho(Q)\tilde c_j$, то $\Phi(\theta\cdot Q)=\Phi(\theta)$, т.е. профиль корректно определён на фактор-пространстве $[\theta]\in\mathcal X/\mathcal G$.%[cite: 1, 2, 3, 4]

\end{enumerate}

\end{proposition}

\begin{proof}

При полном столбцовом ранге $\widetilde{\mathcal A}(\theta)$ матрица $\mathcal G(\theta)=\widetilde{\mathcal A}(\theta)\T\widetilde{\mathcal A}(\theta)$ строго положительно определена; поскольку обращение невырожденной матрицы $A\mapsto A^{-1}$ вещественно-аналитично на открытом конусе СПО-матриц, композиция с $C^s$-гладкими (или $C^\omega$-аналитическими) отображениями $\mathcal A(\theta), \mathcal R(\theta), \Omega(\theta)$ сохраняет класс $C^s$ (или $C^\omega$) для $c^\star(\theta)$ и \eqref{eq:profile-block}.%[cite: 2, 4]

Пункты 2 и 3 следуют из блочной диагональности $\mathcal G(\theta)$ и биективности замены $c_j = \rho(Q)\tilde c_j$.%[cite: 2, 3, 4]

\end{proof}

\subsection{Таксономия частных случаев}

Таблица~\ref{tab:taxonomy} классифицирует все алгоритмические режимы из исходных материалов как специализации пятёрки $(\mathcal A, \mathcal R, \Omega, \chi, \mathcal E)$ в \eqref{eq:master}, строго отделяя эквивалентные преобразования от сужения пространства поиска и от смены оценивателя.%[cite: 2, 3, 9]

\begin{table}[h!]

\centering\footnotesize

\caption{Все конструкции записок как частные случаи мастер-схемы \eqref{eq:master}}\label{tab:taxonomy}

\begin{tabularx}{\linewidth}{@{}P{26mm}P{28mm}P{32mm}L@{}}

\toprule

Модель / Режим & Коэффициенты $c$ и штраф $\mathcal R, \Omega$ & Карта $\chi_\theta$ и подпространство $\mathcal E(\alpha)$ & Математический статус относительно базовой задачи\\

\midrule

Классический AO (\texttt{multiindex}) & Чередование: шаг по $g_j$ ($\tau=0$), шаг по $B\in\R^{m\times d}$ & Аффинная $\chi_B(\Delta)=B+\Delta$, полное $\mathcal E=\R^{m\times d}$ + SVD & Базовая схема; при $\tau=0,\lambda=0$ $\inf_B = \inf_{\St}$, но штраф $\lambda\|B-P\|\F^2$ нарушает $O(m)$-инвариантность.%[cite: 1, 2, 6]

\\

Чистый SVD (\texttt{matrix}) & $g_j$ фиксированы; $\Omega=\lambda\|(B-P)H^{1/2}\|\F^2$ & Аффинная $B_0+\sum_{\ell=1}^r \sigma_\ell a_\ell v_\ell\T$, $\rank B\le r < m$ & \textbf{Сужение пространства поиска} условной задачи; вносит порог $F(B)\ge\lambda(m-r)$ при $H=\Id$.%[cite: 2, 7, 8, 9]

\\

Поправочный SVD (\texttt{correction}) & $g_j$ фиксированы; $B=P+\Delta$, $\Omega=\lambda\|\Delta H^{1/2}\|\F^2$ & Аффинная $P + A M V\T$, $\rank\Delta\le r$, затем QR базиса & \textbf{Сужение пространства поправки}; снимает порог $m-r$; при $Y\T v=0$ совпадает с окружностью в $\Gr$.%[cite: 2, 3, 8, 9]

\\

Сепарабельный SVD ($\mathcal L_0 X = GXH$) & $g_j$ фиксированы; изотропия $H_j=\eta_j H$ & Глобальное SVD матрицы $G^{-1/2}C H^{-1/2}$ & \textbf{Точное глобальное решение} суррогатной квадратичной цели (Эккарт--Янг); предобусловливатель для $\mathcal L$.%[cite: 2, 3, 5, 9]

\\

Профильный $\Gr(m,d)$ (ранг $q$) & Профиль по $g_j$ ($\tau>0$); $\Omega=\lambda d_{\rm ch}^2(Y,Y_0)$ & Спектральная ретракция $R^\varphi_Y(V\Theta A\T)$ на торе $\mathbb T^q$ & \textbf{Смена статистического оценивателя} (синхронный профиль $g_j(Y)$ и хордовый штраф); сохраняет $\rank Y=m$.%[cite: 2, 3, 4]

\\

Касательное поле Такера & Профиль по $g_j$; графовый штраф $\tr(W\T L_{\rm gr}W)$ & $Y_j = R^\varphi_{Y_0}(Z_j)$, $Z_j = V(\sum_\ell w_\ell(x_j)M_\ell)A\T$ & \textbf{Расширение модели} до поля подпространств в оболочке $[Y_0,V]$ размерности $R=m+q>m$.%[cite: 2, 3, 5, 10]

\\

Семейство $\Phi_\kappa$ (OPG $\leftrightarrow$ MAVE) & Профиль по $g_j$; метрика $\Sigma_{j,\delta}^\kappa$, $\kappa\in[0,2]$ & При $\kappa=0$: глобальные собственные векторы $C_{\rm grad}$ & \textbf{Смена оценивателя} при изменении $\kappa$; при $\kappa=0,\tau=0,\lambda=0$ даёт глобальный спектральный оптимум.%[cite: 1, 2, 3, 5]

\\

\bottomrule

\end{tabularx}

\end{table}

\section{Аффинная редукция: теория SVD-решателя и его связь с $\Gr(m,d)$}\label{sec:svd}

При фиксированных коэффициентах $g_j\in\R^m$, текущем базисе $P = Y_0\T\in\R^{m\times d}$ ($PP\T=\Id_m$) и СПО-метрике $H\succ 0$ внутренняя задача действующего решателя (\texttt{ADP/\allowbreak solver/\allowbreak SVD.py}) имеет вид:%[cite: 2, 7, 8, 9]

\begin{equation}\label{eq:fixed-g}

F(B) = \sum_{j=1}^J N_j\|I_j - U_j B\T g_j\|_2^2 + \lambda\|(B-P)H^{1/2}\|\F^2,\qquad \lambda > 0.

\end{equation}

Обратимая замена $\widetilde B = BH^{1/2}$, $\widetilde P = PH^{1/2}$, $\widetilde U_j = U_j H^{-1/2}$ точно сводит \eqref{eq:fixed-g} к случаю $H=\Id_d$ при сохранении рангов $\rank(B)$ и $\rank(B-P)$.%[cite: 2, 3, 8]

Далее полагаем $H=\Id_d$.%[cite: 2]

\subsection{Операторная форма, апостериорные сертификаты и условие неточного спуска}

Прямой оператор $(\mathcal A X)_j = \sqrt{N_j}U_j X\T g_j$ и сопряжённый оператор $\mathcal A^* z = \sum_{j=1}^J \sqrt{N_j} g_j (U_j\T z_j)\T$ порождают самосопряжённый нормальный оператор $\mathcal L = \mathcal A^*\mathcal A + \lambda\Id$ и правую часть $C$:%[cite: 2, 3, 9]

\begin{equation}\label{eq:Lop}

\mathcal L(X) = \sum_{j=1}^J N_j (g_j g_j\T) X H_j + \lambda X,\qquad

C = \sum_{j=1}^J N_j g_j I_j\T U_j + \lambda P,\qquad H_j = U_j\T U_j.

\end{equation}

Тогда $F(B) = C_0 + \ip{B}{\mathcal L(B)} - 2\ip{C}{B}$, антиградиент равен $2Q(B) = 2(C - \mathcal L(B))$, а в векторизованной записи $\vecop(\mathcal L X) = \bigl[\sum_{j=1}^J N_j(H_j\otimes g_j g_j\T) + \lambda\Id_{md}\bigr]\vecop(X)$ все строки $B$ связаны перекрёстными блоками $g_j g_j\T$.%[cite: 2, 3, 9]

\begin{proposition}[Сертификаты полного условного решения и условие неточного спуска]\label{prop:certificates}

Пусть $\mathcal L\succeq\mu\Id$ с константой $\mu > 0$ (в общем случае $\mu=\lambda$, а при известной нижней границе спектра оператора данных $\mu_{\rm data}\ge 0$ можно взять $\mu = \lambda + \mu_{\rm data}$).%[cite: 2, 5, 9]

Тогда для неограниченного минимума $B_\star = \mathcal L^{-1}(C)$ справедливы апостериорные оценки:%[cite: 2, 5, 9]

\begin{equation}\label{eq:cert-bounds}

\|B - B_\star\|\F \le \frac{\|Q(B)\|\F}{\mu},\qquad

F(B) - F(B_\star) = \ip{Q(B)}{\mathcal L^{-1}Q(B)} \le \frac{\|Q(B)\|\F^2}{\mu}.

\end{equation}

Для любой приближённой поправки $D\in\R^{m\times d}$ с невязкой линейного решателя $s = \mathcal L(D) - Q(B)$ выполнено точное тождество:%[cite: 2, 3, 9]

\begin{equation}\label{eq:inexact-step}

F(B+D) - F(B) = -\ip{D}{\mathcal L(D)} + 2\ip{D}{s}.

\end{equation}

В частности, выполнение неравенства $2|\ip{D}{s}| < \ip{D}{\mathcal L(D)}$ (либо достаточного условия $\|s\|\F \le \tfrac12(1-\eta)\mu\|D\|\F$ при $\eta\in(0,1)$) гарантирует строгое убывание $F(B+D) < F(B)$ и задаёт проверяемый критерий остановки итераций Крылова (PCG/LSMR).%[cite: 2, 3, 9]

\end{proposition}

\subsection{Жадный ранк-1 шаг, совместный рефит и полное малое ядро}

Для атома $E = av\T$ ($\|a\|_2=\|v\|_2=1$) квадратичный профиль $F(B+\sigma av\T) - F(B) = D(a,v)\sigma^2 - 2 R_B(a,v)\sigma$ определяется скалярами:%[cite: 2, 7, 9]

\begin{equation}\label{eq:R-and-D}

R_B(a,v) = \ip{Q(B)}{av\T} = a\T Q(B)v,\qquad

D(a,v) = \ip{av\T}{\mathcal L(av\T)} = \sum_{j=1}^J N_j(a\T g_j)^2\|U_j v\|_2^2 + \lambda.

\end{equation}

Оптимальный шаг $\sigma_\star = R_B(a,v)/D(a,v)$ уменьшает цель на $\gamma_{\mathcal L}(av\T) = R_B(a,v)^2/D(a,v)$.%[cite: 2, 7, 9]

Покоординатная максимизация отношения Рэлея $\gamma_{\mathcal L}(av\T)$ даёт альтернирующие СПО-системы:%[cite: 7, 8, 9]

\begin{equation}\label{eq:als-systems}

\Bigl(\sum_{j=1}^J N_j\|U_j v\|_2^2 g_j g_j\T + \lambda\Id_m\Bigr)\tilde a = Q(B)v,\qquad

\Bigl(\sum_{j=1}^J N_j(a\T g_j)^2 U_j\T U_j + \lambda\Id_d\Bigr)\tilde v = Q(B)\T a,

\end{equation}

с нормировкой $a=\tilde a/\|\tilde a\|_2$, $v=\tilde v/\|\tilde v\|_2$.%[cite: 7, 8, 9]

Для накопленных ортонормированных факторов $A\in\St(m,r)$ и $V\in\St(d,r)$ диагональный рефит $B = B_0 + A\diag(\sigma)V\T$ решает $r\times r$ систему $M\sigma = h$, а переход к полному малому ядру $B = B_0 + A M V\T$ со свободной матрицей $M\in\R^{r\times r}$ минимизирует $F$ на всём $r^2$-мерном подпространстве $A\,\R^{r\times r}\,V\T$ при том же кэше $U_j V$.%[cite: 2, 7, 9]

\subsection{Точное решение при сепарабельной метрике и оценка ранк-1 шага при возмущении}

\begin{theorem}[Сепарабельный оптимум Эккарта--Янга и гарантия приближённого ранк-1 шага]\label{thm:separable-svd}

Пусть суррогатный оператор имеет сепарабельный вид $\mathcal L_0(X) = G X H$ с $G\in\R^{m\times m}, H\in\R^{d\times d}$, $G,H\succ 0$ (в частности, в изотропной модели $H_j\approx \eta_j\Id_d$ с $\eta_j=\|U_j\|\F^2/d$ имеем $H=\Id_d$ и $G=\sum_{j=1}^J N_j\eta_j g_j g_j\T + \lambda\Id_m$).%[cite: 2, 3, 5, 9]

Тогда:

\begin{enumerate}

\item \textbf{Замкнутая форма глобального минимума:} Для функционала $F_0(B)$ с оператором $\mathcal L_0$ глобальный минимум при ограничении $\rank(B)\le r$ равен:%[cite: 2, 3, 5, 9]

\begin{equation}\label{eq:whitened-ey}

B_r^\star = G^{-1/2}\bigl[G^{-1/2} C H^{-1/2}\bigr]_r H^{-1/2},

\end{equation}

где $[\cdot]_r$ — усечённое SVD ранга $r$, а зазор до полного минимума $B_\star^0 = G^{-1}CH^{-1}$ равен сумме квадратов отброшенных сингулярных чисел: $F_0(B_r^\star) - F_0(B_\star^0) = \sum_{k>r} s_k^2(G^{-1/2}CH^{-1/2})$.%[cite: 2, 3, 9]

Для задачи поправки $\rank(B-P)\le r$ формула \eqref{eq:whitened-ey} применяется к $\Delta=B-P$ с правой частью $C_\Delta = C - \mathcal L_0(P)$.%[cite: 2, 3, 9]

\item \textbf{Гарантия качества одного ранк-1 шага и предобусловливания:} Пусть истинный оператор $\mathcal L$ удовлетворяет двусторонней спектральной оценке:%[cite: 2, 3, 5, 9]

\begin{equation}\label{eq:spectral-proximity}

(1-\theta)\ip{X}{\mathcal L_0(X)} \le \ip{X}{\mathcal L(X)} \le (1+\theta)\ip{X}{\mathcal L_0(X)}\quad \forall X\in\R^{m\times d},\qquad 0\le\theta < 1.

\end{equation}

(При $H=\Id_d$ достаточным условием служит $\theta = \rho/\lambda_{\min}(G) < 1$, где $\rho = \sum_{j=1}^J N_j\|g_j\|_2^2\|H_j - \eta_j\Id_d\|_2$).%[cite: 9]

Пусть при фиксированной невязке $Q = C - \mathcal L(B)$ выбран атом $E_0 = G^{-1/2} u_1 v_1\T H^{-1/2}$, где $(u_1,v_1)$ — ведущая сингулярная пара матрицы $G^{-1/2} Q H^{-1/2}$, а шаг вдоль $E_0$ выбран точно по \eqref{eq:R-and-D} для истинного $\mathcal L$.%[cite: 9]

Тогда уменьшение истинного функционала $\gamma_{\mathcal L}(E_0)$ удовлетворяет оценкам:%[cite: 9]

\begin{equation}\label{eq:rank1-guarantee}

\gamma_{\mathcal L}(E_0) \ge \frac{1-\theta}{1+\theta}\max_{\rank(E)=1}\gamma_{\mathcal L}(E),\qquad

\max_{\rank(E)=1}\gamma_{\mathcal L}(E) \le \frac{\|G^{-1/2}QH^{-1/2}\|_2^2}{1-\theta},

\end{equation}

а число обусловленности предобусловленного оператора в PCG подчиняется границе $\kappa(\mathcal L_0^{-1/2}\mathcal L\mathcal L_0^{-1/2}) \le \frac{1+\theta}{1-\theta}$.%[cite: 2, 5, 9]

\end{enumerate}

\end{theorem}

\begin{proof}

Пункт 1 следует из замены $Z = G^{1/2}BH^{1/2}$ и теоремы Эккарта--Янга \cite{eckart}.%[cite: 2, 9]

Пункт 2: для любого атома $E\ne 0$ отношение $\gamma_{\mathcal L}(E) = \ip{Q}{E}^2 / \ip{E}{\mathcal L(E)}$ в силу \eqref{eq:spectral-proximity} удовлетворяет $(1+\theta)^{-1}\gamma_{\mathcal L_0}(E) \le \gamma_{\mathcal L}(E) \le (1-\theta)^{-1}\gamma_{\mathcal L_0}(E)$.%[cite: 9]

Поскольку $E_0$ максимизирует $\gamma_{\mathcal L_0}(E)$ на множестве матриц ранга 1 со значением $\|G^{-1/2}QH^{-1/2}\|_2^2$, для любой матрицы $E$ ранга 1 имеем $\gamma_{\mathcal L}(E_0) \ge (1+\theta)^{-1}\gamma_{\mathcal L_0}(E_0) \ge (1+\theta)^{-1}\gamma_{\mathcal L_0}(E) \ge \frac{1-\theta}{1+\theta}\gamma_{\mathcal L}(E)$, что даёт \eqref{eq:rank1-guarantee}.%[cite: 9]

\end{proof}

\subsection{Ранговый пол матричного режима и точное вложение поправки в $\Gr(m,d)$}

В матричном режиме ($\rank(B)\le r < m$) ввиду $PP\T=\Id_m$ всегда выполнено $\min_{\rank(B)\le r}\|B-P\|\F^2 = m-r$, откуда $F(B)\ge \lambda(m-r)$ при $H=\Id_d$: $m-r$ направлений теряются и не могут быть восстановлены самой $B$.%[cite: 2, 3, 9]

Поправочный режим $B = P + \Delta$ ($\rank(\Delta)\le r$) снимает этот порог и вкладывается в геометрию Грассмана.%[cite: 2, 3, 9]

\begin{proposition}[Точное вложение SVD-поправки в геометрию Грассмана]\label{prop:embedding}

Пусть $Y = P\T\in\St(d,m)$, а транспонированная поправка $Z = \Delta\T\in\R^{d\times m}$ горизонтальна ($Y\T Z = 0$) и имеет тонкое SVD $Z = V\Sigma A\T$ со спектром $\sigma_1,\ldots,\sigma_q$.%[cite: 3]

Определим ортонормированный представитель полярным фактором $\polar(Y+Z) = (Y+Z)(\Id_m + Z\T Z)^{-1/2}$ (задающим то же подпространство в $\Gr(m,d)$, что и $\qf(Y+Z)$).%[cite: 2, 3]

Тогда:

\begin{enumerate}

\item При $q=1$ ($Z = \eta v a\T$, $\|v\|_2=\|a\|_2=1$, $Y\T v=0$) полярная ортонормировка поправки совпадает с геодезическим вращением на угол $\theta = \arctan\eta$:%[cite: 1, 3, 5]

\begin{equation}\label{eq:polar-rk1}

\polar(Y + \eta v a\T) = Y(\Id_m - aa\T) + \frac{Ya + \eta v}{\sqrt{1+\eta^2}}\,a\T = \Exp_Y\bigl((\arctan\eta)\,v a\T\bigr).

\end{equation}

\item При любом $q\le\min(m,d-m)$ квадрат хордового расстояния выражается через спектр $Z$ точно:%[cite: 3]

\begin{equation}\label{eq:chordal-exact}

d_{\rm ch}^2\bigl(\polar(Y+Z), Y\bigr) = \sum_{\ell=1}^q \frac{\sigma_\ell^2}{1+\sigma_\ell^2} = \|Z\|\F^2 - \|Z\T Z\|\F^2 + O(\|Z\|\F^6).

\end{equation}

\item Для произвольной поправки $\Delta$ с $\|\Delta\|_2 < 1$ матрица $B=P+\Delta$ сохраняет полный ранг $m$, причём $d_{\rm ch}(\spanop(B\T),\spanop(P\T)) \le \|\Delta\|\F / (1 - \|\Delta\|_2)$.%[cite: 2, 3, 9]

\end{enumerate}

\end{proposition}

\section{Профилированная редукция на $\Gr(m,d)$: плоский тор, Шура и сходимость}\label{sec:grassmann}

Рассмотрим профилированный функционал на $\Gr(m,d)$ при замороженных статистиках:%[cite: 2, 3, 4]

\begin{equation}\label{eq:phi-grass}

\Phi(Y) = \sum_{j=1}^J N_j\min_{g_j\in\R^m}\bigl\{\|I_j - U_j Y g_j\|_2^2 + \tau\|g_j\|_2^2\bigr\} + \lambda d_{\rm ch}^2(Y,Y_0),\qquad Y\in\St(d,m),\quad \tau>0.

\end{equation}

Положим $M_j = U_j Y\in\R^{p\times m}$, $A_j = M_j\T M_j + \tau\Id_m$, $g_j^\star = A_j^{-1}(M_j\T I_j)$ и $r_j = I_j - M_j g_j^\star$.%[cite: 2, 3, 4]

По теореме об огибающей горизонтальный риманов градиент равен \cite{edelman}:%[cite: 2, 3, 4]

\begin{equation}\label{eq:rgrad}

G = \grad\Phi(Y) = (\Id_d - YY\T)\left[-2\sum_{j=1}^J N_j U_j\T r_j(g_j^\star)\T - 2\lambda Y_0 Y_0\T Y\right]\in T_Y\Gr(m,d).

\end{equation}

\subsection{Спектральное семейство ретракций и разложение Картана на плоские вращения}

\begin{theorem}[Семейство ретракций $R^\varphi_Y$ и разложение на коммутирующие вращения]\label{thm:cartan}

Пусть $0<m<d$ и горизонтальный вектор $Z\in T_Y\Gr(m,d)$ ($Y\T Z=0$) ранга $q\le\min(m,d-m)$ имеет тонкое SVD $Z = V\Sigma A\T$, где $V\in\St(d,q)$, $Y\T V=0$, $A\in\St(m,q)$, $\Sigma=\diag(\sigma_1,\ldots,\sigma_q)>0$.%[cite: 2, 3, 4]

Для гладкой нечётной функции $\varphi\colon\R\to\R$ с $\varphi'(0)=1$ положим $\Theta = \varphi(\Sigma) = \diag(\theta_1,\ldots,\theta_q)$ и определим \cite{edelman,absilmalick}:%[cite: 1, 2, 3]

\begin{equation}\label{eq:phi-retr}

R^\varphi_Y(Z) = \Exp_Y\bigl(V\varphi(\Sigma)A\T\bigr) = Y(\Id_m - AA\T) + YA\cos\Theta\,A\T + V\sin\Theta\,A\T.

\end{equation}

Тогда:

\begin{enumerate}

\item Отображение $R^\varphi_Y$ является ретракцией второго порядка на $\Gr(m,d)$ с разложением $R^\varphi_Y(Z) = Y + Z - \frac12 YZ\T Z + O(\|Z\|\F^3)$.%[cite: 1, 2, 3, 4]

Выборы $\varphi(t)=t$, $\varphi(t)=\arctan t$ и $\varphi(t)=2\arctan(t/2)$ дают соответственно геодезическую экспоненту, полярную ретракцию $\polar(Y+Z)$ и ретракцию Кэли \cite{wenyin}.%[cite: 1, 2, 3, 5]

\item Для $y_\ell = Y a_\ell$ элементарные повороты $\rot_{\theta_\ell}(y_\ell,v_\ell) = \Id_d + (\cos\theta_\ell - 1)(y_\ell y_\ell\T + v_\ell v_\ell\T) + \sin\theta_\ell(v_\ell y_\ell\T - y_\ell v_\ell\T)$ действуют в попарно ортогональных двумерных плоскостях $\Pi_\ell = \spanop\{y_\ell,v_\ell\}$, коммутируют между собой и дают \textbf{точное} разложение:%[cite: 2, 3, 4]

\begin{equation}\label{eq:cartan-prod}

R^\varphi_Y(Z) = \left(\prod_{\ell=1}^q \rot_{\theta_\ell}(y_\ell,v_\ell)\right)Y,\qquad \rank\bigl(R^\varphi_Y(Z) - Y\bigr)\le q,\qquad \rank\bigl(R^\varphi_Y(Z)\bigr) = m.

\end{equation}

При фиксированных $(V,A)$ множество $\{\Exp_Y(V\Theta A\T): \theta\in\R^q\}$ образует вполне геодезический плоский тор $\mathbb T^q\subset\Gr(m,d)$ периода $\pi$ по каждому углу.%[cite: 1, 2, 3]

\end{enumerate}

\end{theorem}

\subsection{Замкнутый одномерный профиль Шура и связь углов при $q>1$}

Для вращения в одной плоскости $(Ya,v)$ ($q=1$) дополним $a\in\R^m$ ($\|a\|_2=1$) до ортогональной матрицы $[A_\perp,a]\in O(m)$, $A_\perp\in\R^{m\times(m-1)}$ (при $m=1$ блок $A_\perp$ пуст).%[cite: 2, 3]

\begin{theorem}[Скалярный профиль Шура в тригонометрической и рациональной картах]\label{thm:schur-profile}

Обозначим $C_j = U_j Y A_\perp\in\R^{p\times(m-1)}$, $u_j = U_j Y a\in\R^p$, $v_j = U_j v\in\R^p$ и введём СПО-матрицу дополнения Шура по $(m-1)$ неподвижным направлениям базиса:%[cite: 1, 2, 3]

\begin{equation}\label{eq:pi-j}

\Pi_j = \Id_p - C_j(C_j\T C_j + \tau\Id_{m-1})^{-1}C_j\T \succ 0.

\end{equation}

Один раз перед поиском угла вычислим для каждого узла $j=1,\ldots,J$ шесть скаляров (без сборки $p\times p$ матрицы $\Pi_j$, по формуле $x\T\Pi_j y = x\T y - (C_j\T x)\T(C_j\T C_j+\tau\Id_{m-1})^{-1}(C_j\T y)$ за $O(J(pm^2+m^3))$ суммарно):%[cite: 1, 2, 3]

\begin{equation}\label{eq:six-scalars}

k_j = I_j\T\Pi_j I_j,\quad \alpha_j = u_j\T\Pi_j I_j,\quad \beta_j = v_j\T\Pi_j I_j,\quad

a_j = u_j\T\Pi_j u_j,\quad b_j = u_j\T\Pi_j v_j,\quad e_j = v_j\T\Pi_j v_j,

\end{equation}

и скаляры штрафа $\xi_0 = m - \|Y_0\T Y A_\perp\|\F^2$, $\xi_{yy}=\|Y_0\T Ya\|_2^2$, $\xi_{yv}=(Y_0\T Ya)\T(Y_0\T v)$, $\xi_{vv}=\|Y_0\T v\|_2^2$.%[cite: 2, 3]

Тогда:

\begin{enumerate}

\item В геодезической параметризации $Y(t) = \Exp_Y(t v a\T)$ ($t\in[-\pi/2,\pi/2]$):%[cite: 1, 2, 3]

\begin{align}

\Phi(Y(t)) &= \sum_{j=1}^J N_j\left[k_j - \frac{(\alpha_j\cos t + \beta_j\sin t)^2}{a_j\cos^2 t + 2b_j\sin t\cos t + e_j\sin^2 t + \tau}\right]\nonumber\\

&\quad + \lambda\bigl(\xi_0 - \xi_{yy}\cos^2 t - 2\xi_{yv}\sin t\cos t - \xi_{vv}\sin^2 t\bigr).\label{eq:trig-profile}

\end{align}

\item В полярной параметризации $\widetilde Y(\eta) = \polar(Y + \eta v a\T)$ ($\eta=\tan t\in\R$, покрывающей окружность кроме граничной точки $t=\pm\pi/2$, достигаемой в пределе $|\eta|\to\infty$):%[cite: 2, 3]

\begin{align}

\Phi(\widetilde Y(\eta)) &= \sum_{j=1}^J N_j\left[k_j - \frac{(\alpha_j + \beta_j\eta)^2}{(e_j+\tau)\eta^2 + 2b_j\eta + (a_j+\tau)}\right]\nonumber\\

&\quad + \lambda\left(\xi_0 - \frac{\xi_{yy} + 2\xi_{yv}\eta + \xi_{vv}\eta^2}{1+\eta^2}\right).\label{eq:rat-profile}

\end{align}

\end{enumerate}

После однократной подготовки скаляров \eqref{eq:six-scalars} вычисление $\Phi$, $\Phi'$ и $\Phi''$ в любой точке $t$ или $\eta$ стоит $O(J)$ операций и $O(1)$ дополнительной памяти на узел.%[cite: 1, 2, 3]

\end{theorem}

\begin{remark}[Многоэкстремальность и связанность углов при $q>1$]\label{rem:angle-caveats}

Формулы \eqref{eq:trig-profile}--\eqref{eq:rat-profile} дают замкнутую форму \emph{значения} функции, но сумма рациональных дробей по $j$ может иметь несколько локальных минимумов.%[cite: 2, 3]

При $q>1$, хотя повороты $\rot_{\theta_\ell}(y_\ell,v_\ell)$ коммутируют геометрически, локальные операторы $U_j$ связывают углы $\theta_1,\ldots,\theta_q$ внутри матриц $(M_j(\theta)\T M_j(\theta)+\tau\Id_m)^{-1}$.%[cite: 2, 3]

Поэтому при покоординатном обновлении углов коэффициенты Шура \eqref{eq:six-scalars} требуют пересчёта после каждого поворота; при наличии кэша $W_j = U_j[Y,V]\in\R^{p\times(m+q)}$ как этот пересчёт, так и совместная оценка функции и якобиана при любом $\theta\in\mathbb T^q$ выполняются внутри малой размерности $m+q$ за $O(J(p(m+q)^2+(m+q)^3))$ без обращения к размерности $d$.%[cite: 2, 3]

\end{remark}

\subsection{Полный якобиан VARPRO, спектральный бюджет и строгая теорема сходимости}

Дифференцирование локальной системы $A_j g_j^\star = M_j\T I_j$ вдоль горизонтального вектора $Z\in T_Y\Gr(m,d)$ ($T_j = U_j Z$) даёт производную коэффициентов и расширенной невязки $\tilde r_j = [r_j; -\sqrt\tau g_j^\star]$ \cite{varpro}:%[cite: 2, 3]

\begin{equation}\label{eq:varpro-jacobian}

A_j\dot g_j = T_j\T r_j - M_j\T T_j g_j^\star,\qquad

\dot{\tilde r}_j = \begin{bmatrix}-T_j g_j^\star - M_j\dot g_j \\ -\sqrt\tau\,\dot g_j\end{bmatrix},

\end{equation}

что вместе с производной хордового штрафа $\sqrt\lambda(\Id_d - Y_0 Y_0\T)Z$ определяет полный якобиан $\mathcal J_Y$ для метода Гаусса--Ньютона / доверительной области на торе $\mathbb T^q$.%[cite: 2, 3]

\begin{theorem}[Спектральный выбор ранга шага, backtracking Армихо и сходимость на $\Gr(m,d)$]\label{thm:convergence}

Пусть $\tau>0$, статистики фиксированы, а риманов градиент $G = \grad\Phi(Y)$ имеет тонкое SVD $(-G) = V\Sigma A\T$ с сингулярными числами $s_1\ge\cdots\ge s_{\bar m}\ge 0$, где $\bar m = \min(m,d-m)$.%[cite: 2, 3]

\begin{enumerate}

\item \textbf{Спектральный бюджет:} Задача $\min_{Y\T Z=0,\,\rank Z\le q,\,\|Z\|\F\le 1}\ip{G}{Z}$ при любом $1\le q\le\bar m$ имеет решение $Z_q^\star = -[G]_q / \|[G]_q\|\F$ со значением $-\|[G]_q\|\F = -(\sum_{\ell=1}^q s_\ell^2)^{1/2}$.%[cite: 2, 3]

Для порога $\varepsilon\in[0,1)$ и предела $q_{\max}\ge 1$ правило:%[cite: 2, 3]

\begin{equation}\label{eq:q-rule}

q = \min\left\{\bigl\{r\in\{1,\ldots,\bar q\}: \textstyle\sum_{\ell=1}^r s_\ell^2 \ge (1-\varepsilon)\|G\|\F^2\bigr\}\cup\{\bar q\}\right\},\qquad \bar q = \min(q_{\max}, m, d-m),

\end{equation}

корректно определено и гарантирует $\|[G]_q\|\F^2 \ge \max\bigl(\min(1-\varepsilon,\frac{\bar q}{\bar m}), \frac{q}{\bar m}\bigr)\|G\|\F^2 \ge \frac{1}{\bar m}\|G\|\F^2$.%[cite: 2, 3]

\item \textbf{Гарантия убывания для фиксированного шага и для backtracking Армихо:} В силу компактности $\Gr(m,d)$ и аналитичности $\Phi$ при $\tau>0$ существует константа $L>0$ такая, что $\frac{d^2}{dt^2}\Phi(R^\varphi_Y(tZ))\le L$ для всех $Y\in\St(d,m)$ и $Z\in T_Y\Gr(m,d)$ с $\|Z\|\F=1$.%[cite: 1, 2, 3]

Пусть при $G_k\ne 0$ единичное направление спуска равно $U_k = -[G_k]_q / \|[G_k]_q\|\F$, $Y_{k+1}=R^\varphi_{Y_k}(t_k U_k)$:

\begin{itemize}

\item[(а)] При выборе теоретического шага $t_k = \|[G_k]_q\|\F / L$ справедливо:%[cite: 2, 3]

\begin{equation}\label{eq:fixed-decrease}

\Phi(Y_k) - \Phi(Y_{k+1}) \ge \frac{\|[G_k]_q\|\F^2}{2L} \ge \frac{\|G_k\|\F^2}{2L\,\min(m,d-m)}.

\end{equation}

\item[(б)] При выборе $t_k$ методом \textbf{backtracking-поиска Армихо} с начальным шагом $t_k^{(0)} = \alpha_0\|[G_k]_q\|\F$ ($\alpha_0>0$), коэффициентом дробления $\beta\in(0,1)$ и параметром наклона $c_1\in(0,1)$ (где $t_k = \beta^m t_k^{(0)}$ — наибольший шаг, удовлетворяющий $\Phi(R^\varphi_{Y_k}(t_k U_k))\le \Phi(Y_k) - c_1 t_k\|[G_k]_q\|\F$) принятая длина шага всегда ограничена снизу и сверху:%[cite: 1, 3]

\begin{equation}\label{eq:armijo-bounds}

c_2\|[G_k]_q\|\F \le t_k \le \alpha_0\|[G_k]_q\|\F,\qquad

c_2 = \min\Bigl\{\alpha_0,\, \frac{2\beta(1-c_1)}{L}\Bigr\} > 0,

\end{equation}

откуда следует гарантированное убывание $\Phi(Y_k) - \Phi(Y_{k+1}) \ge c_1 c_2\|[G_k]_q\|\F^2 \ge \frac{c_1 c_2}{\bar m}\|G_k\|\F^2$.

\end{itemize}

В обоих случаях (а) и (б) выполнено $\sum_{k=0}^\infty \|G_k\|\F^2 < \infty$, $\lim_{k\to\infty}\|\grad\Phi(Y_k)\|\F = 0$, и все предельные точки последовательности $\{[Y_k]\}$ на $\Gr(m,d)$ стационарны.%[cite: 1, 2, 3]

\item \textbf{Сходимость всей последовательности при контролируемой длине шага:} Пусть на каждом шаге $Y_{k+1} = R^\varphi_{Y_k}(Z_k)$ с вещественно-аналитической ретракцией $R^\varphi$ ($\varphi(t)=t$ или $\arctan t$) выбор финального касательного вектора $Z_k$ (включая доводку методом Брента или шаг доверительной области на торе $\mathbb T^q$) подчиняется трём условиям сильного спуска с контролем длины \cite{ama05,ams}:%[cite: 1, 3, 5]

\begin{align}

\Phi(Y_k) - \Phi(Y_{k+1}) &\ge \tilde c_1\|G_k\|\F\|Z_k\|\F, &

\tilde c_2\|G_k\|\F &\le \|Z_k\|\F \le C_{\max}\|G_k\|\F,\label{eq:ama-12}\\

\bigl(G_k = 0 \;\;\text{или}\;\; \Phi(Y_{k+1}) = \Phi(Y_k)\bigr) &\implies Z_k = 0\;\; (Y_{k+1}=Y_k),\label{eq:ama-3}

\end{align}

где $\tilde c_1, \tilde c_2, C_{\max} > 0$ — фиксированные константы. (Если пробный шаг $\widehat Z_k$ от поиска Брента или TR улучшает значение Армихо $\Phi(R^\varphi_{Y_k}(\widehat Z_k))\le \Phi(R^\varphi_{Y_k}(t_{\rm Arm}U_k))$ и удовлетворяет барьеру длины $\tilde c_2\|G_k\|\F \le \|\widehat Z_k\|\F \le C_{\max}\|G_k\|\F$, то \eqref{eq:ama-12} выполнено с $\tilde c_1 = c_1 c_2 / (\bar m C_{\max})$). Тогда $\sum_{k=0}^\infty \|Z_k\|\F < \infty$, $\sum_{k=0}^\infty d_{\rm ch}(Y_{k+1},Y_k) < \infty$, и вся последовательность подпространств $\{[Y_k]\}$ сходится на $\Gr(m,d)$ к единственной стационарной точке $[Y_\infty]$.%[cite: 1, 3, 5]

\end{enumerate}

\end{theorem}

\begin{proof}

В п. 2(б) при $m=0$ имеем $t_k = \alpha_0\|[G_k]_q\|\F\ge c_2\|[G_k]_q\|\F$, а при $m\ge 1$ пробный шаг $\tilde t = \beta^{-1}t_k$ нарушил условие Армихо: $\Phi(Y_k) - c_1\tilde t\|[G_k]_q\|\F < \Phi(R^\varphi_{Y_k}(\tilde t U_k)) \le \Phi(Y_k) - \tilde t\|[G_k]_q\|\F + \frac{L}{2}\tilde t^2$, откуда $t_k = \beta\tilde t > \frac{2\beta(1-c_1)}{L}\|[G_k]_q\|\F\ge c_2\|[G_k]_q\|\F$. В п. 3 верхняя оценка $\|Z_k\|\F \le C_{\max}\|G_k\|\F$ гарантирует $\|Z_k\|\F\to 0$ при $\|G_k\|\F\to 0$, поэтому в канонической аналитической карте вокруг предельной точки $[Y_\infty]$ шаг остаётся в области билипшицевой эквивалентности хордовой и касательной метрик ($d_{\rm ch}(Y_{k+1},Y_k)\le \|Z_k\|\F$), а оценка неравенства Лоясевича $(\Phi(Y_k)-\Phi(Y_\infty))^\vartheta - (\Phi(Y_{k+1})-\Phi(Y_\infty))^\vartheta \ge \frac{\vartheta\tilde c_1}{C_{\rm L}}\|Z_k\|\F$ \cite{ama05,ams} даёт $\sum_{k=0}^\infty\|Z_k\|\F < \infty$.%[cite: 1, 3, 5]

\end{proof}

\section{Пространственное поле подпространств и касательная оболочка Такера}\label{sec:field}

При переходе к пространственно-переменному полю $Y_j = Y(x_j)$ независимая оптимизация $B_1,\ldots,B_J$ распадается на $J$ несвязанных задач, в которых при скалярном отклике и $m>1$ локальное подпространство в отдельном узле не идентифицируется.%[cite: 2, 3, 10]

\begin{proposition}[Необходимость широкой оболочки $R>m$, касательная модель и гладкость]\label{prop:tucker}

\begin{enumerate}

\item В линейной параметризации Такера $B_j = A M_j V\T$ \cite{tucker} с глобальным базисом оболочки $V\in\St(d,R)$ (envelope \cite{cook}) всегда $\spanop(B_j\T)\subseteq\spanop(V)$.%[cite: 2, 3, 5, 10]

Если $R=m$ и $\rank(B_j)=m$, то $\spanop(B_j\T)=\spanop(V)$ не зависит от $j$ (вращения нет).%[cite: 2, 3, 10]

Для вращения $m$-мерных плоскостей необходимо строгое неравенство $R > m$.%[cite: 2, 3, 5, 10]

\item В одной карте вокруг опорного репера $Y_0\in\St(d,m)$ геометрически корректная малоранговая параметризация поля строится в горизонтальном касательном пространстве до ретракции:%[cite: 2, 3, 5]

\begin{equation}\label{eq:tangent-tucker}

Z_j = V\left(\sum_{\ell=1}^s w_\ell(x_j) M_\ell\right)A\T \in T_{Y_0}\Gr(m,d),\qquad

Y_j = R^\varphi_{Y_0}(Z_j)\in\St(d,m),

\end{equation}

где $V\in\St(d,q)$, $Y_0\T V = 0$ (оболочка $[Y_0,V]$ имеет размерность $R=m+q > m$), $A\in\St(m,k)$ и $M_\ell\in\R^{q\times k}$.%[cite: 2, 3, 5]

Память сокращается с $O(Jdm)$ до $O(d(m+q)+mk+sqk+Js)$, а все $U_j Y_j$ выражаются через предвычисленный кэш $U_j[Y_0,V]$.%[cite: 2, 3, 5]

\item \textbf{Различие дискретной вариации и непрерывной гладкости:} Квадратичный графовый штраф со строго положительным знаком $+\lambda_{\rm gr}\tr(W\T L_{\rm graph}W) + \lambda_{\rm ch}\sum_{(i,j)\in\mathcal E}\omega_{ij}d_{\rm ch}^2(Y_i,Y_j)$ на связном графе центров ограничивает \emph{дискретную вариацию} значений в узлах $\{x_j\}$.%[cite: 2, 3]

Для непрерывной $C^s$-гладкости поля $x\mapsto [Y(x)]$ на области $\Omega_x\subset\R^d$ \emph{достаточно}, чтобы базисные функции словаря $w_\ell(x)$ принадлежали $C^s(\Omega_x)$, а значения касательного поля $Z(x)$ лежали в области диффеоморфности выбранной карты $R^\varphi_{Y_0}$ (необходимым это условие для каждого отдельного $w_\ell$ не является, так как негладкие компоненты могут взаимно компенсироваться в сумме $\sum_\ell w_\ell(x)M_\ell$).

\end{enumerate}

\end{proposition}

\section{Анизотропный тензор, геометрия матриц ранга $m$ и $\kappa$-семейство}\label{sec:anisotropy}

\subsection{Геометрия структурного тензора и быстрые спектральные действия}

На внешнем шаге структурной адаптации \cite{hjps} по текущему реперу $Y = P\T\in\St(d,m)$ и СПО-матрице энергетических весов $\Lambda\in\mathrm{Sym}_{++}(m)$ ($\|\Lambda\|_2=1$) строится структурная матрица $A = Y\Lambda Y\T$ и локализующий оператор $\mathcal T_k^2 = h_k^{-2}D_k$:%[cite: 2, 3, 6]

\begin{equation}\label{eq:D-forms}

D_{\rm full} = \alpha^2\Id_d + Y\Lambda Y\T \qquad\text{или}\qquad

D_{\rm orth} = \alpha^2(\Id_d - YY\T) + Y\Lambda Y\T,\qquad \alpha\in(0,1].

\end{equation}

\begin{proposition}[Геометрия пары $(Y,\Lambda)$ и матрично-свободные спектральные функции]\label{prop:tensor-geom}

\begin{enumerate}

\item При $m < d$ и $\Lambda\in\mathrm{Sym}_{++}(m)$ матрица $A = Y\Lambda Y\T$ является \textbf{положительно полуопределённой} (PSD) матрицей фиксированного ранга $m$.%[cite: 2, 3]

Отображение $(Y,\Lambda)\mapsto Y\Lambda Y\T$ инвариантно относительно $(Y,\Lambda)\mapsto(YQ, Q\T\Lambda Q)$, $Q\in O(m)$, и задаёт гладкое фактор-многообразие $\mathrm{Sym}_+^m(d) \cong (\St(d,m)\times\mathrm{Sym}_{++}(m))/O(m)$ \cite{mishra}.%[cite: 2, 3]

Если $\Lambda=\diag(\lambda_1>\cdots>\lambda_m>0)$ фиксирована диагональной с простым спектром, то калибровочная свобода $O(m)$ редуцируется до знаков столбцов $Y$.%[cite: 6]

\item При $\alpha > 0$ и $\Lambda\succ 0$ обе матрицы $D_{\rm full}$ и $D_{\rm orth}$ являются \textbf{строго положительно определёнными} (SPD) матрицами полного ранга $d$.%[cite: 2, 8]

При $\alpha=0$ или вырожденной $\Lambda$ матрица $D_{\rm orth}$ теряет ранг, и отрицательные степени $D_{\rm orth}^{-1/2}$ без положительной подрезки спектра (floor $\rho>0$) не определены.%[cite: 2, 8]

\item Для любой функции $f$, определённой на спектре $D\in\{D_{\rm full},D_{\rm orth}\}$, действие $f(D)x$ вычисляется точно за $O(dm)$ операций без сборки $d\times d$ матриц:%[cite: 2, 3, 5]

\begin{equation}\label{eq:fast-fD}

f(D)x = f(\alpha^2)x + Y\bigl(f(\Delta_{\rm in}) - f(\alpha^2)\Id_m\bigr)Y\T x,\qquad

\Delta_{\rm in} = \begin{cases}\alpha^2\Id_m + \Lambda, & D=D_{\rm full},\\ \Lambda, & D=D_{\rm orth}.\end{cases}

\end{equation}

\end{enumerate}

\end{proposition}

\subsection{Классификация локальных метрик ($\kappa$-семейство) и режим вырожденной $\Sigma_j$ ($n_j \le d$)}

В высокоразмерном режиме или на поздних итерациях сжатия полосы $h_k$ число эффективных точек в окне $n_j$ часто удовлетворяет $n_j \le d$, вследствие чего $\rank(\Sigma_j)\le n_j - 1 < d$ и матрица $\Sigma_j$ вырождена. Введём регуляризованную ковариацию $\Sigma_{j,\delta} = \Sigma_j + \delta\Id_d$ ($\delta\ge 0$) и при $\delta>0$ (либо при $\Sigma_j\succ 0$ и $\delta=0$) определим регуляризованную локальную оценку градиента $\hat\nabla_{j,\delta} = \Sigma_{j,\delta}^{-1}c_j$.

\begin{proposition}[Регуляризованное $\kappa$-семейство, смещение при $n_j\le d$ и структура шума]\label{prop:kappa}

Для $\kappa\in[0,2]$ и $\delta\ge 0$ (где при $\delta=0$ предполагается $\Sigma_j\succ 0$) рассмотрим семейство функционалов:%[cite: 1, 2, 3, 5]

\begin{equation}\label{eq:phi-kappa}

\Phi_{\kappa,\delta}(Y) = \sum_{j=1}^J N_j\min_{g_j\in\R^m}(\hat\nabla_{j,\delta} - Y g_j)\T\Sigma_{j,\delta}^\kappa(\hat\nabla_{j,\delta} - Y g_j) + \lambda d_{\rm ch}^2(Y,Y_0).

\end{equation}

Тогда при фиксированных центрах и весах $w_{ij}$ справедливы утверждения:

\begin{enumerate}

\item \textbf{Случай $\kappa=0$ (OPG):} При $\lambda=0$ профиль сводится к $\Phi_{0,\delta}(Y) = \mathrm{const} - \tr(Y\T C_{{\rm grad},\delta}Y)$, где $C_{{\rm grad},\delta} = \sum_{j=1}^J N_j\hat\nabla_{j,\delta}\hat\nabla_{j,\delta}\T$, и его глобальный минимум даётся $m$ старшими собственными векторами матрицы $C_{{\rm grad},\delta}$.%[cite: 1, 2, 3, 5, 6]

\item \textbf{Случай $\kappa=1$ и связь с MAVE:} При $\delta=0$ и $\Sigma_j\succ 0$ центрированная локальная МНК по наблюдениям удовлетворяет тождеству $\sum_{i=1}^n w_{ij}(y_i - \bar y_j - z_{ij}\T Y g_j)^2 = \mathrm{const}_j + N_j(\hat\nabla_{j,0} - Y g_j)\T\Sigma_j(\hat\nabla_{j,0} - Y g_j)$ \cite{mave}.%[cite: 2, 3]

Левая часть этого тождества корректно определена при любом ранге $\Sigma_j$ без обращения $\Sigma_j$.%[cite: 2]

\item \textbf{Связь со скетчем ADP ($\kappa=2$) и устойчивость к вырождению $\Sigma_j$:} Невязка направленного ADP равна $\|I_j - U_j Y g_j\|_2^2 = \|S_j(c_j - \Sigma_j Y g_j)\|_2^2$, а при $\Sigma_j\succ 0$ имеет вид $(\hat\nabla_{j,0} - Y g_j)\T\Sigma_j S_j\T S_j\Sigma_j(\hat\nabla_{j,0} - Y g_j)$.%[cite: 2, 3, 5]

Исходный критерий $\|I_j - U_j Y g_j\|_2^2$ ($\kappa=2$) не требует обращения $\Sigma_j$ и остаётся алгебраически корректным при $n_j\le d$.

\item \textbf{Смещение от регуляризации $\delta>0$ против псевдообратной матрицы $\Sigma_j^+$:} Пусть в узле $j$ выполнено $c_j = \Sigma_j Y_* g_j^* + \xi_j$ с $\mathbb E[\xi_j\mid X]=0$ и $\operatorname{Cov}(\xi_j\mid X) = \frac{\sigma^2}{N_j}\widetilde\Sigma_j$, где $\widetilde\Sigma_j = N_j^{-1}\sum_{i=1}^n w_{ij}^2 z_{ij}z_{ij}\T$.%[cite: 2, 3, 5]

Тогда для $\hat\nabla_{j,\delta} = (\Sigma_j+\delta\Id_d)^{-1}c_j$:

\begin{equation}\label{eq:delta-bias}

\mathbb E[\hat\nabla_{j,\delta}\mid X] - Y_* g_j^* = -\delta(\Sigma_j + \delta\Id_d)^{-1}Y_* g_j^*,\qquad

\operatorname{Cov}(\hat\nabla_{j,\delta}\mid X) = \frac{\sigma^2}{N_j}\Sigma_{j,\delta}^{-1}\widetilde\Sigma_j\Sigma_{j,\delta}^{-1}.

\end{equation}

При замене $(\Sigma_j+\delta\Id_d)^{-1}$ на псевдообратную матрицу $\Sigma_j^+$ ($\delta\to 0^+$) вектор $\mathbb E[\Sigma_j^+ c_j\mid X] = \Pi_{\range(\Sigma_j)}Y_* g_j^*$ проектируется на $\range(\Sigma_j)$ и при $\spanop(Y_*)\not\subseteq\range(\Sigma_j)$ יוצר угловое смещение относительно $\spanop(Y_*)$.

\end{enumerate}

\end{proposition}

\section{Вычислительная организация и единый профильно-плоский алгоритм}\label{sec:algorithm}

\subsection{Точные условия эффективности сжатия в режимах $p > d+1$ и $p \ll d$}

\begin{enumerate}[label=\textbf{U\arabic*.}]

\item \textbf{Режим избыточного скетча ($p > d+1$) — тонкое QR-сжатие:} Тонкое QR-разложение $[U_j, I_j] = Q_j R_j$ с сохранением верхнетреугольного фактора $[\bar U_j, \bar I_j] = R_j\in\R^{(d+1)\times(d+1)}$ тождественно сохраняет $\|I_j - U_j Y g_j\|_2^2$ при всех $(Y,g_j)$ и сокращает высоту локальных матриц с $p$ до $d+1$.%[cite: 2, 3]

При $p \le d+1$ эта операция не уменьшает размерность и не применяется.

\item \textbf{Режим малого скетча ($p \ll d$) — точный порог факторизации и кэширование:}

\begin{itemize}

\item[(а)] \emph{Порог выгодности факторизации $U_j = F_j Z_j$:} если в окне $j$ активны $n_j$ точек с матрицей разностей $Z_j\in\R^{n_j\times d}$ и предвычисленным фактором $F_j = N_j^{-1}(S_j Z_j\T)\diag(w_j)\in\R^{p\times n_j}$, то последовательное умножение $F_j(Z_j V_q)$ требует $n_j(p+d)q$ операций против $pdq$ для плотной $U_j$, а явное хранение пары $(F_j,Z_j)$ занимает $n_j(p+d)$ чисел против $pd$. Поэтому явная факторизация даёт выигрыш по операциям и памяти тогда и только тогда, когда выполнено строгое условие:

\begin{equation}\label{eq:factor-threshold}

n_j(p + d) < pd \quad\Longleftrightarrow\quad n_j < \frac{pd}{p+d} < \min(p,d).

\end{equation}

Если $n_j \ge \frac{pd}{p+d}$ (например, при $p=5, d=100, n_j=100$ плотная $U_j$ содержит $500$ элементов, тогда как пара $(F_j,Z_j)$ требует $10\,500$ элементов), хранение плотной матрицы $U_j$ строго выгоднее;

\item[(б)] \emph{Пакетная проекция и кэширование оболочки:} для плотных $U_j$ базовый кэш $M_j = U_j Y\in\R^{p\times m}$ и кэш активных направлений $W_j = U_j V_q\in\R^{p\times q}$ вычисляются одним блочным вызовом GEMM массива $(Jp)\times d$ на $d\times q$, а после полярного шага кэш $M_j$ обновляется линейной комбинацией столбцов $M_j$ и $W_j$ за $O(Jpmq)$ без обращения к массиву $U$;%[cite: 2, 3, 8, 9]

\item[(в)] \emph{Двухэтапная схема с сертификатом \eqref{eq:cert-bounds}:} при фиксированных $g_j$ сначала решается малая аффинная система поправки $B_V = P + YV\T$ ($\lambda Y + \sum_j N_j(g_j g_j\T)Y(U_j V)\T(U_j V) = Q(P)V$), после чего при необходимости запускается PCG/LSMR с остановкой по неравенству неточного спуска $2|\ip{D}{s}| < \ip{D}{\mathcal L(D)}$.%[cite: 2, 3, 9]

\end{itemize}

\end{enumerate}

\subsection{Исполнимая спецификация профильно-плоского алгоритма}

\begin{center}

\fbox{\begin{minipage}{0.96\linewidth}

\textbf{Алгоритм 1: Внутренний шаг профильно-плоского решателя на $\Gr(m,d)$}\\

\textbf{Вход:} Начальный репер $Y_0\in\St(d,m)$ ($0<m<d$), статистики $\{I_j,U_j,N_j\}_{j=1}^J$ (сжатые по правилу U1 при $p>d+1$), параметры $\tau>0$, $\lambda\ge 0$, порог энергобюджета $\varepsilon\in[0,1)$, предел ранга шага $q_{\max}\ge 1$, параметры Армихо $\alpha_0>0$, $\beta\in(0,1)$, $c_1\in(0,1)$, константа барьера длины шага $C_{\max}\ge \alpha_0$, допуск градиента $\mathrm{tol}_{\rm grad}>0$, предел итераций $K_{\max}$.\\

\textbf{Выход:} Обновлённый репер $Y\in\St(d,m)$, согласованные коэффициенты $\{g_j^\star(Y)\}_{j=1}^J$, энергетическая матрица $\mathcal J_{\rm EDR}$, канонический базис $P_{\rm out}$ и спектр $\Lambda_{\rm out}$.

\begin{enumerate}

\item Положить $Y\leftarrow Y_0$, $\bar q \leftarrow \min(q_{\max}, m, d-m)$. Вычислить кэш базовой проекции $M_j = U_j Y\in\R^{p\times m}$ для всех $j=1,\ldots,J$ за $O(Jpdm)$.%[cite: 2, 3]

\item \textbf{Цикл $k = 0, 1, \ldots, K_{\max}-1$:}

\begin{enumerate}[label=2.\arabic*.]

\item \emph{Профиль VARPRO:} Для каждого $j$ решить СПО-систему $(M_j\T M_j + \tau\Id_m)g_j^\star = M_j\T I_j$ методом Холецкого, вычислить $r_j = I_j - M_j g_j^\star$ и текущее значение $\Phi(Y)$ по \eqref{eq:phi-grass} за $O(J(pm^2+m^3))$.%[cite: 2, 3, 4]

\item \emph{Риманов градиент и проверка остановки:} Вычислить $G = \grad\Phi(Y)$ по \eqref{eq:rgrad} за $O(Jpdm + dm^2)$. Если $\|G\|\F \le \mathrm{tol}_{\rm grad}$, прервать цикл и перейти к шагу 3.%[cite: 2, 3, 4]

\item \emph{Спектральный выбор плоскостей:} Выполнить тонкое SVD $(-G) = V\Sigma A\T$ за $O(dm^2)$ ($\Sigma=\diag(s_1,\ldots,s_{\bar m})$). Выбрать ранг шага $q\in\{1,\ldots,\bar q\}$ по правилу \eqref{eq:q-rule}. Положить $V_q = V_{:,1:q}\in\St(d,q)$, $A_q = A_{:,1:q}\in\St(m,q)$ и $U_k = V_q\Sigma_q A_q\T / \|[G]_q\|\F$.%[cite: 2, 3]

\item \emph{Блочный кэш направлений:} Вычислить $W_j = U_j V_q\in\R^{p\times q}$ одним пакетным умножением за $O(Jpdq)$.%[cite: 2, 3]

\item \emph{Оптимизация на плоском торе $\mathbb T^q$ с контролем длины шага \eqref{eq:ama-12}:}

\begin{itemize}

\item Вычислить эталонный шаг backtracking-Армихо $t_{\rm Arm} = \beta^m \alpha_0\|[G]_q\|\F$ вдоль направления $U_k$, обеспечивающий $\Phi(R^{\rm pol}_Y(t_{\rm Arm}U_k))\le \Phi(Y) - c_1 t_{\rm Arm}\|[G]_q\|\F$ (при $q=1$ вычисление ведётся по формуле Шура \eqref{eq:rat-profile} за $O(J)$ на пробу setelah подготовки 6 скаляров \eqref{eq:six-scalars}; при $q>1$ — через кэш $[M_j,W_j]$). Положить $\eta_{\rm ref} = t_{\rm Arm}\diag(\Sigma_q)/\|[G]_q\|\F$.

\item Выполнить уточняющий поиск кандидата $\hat\eta\in\R^q$ (при $q=1$ — метод Брента по \eqref{eq:rat-profile} на интервале $[-C_{\max}\|G\|\F, C_{\max}\|G\|\F]$; при $q>1$ — шаг доверительной области радиуса $\Delta_{\rm TR}\le C_{\max}\|G\|\F$ с якобианом \eqref{eq:varpro-jacobian}).

\item \emph{Двусторонний фильтр принятия:} Принять $\eta^\star = \hat\eta$, если и только если одновременно выполнены два условия:

\[

    \Phi\bigl(\widetilde Y(\hat\eta)\bigr) \le \Phi\bigl(\widetilde Y(\eta_{\rm ref})\bigr) \qquad\text{и}\qquad

    \|\eta_{\rm ref}\|_2 \le \|\hat\eta\|_2 \le C_{\max}\|G\|\F

\]

(либо прямо проверено $\Phi(Y)-\Phi(\widetilde Y(\hat\eta))\ge \tilde c_1\|G\|\F\|\hat\eta\|_2$ при $\tilde c_2\|G\|\F\le \|\hat\eta\|_2\le C_{\max}\|G\|\F$); в противном случае положить $\eta^\star = \eta_{\rm ref}$.

\end{itemize}

\item \emph{Обновление репера и кэша:} Обновить $Y_+ = R^{\rm pol}_Y(V_q\diag(\eta^\star)A_q\T)$ по формуле Положения~\ref{prop:embedding}. Обновить кэш $M_j \leftarrow M_j(\Id_m - A_q A_q\T) + M_j A_q\diag((1+\eta_\ell^{\star 2})^{-1/2})A_q\T + W_j\diag(\eta_\ell^\star(1+\eta_\ell^{\star 2})^{-1/2})A_q\T$ за $O(Jpmq)$ (при периодической стабилизации $Y_+\leftarrow\qf(Y_+)$ пересчитать $M_j = U_j Y_+$), положить $Y\leftarrow Y_+$.%[cite: 2, 3]

\end{enumerate}

\item \textbf{Безусловная финальная синхронизация коэффициентов и спектра:} Для итогового репера $Y$ и кэша $M_j = U_j Y$ решить при всех $j=1,\ldots,J$ системы $(M_j\T M_j + \tau\Id_m)g_j^\star(Y) = M_j\T I_j$. Сформировать $\mathcal J_{\rm EDR} = Y\bigl(\sum_{j=1}^J N_j g_j^\star(Y)g_j^\star(Y)\T\bigr)Y\T$, вычислить её собственное разложение $\mathcal J_{\rm EDR} = P_{\rm out}\T\Lambda P_{\rm out}$ и вернуть $P_{\rm out}$, $\Lambda_{\rm out} = \Lambda/\|\Lambda\|_2$.%[cite: 6]

\end{enumerate}

\end{minipage}}

\end{center}

\section{Два направления развития: оптимальные аппроксимации оператора и совместная наблюдаемость}\label{sec:extensions}

\subsection{Результат B: оптимальное по затратам приближение нормального оператора}

В Теореме~\ref{thm:separable-svd} относительное спектральное искажение $\theta(\mathcal L_0) = \|\mathcal L_0^{-1/2}(\mathcal L - \mathcal L_0)\mathcal L_0^{-1/2}\|_2$ напрямую определяет как гарантированную долю ранк-1 выигрыша $\frac{1-\theta}{1+\theta}$ по отношению к наилучшему возможному ранк-1 атому $\Gamma_1(Q) = \max_{\rank E=1}\gamma_{\mathcal L}(E)$, так и скорость сходимости PCG.%[cite: 2, 9]

Однако усложнение структуры $\mathcal L_0$ само по себе не гарантирует уменьшения $\theta(\mathcal L_0)$.

\begin{remark}[Контрпример к наивному двустороннему кронекерову разнесению регуляризатора]\label{rem:kron-counterexample}

Если определить двустороннее приближение $\mathcal L_{\rm kron}(X) = G X H$, включив регуляризатор $\lambda$ одновременно в оба множителя: $G = \sum_j N_j\eta_j g_j g_j\T + \lambda\Id_m$ и $H = \lambda\Id_d + \sum_j w_j H_j$ ($w_j = N_j\|g_j\|_2^2/\tr(G)$), то качество приближения может стать \emph{хуже}, чем у простого левого изотропного оператора $\mathcal L_{\rm iso}(X) = G X$. Действительно, возьмём $m=1, d=p=2, J=1, U_1=\Id_2, g_1=1, N_1=1, \lambda=1$. Тогда истинный оператор равен $\mathcal L(X) = 2X$, изотропный оператор точен: $\mathcal L_{\rm iso}(X) = 2X$ ($\theta_{\rm iso}=0$), тогда как $\mathcal L_{\rm kron}(X) = 2 X (1 + 1/2)\Id_2 = 3X$, что даёт $\theta_{\rm kron} = \|(3\Id)^{-1/2}(2\Id - 3\Id)(3\Id)^{-1/2}\|_2 = 1/3 > \theta_{\rm iso}$. Причина состоит в том, что произведение $(G_0+\lambda\Id_m)X(H_0+\lambda\Id_d)$ искажает аддитивный член $\lambda X$. Более того, даже минимизация фробениусовой невязки $\|\mathcal L - (H\T\otimes G)\|\F^2$ по $G,H\succ 0$ \cite{kron} не гарантирует уменьшения относительной спектральной нормы $\theta(\mathcal L_0)$.%[cite: 2]

\end{remark}

\begin{proposition}[Семейство аддитивно-кронекеровых приближений и сравнение с ALS]\label{prop:cost-optimal-L0}

Рассмотрим семейство структурных приближений ранга $s\ge 1$, сохраняющих точную структуру регуляризатора:

\begin{equation}\label{eq:Ls-family}

\mathcal L_0^{(s)}(X) = \sum_{\ell=1}^s G_\ell X H_\ell + \lambda X,\qquad G_\ell\in\mathrm{Sym}_+(m),\quad H_\ell\in\mathrm{Sym}_+(d).

\end{equation}

Пусть $\operatorname{Cost}(\mathcal L_0) = \operatorname{Cost}_{\rm build}(\mathcal L_0) + k_{\rm apply}\operatorname{Cost}_{\rm apply}(\mathcal L_0)$ обозначает суммарные затраты на построение и применение приближения. Тогда:

\begin{enumerate}

\item \textbf{Одночленное приближение ($s=1$, Сильвестров предобусловливатель):} При $s=1$ оператор $\mathcal L_0^{(1)}(X) = G_1 X H_1 + \lambda X$ обращается точно с помощью спектральных разложений $G_1 = Q_G\Lambda_G Q_G\T$ ($O(m^3)$) и $H_1 = Q_H\Lambda_H Q_H\T$: решение уравнения $G_1 X H_1 + \lambda X = R$ в базисе $(Q_G,Q_H)$ имеет диагональный вид $(\widetilde X)_{ab} = (\widetilde R)_{ab} / (\lambda_{G,a}\lambda_{H,b} + \lambda)$. Если $H_1 = \Id_d$ и $G_1 = \sum_j N_j\eta_j g_j g_j\T$, мы возвращаемся к $\mathcal L_{\rm iso}(X) = (G_1+\lambda\Id_m)X$ со стоимостью $O(Jpd + dm^2 + m^3)$.

\item \textbf{Вариационный выбор приближения по бюджету:} Задача оптимального синтеза предобусловливателя/оракула для заданного порога качества $\theta\in(0,1)$ имеет вид:

\begin{equation}\label{eq:opt-precond}

\min_{\mathcal L_0\in\mathfrak M}\operatorname{Cost}(\mathcal L_0)\qquad\text{при условии}\qquad

\bigl\|\mathcal L_0^{-1/2}(\mathcal L - \mathcal L_0)\mathcal L_0^{-1/2}\bigr\|_2 \le \theta.

\end{equation}

\item \textbf{Корректное сравнение одного шага с циклом ALS:} Пусть $E_{\mathrm{ALS}_t}$ — ранк-1 атом, полученный после $t$ шагов альтернирующего поиска \eqref{eq:als-systems} при текущем $Q(B)$. Поскольку $\rank(E_{\mathrm{ALS}_t}) = 1$, его выигрыш не превосходит глобального максимума $\Gamma_1(Q) = \max_{\rank E=1}\gamma_{\mathcal L}(E)$. Следовательно, при выполнении \eqref{eq:opt-precond} однократный спектральный атом $E_0(\mathcal L_0)$ гарантирует по отношению к любому $t$-шаговому результату ALS одного атома оценку:

\begin{equation}\label{eq:als-bound}

\gamma_{\mathcal L}\bigl(E_0(\mathcal L_0)\bigr) \ge \frac{1-\theta}{1+\theta}\,\Gamma_1(Q) \ge \frac{1-\theta}{1+\theta}\,\gamma_{\mathcal L}(E_{\mathrm{ALS}_t}).

\end{equation}

Для сравнения многошаговых последовательностей (нескольких атомов подряд) сравнение должно проводиться при фиксированном суммарном бюджете арифметических операций $\sum_{\text{шаги}}\operatorname{Cost}$.

\end{enumerate}

\end{proposition}

\subsection{Результат A: совместная наблюдаемость при $p < d$, идентифицируемость и внешний цикл}

В высокоразмерном режиме $p < d$ каждый локальный оператор $U_j\in\R^{p\times d}$ имеет нетривиальное ядро размерности $\dim\ker(U_j) \ge d - p \ge 1$. Поэтому поузловое условие $\sigma_{\min}(U_j)\ge\underline{\sigma}_j > 0$ \textbf{не обеспечивает} наблюдаемости векторов $v_j\in\R^d$ по значениям $U_j v_j$.

\begin{remark}[Контрпример к поузловой наблюдаемости при $p < d$]\label{rem:obs-counterexample}

Пусть $d=2$, $m=p=J=1$, $N_1=1$, $U_1 = (1, 0)\in\R^{1\times 2}$, $Y_* = (1, 0)\T$ и $g_1^* = 1$. Тогда истинный отклик равен $I_1 = U_1 Y_* g_1^* = 1$, причём $\nu_1 = (g_1^*)^2 = 1 > 0$ и $\sigma_{\min}(U_1) = 1 > 0$. Однако для ложного подпространства $\widehat Y = \frac{1}{\sqrt 2}(1, 1)\T$ и коэффициента $\hat g_1 = \sqrt 2$ имеем $U_1\widehat Y\hat g_1 = 1 = I_1$, так что невязка данных тождественно равна нулю ($\mathcal R_k = \|U_1(\widehat Y\hat g_1 - Y_* g_1^*)\|_2 = 0$), тогда как ошибка подпространства составляет $d_{\rm ch}(\widehat Y, Y_*) = 1/\sqrt 2 \approx 0.707 > 0$. Причина в том, что вектор разности $\widehat Y\hat g_1 - Y_* g_1^* = (0, 1)\T$ целиком лежит в $\ker(U_1)$.

\end{remark}

Следовательно, восстановление $Y_*$ при $p < d$ возможно лишь за счёт того, что разные узлы $j=1,\ldots,J$ просвечивают разные направления в $\R^d$ при совместном ограничении $\rank(Y)=m$. Введём нелинейный оператор совместных измерений:

\begin{equation}\label{eq:collective-op}

\mathscr A\colon \St(d,m)\times(\R^m)^J \longrightarrow (\R^p)^J,\qquad

\mathscr A\bigl(Y, \{g_j\}_{j=1}^J\bigr) = \bigl\{\sqrt{N_j}\,U_j Y g_j\bigr\}_{j=1}^J.

\end{equation}

\begin{proposition}[Результат A: локальная идентифицируемость оператора $\mathscr A$ на $\Gr(m,d)$]\label{prop:identifiability}

Рассмотрим профилированный оператор измерений в окрестности истинной пары $(Y_*, \{g_j^*\})$ при $U_j Y_*$ полного столбцового ранга $m$ ($p \ge m$). Обозначим ортопроектор на ортогональное дополнение к $\range(U_j Y_*)$ в $\R^p$:

\begin{equation}\label{eq:proj-orth-j}

\Pi_{j,_}^\perp = \Id_p - U_j Y__\bigl(Y__\T U_j\T U_j Y__\bigr)^{-1}Y_*\T U_j\T \in \R^{p\times p}.

\end{equation}

Тогда:

\begin{enumerate}

\item \textbf{Критерий локальной идентифицируемости первого порядка:} Подпространство $[Y_*]\in\Gr(m,d)$ локально идентифицируемо по бесшумным наблюдениям $I_j^* = U_j Y_* g_j^*$ с точностью до $O(m)$-калибровки в первом порядке тогда и только тогда, когда профилированный оператор Гаусса--Ньютона (информационный оператор) на горизонтальном касательном пространстве $T_{Y_*}\Gr(m,d)$ строго положительно определён:%[cite: 1, 3]

\begin{equation}\label{eq:mu-id}

\ip{Z}{\mathcal H_{\rm GN}(Y__)Z} \coloneqq \sum\_{j=1}^J N_j\bigl\|\Pi\_{j,_}^\perp U_j Z g_j^_\bigr\|*2^2 \ge \mu*{\rm id}\,\|Z\|\F^2\quad \forall Z\in T\_{Y__}\Gr(m,d),\qquad \mu_{\rm id} > 0.

\end{equation}

\item \textbf{Необходимое размерностное условие:} Поскольку $\rank(\Pi_{j,*}^\perp)\le p-m$, каждое слагаемое в \eqref{eq:mu-id} доставляет не более $p-m$ скалярных линейных ограничений на вектор $Z\in T_{Y_*}\Gr(m,d)$ размерности $m(d-m)$. Поэтому для выполнения $\mu_{\rm id} > 0$ необходимо:

\begin{equation}\label{eq:dim-condition}

p > m \qquad\text{и}\qquad J(p - m) \ge m(d - m) \quad\Longleftrightarrow\quad Jp \ge m(d - m + J)

\end{equation}

(а в задаче с известными фиксированными $g_j^*$, где $\Pi_{j,*}^\perp$ заменяется на $\Id_p$, необходимо $Jp \ge m(d-m)$). При $p \le m$ без регуляризации $\tau>0$ любое возмущение $Z$ может быть локально компенсировано выбором $g_j$ в каждом узле отдельно, и $\mu_{\rm id} = 0$.

\end{enumerate}

\end{proposition}

\begin{proof}

Линеаризуем уравнение $U_j(Y_* + Z)(g_j^* + \delta g_j) = U_j Y_* g_j^*$ при $Y_*\T Z = 0$: получаем $U_j Z g_j^* + U_j Y_*\delta g_j = 0$. Исключение свободного вектора $\delta g_j\in\R^m$ равносильно проектированию на $(\range(U_j Y_*))^\perp$ с помощью $\Pi_{j,*}^\perp$, что даёт систему $\Pi_{j,*}^\perp U_j Z g_j^* = 0$ ($j=1,\ldots,J$). Её тривиальность на $T_{Y_*}\Gr(m,d)$ эквивалентна \eqref{eq:mu-id}, а подсчёт рангов отображений $Z\mapsto \Pi_{j,*}^\perp U_j Z g_j^*\in\range(\Pi_{j,*}^\perp)\cong\R^{p-m}$ даёт \eqref{eq:dim-condition}.

\end{proof}

Сформулируем теперь исправленную теорему об устойчивости восстановления подпространства и рекурсии внешнего цикла.

\begin{theorem}[Условная оценка возмущения EDR-подпространства и рекурсия внешнего цикла]\label{thm:perturbation-edr}

Пусть на $k$-м внешнем шаге локальные статистики имеют вид $I_j^{(k)} = U_j^{(k)} Y_* g_j^* + \xi_j^{(k)}$, где $g_j^* = \nabla\psi(Y_*\T x_j)\in\R^m$, а $\xi_j^{(k)}\in\R^p$ — суммарная ошибка аппроксимации и шума.%[cite: 6]

Пусть $(\widehat B_k, \{\hat g_j\})$ — приближённое решение внутренней задачи, $Y_k\in\St(d,m)$ — ведущее $m$-мерное собственное подпространство матрицы $\widehat{\mathcal J}_k = \sum_{j=1}^J N_j^{(k)} (\widehat B_k\T \hat g_j)(\widehat B_k\T \hat g_j)\T$, и пусть выполнены два условия:

\begin{itemize}

\item[(I1)] \emph{Возбуждение истинных градиентов:} $\lambda_{\min}\Bigl(\sum_{j=1}^J N_j^{(k)} g_j^*(g_j^*)\T\Bigr) \ge \nu_m^{(k)} > 0$;%[cite: 1, 6, 9]

\item[(I2*)] \emph{Ограниченная совместная наблюдаемость (Restricted Observability):} для набора векторов разностей $v_j = \widehat B_k\T\hat g_j - Y_* g_j^*$ ($j=1,\ldots,J$) выполнено неравенство:

\begin{equation}\label{eq:restricted-obs}

\sum_{j=1}^J N_j^{(k)}\|U_j^{(k)} v_j\|_2^2 \ge \mu_{\rm obs}^{(k)}\sum_{j=1}^J N_j^{(k)}\|v_j\|_2^2,\qquad \mu_{\rm obs}^{(k)} > 0,

\end{equation}

и обозначим совместный энергетический зазор $\Delta_k \coloneqq \mu_{\rm obs}^{(k)}\,\nu_m^{(k)} > 0$.

\end{itemize}

Тогда:

\begin{enumerate}

\item \textbf{Оценка ошибки через статистический шум и невязку решателя:} Положим

\[

\mathcal R_k \coloneqq \Bigl(\sum_{j=1}^J N_j^{(k)}\|U_j^{(k)}(\widehat B_k\T\hat g_j - Y_* g_j^*)\|_2^2\Bigr)^{1/2}

\le 2\,\mathcal E_{\rm stat}(D_k,h_k) + \delta_{\rm solver}^{(k)},

\]

где $\mathcal E_{\rm stat}(D_k,h_k) = (\sum_j N_j^{(k)}\|\xi_j^{(k)}\|_2^2)^{1/2}$ — статистическая ошибка данных, а $\delta_{\rm solver}^{(k)} = \max\bigl(0,\, (\sum_j N_j^{(k)}\|I_j^{(k)} - U_j^{(k)}\widehat B_k\T\hat g_j\|_2^2)^{1/2} - \mathcal E_{\rm stat}(D_k,h_k)\bigr)$ — избыточная невязка неточного решателя. Если $\mathcal R_k < \sqrt{\Delta_k}$, то по теореме Уэдина хордовая ошибка восстановления подпространства удовлетворяет оценке:

\begin{equation}\label{eq:wedin-edr}

e_k \coloneqq d_{\rm ch}(Y_k, Y_*) \le \frac{\sqrt{2}\,\mathcal R_k}{\sqrt{\Delta_k} - \mathcal R_k}

\le \frac{\frac{2\sqrt{2}}{\sqrt{\Delta_k}}\,\mathcal E_{\rm stat}(D_k,h_k) + \frac{\sqrt{2}}{\sqrt{\Delta_k}}\,\delta_{\rm solver}^{(k)}}{1 - \mathcal R_k/\sqrt{\Delta_k}}.

\end{equation}

\item \textbf{Нелинейная рекурсия внешнего цикла и точный порог сжимаемости:} Пусть при обновлении тензора $D_{k+1} = \mathcal U(Y_k,\Lambda_k)$ по правилу \eqref{eq:D-forms} и сужении полосы $h_{k+1} = h_k / a$ статистический член удовлетворяет оценке чувствительности:

\begin{equation}\label{eq:outer-lipschitz}

\frac{2\sqrt{2}\,\mathcal E_{\rm stat}\bigl(\mathcal U(Y_k,\Lambda_k), h_{k+1}\bigr)}{\sqrt{\Delta_{k+1}}}

\le \bar e_{\rm stat}(h_{k+1}) + \kappa_{\rm adapt}^{(k)}\, e_k,

\end{equation}

где $\bar e_{\rm stat}(h_{k+1})$ — ошибка при оракульном ядре $D_* = \mathcal U(Y_*,\Lambda_*)$. Тогда на шаге $k+1$ при $\mathcal R_{k+1}/\sqrt{\Delta_{k+1}} \le \rho_{k+1} < 1$ выполнена рекурсия:

\begin{equation}\label{eq:outer-recursion}

e_{k+1} \le \frac{\kappa_{\rm adapt}^{(k)}\, e_k + \bar e_{\rm stat}(h_{k+1}) + \sqrt{2}\,\delta_{\rm solver}^{(k+1)}/\sqrt{\Delta_{k+1}}}{1 - \mathcal R_{k+1}/\sqrt{\Delta_{k+1}}}

\le \frac{\kappa_{\rm adapt}^{(k)}}{1 - \rho_{k+1}}\,e_k + \frac{\bar e_{\rm stat}(h_{k+1}) + \sqrt{2}\,\delta_{\rm solver}^{(k+1)}/\sqrt{\Delta_{k+1}}}{1 - \rho_{k+1}}.

\end{equation}

В частности, если $\mathcal R_{k+1}/\sqrt{\Delta_{k+1}} \le \rho < 1$ равномерно в рабочей окрестности, то достаточным условием сжимаемости однородной части рекурсии является строгое неравенство:

\begin{equation}\label{eq:contraction-cond}

\kappa_{\rm adapt}^{(k)} < 1 - \rho.

\end{equation}

\end{enumerate}

\end{theorem}

\begin{proof}

Для матриц $G_* = [\sqrt{N_1^{(k)}}Y_* g_1^*,\ldots,\sqrt{N_J^{(k)}}Y_* g_J^*]\in\R^{d\times J}$ и $\widehat G = [\sqrt{N_1^{(k)}}\widehat B_k\T\hat g_1,\ldots,\sqrt{N_J^{(k)}}\widehat B_k\T\hat g_J]\in\R^{d\times J}$ условие совместной наблюдаемости \eqref{eq:restricted-obs} даёт $\|\widehat G - G_*\|\F = (\sum_j N_j^{(k)}\|v_j\|_2^2)^{1/2} \le \mathcal R_k / \sqrt{\mu_{\rm obs}^{(k)}}$. По условию (I1) имеем $\sigma_m(G_*)\ge\sqrt{\nu_m^{(k)}}$ и $\rank(G_*)=m$. Применение теоремы Уэдина к левым сингулярным подпространствам $G_*$ и $\widehat G$ при $\|\widehat G - G_*\|\F < \sqrt{\nu_m^{(k)}}$ (т.е. $\mathcal R_k < \sqrt{\Delta_k}$) даёт $d_{\rm ch}(Y_k,Y_*) \le \frac{\sqrt 2\|\widehat G - G_*\|\F}{\sqrt{\nu_m^{(k)}} - \|\widehat G - G_*\|\F} \le \frac{\sqrt 2\,\mathcal R_k}{\sqrt{\Delta_k} - \mathcal R_k}$, откуда следуют \eqref{eq:wedin-edr} и \eqref{eq:outer-recursion}.

\end{proof}

\begin{remark}[Физический смысл условия $\kappa_{\rm adapt}^{(k)} < 1 - \rho$ и открытая проблема при $p \ll d$]

Формула \eqref{eq:outer-recursion} показывает, что вблизи порога идентифицируемости ($\rho = \mathcal R_{k+1}/\sqrt{\Delta_{k+1}} \to 1$) множитель $(1-\rho)^{-1}$ неограниченно растёт, и даже умеренная чувствительность ядра $\kappa_{\rm adapt}^{(k)} < 1$ может приводить к взрывному росту ошибки, если нарушено условие \eqref{eq:contraction-cond}. Кроме того, если при $p \ge d$ и невырожденных $\Sigma_j, S_j$ константа $\mu_{\rm obs}^{(k)} \ge \min_j\sigma_{\min}^2(U_j^{(k)}) > 0$ положительна детерминированно, то при $p < d$ положительность $\mu_{\rm obs}^{(k)} > 0$ в \eqref{eq:restricted-obs} опирается на геометрическое разнообразие подпространств $\range((U_j^{(k)})\T)$ между разными узлами $j$ (см. Предложение~\ref{prop:identifiability}); строгое вероятностное доказательство оценки $\mu_{\rm obs}^{(k)} > 0$ для анизотропных ядер и скетчей при $p \ll d$ составляет главную открытую математическую задачу статистической теории ADP.

\end{remark}

\appendix

\section{Сопоставление с научной литературой}\label{app:literature}

\begin{table}[h!]

\centering\footnotesize

\caption{Научные предшественники и границы переноса известных результатов}\label{tab:lit}

\begin{tabularx}{\linewidth}{@{}P{36mm}P{36mm}L@{}}

\toprule

Направление и источники & Что именно используется & Где заканчивается перенос и что добавлено здесь\\

\midrule

Адаптивное снижение размерности \cite{hjps,mave} & Схема структурной адаптации ADE и локальная регрессия MAVE & В \cite{hjps,mave} не используются конечный скетч $S_j$, профильные вращения на $\Gr(m,d)$ и регуляризованное $\kappa$-семейство (Предложение~\ref{prop:kappa}).%[cite: 2, 3, 5, 6]

\\

Сепарабельные нелинейные МНК \cite{varpro,kaufman,rw} & Исключение линейных параметров (VARPRO) и якобиан \eqref{eq:varpro-jacobian} & Классический VARPRO строится в евклидовом пространстве; здесь он объединён с $O(m)$-факторизацией на $\Gr(m,d)$ и скалярным профилем Шура за $O(J)$.%[cite: 2, 3, 5]

\\

Геометрия $\Gr(m,d)$ и ретракции \cite{edelman,ams,absilmalick,wenyin} & Геодезическая экспонента, разложение Картана, полярная и Кэли карты & Геометрические формулы стандартны; новыми для задачи ADP являются вложение SVD-поправки (Предложение~\ref{prop:embedding}) и рациональный профиль \eqref{eq:rat-profile}.%[cite: 2, 3, 5]

\\

Онлайн-подпространства и методы Якоби \cite{grouse,jacobi,tsd} & Ранк-1 геодезические шаги (GROUSE) и плоские вращения Якоби / Гивенса & GROUSE работает с неполными векторами и шагом первого порядка; здесь в каждом узле действует оператор $U_j$, а шаг ищется по профилю Шура.%[cite: 2, 3, 5]

\\

Сходимость по Лоясевичу \cite{ama05} & Теорема сходимости аналитических функций к одной точке & Доказана нижняя оценка шага для backtracking Армихо \eqref{eq:armijo-bounds} и введён двусторонний фильтр длины шага \eqref{eq:ama-12} для доводки Брента/TR.%[cite: 1, 3, 5]

\\

Малоранговая аппроксимация и Кронекер \cite{eckart,pursuit,weighted,mishra,kron} & Теорема Эккарта--Янга, жадный поиск ORMP, кронекерова аппроксимация & Доказана гарантия сохранения доли $(1-\theta)/(1+\theta)$ ранк-1 выигрыша (Теорема~\ref{thm:separable-svd}) и сформулирована задача синтеза \eqref{eq:opt-precond}.%[cite: 2, 3, 5, 9]

\\

Тензорные разложения и оболочки \cite{tucker,ishteva,cook,manifold} & Модели Такера, envelopes Кука, локальная регрессия на многообразиях & Выведено условие вращения $R>m$, предложена касательная модель \eqref{eq:tangent-tucker} и разделены дискретная вариация и достаточные условия гладкости.%[cite: 2, 3, 5, 10]

\\

\bottomrule

\end{tabularx}

\end{table}

\begin{thebibliography}{99}\small

\bibitem{hjps} M. Hristache, A. Juditsky, J. Polzehl, V. Spokoiny. \emph{Structure adaptive approach for dimension reduction}. Ann. Statist. 29(6):1537--1566, 2001.

\bibitem{mave} Y. Xia, H. Tong, W. K. Li, L.-X. Zhu. \emph{An adaptive estimation of dimension reduction space} (MAVE). J. R. Stat. Soc. B 64(3):363--410, 2002.

\bibitem{varpro} G. H. Golub, V. Pereyra. \emph{The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate}. SIAM J. Numer. Anal. 10(2):413--432, 1973.

\bibitem{kaufman} L. Kaufman. \emph{A variable projection method for solving separable nonlinear least squares problems}. BIT 15(1):49--57, 1975.

\bibitem{rw} A. Ruhe, P. \AA. Wedin. \emph{Algorithms for separable nonlinear least squares problems}. SIAM Rev. 22(3):318--337, 1980.

\bibitem{edelman} A. Edelman, T. A. Arias, S. T. Smith. \emph{The geometry of algorithms with orthogonality constraints}. SIAM J. Matrix Anal. Appl. 20(2):303--353, 1998.

\bibitem{ams} P.-A. Absil, R. Mahony, R. Sepulchre. \emph{Optimization Algorithms on Matrix Manifolds}. Princeton University Press, 2008.

\bibitem{absilmalick} P.-A. Absil, J. Malick. \emph{Projection-like retractions on matrix manifolds}. SIAM J. Optim. 22(1):135--158, 2012.

\bibitem{wenyin} Z. Wen, W. Yin. \emph{A feasible method for optimization with orthogonality constraints}. Math. Program. 142:397--434, 2013.

\bibitem{ama05} P.-A. Absil, R. Mahony, B. Andrews. \emph{Convergence of the iterates of descent methods for analytic cost functions}. SIAM J. Optim. 16(2):531--547, 2005.

\bibitem{grouse} L. Balzano, R. Nowak, B. Recht. \emph{Online identification and tracking of subspaces from highly incomplete information} (GROUSE). Allerton, 2010.

\bibitem{jacobi} C. G. J. Jacobi. \emph{\"Uber ein leichtes Verfahren, die in der Theorie der S\"acularst\"orungen vorkommenden Gleichungen numerisch aufzul\"osen}. J. Reine Angew. Math. 30:51--94, 1846.

\bibitem{tsd} D. Gutman, N. Ho-Nguyen. \emph{Coordinate descent without coordinates: tangent subspace descent on Riemannian manifolds}. Math. Oper. Res. 48(1):123--157, 2023.

\bibitem{eckart} C. Eckart, G. Young. \emph{The approximation of one matrix by another of lower rank}. Psychometrika 1:211--218, 1936.

\bibitem{pursuit} Z. Wang, M.-J. Lai, Z. Lu, W. Fan, H. Davulcu, J. Ye. \emph{Rank-one matrix pursuit for matrix completion}. ICML (PMLR 32), 91--99, 2014.

\bibitem{weighted} N. Srebro, T. Jaakkola. \emph{Weighted low-rank approximations}. ICML, 720--727, 2003.

\bibitem{mishra} B. Mishra, G. Meyer, S. Bonnabel, R. Sepulchre. \emph{Fixed-rank matrix factorizations and Riemannian low-rank optimization}. Comput. Stat. 29:591--621, 2014.

\bibitem{kron} C. F. Van Loan, N. Pitsianis. \emph{Approximation with Kronecker products}. In: Linear Algebra for Large Scale and Real-Time Applications, 293--314, Kluwer, 1993.

\bibitem{tucker} T. G. Kolda, B. W. Bader. \emph{Tensor decompositions and applications}. SIAM Rev. 51(3):455--500, 2009.

\bibitem{ishteva} M. Ishteva, P.-A. Absil, S. Van Huffel, L. De Lathauwer. \emph{Best low multilinear rank approximation of higher-order tensors, based on the Riemannian trust-region scheme}. SIAM J. Matrix Anal. Appl. 32(1):115--135, 2011.

\bibitem{cook} R. D. Cook, B. Li, F. Chiaromonte. \emph{Envelope models for parsimonious and efficient multivariate linear regression}. Statist. Sinica 20(3):927--960, 2010.

\bibitem{manifold} M.-Y. Cheng, H.-T. Wu. \emph{Local linear regression on manifolds and its geometric interpretation}. J. Amer. Statist. Assoc. 108(504):1421--1434, 2013.

\bibitem{lsmr} D. C.-L. Fong, M. Saunders. \emph{LSMR: An iterative algorithm for sparse least-squares problems}. SIAM J. Sci. Comput. 33(5):2950--2971, 2011.

\end{thebibliography}

\end{document}

1. Архитектура обобщения: в чём состоит единая суть теории

Ваша интуиция о том, что SVD-решатель и геодезические/полярные шаги на многообразии Грассмана Gr(m,d) являются частными случаями единого механизма разложения многообразия в более простое пространство, имеет точное дифференциально-геометрическое и алгебраическое выражение.

Вычислительная трудность всех вариантов внутренней задачи EDR/ADE вызвана тем, что искомая структура θ (матрица базиса B∈Rm×d, ортонормированный репер Y∈St(d,m), пространственное поле {Yj​}j=1J​ или структурный тензор A=YΛYT) взаимодействует с высокой размерностью d в каждом из J узлов через проекции Uj​Y∈Rp×m, а локальные ковариации Hj​=UjT​Uj​ сцеплены с направлениями градиентов gj​gjT​.

Вся теория сжимается в принцип трёхступенчатой декомпозиции (Канонический спектральный подъём — Калибровочный мост — Блочная редукция Шура):
Уровень 1. Единый закон спектрального подъёма малорангового ядра (Spectral Chart Lifting)

Вокруг текущего ортонормированного репера Y0​=PT∈St(d,m) любое обновление параметризуется би-факторизованным возмущением ранга q≤min(m,d−m):
Z=VMAT(или в главных осях Z=VΣAT),A∈St(m,q),V∈St(d,q),M∈Rq×q,

которое поднимается в пространство состояний универсальным спектральным отображением карты χY0​(π,ψ0​,ψ1​)​, задаваемым параметром сохранения неактивного подпространства π∈{0,1} и парой спектральных функций (ψ0​,ψ1​):
Y+​(Z)=χY0​(π,ψ0​,ψ1​)​(VΣAT):=πY0​(Im​−AAT)+Y0​Aψ0​(Σ)AT+Vψ1​(Σ)AT.​

Для произвольного полного ядра M∈Rq×q формула записывается без SVD через матричные функции от MTM​:
χY0​(π,ψ0​,ψ1​)​(VMAT)=πY0​(Im​−AAT)+Y0​AΨ0​(M)AT+VΨ1​(M)AT.

Все алгоритмы из исходных записок возникают как выбор тройки (π,ψ0​,ψ1​) в одной и той же формуле:

    Чистый матричный SVD-решатель (matrix mode, rankB≤q<m): π=0, ψ0​(σ)=0, ψ1​(σ)=σ. Зануление π=0 уничтожает (m−q)-мерное подпространство Y0​(Im​−AAT), что мгновенно объясняет структурный порог потерь F(B)≥λ(m−q).

    Поправочный SVD-решатель (correction mode, B=P+AMTVT): π=1, ψ0​(σ)=1, ψ1​(σ)=σ, что даёт аффинный подъём Yaff​(Z)=Y0​+VMAT.

    Полярная ретракция на Gr(m,d): π=1, ψ0​(σ)=(1+σ2)−1/2, ψ1​(σ)=σ(1+σ2)−1/2 при Y0T​V=0 (в матричной форме Ψ0​(M)=(Iq​+MTM)−1/2, Ψ1​(M)=M(Iq​+MTM)−1/2).

    Геодезическая экспонента на Gr(m,d) (плоский тор Картана Tq): π=1, ψ0​(θ)=cosθ, ψ1​(θ)=sinθ при Y0T​V=0.

    Ретракция Кэли и общее изометрическое семейство RY0​φ​: π=1, ψ0​(σ)=cosφ(σ), ψ1​(σ)=sinφ(σ), где φ(σ)=2arctan(σ/2) для Кэли.

    Касательное поле подпространств Такера: поузловое применение χY0​(1,ψ0​,ψ1​)​(Zj​) к ядру M(xj​)=∑ℓ=1s​wℓ​(xj​)Mℓ​ внутри оболочки [Y0​,V] размерности R=m+q>m.

Уровень 2. Точный калибровочный мост (Gauge Bridge): как SVD-решатель вкладывается в риманову оптимизацию на Gr(m,d)

В редакции v3 аффинный SVD-решатель (Раздел 3) и профилированный решатель на Gr(m,d) (Раздел 4) рассматривались параллельно. Между ними существует точное алгебраическое тождество.

Пусть возмущение горизонтально: Z=ΔT=VMAT∈TY0​​Gr(m,d) (Y0T​V=0). В дифференциальной геометрии отображение Z↦span(Y0​+Z) задаёт каноническую аффинную (графовую / проективную) карту на открытой плотной клетке Шуберта UY0​​={[Y]∈Gr(m,d):det(Y0T​Y)=0}. Ортонормированное сечение Штифеля этой карты — полярная ретракция — связано с аффинной поправкой SVD-решателя Yaff​(Z)=Y0​+Z правым калибровочным множителем C(Z)∈Sym++​(m):
Ypol​(Z)=polar(Y0​+Z)=(Y0​+Z)C(Z)(Im​+ZTZ)−1/2​​,C(Z)=Im​−A(Iq​−(Iq​+MTM)−1/2)AT.

Если в профилированном функционале Φ(Ypol​(Z)) на Gr(m,d) выполнить биективную калибровочную замену локальных коэффициентов g~~​j​=C(Z)gj​∈Rm, то Ypol​(Z)gj​≡(Y0​+Z)g~~​j​, а норма коэффициентов преобразуется по формуле ∥gj​∥22​=g~~​jT​(Im​+ZTZ)g~~​j​=∥g~~​j​∥22​+∥Zg~~​j​∥22​. Отсюда следует точное представление риманова функционала в аффинной карте SVD-решателя:
Φ(polar(Y0​+Z))=j=1∑J​Nj​g~~​j​∈Rmmin​{​Ij​−Uj​(Y0​+Z)g~~​j​​22​+τ∥g~~​j​∥22​+τ∥Zg~~​j​∥22​}+λtr(ZTZ(Im​+ZTZ)−1).​

Это тождество доказывает три фундаментальных факта:

    При заморозке g~​j​=gj​ и квадратичной аппроксимации хордового штрафа tr(ZTZ(Im​+ZTZ)−1)=∥Z∥F2​−∥ZTZ∥F2​+O(∥Z∥F6​) риманов функционал на Gr(m,d) в точности переходит в функционал поправочного SVD-решателя F(P+Δ) с индуцированным левым метрическим регуляризатором tr(ΔT(λIm​+τ∑j=1J​Nj​gj​gjT​)Δ).

    Изотропная левая метрика G=∑j​Nj​ηj​gj​gjT​+λIm​ из Теоремы 3.2 возникает не только как приближение гессиана данных, но и как точный вклад тихоновского регуляризатора τ∥gj​∥22​ при переносе с многообразия Грассмана в аффинную карту.

    Любой шаг аффинного поправочного SVD-решателя B+​=P+AMTVT (PV=0) после ортонормировки polar(B+T​) лежит на том же q2-мерном подмногообразии в Gr(m,d), что и шаг геодезической экспоненты ExpY0​​(VΘAT).

2. Четыре математических и алгоритмических улучшения

На основе построенной общей теории в документ v4 внедрены четыре конкретных улучшения, повышающих скорость, качество восстановления и экономию памяти (а также исправлена типографская ошибка в v3 — вкрапление ивритского слова יוצר в п. 4 Предложения 6.2).
Улучшение 1. Матрично-рациональный блочный профиль Шура для произвольного ранга q≥1 (O(Jq3) вместо O(J(p(m+q)2+(m+q)3)))

В редакции v3 замкнутый профиль Шура (Теорема 4.2) был выведен только для одномерного вращения (q=1), а в Замечании 4.3 указывалось, что при q>1 пересчёт на торе Tq требует O(J(p(m+q)2+(m+q)3)) операций на каждый пробный шаг.

В v4 доказана общая теорема о блочном профиле Шура ранга q≥1 на всём q2-мерном полярном ядре M∈Rq×q:

    Дополним активные внутренние направления A∈St(m,q) до ортогональной матрицы [A⊥​,A]∈O(m), A⊥​∈St(m,m−q). Поскольку блок Cj​=Uj​Y0​A⊥​∈Rp×(m−q) неподвижен при любом выборе M∈Rq×q, дополнение Шура Πj​=Ip​−Cj​(CjT​Cj​+τIm−q​)−1CjT​≻0 исключает m−q стационарных координат gj​ один раз до начала поиска по M.

    В полярной карте Ypol​(VMAT) множитель (Iq​+MTM)−1/2 аналитически сокращается внутри локальной обратной матрицы, и профилированный функционал на всём пространстве матриц M∈Rq×q принимает матрично-рациональный вид без квадратных корней и тригонометрических функций:
    Φ(polar(Y0​+VMAT))=j=1∑J​Nj​[kj​−(αj​+MTβj​)TQj​(M)−1(αj​+MTβj​)]+λ(ξ0​−tr(Ξ(M)(Iq​+MTM)−1)),

    где Qj​(M)=(Aj​+τIq​)+Bj​M+MTBjT​+MT(Ej​+τIq​)M∈Sym++​(q), а матрицы Aj​,Bj​,Ej​∈Rq×q и векторы αj​,βj​∈Rq предвычисляются однократно.

    Выигрыш по скорости и памяти: Вычисление Φ(M), точного градиента ∇M​Φ(M)=2∑j=1J​Nj​[(BjT​+(Ej​+τIq​)M)uj​(M)−βj​]uj​(M)T+∇M​Ω(M) (где uj​(M)=Qj​(M)−1(αj​+MTβj​)) и гессиана в любой пробной точке M∈Rq×q стоит O(Jq3) операций (не зависит от d, p и m) и требует лишь O(Jq2) чисел в памяти. При q=1 формула тождественно переходит в (4.7) из v3.

Улучшение 2. Аналитический шаг Гаусса–Ньютона в профиле Шура за O(J)

В Алгоритме 1 редакции v3 начальный шаг Армихо выбирался как tk(0)​=α0​∥[Gk​]q​∥F​ со свободным гиперпараметром α0​, не учитывающим локальную кривизну вдоль геодезической.
Из формулы блочного профиля Шура при M=0 (η=0) выведены замкнутые выражения для первой и второй производных Φ′(0), Φ′′(0) и строго положительной мажорантной кривизны Гаусса–Ньютона:
LGN​(0)=2j=1∑J​Nj​((ej​+τ)uj2​+aj​+τ(ρj​−bj​uj​)2​)+2λ(ξyy​+ξvv​)>0,uj​=aj​+ταj​​,ρj​=βj​−bj​uj​.

Инициализация поиска шагом Ньютона/Гаусса–Ньютона η(0)=−Φ′(0)/max{Φ′′(0),LGN​(0)} (проектируемым в доверительный интервал [c~2​∥G∥F​,Cmax​∥G∥F​]) вычисляется за O(J) операций, обычно принимается с первой попытки Армихо (m=0) и обеспечивает квадратичную/сверхлинейную локальную доводку.
Улучшение 3. Масштабно-инвариантный (метрически-белёный) выбор плоскостей вращения на Gr(m,d)

В v3 выбор активных плоскостей (Vq​,Aq​) в Алгоритме 1 производился по SVD ненормированного риманова градиента (−G)=VΣAT. Если в многоиндексной модели m>1 сингулярные числа матрицы энергии градиентов ∑j​Nj​gj⋆​(gj⋆​)T сильно различаются (λ1​≫λm​), евклидов градиент G доминируется первой компонентой индекса и «замораживает» вращение слабых компонент.
Введение на TY​Gr(m,d) правой римановой метрики GY​=∑j=1J​Nj​ηj​gj⋆​(gj⋆​)T+(λ+τηˉ​)Im​≻0 и переход к SVD белёного градиента (−GGY−1/2​)=VΣAT:

    Сохраняет горизонтальность YTV=0 автоматически;

    Выравнивает чувствительность по всем m направлениям подпространства (повышая качество восстановления слабых индексов);

    Объединяет Теорему 3.2 (белёный Эккарт–Янг) и Теорему 4.4 (спектральный бюджет на Gr(m,d)) в единое утверждение.

Улучшение 4. Точное симметрично-конусное кронекерово сжатие нормального оператора по узлам (rK​≤2m(m+1)​≪J)

В разделе 7.1 v3 сумма кронекеровых произведений L0(s)​(X)=∑ℓ=1s​Gℓ​XHℓ​+λX рассматривалась только как приближение истинного нормального оператора L(X)=∑j=1J​Nj​(gj​gjT​)XHj​+λX.
Однако все матрицы gj​gjT​ лежат в пространстве симметричных матриц Sym(m), размерность которого равна dimSym(m)=2m(m+1)​. Следовательно, оператор L−λId имеет точный кронекеров ранг
rK​=rank{Nj​​svec(gj​gjT​)}j=1J​≤min(J,2m(m+1)​)≪J!

Разложение матрицы коэффициентов по ортонормированному базису G1​,…,GrK​​∈Sym(m) позволяет точно (без погрешности аппроксимации) записать:
L(X)=α=1∑rK​​Gα​XHα​+λX,Hα​=j=1∑J​Nj​⟨Gα​,gj​gjT​⟩F​UjT​Uj​∈Sym(d).

При Jp>4m(m+1)​d предвычисление rK​≤2m(m+1)​ матриц Hα​ полностью исключает число узлов J из всех внутренних итераций ALS, PCG и вычисления D(a,v), сокращая память с O(Jpd) до O(m2d2) и сложность итерации с O(Jpdm) до O(m3d2). 3. Исследование связей и аналогов в мировой научной литературе
Направление в науке Ключевые работы Суть известного метода Отличие и новизна построенной теории
Сепарабельные МНК (VARPRO) на многообразиях Golub & Pereyra (1973); Kaufman (1975); Ruhe & Wedin (1980); Usevich & Markovsky (2014) Исключение линейных параметров c⋆(θ)=G(θ)−1A(θ)Tb и дифференцирование псевдообратных на Rn и Gr(m,d). В классическом и грассмановом VARPRO внутренняя линейная задача глобальна. В задаче ADE/ADP она блочно-сепарабельна по J узлам размера m×m, что позволяет применить блочное дополнение Шура по m−q неподвижным столбцам и свести вычисление функции и градиента на q×q ядре M к O(Jq3) операциям.
Касательный подпространственный спуск (TSD) и вращения Якоби/Гивенса Jacobi (1846); Balzano et al. (2010, GROUSE); Shalit & Chechik (2014); Gutman & Ho-Nguyen (2023) Ограничение шага на многообразии маломерным подпространством в TY​M или геодезическими вращениями ранга 1. В случайном/координатном TSD (Gutman & Ho-Nguyen, 2023) проекция градиента сохраняет лишь долю m(d−m)q​ энергии ∥G∥F2​. Выбор подпространства через усечённое SVD белёного матричного градиента на TY​Gr(m,d) по теореме Эккарта–Янга сохраняет долю ≥min(m,d−m)q​ (выигрыш в d−m раз) и одновременно переводит шаг на вполне геодезический плоский тор Картана Tq.
Графовые (проективные) карты и ретракции на Gr(m,d) Edelman, Arias & Smith (1998); Absil, Mahony & Sepulchre (2008); Absil & Malick (2012); Wen & Yin (2013) Аффинные координаты Z↦span(Y0​+Z) (Y0T​Z=0), полярная, геодезическая и Кэли ретракции. Доказан точный калибровочный мост: в сепарабельной модели правый нормирующий фактор полярной ретракции (Im​+ZTZ)−1/2 поглощается калибровкой векторов g~​j​, благодаря чему аффинный поправочный SVD-решатель оказывается квадратичной моделью риманова функционала в графовой карте.
Жадный малоранговый поиск (ORMP) и кронекеровы предобусловливатели Van Loan & Pitsianis (1993); Srebro & Jaakkola (2003); Wang et al. (2014); Simoncini (2016) Приближение оператора суммой кронекеровых произведений ∑ℓ​Hℓ​⊗Gℓ​ и жадное добавление ранк-1 атомов. Доказана точная мультипликативная гарантия γL​(E0​)≥1+θ1−θ​maxrankE=1​γL​(E) при относительном спектральном искажении θ<1, и установлено, что в силу dimSym(m)=2m(m+1)​ нормальный оператор имеет конечный точный кронекеров ранг rK​≤2m(m+1)​.
Оболочки Кука (Envelopes) и регрессия на многообразиях Cook, Li & Chiaromonte (2010); Kolda & Bader (2009); Ishteva et al. (2011); Cheng & Wu (2013) Сжатие предикторов в маломерную инвариантную оболочку E и тензорные разложения Такера. Показано, что прямое разложение Такера Bj​=AMj​VT при R=m не способно вращать подпространство, а перенос модели Такера в горизонтальное касательное пространство Zj​=V(∑ℓ​wℓ​(xj​)Mℓ​)AT∈TY0​​Gr(m,d) внутри оболочки [Y0​,V] ранга R=m+q>m обеспечивает гладкое поле ортонормированных реперов с кэшированием Uj​[Y0​,V]. 4. Сильные альтернативы, ограничения и открытые вопросы

    Профилированный Gr(m,d) решатель против двухэтапного аффинного SVD + PCG:

        Когда профилированный решатель на Gr(m,d) строго лучше: при умеренных m (m≤10) и плохой обусловленности или неточной инициализации gj​. Синхронное обновление gj⋆​(Y) по блочному профилю Шура исключает эффект «запирания» шага старыми коэффициентами gj​, характерный для альтернирования (AO).

        Когда аффинный поправочный SVD + PCG с кронекеровым сжатием конкурентоспособен: на финальных внешних итерациях при J≫d и уже стабилизировавшихся gj​. В этом режиме точное кронекерово сжатие до rK​≤m(m+1)/2 слагаемых позволяет решить задачу поправки за время, вообще не зависящее от J.

    Ограничения и установленные контрпримеры:

        Невозможность восстановления полного ранга в режиме matrix (r<m): доказан неустранимый порог F(B)≥λ(m−r) при H=Id​. Режим matrix допустим только при r=m либо как инициализатор до включения correction.

        Некорректность двустороннего включения λ в кронекеров предобусловливатель: произведение (G0​+λIm​)X(H0​+λId​) искажает член λX и может ухудшать обусловленность по сравнению с простым левым предобусловливателем (Замечание 7.1). Корректная форма — сильвестров оператор G1​XH1​+λX.

        Недостаточность поузловой невырожденности Uj​ при p<d: при p<d каждый оператор Uj​ имеет ядро размерности ≥d−p. В профилированной задаче без регуляризации τ для локальной идентифицируемости первого порядка на Gr(m,d) необходимо p>m и J(p−m)≥m(d−m) (Предложение 7.3).

    Нерешённые теоретические задачи (отделены от доказанных теорем):

        Вероятностная нижняя граница константы совместной наблюдаемости μobs(k)​>0 при p≪d: для детерминированно заданных узлов константа μid​ проверяется спектрально по оператору Гаусса–Ньютона (7.7), однако неасимптотическая высоковероятностная оценка μobs(k)​ для случайных центров и зависимых анизотропных окон Dk​ остаётся главной открытой задачей статистической теории сжатого ADP.

        Глобальная сходимость κ-гомотопии (κ:0→1→2): переход от спектрального глобального минимума OPG (κ=0) к MAVE (κ=1) и направленному ADP (κ=2) алгоритмически устраняет плохие локальные минимумы, но точные условия отсутствия бифуркаций вдоль пути κ∈[0,2] при конечной выборке требуют отдельного исследования.

5. Полный самодостаточный LaTeX-документ (EDR_master_theory_v4.tex)

Документ полностью готов к сборке lualatex, очищен от технических артефактов и опечаток, включает единый закон спектрального подъёма χY0​(π,ψ0​,ψ1​)​, теорему о калибровочном мосте между аффинным SVD-решателем и Gr(m,d), матрично-рациональный блочный профиль Шура ранга q≥1 за O(Jq3), шаг Гаусса–Ньютона за O(J), метрически-белёный выбор плоскостей и точное кронекерово сжатие по конусу Sym(m).
Code snippet

% !TEX program = lualatex
% Самодостаточная сборка без предупреждений Overfull \hbox (выполнить дважды):
% lualatex -halt-on-error -interaction=nonstopmode EDR_master_theory_v4.tex
\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Noto Serif}
\setsansfont{Noto Sans}
\setmonofont{DejaVu Sans Mono}[Scale=MatchLowercase]
\usepackage[russian]{babel}
\usepackage[margin=19mm]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools,booktabs,tabularx,array}
\usepackage{microtype,enumitem,xcolor}
\usepackage[colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black]{hyperref}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{4em}
\tolerance=2500
\hbadness=10000
\hfuzz=1.5pt
\sloppy
\setlist{nosep,leftmargin=*}
\setcounter{tocdepth}{2}
\numberwithin{equation}{section}

% Макросы
\newcommand{\R}{\mathbb R}
\newcommand{\Id}{\mathrm I}
\newcommand{\T}{^{\mathsf T}}
\newcommand{\F}{_{\mathrm F}}
\newcommand{\ip}[2]{\langle #1,#2\rangle_{\mathrm F}}
\newcommand{\Gr}{\mathrm{Gr}}
\newcommand{\St}{\mathrm{St}}
\newcommand{\Sym}{\mathrm{Sym}}
\newcommand{\GL}{\mathrm{GL}}
\newcommand{\Exp}{\operatorname{Exp}}
\newcommand{\polar}{\operatorname{polar}}
\newcommand{\grad}{\operatorname{grad}}
\newcommand{\qf}{\operatorname{qf}}
\newcommand{\rot}{\operatorname{rot}}
\newcommand{\svec}{\operatorname{svec}}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\spanop}{span}
\DeclareMathOperator{\range}{range}
\DeclareMathOperator{\vecop}{vec}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}

\newtheorem{theorem}{Теорема}[section]
\newtheorem{proposition}[theorem]{Утверждение}
\newtheorem{lemma}[theorem]{Лемма}
\newtheorem{corollary}[theorem]{Следствие}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Определение}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Замечание}

\newcolumntype{L}{>{\raggedright\arraybackslash\hspace{0pt}}X}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash\hspace{0pt}}p{#1}}

\title{\textbf{Спектральные подъёмы в картах, калибровочный мост и блочные профили Шура}\\[4pt]
\large Единая математическая теория SVD-решателя, римановой оптимизации на $\Gr(m,d)$,\\
касательных полей Такера и анизотропной адаптации в задачах EDR/ADE}
\author{Сводная исследовательская записка (редакция 4)}
\date{Октябрь 2026 г.}

\begin{document}
\maketitle
\vspace{-7mm}
\begin{abstract}
Построена единая математическая теория внутренних и внешних итераций многоиндексного оценивания эффективного подпространства снижения размерности (EDR) методом средних производных со скетчированием (ADE/ADP) и анизотропной структурной адаптацией. Показано, что все алгоритмические режимы — классическое альтернирование (AO), матричный и поправочный режимы SVD-решателя, геодезические, полярные и Кэли-вращения на многообразии Грассмана $\Gr(m,d)$, пространственно-переменное поле в касательной оболочке Такера и $\kappa$-семейство спектральных оценок (OPG/MAVE/ADP) — являются специализациями сепарабельного функционала $\Phi(\theta)=\min_c\{\|b-\mathcal A(\theta)c\|_2^2+\|\mathcal R(\theta)c\|_2^2\}+\Omega(\theta)$ и универсального спектрального подъёма би-факторизованного ядра $\chi_{Y_0}^{(\pi,\psi_0,\psi_1)}(V M A\T) = \pi Y_0(\Id_m - AA\T) + Y_0 A\Psi_0(M)A\T + V\Psi_1(M)A\T$.
Установлены четыре новых результата, ускоряющих вычисления и связывающих аффинную и риманову модели: (1)~~\emph{точный калибровочный мост} между аффинным поправочным SVD-решателем и профилированной задачей на $\Gr(m,d)$ в графовой карте $Y_{\rm pol}(Z)=(Y_0+Z)(\Id_m+Z\T Z)^{-1/2}$; (2)~~\emph{матрично-рациональный блочный профиль Шура} произвольного ранга $q\ge 1$ на полном ядре $M\in\R^{q\times q}$ со стоимостью $O(Jq^3)$ и аналитическим шагом Гаусса--Ньютона $O(J)$; (3)~~\emph{масштабно-инвариантный выбор плоскостей} по метрически-белёному риманову градиенту; (4)~~\emph{точное кронекерово сжатие} нормального оператора по конусу $\Sym(m)$ до $r_K\le m(m+1)/2 \ll J$ слагаемых. Доказаны критерии локальной идентифицируемости при неполных локальных измерениях ($p < d$) и порог сжимаемости внешнего цикла.
\end{abstract}

\tableofcontents
\medskip

\textbf{Статус результатов и разграничение утверждений.} В документе строго разделены три уровня результатов:
\begin{enumerate}
\item \textbf{Доказанные алгебраические, геометрические и сложностные теоремы:} регулярность профиля VARPRO (Предложение~\ref{prop:profile}), универсальный спектральный подъём и калибровочный мост между SVD-решателем и $\Gr(m,d)$ (Теорема~\ref{thm:gauge-bridge}), точное кронекерово сжатие ранга $r_K\le m(m+1)/2$ и гарантия доли $(1-\theta)/(1+\theta)$ ранк-1 шага (Теорема~\ref{thm:separable-svd}), матрично-рациональный блочный профиль Шура за $O(Jq^3)$ (Теорема~\ref{thm:schur-profile}), глобальная сходимость с белёным выбором плоскостей и фильтром длины шага по Лоясевичу (Теорема~\ref{thm:convergence}).
\item \textbf{Условные оценки возмущения и идентифицируемости:} критерий локальной идентифицируемости при $p < d$ (Предложение~\ref{prop:identifiability}) и теорема Уэдина для сквозного цикла EDR (Теорема~\ref{thm:perturbation-edr}) опираются на условие совместной наблюдаемости $\mu_{\rm obs}>0$, которое при $p < d$ требует разнообразия подпространств $\range(U_j\T)$ между узлами $j=1,\ldots,J$.
\item \textbf{Открытые математические проблемы:} неасимптотическая высоковероятностная оценка константы $\mu_{\rm obs}>0$ для зависимых анизотропных окон при $p\ll d$ и условия отсутствия бифуркаций вдоль гомотопии $\kappa\in[0,2]$ при конечной выборке.
\end{enumerate}

\section{Наблюдаемая модель, статистики ADP и четыре роли метрик}\label{sec:model}

Рассматривается многоиндексная регрессионная модель наблюдений $(x_i,y_i)\in\R^d\times\R$, $i=1,\ldots,n$ \cite{hjps,mave}:
\begin{equation}\label{eq:mim}
y_i = f(P_* x_i) + \varepsilon_i = \psi(Y__\T x_i) + \varepsilon_i,\qquad
Y__ = P__\T \in \St(d,m)=\{Y\in\R^{d\times m}: Y\T Y = \Id_m\},\quad m\ll d.
\end{equation}
Так как $\nabla f(x) = Y_*\nabla\psi(Y_*\T x)$, при невырожденности матрицы вторых моментов $\mathbb E[\nabla\psi(Y_*\T x)\nabla\psi(Y_*\T x)\T]\succ 0$ линейная оболочка градиентов совпадает с целевым EDR-подпространством $[Y_*]=\spanop(Y_*)\in\Gr(m,d)$.
Ошибка любого ортонормированного репера $Y\in\St(d,m)$ ($P=Y\T$) инвариантно измеряется квадратом хордового расстояния:
\begin{equation}\label{eq:chordal}
\epsilon(Y,Y__) = d_{\rm ch}^2(Y,Y_*) = \tfrac12\|YY\T - Y_*Y__\T\|\F^2 = m - \|Y__\T Y\|\F^2 = \|Y\T(\Id_d - Y__Y__\T)\|\F^2.
\end{equation}

Для центров локализации $\{x_j\}_{j=1}^J$, симметричного положительно определённого (СПО) оператора локализации $D_j\succ 0$, ширины окна $h>0$ и ядра $K(\cdot)$ от квадрата расстояния определим локальные веса и моменты:
\begin{align}
w_{ij} &= K\bigl(h^{-2}(x_i-x_j)\T D_j(x_i-x_j)\bigr), &
N_j &= \sum_{i=1}^n w_{ij} > 0, &
\bar x_j &= N_j^{-1}\sum_{i=1}^n w_{ij}x_i,\label{eq:weights}\\
z_{ij} &= x_i - \bar x_j, &
\Sigma_j &= N_j^{-1}\sum_{i=1}^n w_{ij} z_{ij}z_{ij}\T, &
c_j &= N_j^{-1}\sum_{i=1}^n w_{ij} z_{ij} y_i.\label{eq:moments}
\end{align}
При заданной матрице пробных направлений (скетче) $S_j\in\R^{p\times d}$ направленные ADP-статистики равны $U_j = S_j\Sigma_j\in\R^{p\times d}$ и $I_j = S_j c_j\in\R^p$. Локальная линейная аппроксимация функции связи даёт модель
\begin{equation}\label{eq:local-model}
I_j \simeq U_j Y_j g_j,\qquad g_j\in\R^m,\quad j=1,\ldots,J,
\end{equation}
где в глобальной многоиндексной постановке $Y_1=\cdots=Y_J=Y$ (или $Y=B\T$). Поскольку нормировка на $N_j$ уже входит в определения $U_j$ и $I_j$, в целевом функционале вес $N_j$ применяется ровно один раз.

\begin{remark}[Разделение четырёх метрических объектов и сохранение оценивателя]\label{rem:four-metrics}
В схеме действуют четыре функционально различных метрических оператора:
\begin{enumerate}
\item \textbf{Оператор локализации $D_j\succ 0$:} входит в ядро \eqref{eq:weights}, пересчитывает $(N_j,U_j,I_j)$ и управляет внешней итерацией структурной адаптации \cite{hjps}.
\item \textbf{Метрика пространства отклика $W_j\succeq 0$:} задаёт взвешивание невязки $\|I_j - U_j Y g_j\|_{W_j}^2$ и сводится к евклидовой норме заменой $(U_j,I_j)\leftarrow (W_j^{1/2}U_j, W_j^{1/2}I_j)$.
\item \textbf{Метрика регуляризатора базиса $H\succ 0$ и матрица весов $\Gamma\succ 0$:} задают штраф близости $\tr(\Gamma(B-P)H(B-P)\T)$ в аффинной задаче при фиксированных коэффициентах.
\item \textbf{Риманова метрика / предобусловливатель касательного пространства $\mathcal M_\theta\succ 0$:} определяет выбор направления и активных плоскостей поиска при фиксированной целевой функции.
\end{enumerate}
Изменение $D_j$, $W_j$, $(\Gamma,H)$ либо переход от фиксированных $g_j$ к профилированию меняет статистический оцениватель; изменение предобусловливателя $\mathcal M_\theta$ и точные алгебраические сжатия кэшей оцениватель не меняют.
\end{remark}

\section{Единый математический каркас: VARPRO, спектральный подъём и калибровочный мост}\label{sec:framework}

\subsection{Сепарабельный мастер-функционал и корректность профилирования}

Пусть $\theta\in\mathcal X$ обозначает искомую геометрическую структуру, а $\Psi_j(\theta)\in\R^{p\times k}$ — локальный матричный словарь (в базовой модели $\Psi_j(\theta)=U_j Y$, $k=m$). Сформируем блочно-диагональный оператор дизайна $\mathcal A(\theta)\colon \R^{Jk}\to\R^{Jp}$ с блоками $(\mathcal A(\theta)c)_j = \sqrt{N_j}\Psi_j(\theta)c_j$ и вектор откликов $b_j = \sqrt{N_j}I_j$.

\begin{definition}[Сепарабельная мастер-задача]\label{def:master}
Общая задача внутренней оптимизации определяется функционалом типа VARPRO \cite{varpro}:
\begin{equation}\label{eq:master}
\Phi(\theta) = \min_{c\in\R^{Jk}}\left\{
\|b - \mathcal A(\theta)c\|_2^2 + \|\mathcal R(\theta)c\|*2^2
\right\} + \Omega(\theta),\qquad
\theta*+(M) = \chi_{\theta_0}\bigl(V M A\T\bigr),
\end{equation}
где $\mathcal R(\theta)$ — линейный оператор регуляризации коэффициентов, $\Omega(\theta)$ — структурный штраф, а $\chi_{\theta_0}$ — локальный подъём малорангового ядра $Z = V M A\T$ ранга $q\le\min(m,d-m)$. При фиксированных коэффициентах $c$ операция $\min_c$ опускается, порождая условный квадратичный функционал $F(\theta) = \Phi(\theta;c)$.
\end{definition}

\begin{proposition}[Регулярность профиля VARPRO и фактор-инвариантность]\label{prop:profile}
Пусть на открытом множестве $\mathcal U\subset\mathcal X$ отображения $\theta\mapsto\mathcal A(\theta)$, $\theta\mapsto\mathcal R(\theta)$ и $\theta\mapsto\Omega(\theta)$ принадлежат классу $C^s$ ($s\ge 1$) либо вещественно-аналитичны ($C^\omega$). Тогда:
\begin{enumerate}
\item Если расширенный оператор $\widetilde{\mathcal A}(\theta) = [\mathcal A(\theta); \mathcal R(\theta)]$ имеет на $\mathcal U$ полный столбцовый ранг $Jk$, то оператор Грама $\mathcal G(\theta) = \mathcal A(\theta)\T\mathcal A(\theta) + \mathcal R(\theta)\T\mathcal R(\theta)\succ 0$ обратим и принадлежит тому же классу $C^s$ (или $C^\omega$), единственный минимум равен $c^\star(\theta) = \mathcal G(\theta)^{-1}\mathcal A(\theta)\T b$, а профилированный функционал имеет вид
\begin{equation}\label{eq:profile-block}
\Phi(\theta) = \|b\|_2^2 - b\T\mathcal A(\theta)\mathcal G(\theta)^{-1}\mathcal A(\theta)\T b + \Omega(\theta) \in C^s(\mathcal U)\quad (\text{или } C^\omega(\mathcal U)).
\end{equation}
\item \textbf{Поузловая декомпозиция:} Если $\|\mathcal R(\theta)c\|_2^2 = \tau\sum_{j=1}^J N_j\|c_j\|_2^2$ с $\tau > 0$, а $\theta\mapsto\Psi_j(\theta)$ и $\Omega(\theta)$ вещественно-аналитичны, то $\mathcal G(\theta)\succ 0$ всюду (включая точки падения ранга $\Psi_j(\theta)$), и \eqref{eq:profile-block} распадается на $J$ независимых $k\times k$ СПО-систем, задавая вещественно-аналитическую функцию:
\begin{equation}\label{eq:profile-node}
\Phi(\theta) = \sum_{j=1}^J N_j\bigl(\|I_j\|_2^2 - b_j(\theta)\T A_j(\theta)^{-1} b_j(\theta)\bigr) + \Omega(\theta),\quad
A_j = \Psi_j\T\Psi_j + \tau\Id_k,\quad b_j = \Psi_j\T I_j.
\end{equation}
\item \textbf{Калибровочная эквивариантность:} Если под действием группы $Q\in\mathcal G$ выполнено $\Psi_j(\theta\cdot Q)=\Psi_j(\theta)\rho(Q)$ с обратимым представлением $\rho(Q)$, $\|\mathcal R(\theta\cdot Q)\tilde c\|_2 = \|\mathcal R(\theta)c\|_2$ при $c_j=\rho(Q)\tilde c_j$ и $\Omega(\theta\cdot Q)=\Omega(\theta)$, то $\Phi(\theta\cdot Q)=\Phi(\theta)$, т.е. функционал корректно спускается на фактор-многообразие $\mathcal X/\mathcal G$.
\end{enumerate}
\end{proposition}
\begin{proof}
Невырожденность $\widetilde{\mathcal A}(\theta)$ гарантирует $\mathcal G(\theta)\succ 0$; так как обращение $A\mapsto A^{-1}$ вещественно-аналитично на открытом конусе СПО-матриц, композиция сохраняет класс гладкости $C^s$ ($C^\omega$). Пункты 2 и 3 следуют из блочной диагональности $\mathcal G(\theta)$ и биективности подстановки $c_j=\rho(Q)\tilde c_j$.
\end{proof}

\subsection{Универсальный закон спектрального подъёма и точный калибровочный мост}

Покажем теперь, что все способы обновления матрицы подпространства в исследуемых алгоритмах описываются единым матрично-функциональным законом подъёма, а SVD-решатель и оптимизация на $\Gr(m,d)$ связаны точным калибровочным преобразованием в аффинной графовой карте.

\begin{definition}[Канонический спектральный подъём малорангового ядра]\label{def:spectral-lift}
Пусть $Y_0 = P\T\in\St(d,m)$ — опорный репер, $A\in\St(m,q)$ — матрица активных внутренних координат подпространства, $V\in\St(d,q)$ — матрица активных внешних направлений, а $M\in\R^{q\times q}$ — ядро взаимодействия с сингулярным разложением $M = U_M\Sigma W_M\T$, $\Sigma=\diag(\sigma_1,\ldots,\sigma_q)\ge 0$ ($Z = V M A\T\in\R^{d\times m}$). Для индикатора сохранения неактивного подпространства $\pi\in\{0,1\}$ и пары гладких спектральных функций $(\psi_0,\psi_1)$ (где $\psi_0$ чётна, $\psi_1$ нечётна) определим \emph{спектральный подъём карты} $\chi_{Y_0}^{(\pi,\psi_0,\psi_1)}\colon \R^{d\times m}\to\R^{d\times m}$:
\begin{equation}\label{eq:canonical-lift}
\chi_{Y_0}^{(\pi,\psi_0,\psi_1)}(V M A\T) \coloneqq \pi\, Y_0(\Id_m - AA\T) + Y_0 A\,\Psi_0(M)\,A\T + V\,\Psi_1(M)\,A\T,
\end{equation}
где $\Psi_0(M) = W_M\psi_0(\Sigma)W_M\T = \psi_0(\sqrt{M\T M})\in\Sym(q)$ и $\Psi_1(M) = U_M\psi_1(\Sigma)W_M\T = M\,\frac{\psi_1(\sqrt{M\T M})}{\sqrt{M\T M}}\in\R^{q\times q}$.
\end{definition}

\begin{theorem}[Единая классификация карт и точный калибровочный мост $\text{SVD}\leftrightarrow\Gr(m,d)$]\label{thm:gauge-bridge}
Пусть $Y_0 = P\T\in\St(d,m)$ и $Z = \Delta\T = V M A\T\in\R^{d\times m}$ ($\rank Z\le q\le\min(m,d-m)$). Тогда:
\begin{enumerate}
\item \textbf{Полнота охвата всех режимов формулой \eqref{eq:canonical-lift}:}
\begin{itemize}
\item[(а)] Выбор $\pi=0$, $\psi_0(\sigma)=0$, $\psi_1(\sigma)=\sigma$ даёт \emph{чистый матричный режим SVD-решателя} $Y_+ = V M A\T$ ($B_+ = A M\T V\T$, $\rank B_+\le q$); обращение $\pi$ в нуль уничтожает подпространство $Y_0(\Id_m - AA\T)$ размерности $m-q$ и порождает неустранимый порог $\min_{\rank B\le q}\|B-P\|\F^2 = m-q$.
\item[(б)] Выбор $\pi=1$, $\psi_0(\sigma)=1$, $\psi_1(\sigma)=\sigma$ даёт \emph{аффинный поправочный режим SVD-решателя} $Y_{\rm aff}(Z) = Y_0 + V M A\T$ ($B_+ = P + A M\T V\T$).
\item[(в)] При горизонтальности $Y_0\T V = 0$ любой выбор $\pi=1$ с пифагоровым тождеством $\psi_0^2(\sigma) + \psi_1^2(\sigma) = 1$, $\psi_0(0)=1$, $\psi_1'(0)=1$ задаёт ортонормированную ретракцию второго порядка $\chi_{Y_0}^{(1,\psi_0,\psi_1)}(Z)\in\St(d,m)$ на $\Gr(m,d)$, включая:
\begin{align*}
\text{Полярную (графовую) карту } R^{\rm pol}_{Y_0}(Z) &= \polar(Y_0+Z)\colon &
\psi_0(\sigma) &= (1+\sigma^2)^{-1/2}, & \psi_1(\sigma) &= \sigma(1+\sigma^2)^{-1/2};\\
\text{Геодезическую экспоненту } \Exp_{Y_0}(Z)&\colon &
\psi_0(\sigma) &= \cos\sigma, & \psi_1(\sigma) &= \sin\sigma;\\
\text{Ретракцию Кэли } R^{\rm Cay}_{Y_0}(Z)&\colon &
\psi_0(\sigma) &= \frac{1-\sigma^2/4}{1+\sigma^2/4}, & \psi_1(\sigma) &= \frac{\sigma}{1+\sigma^2/4}.
\end{align\*}
\end{itemize}
\item \textbf{Проективная эквивалентность аффинной SVD-поправки и римановых ретракций:} При $Y_0\T V = 0$ аффинная поправка $Y_{\rm aff}(Z) = Y_0 + V M A\T$ имеет полный столбцовый ранг $m$ для любой матрицы $M\in\R^{q\times q}$ с метрическим тензором графовой карты $G_{\rm gr}(Z) = Y_{\rm aff}(Z)\T Y_{\rm aff}(Z) = \Id_m + Z\T Z = \Id_m + A M\T M A\T \succ 0$. Умножение $Y_{\rm aff}(Z)$ справа на обратимую калибровочную матрицу $C(Z) = G_{\rm gr}(Z)^{-1/2} = (\Id_m - AA\T) + A(\Id_q + M\T M)^{-1/2}A\T \in \Sym_{++}(m)$ переводит аффинную поправку SVD-решателя в полярную и геодезическую ретракции:
\begin{equation}\label{eq:gauge-factor}
Y_{\rm pol}(Z) = (Y_0 + V M A\T)\,C(Z) = \Exp_{Y_0}\bigl(V U_M(\arctan\Sigma)W_M\T A\T\bigr)\in\St(d,m),
\end{equation}
вследствие чего $\spanop(Y_0 + V M A\T) \equiv \spanop(Y_{\rm pol}(Z)) \in \Gr(m,d)$ при всех $M\in\R^{q\times q}$.
\item \textbf{Точный перенос профилированного функционала в пространство SVD-поправки:} Для любого горизонтального возмущения $Z = \Delta\T = V M A\T\in T_{Y_0}\Gr(m,d)$ подстановка калибровки $\tilde g_j = C(Z)g_j\in\R^m$ обращает риманов профилированный функционал $\Phi(Y_{\rm pol}(Z))$ на $\Gr(m,d)$ (с опорной точкой штрафа $Y_0$) в точную аффинную задачу относительно линейной поправки $Y_0 + Z = (P+\Delta)\T$:
\begin{align}\label{eq:exact-pullback}
\Phi\bigl(\polar(Y_0 + Z)\bigr) &= \sum_{j=1}^J N_j\min_{\tilde g_j\in\R^m}\Bigl\{
\bigl\|I_j - U_j(Y_0 + Z)\tilde g_j\bigr\|_2^2 + \tau\|\tilde g_j\|*2^2 + \tau\|Z\tilde g_j\|*2^2
\Bigr\}\nonumber\\
&\quad + \lambda\tr\bigl(M\T M(\Id_q + M\T M)^{-1}\bigr).
\end{align}
В частности, при фиксации аффинных координат $\tilde g_j = g_j$ и квадратичной аппроксимации хордового штрафа $\tr(M\T M(\Id_q + M\T M)^{-1}) = \|\Delta\|\F^2 - \|\Delta\Delta\T\|\F^2 + O(\|\Delta\|\F^6)$ функционал \eqref{eq:exact-pullback} переходит в квадратичный функционал поправочного SVD-решателя $F(P+\Delta)$ с индуцированной левой метрикой $\Gamma_{\rm ind} = \lambda\Id_m + \tau\sum_{j=1}^J N_j g_j g_j\T \succ 0$:
\begin{equation}\label{eq:induced-svd}
F*{\rm ind}(P+\Delta) = \sum*{j=1}^J N_j\bigl\|I_j - U_j(P+\Delta)\T g_j\bigr\|*2^2 + \tr\bigl(\Delta\T \Gamma*{\rm ind}\Delta\bigr) + \tau\sum_{j=1}^J N_j\|g_j\|_2^2.
\end{equation}
\end{enumerate}
\end{theorem}
\begin{proof}
Пункт 1 проверяется прямой подстановкой $(\pi,\psi_0,\psi_1)$ в \eqref{eq:canonical-lift}; при $Y_0\T V=0$ имеем $Y_+(Z)\T Y_+(Z) = (\Id_m - AA\T) + A(\psi_0^2(\Sigma) + \psi_1^2(\Sigma))A\T = \Id_m$.
В п. 2 равенство $Y_{\rm aff}(Z)\T Y_{\rm aff}(Z) = \Id_m + Z\T Z \succ 0$ следует из $Y_0\T Z = 0$, откуда по единственности полярного разложения $\polar(Y_0+Z) = (Y_0+Z)(\Id_m+Z\T Z)^{-1/2} = (Y_0+Z)C(Z)$, а подстановка $\cos(\arctan\Sigma)=(\Id_q+\Sigma^2)^{-1/2}$ и $\sin(\arctan\Sigma)=\Sigma(\Id_q+\Sigma^2)^{-1/2}$ даёт совпадение с $\Exp_{Y_0}(V U_M(\arctan\Sigma)W_M\T A\T)$.
В п. 3 биективная замена $\tilde g_j = C(Z)g_j$ даёт $Y_{\rm pol}(Z)g_j = (Y_0+Z)C(Z)g_j = (Y_0+Z)\tilde g_j$ и $\|g_j\|_2^2 = \tilde g_j\T C(Z)^{-2}\tilde g_j = \tilde g_j\T(\Id_m + Z\T Z)\tilde g_j = \|\tilde g_j\|_2^2 + \|Z\tilde g_j\|_2^2$. Наконец, $d_{\rm ch}^2(Y_{\rm pol}(Z),Y_0) = m - \|Y_0\T(Y_0+Z)C(Z)\|\F^2 = m - \tr(C(Z)^2) = \tr(Z\T Z(\Id_m+Z\T Z)^{-1}) = \tr(M\T M(\Id_q+M\T M)^{-1})$, что доказывает \eqref{eq:exact-pullback} и \eqref{eq:induced-svd}.
\end{proof}

\begin{table}[h!]
\centering\footnotesize
\caption{Таксономия всех режимов как частных случаев мастер-функционала \eqref{eq:master} и подъёма \eqref{eq:canonical-lift}}\label{tab:taxonomy}
\begin{tabularx}{\linewidth}{@{}P{27mm}P{28mm}P{33mm}L@{}}
\toprule
Модель / Режим & Коэффициенты $c$ и штраф $\mathcal R, \Omega$ & Параметры подъёма $(\pi,\psi_0,\psi_1)$ и ядро $M$ & Математический статус в единой теории\\
\midrule
Классический AO (\texttt{multiindex}) & Шаг по $g_j$ ($\tau=0$), шаг по $B\in\R^{m\times d}$ & $\pi=1,\psi_0=1,\psi_1(\sigma)=\sigma$, полное $q=m$ + SVD & Покоординатный спуск по $(B,g)$; штраф $\lambda\|B-P\|\F^2$ фиксирует калибровку в слое $\GL(m)$.\\
Чистый SVD (\texttt{matrix}) & $g_j$ фиксированы; $\Omega=\lambda\|(B-P)H^{1/2}\|\F^2$ & $\pi=0,\psi_0=0,\psi_1(\sigma)=\sigma$, $\rank B\le q < m$ & Вырожденная карта ($\pi=0$): стирает $Y_0(\Id_m-AA\T)$ и создаёт ранговый пол $\lambda(m-q)$.\\
Поправочный SVD (\texttt{correction}) & $g_j$ фиксированы; $\Omega=\tr(\Delta\T\Gamma\Delta H)$ & $\pi=1,\psi_0=1,\psi_1(\sigma)=\sigma$, $B=P+AM\T V\T$ & Квадратичная модель риманова функционала \eqref{eq:exact-pullback} в графовой карте при замороженных $\tilde g_j$.\\
Сепарабельный SVD ($\mathcal L_0 X = GXH$) & $g_j$ фиксированы; суррогат $H_j\approx\eta_j H$ & Глобальное SVD матрицы $G^{-1/2}Q H^{-1/2}$ & Точный минимум изотропной модели (Эккарт--Янг) и масштабно-инвариантный выбор плоскостей.\\
Профильный $\Gr(m,d)$ (ранг $q$) & Синхронный профиль $g_j^\star(Y)$ ($\tau>0$) & $\pi=1$, $\psi_0^2+\psi_1^2=1$, $Y_0\T V=0$, ядро $M\in\R^{q\times q}$ & Точная оптимизация в фактор-пространстве $\Gr(m,d)$ со сжатием до $q\times q$ профиля Шура за $O(Jq^3)$.\\
Поле Такера в оболочке $[Y_0,V]$ & Профиль по $g_j$; графовый штраф Лапласа & $\pi=1$, $\psi_0^2+\psi_1^2=1$, $M(x_j)=\sum_{\ell=1}^s w_\ell(x_j)M_\ell$ & Гладкое сечение расслоения над $\Gr(m,d)$ внутри оболочки размерности $R=m+q > m$.\\
Семейство $\Phi_{\kappa,\delta}$ (OPG--MAVE--ADP) & Профиль по $g_j$; вес $\Sigma_{j,\delta}^\kappa$, $\kappa\in[0,2]$ & При $\kappa=0$: глобальные собственные векторы $C_{\rm grad,\delta}$ & Гомотопия метрики невязки от спектрального метода ($\kappa=0$) к MAVE ($\kappa=1$) и скетчу ADP ($\kappa=2$).\\
\bottomrule
\end{tabularx}
\end{table}

\section{Аффинная карта: теория SVD-решателя, кронекерово сжатие и гарантии}\label{sec:svd}

Рассмотрим задачу при фиксированных коэффициентах $g_j\in\R^m$, базовом репере $P=Y_0\T\in\R^{m\times d}$ ($PP\T=\Id_m$), правой метрике $H\succ 0$ и левой матрице весов $\Gamma\in\Sym_{++}(m)$ (в стандартном режиме $\Gamma=\lambda\Id_m$, в индуцированном режиме \eqref{eq:induced-svd} $\Gamma = \lambda\Id_m + \tau\sum_j N_j g_j g_j\T$):
\begin{equation}\label{eq:fixed-g}
F(B) = \sum_{j=1}^J N_j\|I_j - U_j B\T g_j\|_2^2 + \tr\bigl((B-P)\T\Gamma(B-P)H\bigr).
\end{equation}
Обратимая замена $\widetilde B = B H^{1/2}$, $\widetilde P = P H^{1/2}$, $\widetilde U_j = U_j H^{-1/2}$ сводит \eqref{eq:fixed-g} к случаю $H=\Id_d$ с сохранением рангов $\rank(B)$ и $\rank(B-P)$; далее полагаем $H=\Id_d$.

\subsection{Нормальный оператор, точное кронекерово сжатие по конусу $\Sym(m)$ и сертификаты}

Самосопряжённый нормальный оператор $\mathcal L$ и правая часть $C$ задачи \eqref{eq:fixed-g} равны:
\begin{equation}\label{eq:Lop}
\mathcal L(X) = \sum_{j=1}^J N_j (g_j g_j\T) X H_j + \Gamma X,\qquad
C = \sum_{j=1}^J N_j g_j I_j\T U_j + \Gamma P,\qquad H_j = U_j\T U_j,
\end{equation}
так что $F(B) = C_0 + \ip{B}{\mathcal L(B)} - 2\ip{C}{B}$, а антиградиент равен $2Q(B) = 2(C - \mathcal L(B))$.

\begin{proposition}[Точное кронекерово сжатие по $\Sym(m)$ и апостериорные сертификаты]\label{prop:certificates}
\begin{enumerate}
\item \textbf{Конечный кронекеров ранг $r_K\le m(m+1)/2$:} Поскольку каждая матрица $g_j g_j\T$ принадлежит пространству $\Sym(m)$ размерности $s_{\max} = m(m+1)/2$, оператор данных $\mathcal L_{\rm data}(X) = \sum_{j=1}^J N_j(g_j g_j\T)X H_j$ имеет точный кронекеров ранг $r_K \le \min(J, m(m+1)/2)$. Именно, пусть $\Gamma_{\rm sym}\in\R^{J\times s_{\max}}$ — матрица со строками $\sqrt{N_j}\svec(g_j g_j\T)\T$, и пусть $\{G_\alpha\}_{\alpha=1}^{r_K}\subset\Sym(m)$ ($\ip{G_\alpha}{G_\beta}=\delta_{\alpha\beta}$) — ортонормированный базис её правого сингулярного подпространства. Тогда \textbf{точно} (без ошибки аппроксимации):
\begin{equation}\label{eq:exact-kron}
\mathcal L(X) = \sum_{\alpha=1}^{r_K} G_\alpha X \widehat H_\alpha + \Gamma X,\qquad
\widehat H_\alpha = \sum_{j=1}^J N_j\ip{G_\alpha}{g_j g_j\T}\,U_j\T U_j \in \Sym(d).
\end{equation}
При $J p > \frac{m(m+1)}{4}d$ однократная сборка матриц $\{\widehat H_\alpha\}_{\alpha=1}^{r_K}$ сокращает память с $Jpd$ до $\frac{m(m+1)}{2}\frac{d(d+1)}{2}$ чисел и уменьшает стоимость каждого вычисления $\mathcal L(X)$ с $O(Jpdm)$ до $O(m^3 d^2)$ (независимо от $J$ и $p$).
\item \textbf{Сертификаты и условие неточного спуска:} Пусть $\mathcal L\succeq\mu\Id$ с $\mu = \lambda_{\min}(\Gamma) + \mu_{\rm data} > 0$. Тогда для полного минимума $B_\star = \mathcal L^{-1}(C)$ и любой поправки $D\in\R^{m\times d}$ с невязкой $s = \mathcal L(D) - Q(B)$ справедливы соотношения:
\begin{equation}\label{eq:cert-bounds}
\|B - B_\star\|\F \le \frac{\|Q(B)\|\F}{\mu},\quad
F(B) - F(B_\star) \le \frac{\|Q(B)\|\F^2}{\mu},\quad
F(B+D) - F(B) = -\ip{D}{\mathcal L(D)} + 2\ip{D}{s}.
\end{equation}
Неравенство $2|\ip{D}{s}| < \ip{D}{\mathcal L(D)}$ гарантирует строгое убывание $F(B+D) < F(B)$.
\end{enumerate}
\end{proposition}
\begin{proof}
Пункт 1 следует из разложения каждого тензора $g_j g_j\T = \sum_{\alpha=1}^{r_K}\ip{G_\alpha}{g_j g_j\T}G_\alpha$ по ортонормированному базису линейной оболочки $\{g_j g_j\T\}_{j=1}^J\subset\Sym(m)$ и линейности следа. Пункт 2 следует из спектральной оценки $\mathcal L^{-1}\preceq\mu^{-1}\Id$ и квадратичного разложения $F(B+D)-F(B) = \ip{D}{\mathcal L(D)} - 2\ip{Q(B)}{D}$.
\end{proof}

\subsection{Жадный ранк-1 шаг, альтернирование и полное малое ядро}

Для атома $E = a v\T$ ($\Vert{}a\Vert{}_2=\Vert{}v\Vert{}_2=1$) оптимальный шаг $\sigma_\star = R_B(a,v)/D(a,v)$ даёт выигрыш $\gamma_{\mathcal L}(av\T) = R_B(a,v)^2/D(a,v)$, где:
\begin{equation}\label{eq:R-and-D}
R_B(a,v) = a\T Q(B)v,\qquad
D(a,v) = \sum_{j=1}^J N_j(a\T g_j)^2\|U_j v\|_2^2 + a\T\Gamma a = \

Gemini is AI and can make mistakes, including about people. Your privacy & Gemini Opens in a new window
