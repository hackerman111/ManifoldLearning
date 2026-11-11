# Объединение Vlad_EDR_M → EDR_code, 2026-10-26

Источник — рабочая копия `../Vlad_EDR_M/Code` внутри `Vlad_EDR`.
`Vlad_EDR_M/Overleaf` пуст. По указанию пользователя приоритет имеет EDR_code.
Исходная папка M сохранена. Коммит, push и перенос истории Git не выполнялись.

## Что перенесено

Учтены **23 326 файлов, 650 112 313 байт** исходного комплекта, без `.git`,
локальных конфигураций агентов, окружений, кэшей и Python bytecode.
Перенесены 43 новых файла экспериментов, 6 файлов benchmarks, 20 новых тестовых
модулей, полные `verification/` и `benchmarks/results/outputs/`, TeX, PDF и заметки.
Сохранены исходные снимки, входы, hashes, протоколы, аудиты и неудачные серии.

Все 38 конфликтовавших исходных файлов M сохранены в
[CONFLICTS.md](CONFLICTS.md). В рабочих файлах приоритет остаётся за EDR_code.
Старые численные реализации, LSMR по умолчанию, CG/HYBRID и GPU-возможности
не заменены версией M. Правки существующего исполняемого кода EDR_code:

- Явная опция `--solver grassman` в `ADP.cli` для single/multi: отдельный
  бюджет 20, если пользователь не задаёт его сам. Прежние опции и бюджеты
  сохранены. Grassmann для manifold не допускается.
- Дополнительные R-профили `d25-n5000` и `high-dim-dm`.
- Базовый метод парного отчёта `ADP-Grassmann-combined`, когда отсутствуют
  прежние ADP-Grassmann/ADP. Порядок приоритета старых базовых методов сохранён.

Добавлен `ADP/core/solver_common.py`: импорт того же существующего
`LSMR.HPAOResult`, чтобы результаты новых экспериментов распознавались движком.
Оригинальная версия этого файла M сохранена в `source-adapted/`.
Версии остальных различавшихся модулей M остаются в `source-conflicts/`.
В перенесённых `parameter_search.py` и `gaussian_fourier_study.py` явно
закреплён прежний экспериментальный Grassmann20: default модели EDR_code
остаётся LSMR5. Оригиналы этих двух модулей сохранены в `source-adapted/`.
CLI single использует адаптер rank-one матрицы, не меняющий численный solver.

Документы первоначального автономного экспорта находятся в [source-root/](source-root/).
Их `SOURCE_MANIFEST.json` и `VERIFY.md` описывают старый экспорт; они не являются
манифестом текущего объединённого кода. Первоначальный README M содержит
исторические настройки/ограничения, которые не заменяют текущий [README](../../../README.md).

Существующие правила `.gitignore` сохранены: `benchmarks/results/outputs/` по-прежнему
локальный игнорируемый архив. Перенос на диск не означает его добавления в Git.

## Где читать

- [Исходный каталог экспериментов EDR_code](../../../benchmarks/experiments/README.md)
  сохранён целиком с добавленной ссылкой на [каталог M](../../../benchmarks/experiments/IMPORTED_FROM_M.md).
- [36 сохранённых серий](../../../benchmarks/results/verification/README.md).
- [Общий каталог TeX](../../../docs/tex/README.md), [результаты](../../../docs/tex/result/README.md),
  [идеи/черновики](../../../docs/tex/data/README.md), [архив](../../../docs/tex/archive/README.md).
- [SVD и теория](../../../docs/SVD/README.md).

14 точных копий исходников TeX представлены относительными ссылками на
существующие канонические документы; версии с разным содержимым сохранены
отдельно. Старые пути EDR_code остались действующими. Включаемый
`edr_metric_results.tex` остаётся рядом с основным документом и графиком.

## Проверки

- **235 passed**: все новые тестовые модули и тесты совместимости;
  [лог](imported-tests.log).
- [6 проверок CLI](cli-results.json): каталоги, dry-run двух новых R-профилей,
  multi и single Grassmann; коды выхода 0.
- [Ссылки в новых каталогах](document-links.json) проверены без ошибок.
- **41/41** новых модулей experiments/benchmarks импортированы отдельно;
  [результаты](module-imports.json).
- Четыре самостоятельных result-документа успешно собраны pdfLaTeX минимум
  дважды: `anisotropic_full_moments`, `full_moments_ridge`,
  `multi_index_algorithm`, `edr_metric_corrections`. Последний потребовал
  четырёх проходов. В итоговых логах нет предупреждений, undefined references,
  missing characters или bad boxes: [результат](latex/build-results.json),
  PDF и логи — в соответствующих подкаталогах `latex/`.
- Ruff проверяет изменённые исполняемые файлы, bridge, интеграционные тесты
  и скрипт проверки. Исторические исходники и сохранённые исследования не
  переформатировались. В двух адаптированных импортированных study-модулях
  остаются прежние lint-замечания (40 в parameter_search, 1 в Fourier);
  [сравнение с оригиналами](adapted-lint-comparison.json) показывает отсутствие
  новых замечаний. Это не утверждение о чистом Ruff всего старого репозитория.
- [IMPORT_MANIFEST.json](IMPORT_MANIFEST.json): SHA-256 и маршрут каждого
  исходного файла. [TARGET_BEFORE.json](TARGET_BEFORE.json): fingerprints
  файлов EDR_code до копирования. [integrity-check.json](integrity-check.json):
  проверка сохранности исходного M, импортированных копий и прежних файлов
  EDR_code, с явным перечнем допустимых текущих правок.

Проверка целостности из корня EDR_code:

```bash
python docs/integration/vlad_edr_m_20261026/verify_merge.py
```

Полный `pytest` останавливается на трёх прежних ошибках сбора `multiv2`:
[лог](full-tests.log). Запуск с `--continue-on-collection-errors` дал
**1028 passed, 53 skipped, 8 failed, 3 errors**:
[лог](full-tests-continued.log). Пропуски связаны с отсутствием CUDA-устройства.
Все падения отдельно воспроизведены в изолированной копии файлов до объединения,
сверенных по SHA-256: [сбор тестов](baseline-collection.log),
[восемь падений](baseline-failures.log).

Прежние ограничения:

- `test_experiment.py`, `test_multi_center_diagnostic.py`,
  `test_multiv2_quality.py`: collection `KeyError: mi-spokoini-m2-n`.
- `test_cli.py::test_cli_manifold_smoke[cg/hybrid]`,
  `test_hybrid.py::test_manifold_hybrid_fit_matches_cg`,
  `test_statistics_optimization.py::test_fit_reference[manifold-1/2/3]`:
  несовместимость m=2 с default `local_quadratic`, требующим m=1.
- `test_inverse_moment_init.py::test_recovery_gate_retains_failures_and_checks_each_pair`:
  та же проблема импорта `multiv2`.
- `test_export_bundle.py::test_export_is_complete_and_runs_without_parent_repository`:
  прежний каталог `export/` отсутствует.

Эти существующие проблемы не исправлялись в рамках переноса. Новые научные
серии не запускались; исторические recovery/EDR/time выводы не переоценивались.
Успешная сборка документов проверяет их переносимость, а не научные утверждения.

Независимое review выявило проблему single CLI с вектором; отдельный
регрессионный тест воспроизвёл её до исправления и прошёл после адаптера.
Два теста реальных bounded worker fits аналогично проверили сохранение
Grassmann20 в M-исследованиях при сохранении default LSMR в EDR_code.
Полные source-версии адаптированных модулей остаются в манифесте.
