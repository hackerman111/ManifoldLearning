# TeX: карта документов

Сначала выберите назначение документа. Формулы исторических рукописей,
описание конкретного снимка реализации и экспериментальные предложения
имеют разные области применимости.

| Назначение | Где читать |
|---|---|
| Исходные single/multi-index и manifold спецификации | [multiindex.tex](multiindex.tex), [manifold.tex](manifold.tex), [manifold-ade.tex](manifold-ade.tex); [подробная карта формул](../agent-notes/tex/README.md) |
| Описание реализации Grassmann на 2026-10-23 из EDR_code | [ADP_Grassmann_current.tex](ADP_Grassmann_current.tex), [PDF](ADP_Grassmann_current.pdf), [проверки и hashes](ADP_Grassmann_current.verification.md) |
| Формализация opt-in варианта `LOO1-WLS-ADP` (`feat/loo1-wls-adp`) | [ADP_LOO1_WLS.tex](ADP_LOO1_WLS.tex), [PDF](ADP_LOO1_WLS.pdf) (сборка LuaLaTeX) |
| Описание реализации из отдельного комплекта M | [archive/ADP_Grassmann_current.tex](archive/ADP_Grassmann_current.tex), [result/multi_index_algorithm.tex](result/multi_index_algorithm.tex); это отдельные снимки, не описание всех текущих API EDR_code |
| Выводы и изложение экспериментов из M | [result/README.md](result/README.md) |
| Исходные идеи, ответы и черновики из M | [data/README.md](data/README.md) |
| Теория SVD/Grassmann и история проверок | [SVD/README.md](../SVD/README.md), [SVD_solver.tex](../SVD/SVD_solver.tex), [EDR_unified_theory.tex](../SVD/EDR_unified_theory.tex), [AI_research_synthesis.tex](../SVD/AI_research_synthesis.tex) |
| Исторические копии из M | [archive/README.md](archive/README.md) |

При объединении приоритет имеет `EDR_code`. Его исходные документы и
сопутствующие записи проверки сохранены. Новые архивные копии с одинаковым
SHA-256 представлены относительными символическими ссылками на существующие
документы; разные версии имеют разные файлы. Полный маршрут каждого исходного
файла зафиксирован в [манифесте переноса](../integration/vlad_edr_m_20261026/IMPORT_MANIFEST.json).

Существующие TeX в `docs/tex/` и `docs/SVD/` сохраняют файлы и историю для
совместимости с заметками, сборкой и историческими отчётами. Этот каталог
объединяет навигацию по ним. Копии в снимках `verification/**/source/` являются
частью доказательной базы экспериментов, их не нужно редактировать вместе с
основными документами.

## Сборка результатов из M

Четыре самостоятельных документа `result/` используют pdfLaTeX; файл
`edr_metric_results.tex` является включаемой частью, его отдельно не собирают.
Команды выполняются из `EDR_code/tex/result`:

```bash
mkdir -p /tmp/edr-tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/edr-tex full_moments_ridge.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/edr-tex full_moments_ridge.tex
```

Аналогично собираются `anisotropic_full_moments.tex`,
`multi_index_algorithm.tex` и `edr_metric_corrections.tex`. Последнему могут
потребоваться дополнительные проходы для стабилизации ссылок. PDF из исходного
комплекта сохранены рядом; свежие проверочные сборки и логи лежат в
[протоколе объединения](../integration/vlad_edr_m_20261026/README.md).
Синтез `AI_research_synthesis.tex` собирается XeLaTeX из корня `EDR_code`
по инструкциям в [его записи проверки](../SVD/synthesis_2026_10_24/VERIFICATION.md).
