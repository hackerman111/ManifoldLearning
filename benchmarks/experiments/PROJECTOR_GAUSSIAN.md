# Gaussian projector ADP: кандидат на проверке

Основная экспериментальная версия выбрана Luna по отдельным пилотам:
Gaussian kernel `exp(-2u)`, единичные веса всех m направлений в тензоре
локализации, прежний compact-kernel initializer, прежний full-moment ridge
и двусторонний Grassmann solver. Штатный пакет ADP не изменён.

```python
from experiments.projector_gaussian import fit

result = fit(X, y, m=3)
basis = result.index.T  # d x m
```

Этот API пока исследовательский. Результат большого теста неизвестен.
Разведочный выигрыш не является гарантией EDR; слабый сигнал и неверный m
остаются ограничениями. Популяционный вывод для Gaussian X не переносится
автоматически на другую геометрию данных. При прежней kernel mass Gaussian
имеет гораздо большее эффективное число наблюдений и дороже вычисляется.

Код переиспользует прежние моменты, solver, canonical_index и операции
адаптации; ограниченный CPU outer loop изолирован в experiments. Паритет
compact control с live fit_index проверен, в том числе по стадиям и
коэффициентам. Никаких process-global monkeypatch в этом API нет.

Кампания: `python -m experiments.projector_gaussian_study campaign`.
Команда допускает единственный запуск и не перезаписывает существующие
результаты. Отсутствует автоматический повтор неудавшихся fits.

- [Протокол](../results/verification/weak-localization-20261028/PROTOCOL.md):
  600 selection, условно 900 validation и условно 300 stress fits.
- [Текущее состояние кампании](../results/verification/weak-localization-20261028/STATE.md).
- Сохранённый `campaign.log` отсутствует в этом архиве.
- [Выбор Luna](../results/verification/weak-localization-20261028/LUNA_PILOT_ANALYSIS.md).
- [Математический аудит](../results/verification/weak-localization-20261028/R1_AUDIT.md).

После каждого этапа независимый анализатор Luna пересчитывает метрики из
сохранённых bases и проверяет парность, исходники, полный набор задач и
все gates. При FAIL следующая выборка не генерируется. Полные результаты
и их интерпретация проверяются после завершения кампании.
