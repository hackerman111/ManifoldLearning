import csv
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


class ADP_Characteristics:
    def __init__(self) -> None:
        # Вход: нет.
        # Выход: объект для накопления характеристик ADP.
        # Что делает: создает пустые списки для h_0, h_k, rho_k и записей шагов.
        # Реализация: все данные хранятся в обычных python-структурах, чтобы их легко писать в json/csv.

        self.h_0_values = []
        self.h_k_values = []
        self.rho_k_values = []
        self.step_records = []
        self.run_folder = None

    def Cosine_Calculate(self, beta, beta_hat):
        # Вход: beta - истинное направление, beta_hat - оцененное направление.
        # Выход: float с косинусом между направлениями или None, если косинус посчитать нельзя.
        # Что делает: считает диагностическую характеристику качества восстановления beta.
        # Реализация: векторы приводятся к одномерному виду, затем считается нормированный скалярный продукт.

        if beta is None or beta_hat is None:
            return None

        beta_array = np.asarray(beta, dtype=float).reshape(-1)
        beta_hat_array = np.asarray(beta_hat, dtype=float).reshape(-1)

        if beta_array.shape != beta_hat_array.shape:
            return None

        beta_norm = float(np.linalg.norm(beta_array))
        beta_hat_norm = float(np.linalg.norm(beta_hat_array))

        if beta_norm == 0.0 or beta_hat_norm == 0.0:
            return None

        cosine = float(beta_array @ beta_hat_array / (beta_norm * beta_hat_norm))

        return self._Float_Or_None(cosine)

    def Rho_k_Save(self, rho_k):
        # Вход: rho_k - параметр анизотропии на шаге k.
        # Выход: сохраненное значение rho_k или None.
        # Что делает: добавляет rho_k в отдельную историю скалярных значений.
        # Реализация: значение приводится к float и сохраняется только если оно конечно.

        rho_k = self._Float_Or_None(rho_k)

        if rho_k is not None:
            self.rho_k_values.append(rho_k)

        return rho_k

    def H0_Save(self, h_0):
        # Вход: h_0 - начальная ширина окна.
        # Выход: сохраненное значение h_0 или None.
        # Что делает: добавляет h_0 в отдельную историю скалярных значений.
        # Реализация: значение приводится к float и сохраняется только если оно конечно.

        h_0 = self._Float_Or_None(h_0)

        if h_0 is not None:
            self.h_0_values.append(h_0)

        return h_0

    def H_k_Save(self, h_k):
        # Вход: h_k - ширина окна на шаге k.
        # Выход: сохраненное значение h_k или None.
        # Что делает: добавляет h_k в отдельную историю скалярных значений.
        # Реализация: значение приводится к float и сохраняется только если оно конечно.

        h_k = self._Float_Or_None(h_k)

        if h_k is not None:
            self.h_k_values.append(h_k)

        return h_k

    def Step_Characteristics_Save(self, beta, beta_hat, rho_k, h_0, h_k):
        # Вход: истинная beta, оценка beta_hat, rho_k, h_0, h_k.
        # Выход: словарь с характеристиками шага.
        # Что делает: сохраняет одну строку диагностики для текущего шага алгоритма.
        # Реализация: номер шага берется из длины уже сохраненной истории.

        step_record = self._Step_Record_Create(
            step_name="step_k",
            beta=beta,
            beta_hat=beta_hat,
            rho_k=rho_k,
            h_0=h_0,
            h_k=h_k,
        )

        self.step_records.append(step_record)

        return dict(step_record)

    def Step_0_Characteristics_Save(self, beta, beta_hat, h_0):
        # Вход: истинная beta, оценка beta_hat после step 0, h_0.
        # Выход: словарь с характеристиками step 0.
        # Что делает: сохраняет диагностику начального изотропного шага.
        # Реализация: rho_k и h_k для step 0 отсутствуют и пишутся как None.

        step_record = self._Step_Record_Create(
            step_name="step_0",
            beta=beta,
            beta_hat=beta_hat,
            rho_k=None,
            h_0=h_0,
            h_k=None,
        )

        self.step_records.append(step_record)

        return dict(step_record)

    def Step_k_Characteristics_Save(self, beta, beta_hat, rho_k, h_0, h_k):
        # Вход: истинная beta, оценка beta_hat после step k, rho_k, h_0, h_k.
        # Выход: словарь с характеристиками step k.
        # Что делает: сохраняет диагностику очередного анизотропного шага.
        # Реализация: используется общий метод записи шага, чтобы формат строк был единым.

        return self.Step_Characteristics_Save(
            beta=beta,
            beta_hat=beta_hat,
            rho_k=rho_k,
            h_0=h_0,
            h_k=h_k,
        )

    def Characteristics_Get(self):
        # Вход: нет.
        # Выход: словарь со всеми накопленными характеристиками.
        # Что делает: возвращает копию текущего диагностического состояния.
        # Реализация: списки копируются, чтобы внешний код случайно не изменил внутреннее состояние.

        return {
            "h_0_values": list(self.h_0_values),
            "h_k_values": list(self.h_k_values),
            "rho_k_values": list(self.rho_k_values),
            "step_records": [dict(step_record) for step_record in self.step_records],
            "run_folder": str(self.run_folder) if self.run_folder is not None else None,
        }

    def Characteristics_Reset(self):
        # Вход: нет.
        # Выход: self.
        # Что делает: очищает все накопленные характеристики.
        # Реализация: списки очищаются на месте, чтобы объект можно было переиспользовать.

        self.h_0_values.clear()
        self.h_k_values.clear()
        self.rho_k_values.clear()
        self.step_records.clear()
        self.run_folder = None

        return self

    def Run_Folder_Create(self, output_dir="adp_runs", run_name=None, overwrite=False):
        # Вход: output_dir - базовая папка, run_name - имя запуска, overwrite - можно ли использовать готовую папку.
        # Выход: Path созданной папки запуска.
        # Что делает: создает отдельную папку под один запуск ADP и подпапку plots.
        # Реализация: если имя не задано, используется timestamp; если папка занята, добавляется числовой суффикс.

        output_path = Path(output_dir)

        if run_name is None:
            run_name = datetime.now().strftime("adp_run_%Y%m%d_%H%M%S_%f")

        run_folder = output_path / run_name

        if run_folder.exists() and not overwrite:
            run_folder = self._Unique_Run_Folder_Create(output_path, run_name)

        run_folder.mkdir(parents=True, exist_ok=overwrite)
        (run_folder / "plots").mkdir(parents=True, exist_ok=True)

        self.run_folder = run_folder

        return run_folder

    def Run_Save(self, output_dir="adp_runs", run_name=None, use_latex=True, overwrite=False):
        # Вход: output_dir/run_name задают папку запуска, use_latex включает LaTeX-рендеринг графиков.
        # Выход: Path папки, куда сохранены характеристики.
        # Что делает: создает папку запуска, сохраняет json/csv и строит графики по характеристикам.
        # Реализация: папка создается один раз, затем вызываются отдельные методы для таблиц и графиков.

        run_folder = self.Run_Folder_Create(
            output_dir=output_dir,
            run_name=run_name,
            overwrite=overwrite,
        )

        self.Characteristics_JSON_Save(run_folder)
        self.Characteristics_CSV_Save(run_folder)
        self.Characteristics_Plot_Save(run_folder, use_latex=use_latex)

        return run_folder

    def Characteristics_JSON_Save(self, run_folder=None):
        # Вход: run_folder - папка запуска или None для последней созданной папки.
        # Выход: Path json-файла.
        # Что делает: сохраняет все характеристики в человекочитаемый json.
        # Реализация: используется ensure_ascii=False, чтобы русские комментарии и будущие подписи не экранировались.

        run_folder = self._Run_Folder_Resolve(run_folder)
        json_path = run_folder / "characteristics.json"

        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                self.Characteristics_Get(),
                json_file,
                ensure_ascii=False,
                indent=2,
            )

        return json_path

    def Characteristics_CSV_Save(self, run_folder=None):
        # Вход: run_folder - папка запуска или None для последней созданной папки.
        # Выход: Path csv-файла.
        # Что делает: сохраняет пошаговые характеристики в табличном формате.
        # Реализация: None записывается пустой ячейкой, чтобы csv было удобно открыть в таблицах.

        run_folder = self._Run_Folder_Resolve(run_folder)
        csv_path = run_folder / "characteristics.csv"

        fieldnames = ["step", "step_name", "cosine", "rho_k", "h_0", "h_k"]

        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # Записываем каждую строку step_records в CSV без служебных полей.
            for step_record in self.step_records:
                writer.writerow(
                    {
                        "step": step_record["step"],
                        "step_name": step_record["step_name"],
                        "cosine": self._CSV_Value_Format(step_record["cosine"]),
                        "rho_k": self._CSV_Value_Format(step_record["rho_k"]),
                        "h_0": self._CSV_Value_Format(step_record["h_0"]),
                        "h_k": self._CSV_Value_Format(step_record["h_k"]),
                    }
                )

        return csv_path

    def Characteristics_Plot_Save(self, run_folder=None, use_latex=True):
        # Вход: run_folder - папка запуска, use_latex - нужно ли включать компиляцию LaTeX.
        # Выход: список Path созданных графиков.
        # Что делает: строит отдельные графики для cosine, rho_k, h_0, h_k и общий график 2x2.
        # Реализация: matplotlib импортируется лениво, чтобы обычный запуск алгоритма не зависел от графического стека.

        run_folder = self._Run_Folder_Resolve(run_folder)
        plots_folder = run_folder / "plots"
        plots_folder.mkdir(parents=True, exist_ok=True)

        plt = self._Matplotlib_Configure(use_latex=use_latex)

        metric_configs = self._Metric_Configs_Create()
        saved_paths = []

        # Строим отдельный график для каждой важной характеристики.
        for metric_name, metric_config in metric_configs.items():
            x_values, y_values = self._Metric_Series_Get(metric_name)
            plot_path = plots_folder / f"{metric_name}.png"

            self._Single_Metric_Plot_Save(
                plt=plt,
                x_values=x_values,
                y_values=y_values,
                metric_config=metric_config,
                plot_path=plot_path,
            )

            saved_paths.append(plot_path)

        summary_path = plots_folder / "all_characteristics.png"
        self._Summary_Plot_Save(plt, metric_configs, summary_path)
        saved_paths.append(summary_path)

        return saved_paths

    def _Step_Record_Create(self, step_name, beta, beta_hat, rho_k, h_0, h_k):
        # Вход: имя шага, beta/beta_hat и численные характеристики.
        # Выход: словарь одной строки диагностики.
        # Что делает: приводит все значения к единому формату для json/csv/графиков.
        # Реализация: step равен текущей длине истории, cosine считается отдельной функцией.

        return {
            "step": len(self.step_records),
            "step_name": step_name,
            "cosine": self.Cosine_Calculate(beta, beta_hat),
            "rho_k": self._Float_Or_None(rho_k),
            "h_0": self._Float_Or_None(h_0),
            "h_k": self._Float_Or_None(h_k),
        }

    def _Float_Or_None(self, value):
        # Вход: произвольное численное значение.
        # Выход: float или None.
        # Что делает: нормализует numpy/python числа для сохранения.
        # Реализация: нечисловые, бесконечные и nan-значения превращаются в None.

        if value is None:
            return None

        value = float(np.asarray(value).reshape(()))

        if not np.isfinite(value):
            return None

        return value

    def _CSV_Value_Format(self, value):
        # Вход: значение из записи шага.
        # Выход: строка для csv.
        # Что делает: делает csv компактным и стабильным.
        # Реализация: None становится пустой строкой, числа пишутся с точностью до 12 значащих цифр.

        if value is None:
            return ""

        return f"{float(value):.12g}"

    def _Unique_Run_Folder_Create(self, output_path, run_name):
        # Вход: базовая папка и желаемое имя запуска.
        # Выход: свободный Path.
        # Что делает: подбирает уникальную папку, если запуск с таким именем уже существует.
        # Реализация: перебирает суффиксы 001, 002, ... до первого свободного варианта.

        suffix = 1

        # Ищем первый свободный суффикс для повторного запуска с тем же именем.
        while True:
            candidate = output_path / f"{run_name}_{suffix:03d}"

            if not candidate.exists():
                return candidate

            suffix += 1

    def _Run_Folder_Resolve(self, run_folder):
        # Вход: run_folder или None.
        # Выход: Path существующей папки запуска.
        # Что делает: возвращает явную папку или последнюю созданную папку объекта.
        # Реализация: если папка еще не создана, вызывается Run_Folder_Create().

        if run_folder is None and self.run_folder is None:
            return self.Run_Folder_Create()

        if run_folder is None:
            return self.run_folder

        run_folder = Path(run_folder)
        run_folder.mkdir(parents=True, exist_ok=True)
        (run_folder / "plots").mkdir(parents=True, exist_ok=True)
        self.run_folder = run_folder

        return run_folder

    def _Metric_Configs_Create(self):
        # Вход: нет.
        # Выход: словарь настроек для каждого графика.
        # Что делает: задает подписи, цвета и имена важных характеристик.
        # Реализация: подписи y записаны в LaTeX-совместимом виде.

        return {
            "cosine": {
                "title": "Cosine",
                "ylabel": r"$\cos(\beta,\hat{\beta})$",
                "color": "#1f77b4",
            },
            "rho_k": {
                "title": "Anisotropy",
                "ylabel": r"$\rho_k$",
                "color": "#d62728",
            },
            "h_0": {
                "title": "Initial bandwidth",
                "ylabel": r"$h_0$",
                "color": "#2ca02c",
            },
            "h_k": {
                "title": "Step bandwidth",
                "ylabel": r"$h_k$",
                "color": "#9467bd",
            },
        }

    def _Metric_Series_Get(self, metric_name):
        # Вход: имя метрики из {"cosine", "rho_k", "h_0", "h_k"}.
        # Выход: x_values, y_values для графика.
        # Что делает: извлекает пошаговый ряд из step_records или из отдельной истории скаляров.
        # Реализация: сначала используются записи шагов, потому что они дают правильную ось итераций.

        x_values = []
        y_values = []

        # Собираем значения из пошаговой истории, где известен номер итерации.
        for step_record in self.step_records:
            value = step_record.get(metric_name)

            if value is not None:
                x_values.append(step_record["step"])
                y_values.append(value)

        if y_values:
            return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)

        fallback_values = self._Metric_Fallback_Values_Get(metric_name)

        return np.arange(len(fallback_values), dtype=float), np.asarray(fallback_values, dtype=float)

    def _Metric_Fallback_Values_Get(self, metric_name):
        # Вход: имя метрики.
        # Выход: список значений из отдельной истории.
        # Что делает: дает данные для графиков даже тогда, когда сохранены только h/rho без step_records.
        # Реализация: cosine не имеет отдельной истории, поэтому для него возвращается пустой список.

        if metric_name == "rho_k":
            return list(self.rho_k_values)

        if metric_name == "h_0":
            return list(self.h_0_values)

        if metric_name == "h_k":
            return list(self.h_k_values)

        return []

    def _Matplotlib_Configure(self, use_latex=True):
        # Вход: use_latex - флаг LaTeX-рендеринга.
        # Выход: модуль matplotlib.pyplot.
        # Что делает: настраивает красивый стиль графиков и включает text.usetex при необходимости.
        # Реализация: backend Agg выбирается до импорта pyplot, если pyplot еще не загружен.

        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adp_matplotlib"))

        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        matplotlib.rcParams.update(
            {
                "text.usetex": bool(use_latex),
                "text.latex.preamble": r"\usepackage{amsmath}",
                "font.family": "serif" if use_latex else "DejaVu Sans",
                "figure.figsize": (7.2, 4.2),
                "figure.dpi": 130,
                "savefig.dpi": 180,
                "axes.facecolor": "#f7f8fb",
                "axes.edgecolor": "#293241",
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "grid.color": "#9aa4b2",
                "grid.alpha": 0.28,
                "grid.linewidth": 0.8,
                "lines.linewidth": 2.3,
                "lines.markersize": 6.0,
            }
        )

        return plt

    def _Single_Metric_Plot_Save(self, plt, x_values, y_values, metric_config, plot_path):
        # Вход: pyplot, данные ряда, настройки метрики и путь сохранения.
        # Выход: Path сохраненного png.
        # Что делает: строит один аккуратный график характеристики.
        # Реализация: подпись оси y выставляется с rotation=0, поэтому текст параллелен оси X.

        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

        self._Axis_Style_Apply(axis, metric_config)
        self._Metric_Line_Draw(axis, x_values, y_values, metric_config)

        figure.savefig(plot_path, bbox_inches="tight")
        plt.close(figure)

        return plot_path

    def _Summary_Plot_Save(self, plt, metric_configs, summary_path):
        # Вход: pyplot, настройки всех метрик, путь сохранения.
        # Выход: Path общего png.
        # Что делает: строит сетку 2x2 со всеми важными характеристиками запуска.
        # Реализация: каждый subplot использует тот же стиль и горизонтальную подпись y.

        figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
        axes = axes.reshape(-1)

        # Рисуем каждую характеристику в своем subplot.
        for axis, (metric_name, metric_config) in zip(axes, metric_configs.items()):
            x_values, y_values = self._Metric_Series_Get(metric_name)

            self._Axis_Style_Apply(axis, metric_config)
            self._Metric_Line_Draw(axis, x_values, y_values, metric_config)

        figure.savefig(summary_path, bbox_inches="tight")
        plt.close(figure)

        return summary_path

    def _Axis_Style_Apply(self, axis, metric_config):
        # Вход: axis matplotlib и настройки метрики.
        # Выход: нет.
        # Что делает: применяет единый визуальный стиль к оси.
        # Реализация: y-label имеет rotation=0 и вынесен левее графика через labelpad.

        axis.set_title(metric_config["title"], pad=12)
        axis.set_xlabel(r"$k$")
        axis.set_ylabel(
            metric_config["ylabel"],
            rotation=0,
            labelpad=34,
            ha="right",
            va="center",
        )
        axis.grid(True, axis="y")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.08, y=0.18)

    def _Metric_Line_Draw(self, axis, x_values, y_values, metric_config):
        # Вход: axis matplotlib, x/y значения и настройки метрики.
        # Выход: нет.
        # Что делает: рисует линию характеристики или аккуратную заглушку no data.
        # Реализация: для одного значения используется scatter, для нескольких - line plot с маркерами.

        if y_values.size == 0:
            axis.text(
                0.5,
                0.5,
                "no data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)

            return

        if y_values.size == 1:
            axis.scatter(
                x_values,
                y_values,
                s=70,
                color=metric_config["color"],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
        else:
            axis.plot(
                x_values,
                y_values,
                marker="o",
                color=metric_config["color"],
                markerfacecolor="white",
                markeredgewidth=1.8,
            )

        axis.set_xticks(x_values)
