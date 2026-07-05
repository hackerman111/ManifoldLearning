import csv
import os
from pathlib import Path

import numpy as np


def CreateTrace(enabled=True, store_arrays=False):
    """
    Создает контейнер для отслеживания переменных ADP.

    store_arrays=False хранит только сводки, чтобы не раздувать память.
    store_arrays=True дополнительно сохраняет копии массивов.
    """
    return {
        "enabled": enabled,
        "store_arrays": store_arrays,
        "steps": [],
    }


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    return value


def _summarize_value(value):
    if isinstance(value, np.ndarray):
        summary = {
            "type": "ndarray",
            "shape": value.shape,
            "dtype": str(value.dtype),
        }

        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            finite = value[np.isfinite(value)]
            if finite.size > 0:
                summary.update(
                    {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                        "norm": float(np.linalg.norm(finite)),
                    }
                )

        return summary

    if isinstance(value, (int, float, str, bool, type(None))):
        return {"type": type(value).__name__, "value": value}

    if isinstance(value, (np.integer, np.floating)):
        return {"type": type(value).__name__, "value": value.item()}

    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value.keys())}

    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}

    return {"type": type(value).__name__, "repr": repr(value)}


def TraceStep(trace, name, **variables):
    """
    Добавляет запись о шаге алгоритма.
    """
    if trace is None or not trace.get("enabled", True):
        return trace

    record = {
        "name": name,
        "summaries": {},
    }

    if trace.get("store_arrays", False):
        record["values"] = {}

    for variable_name, value in variables.items():
        record["summaries"][variable_name] = _summarize_value(value)

        if trace.get("store_arrays", False):
            record["values"][variable_name] = _copy_value(value)

    trace["steps"].append(record)
    return trace


def GetTraceTable(trace):
    """
    Возвращает плоскую таблицу summary по всем переменным всех шагов.
    """
    rows = []

    if trace is None:
        return rows

    for step_index, step in enumerate(trace.get("steps", [])):
        for variable_name, summary in step.get("summaries", {}).items():
            row = {
                "step_index": step_index,
                "step_name": step["name"],
                "variable": variable_name,
            }
            row.update(summary)
            rows.append(row)

    return rows


def SaveTraceSummary(trace, path):
    """
    Сохраняет summary трассировки в CSV.
    """
    rows = GetTraceTable(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step_index",
        "step_name",
        "variable",
        "type",
        "shape",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "norm",
        "value",
        "keys",
        "length",
        "repr",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return str(path)


def _plot_bar(values, title, ylabel, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(values)), values)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_line(values, title, ylabel, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.asarray(values), marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def PlotADPDiagnostics(result, output_dir="adp_trace_plots"):
    """
    Строит базовые графики по результату ADP.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    beta = result.get("beta")
    if beta is not None:
        path = output_dir / "beta_components.png"
        _plot_bar(beta, "Оценка beta", "value", path)
        paths["beta_components"] = str(path)

    average_gradient = result.get("average_gradient")
    if average_gradient is not None:
        path = output_dir / "average_gradient_components.png"
        _plot_bar(average_gradient, "Средний градиент", "value", path)
        paths["average_gradient_components"] = str(path)

    local_gradients = result.get("local_gradients")
    if local_gradients is not None:
        path = output_dir / "local_gradient_norms.png"
        _plot_line(
            np.linalg.norm(local_gradients, axis=1),
            "Нормы локальных градиентов",
            "norm",
            path,
        )
        paths["local_gradient_norms"] = str(path)

    weights = result.get("weights")
    if weights is not None:
        path = output_dir / "weight_sums.png"
        _plot_line(weights.sum(axis=1), "Суммы весов по центрам", "sum weights", path)
        paths["weight_sums"] = str(path)

    return paths


def SaveADPDiagnostics(result, output_dir="adp_trace_plots"):
    """
    Сохраняет CSV трассировки и графики, если они есть в result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {
        "plots": PlotADPDiagnostics(result, output_dir=output_dir),
    }

    trace = result.get("trace")
    if trace is not None:
        saved["trace_summary"] = SaveTraceSummary(trace, output_dir / "trace_summary.csv")

    return saved
