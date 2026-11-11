"""Summaries specific to the Spokoini multi-index simulation grid."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_CHECKPOINTS = (("1", 1), ("2", 2), ("4", 4), ("8", 8), ("last", None))
_SUMMARY_FIELDS = (
    "experiment",
    "point",
    "point_label",
    "index_dim",
    "d",
    "n",
    "sigma_eps",
    "tau",
    "checkpoint",
    "iteration_min",
    "iteration_median",
    "iteration_max",
    "n_total",
    "n_with_trace",
    "n_numerical_failure",
    "mean_loss",
    "loss_q25",
    "loss_q75",
    "loss_iqr",
)


def write_checkpoint_summary(runs_path: Path) -> Path:
    """Write mean and IQR of sum(sin(theta)^2) from each saved ADP trace."""
    with runs_path.open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))

    by_point: dict[str, list[dict[str, str]]] = defaultdict(list)
    for run in runs:
        by_point[run["point"]].append(run)

    result: list[dict[str, object]] = []
    for point, point_runs in by_point.items():
        first = point_runs[0]
        checkpoint_values: dict[str, list[tuple[float, int]]] = {
            name: [] for name, _ in _CHECKPOINTS
        }
        for run in point_runs:
            try:
                trace = json.loads(run["trace"])
            except (KeyError, TypeError, json.JSONDecodeError):
                trace = []
            if not isinstance(trace, list):
                continue
            for name, number in _CHECKPOINTS:
                step = trace[-1] if number is None and trace else None
                if number is not None and len(trace) >= number:
                    step = trace[number - 1]
                if not isinstance(step, dict):
                    continue
                try:
                    score = float(step["quality"])
                    iteration = int(step["iteration"]) + 1
                    index_dim = int(run["index_dim"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(score):
                    continue
                loss = max(0.0, index_dim * (1.0 - min(1.0, score)))
                checkpoint_values[name].append((loss, iteration))

        for name, _ in _CHECKPOINTS:
            values = checkpoint_values[name]
            losses = np.asarray([item[0] for item in values], dtype=float)
            iterations = np.asarray([item[1] for item in values], dtype=int)
            q25, q75 = (
                np.quantile(losses, (0.25, 0.75))
                if len(losses)
                else (np.nan, np.nan)
            )
            result.append(
                {
                    "experiment": first["experiment"],
                    "point": point,
                    "point_label": first["point_label"],
                    "index_dim": first["index_dim"],
                    "d": first["d"],
                    "n": first["n"],
                    "sigma_eps": first["sigma_eps"],
                    "tau": first["tau"],
                    "checkpoint": name,
                    "iteration_min": int(iterations.min()) if len(iterations) else "",
                    "iteration_median": float(np.median(iterations)) if len(iterations) else "",
                    "iteration_max": int(iterations.max()) if len(iterations) else "",
                    "n_total": len(point_runs),
                    "n_with_trace": len(values),
                    "n_numerical_failure": sum(
                        row["status"] == "numerical_failure" for row in point_runs
                    ),
                    "mean_loss": float(losses.mean()) if len(losses) else "",
                    "loss_q25": float(q25) if len(losses) else "",
                    "loss_q75": float(q75) if len(losses) else "",
                    "loss_iqr": float(q75 - q25) if len(losses) else "",
                }
            )

    output_path = runs_path.with_name("checkpoint_summary.csv")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(result)
    return output_path
