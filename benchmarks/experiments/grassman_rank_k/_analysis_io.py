"""Shared JSONL and report-output helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except json.JSONDecodeError as exc:
                    rows.append(
                        {"status": "malformed", "error": str(exc), "line": line_no}
                    )
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)
