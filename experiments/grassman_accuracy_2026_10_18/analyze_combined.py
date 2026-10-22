"""Парные абляции состава и бюджета из одного сохранённого R-SDR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze

PAIRS = {
    "relative_vs_old": ("ADP-Grassmann-relative", "ADP-Grassmann"),
    "combined10_vs_old": ("ADP-Grassmann-combined10", "ADP-Grassmann"),
    "combined20_vs_old": ("ADP-Grassmann-combined", "ADP-Grassmann"),
    "integration": ("ADP-Grassmann-combined10", "ADP-Grassmann-relative"),
    "budget": ("ADP-Grassmann-combined", "ADP-Grassmann-combined10"),
    "combined20_vs_relative": ("ADP-Grassmann-combined", "ADP-Grassmann-relative"),
    "combined20_vs_R": ("ADP-Grassmann-combined", "MAVE-package"),
}


def run(directory: Path) -> dict:
    output = {}
    for name, (candidate, baseline) in PAIRS.items():
        result = analyze(
            directory, 2.0, write=False, candidate=candidate, baseline=baseline
        )
        output[name] = result
        print(
            json.dumps(
                dict(comparison=name, **result["overall"], gates=result["gates"])
            ),
            flush=True,
        )
    (directory / "combined_analysis.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory.resolve())
