"""Observational extension of frozen workers: count refits and Schur trials.

The original 32 numerical/launcher sources stay frozen. This separate wrapper
records its own manifest before launch and leaves all solver decisions intact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import inner
from .common import OUT, _assert_frozen, environment


def counted_solve(*args, **kwargs) -> dict[str, Any]:
    from ADP.solver import grassman as core, grassman_exp as exp

    originals = [
        (module, name, getattr(module, name))
        for module in (core, exp)
        for name in ("_profile", "_gradient")
    ]
    original_value = core._AngleProfile.value
    total, curve_depth = 0, 0
    gradient_counts: list[int] = []

    def profile(original):
        def call(*call_args, **call_kwargs):
            nonlocal total
            if curve_depth == 0:
                total += 1
            return original(*call_args, **call_kwargs)

        return call

    def gradient(original):
        def call(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            gradient_counts.append(total)
            return result

        return call

    def curve_value(self, t):
        nonlocal total, curve_depth
        total += 1
        curve_depth += 1
        try:
            return original_value(self, t)
        finally:
            curve_depth -= 1

    for module, name, original in originals:
        setattr(
            module,
            name,
            profile(original) if name == "_profile" else gradient(original),
        )
    core._AngleProfile.value = curve_value
    try:
        row = _frozen_solve(*args, **kwargs)
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
        core._AngleProfile.value = original_value
    if len(row["trace"]) != len(gradient_counts):
        raise AssertionError("objective counter and gradient trace disagree")
    for point, count in zip(row["trace"], gradient_counts, strict=True):
        point["objective_evaluations"] = count
    row["objective_evaluation_count"] = total
    expected = row["diagnostics"]["profile_evaluations"] + 2
    if total != expected:
        raise AssertionError(f"objective count {total} != diagnostic count {expected}")
    row["counting_note"] = (
        "Each exact profile trial counts once, including Schur trials; "
        "initial/final included."
    )
    return row


def worker() -> None:
    original = inner.solve_snapshot
    inner.solve_snapshot = counted_solve
    try:
        inner.main()
    finally:
        inner.solve_snapshot = original
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    row = json.loads(output.read_text())
    row.setdefault("environment", environment())
    row["objective_counter_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    output.write_text(json.dumps(row, allow_nan=False) + "\n")


_frozen_solve = inner.solve_snapshot


def main() -> None:
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker()
        return
    from . import run_inner

    _assert_frozen()
    manifest = OUT / "results" / "objective_counter_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(
        json.dumps(
            {
                "base_manifest_sha256": hashlib.sha256(
                    (OUT / "freeze_manifest.json").read_bytes()
                ).hexdigest(),
                "source": str(Path(__file__).relative_to(OUT)),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "purpose": "Objective counter; no numerical source changes.",
                "environment": environment(),
            },
            indent=2,
        )
        + "\n"
    )
    launch = run_inner._run_child

    def counted_child(module, arguments, timeout):
        if module == "experiments.grassman_rank_k.inner":
            return launch(
                "experiments.grassman_rank_k.run_counted_inner",
                ["--worker", *arguments],
                timeout,
            )
        return launch(module, arguments, timeout)

    run_inner._run_child = counted_child
    try:
        run_inner.inner_runs()
    finally:
        run_inner._run_child = launch


if __name__ == "__main__":
    main()
