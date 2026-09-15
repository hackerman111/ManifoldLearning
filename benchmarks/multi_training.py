from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy


def main() -> None:
    """Воспроизводимый отдельный процесс для одной точки multi-index ADP."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--solver-source", type=Path)
    parser.add_argument("--initialization-source", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--no-tracemalloc", action="store_true")
    settings, arguments = parser.parse_known_args()
    if settings.output.exists() or settings.output.with_suffix(".npy").exists():
        parser.error("output prefix already exists; choose a new --output path")
    if settings.reference_root is not None:
        if not (settings.reference_root / "ADP" / "__init__.py").is_file():
            parser.error("--reference-root must contain the reference ADP package")
        sys.path.insert(0, str(settings.reference_root.resolve()))
    from ADP.cli.main import _quality, _run, build_parser

    source_root = Path(importlib.import_module("ADP").__file__).parent
    args = build_parser().parse_args(["--mode", "multi", *arguments])
    if settings.solver_source is not None:
        runner = importlib.import_module("ADP.cli.main")
        spec = importlib.util.spec_from_file_location(
            "reference_solver", settings.solver_source
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if args.solver != "lsmr":
            raise ValueError("--solver-source supports lsmr only")
        runner.solve_lsmr = module.solve
    if settings.initialization_source is not None:
        runner = importlib.import_module("ADP.cli.main")
        spec = importlib.util.spec_from_file_location(
            "ADP.engine.reference_initialize", settings.initialization_source
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner.initialize_basis_local_with_spectrum = (
            module.initialize_basis_local_with_spectrum
        )
    row: dict[str, object] = {
        "label": settings.label,
        "arguments": arguments,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "dtype": "float64",
        "tracemalloc_enabled": not settings.no_tracemalloc,
        "source_root": str(source_root),
        "source_sha256": {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(source_root.rglob("*.py"))
        },
        "kernel": "max(1-t**2,0), t=squared anisotropic distance/h**2",
        "config": {key: value for key, value in vars(args).items() if key != "kernel"},
        "solver_source_sha256": (
            hashlib.sha256(settings.solver_source.read_bytes()).hexdigest()
            if settings.solver_source is not None
            else None
        ),
        "initialization_source_sha256": (
            hashlib.sha256(settings.initialization_source.read_bytes()).hexdigest()
            if settings.initialization_source is not None
            else None
        ),
    }
    try:
        from threadpoolctl import threadpool_info

        row["blas"] = threadpool_info()
    except ImportError:
        row["blas"] = "threadpoolctl unavailable"
    started = perf_counter()
    try:
        # Сохраняем таймеры исходного CLI; отключаются только измерения памяти.
        tracing = (
            patch("tracemalloc.start") if settings.no_tracemalloc else nullcontext()
        )
        with tracing:
            index, truth, profile, metadata = _run(args)
        if settings.no_tracemalloc:
            for record in profile.values():
                record.pop("traced_peak_bytes", None)
        row.update(profile=profile, metadata=metadata)
        row["quality"] = _quality(args.mode, index, truth)
        if args.mode == "manifold":
            row["quality_metric"] = "RMS principal sine (lower is better)"
            residual = index - (index @ truth.swapaxes(1, 2)) @ truth
            row["projector_distance"] = float(
                np.square(residual).sum(axis=(1, 2)).mean()
            )
        elif args.mode == "single":
            row["quality_metric"] = "absolute cosine (higher is better)"
            residual = index - (index @ truth) @ truth.T
            row["projector_distance"] = float(np.square(residual).sum())
        else:
            row["quality_metric"] = "trace_score (normalized overlap, higher is better)"
            residual = index - (index @ truth) @ truth.T
            row["projector_distance"] = float(np.square(residual).sum())
        row["error"] = None
        np.save(settings.output.with_suffix(".npy"), index)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["elapsed_seconds"] = perf_counter() - started
    row["rss_peak_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    settings.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({key: row.get(key) for key in ("label", "quality", "error")}))


if __name__ == "__main__":
    main()
