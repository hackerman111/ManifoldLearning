"""Paired Multi v2 sentinel experiments with a separate held-out validation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from ADP.core.ADP_Config import ADP_Config

from .diagnostic import _code_fingerprint, _paired, _select_candidate, _summary
from .models import Build, Experiment, ExperimentPoint
from .multiv2 import CATALOG
from .runner import _effective_config, _experiment_id

SENTINELS = {"d10": ("multiv2-mi-d", 0), "n1000": ("multiv2-mi-n", 2)}
SELECTION_SEED = 1000
VALIDATION_SEED = 2000
SELECTION_RUNS = 10
VALIDATION_RUNS = 20
VARIABLE_FIELDS = (
    "solver_max_steps",
    "outer_steps",
    "N_loc",
    "N_phi",
    "N_J",
    "lambda_penalty",
    "index_init",
)


def _baseline(name: str) -> ExperimentPoint:
    selector, index = SENTINELS[name]
    return CATALOG[selector].full[index]


def _candidate_values(stage: str, base: ExperimentPoint, args: argparse.Namespace):
    if stage == "inner":
        return ("baseline", "solver_max_steps=50", "solver_max_steps=80"), (
            base,
            replace(base, solver_max_steps=50),
            replace(base, solver_max_steps=80),
        )
    if stage == "outer":
        if args.inner_steps is None or args.inner_steps < 1:
            raise ValueError("outer stage requires --inner-steps >= 1")
        frozen = replace(base, solver_max_steps=args.inner_steps)
        return ("outer_steps=3", "outer_steps=6", "outer_steps=9"), (
            frozen,
            replace(frozen, outer_steps=6),
            replace(frozen, outer_steps=9),
        )
    if stage == "factor":
        if args.inner_steps is None or args.outer_steps is None or not args.factor:
            raise ValueError(
                "factor stage requires --inner-steps, --outer-steps and --factor"
            )
        field, separator, raw = args.factor.partition("=")
        if not separator or field not in VARIABLE_FIELDS[2:]:
            raise ValueError(
                "factor must be one of N_loc,N_phi,N_J,lambda_penalty,index_init"
            )
        value = (
            raw
            if field == "index_init"
            else (float(raw) if field == "lambda_penalty" else int(raw))
        )
        frozen = replace(
            base, solver_max_steps=args.inner_steps, outer_steps=args.outer_steps
        )
        return ("frozen", args.factor), (frozen, replace(frozen, **{field: value}))
    if stage == "validation":
        if args.candidate_file is None:
            raise ValueError("validation requires --candidate-file")
        variants = json.loads(args.candidate_file.read_text(encoding="utf-8"))
        if not isinstance(variants, dict) or set(variants) != set(SENTINELS):
            raise ValueError("candidate file must contain exactly d10 and n1000")
        return variants
    raise ValueError(f"unknown stage: {stage}")


def make_case(
    name: str, stage: str, args: argparse.Namespace
) -> tuple[Experiment, tuple[str, ...]]:
    base = _baseline(name)
    if stage == "validation":
        variants = _candidate_values(stage, base, args)
        changes = variants[name]
        if (
            not isinstance(changes, dict)
            or not changes
            or set(changes) - set(VARIABLE_FIELDS)
        ):
            raise ValueError(f"invalid frozen candidate for {name}")
        point = replace(base, **changes)
        if point == base:
            raise ValueError("validation candidate must differ from baseline")
        labels, points = ("baseline", "frozen_candidate"), (base, point)
    else:
        labels, points = _candidate_values(stage, base, args)
    experiment = Experiment(
        selector=f"multiv2-quality-{name}",
        title=f"Multi v2 paired quality: {name}, {stage}",
        smoke=points[0],
        full=points,
        report_fields=VARIABLE_FIELDS,
        full_runs=VALIDATION_RUNS if stage == "validation" else SELECTION_RUNS,
        quality_threshold=0.95,
        common_random_fields=VARIABLE_FIELDS,
        hypothesis="Compare inner, outer and estimator factors on paired data.",
    )
    return experiment, labels


def analyze(
    experiment: Experiment,
    labels: tuple[str, ...],
    series_dir: Path,
    *,
    runs: int,
    seed: int,
    stage: str,
) -> dict[str, object]:
    saved = json.loads((series_dir / "series.json").read_text(encoding="utf-8"))
    if saved["points"] != [asdict(point) for point in experiment.full]:
        raise ValueError("saved points differ from frozen candidate set")
    with (series_dir / "runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != runs * len(labels):
        raise ValueError(f"incomplete series: {len(rows)} of {runs * len(labels)} fits")
    by_point = [[] for _ in labels]
    for row in rows:
        index = int(row["point"])
        if index < 0 or index >= len(labels):
            raise ValueError("unexpected point index")
        by_point[index].append(row)
    expected = set(range(runs))
    for index, group in enumerate(by_point):
        if len(group) != runs or {int(row["run"]) for row in group} != expected:
            raise ValueError("candidate run indices incomplete or duplicated")
        if {int(row["seed"]) for row in group} != set(range(seed, seed + runs)):
            raise ValueError("seed range differs from frozen phase")
        point = experiment.full[index]
        for row in group:
            effective = (
                json.loads(row["effective_config"]) if row["effective_config"] else None
            )
            expected_config = _effective_config(
                ADP_Config(), point, int(row["model_seed"])
            )
            if effective is not None and any(
                effective[field]
                != (
                    point.solver_max_steps
                    if field == "solver_max_steps"
                    else getattr(expected_config, field)
                )
                for field in VARIABLE_FIELDS
            ):
                raise ValueError("saved effective config differs from candidate")
    for run in expected:
        if (
            len(
                {
                    next(row["seed_bundle"] for row in group if int(row["run"]) == run)
                    for group in by_point
                }
            )
            != 1
        ):
            raise ValueError("candidate seed bundles are not paired")

    summaries = []
    for index, group in enumerate(by_point):
        traces = [step for row in group for step in json.loads(row["trace"] or "[]")]
        ratios = [float(step["solver"]["normal_residual_ratio"]) for step in traces]
        solver = [step["solver"] for step in traces]
        summaries.append(
            {
                "point": index,
                "candidate": labels[index],
                "phase": stage,
                "requested_config": group[0]["requested_config"],
                "effective_config": next(
                    (
                        row["effective_config"]
                        for row in group
                        if row["effective_config"]
                    ),
                    "",
                ),
                **_summary(group, expected=runs),
                **_paired(group, by_point[0], "higher"),
                "trace_steps": len(traces),
                "inner_converged": sum(step["converged"] is True for step in solver),
                "accepted_steps": dict(
                    Counter(str(step["accepted_steps"]) for step in solver)
                ),
                "lsmr_stop": dict(Counter(str(step["lsmr_stop"]) for step in solver)),
                "correction_certified": sum(
                    math.isfinite(ratio) and ratio <= 0.1 for ratio in ratios
                ),
                "normal_residual_ratio_median": median(ratios) if ratios else None,
                "inner_by_outer": {
                    str(iteration): {
                        "calls": sum(step["iteration"] == iteration for step in traces),
                        "converged": sum(
                            step["iteration"] == iteration
                            and step["solver"]["converged"]
                            for step in traces
                        ),
                    }
                    for iteration in sorted({step["iteration"] for step in traces})
                },
            }
        )
    result = {"stage": stage, "seed": seed, "runs": runs, "summaries": summaries}
    if stage != "validation":
        result["selected_candidate"] = _select_candidate(summaries, "higher")[
            "candidate"
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("inner", "outer", "factor", "validation"), required=True
    )
    parser.add_argument("--point", choices=(*SENTINELS, "all"), default="all")
    parser.add_argument("--inner-steps", type=int)
    parser.add_argument("--outer-steps", type=int)
    parser.add_argument("--factor")
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_outputs/diagnostic")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.stage == "validation" and args.point != "all":
        parser.error("validation always runs both sentinel points")
    names = tuple(SENTINELS) if args.point == "all" else (args.point,)
    try:
        cases = [(name, *make_case(name, args.stage, args)) for name in names]
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    seed = VALIDATION_SEED if args.stage == "validation" else SELECTION_SEED
    runs = VALIDATION_RUNS if args.stage == "validation" else SELECTION_RUNS
    for name, _, labels in cases:
        print(
            f"{name}: {len(labels)} variants x {runs} seeds "
            f"= {len(labels) * runs} fits; {', '.join(labels)}"
        )
    total = sum(len(labels) * runs for _, _, labels in cases)
    print(f"phase={args.stage}; seeds={seed}-{seed + runs - 1}; total={total} fits")
    if args.dry_run:
        return 0
    import threadpoolctl

    root = args.output_dir / f"{_experiment_id()}-multiv2-quality-{args.stage}"
    root.mkdir(parents=True)
    manifest = {
        "stage": args.stage,
        "seed": seed,
        "runs": runs,
        "threads": args.threads,
        "points": list(names),
        "candidates": {
            name: [asdict(p) for p in experiment.full] for name, experiment, _ in cases
        },
        "source_sha256": _code_fingerprint(),
        "status": "running",
    }
    (root / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = []
    try:
        with threadpoolctl.threadpool_limits(limits=args.threads):
            for _, experiment, labels in cases:
                series_dir = experiment.run(
                    Build("ADP", ADP_Config(), solver="lsmr"),
                    profile="full",
                    runs=runs,
                    seed=seed,
                    output_dir=root,
                    plots=False,
                    experiment_id="series",
                    progress=True,
                )
                if _code_fingerprint() != manifest["source_sha256"]:
                    raise RuntimeError("source changed during diagnostic benchmark")
                results.append(
                    analyze(
                        experiment,
                        labels,
                        series_dir,
                        runs=runs,
                        seed=seed,
                        stage=args.stage,
                    )
                )
                (root / "summary.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        return 130
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "error"
        (root / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Result: {root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
