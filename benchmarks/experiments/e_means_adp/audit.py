"""Независимый read-back базисов/данных; отдельная формула метрик."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.r_sdr.design import Case, generate


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_readback(rows: list[dict], spec: dict, method: str) -> dict:
    """Отдельный матричный расчёт gates, без импорта study.evaluate."""
    table = {(r["case"], r["seed"], r["method"]): r for r in rows}
    grid = [
        (
            table.get((c["name"], seed, "original20")),
            table.get((c["name"], seed, method)),
        )
        for c in spec["cases"]
        for seed in spec["seeds"]
    ]
    pairs = [(a, b) for a, b in grid if a and b and a["status"] == b["status"] == "ok"]
    if len(pairs) != len(grid) or not pairs:
        return {"pass": False, "pairs": len(pairs)}
    delta = np.array([b["edr"] - a["edr"] for a, b in pairs]).reshape(
        len(spec["cases"]), len(spec["seeds"])
    )
    seed_means = delta.mean(axis=0)
    rng = np.random.default_rng(1133400)
    idx = rng.integers(0, len(seed_means), size=(10000, len(seed_means)))
    upper = float(np.quantile(seed_means[idx].mean(axis=1), 0.95))
    time_ratio = float(
        np.median([b["fit_seconds"] / a["fit_seconds"] for a, b in pairs])
    )
    rss_ratio = float(
        np.median([b["peak_rss_mib"] / a["peak_rss_mib"] for a, b in pairs])
    )
    gates = {
        "complete": True,
        "orthogonality": all(b["orthogonality_error"] <= 1e-8 for _, b in pairs),
        "full_profile_guard": all(
            b["full_guard_max_increase"] <= 1e-12 for _, b in pairs
        ),
        "worst_edr_regression": float(delta.max()) <= 0.05,
        "per_case_mean_regression": float(delta.mean(axis=1).max()) <= 0.02,
        "rss": rss_ratio <= 1.1,
        "speed_or_quality": (upper <= 0.01 and time_ratio <= 0.9)
        or (upper <= -0.01 and time_ratio <= 1.1),
    }
    return {
        "pass": all(gates.values()),
        "pairs": len(pairs),
        "gates": gates,
        "mean_delta": float(delta.mean()),
        "upper95_mean_delta": upper,
        "worst_delta": float(delta.max()),
        "median_time_ratio": time_ratio,
        "median_rss_ratio": rss_ratio,
    }


def audit_stage(root: Path, name: str) -> dict:
    """Проверить полноту, frozen source/data, pairing, метрики и true trace."""
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    stage = root / name
    spec = json.loads((stage / "manifest.json").read_text())
    rows = [
        json.loads(line) for line in (stage / "runs.jsonl").read_text().splitlines()
    ]
    failures: list[str] = []
    checks = 0

    def check(condition: bool, label: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    if name == "selection":
        seeds, methods = protocol["selection_seeds"], protocol["methods"]
    elif name == "validation":
        selection_audit = audit_stage(root, "selection")
        check(selection_audit["pass"], "selection_audit_required")
        selection = json.loads((root / "selection/gates.json").read_text())
        eligible = [
            m
            for m in ("mini", "mini_ema")
            if selection["candidates"].get(m, {}).get("pass", False)
        ]
        check(bool(eligible), "selection_gate_required")
        seeds = protocol["validation_seeds"]
        methods = [*protocol.get("controls", ["original20", "full6"]), *eligible]
    else:
        raise ValueError("only frozen selection/validation stages can be audited")
    expected_spec = {"cases": protocol["cases"], "seeds": seeds, "methods": methods}
    check(spec == expected_spec, "stage_matches_frozen_protocol")
    check(
        protocol.get("center_policy", "bounded") in {"all", "bounded"}, "center_policy"
    )
    if failures:
        return {
            "pass": False,
            "checks": checks,
            "failures": failures,
            "valid_fits": 0,
            "artifact_hashes": {},
        }

    for rel, expected in protocol["sources_sha256"].items():
        path = root / "source" / rel
        check(path.is_file() and _hash(path) == expected, f"source:{rel}")
    for rel, expected in json.loads((stage / "data_hashes.json").read_text()).items():
        check(_hash(root / rel) == expected, f"data:{rel}")
    expected_keys = {
        (c["name"], seed, method)
        for c in spec["cases"]
        for seed in spec["seeds"]
        for method in spec["methods"]
    }
    keys = [(r["case"], r["seed"], r["method"]) for r in rows]
    check(len(keys) == len(set(keys)), "duplicate_rows")
    check(set(keys) == expected_keys, "complete_expected_rows")
    lookup = dict(zip(keys, rows, strict=True))
    declared = json.loads((stage / "gates.json").read_text())["candidates"]
    for method in methods:
        if method != "original20":
            actual = _gate_readback(rows, expected_spec, method)
            given = declared.get(method, {})
            for field, value in actual.items():
                if isinstance(value, float):
                    check(
                        abs(given.get(field, float("inf")) - value) < 2e-12,
                        f"gate_metric:{method}:{field}",
                    )
                else:
                    check(given.get(field) == value, f"gate:{method}:{field}")
    valid = 0
    artifacts = {}
    for c in spec["cases"]:
        case = Case(**c)
        for seed in spec["seeds"]:
            block = stage / f"{case.name}-{seed}"
            data = np.load(block / "data.npz")
            for field, generated in zip(
                ("X", "y", "truth"), generate(case, seed), strict=True
            ):
                check(
                    np.array_equal(data[field], generated),
                    f"regenerate:{block.name}:{field}",
                )
            initials = {}
            for method in spec["methods"]:
                fit = block / method
                job = json.loads((fit / "job.json").read_text())
                expected_j = (
                    case.n
                    if protocol.get("center_policy") == "all"
                    and method != "original20_bounded"
                    else min(case.n, 200 if case.d <= 20 else 300)
                )
                expected_config = {
                    "seed": int(
                        np.random.SeedSequence([seed, 4101]).generate_state(1)[0]
                    ),
                    "N_J": expected_j,
                    "N_phi": 12,
                    "N_loc": 20,
                    "N_lin": 2 * case.d,
                }
                check(job["config"] == expected_config, f"job_config:{fit}")
                check(job["method"] == method, f"job_method:{fit}")
                check(
                    job["mini_seed"]
                    == int(np.random.SeedSequence([seed, 4102]).generate_state(1)[0]),
                    f"job_mini_seed:{fit}",
                )
                check(
                    Path(job["data"]) == block / "data.npz"
                    and Path(job["output"]) == fit,
                    f"job_paths:{fit}",
                )
                row = lookup.get((case.name, seed, method))
                if row is None or row["status"] != "ok":
                    continue
                valid += 1
                saved = np.load(fit / "basis.npz")
                B = saved["basis"]
                check(
                    B.shape == data["truth"].shape and np.isfinite(B).all(),
                    f"basis:{fit}",
                )
                Q = np.linalg.qr(B)[0]
                singular = np.linalg.svd(data["truth"].T @ Q, compute_uv=False)
                singular = np.clip(singular, 0, 1)
                edr = float(
                    np.sqrt(max(0, len(singular) - (singular**2).sum()) / len(singular))
                )
                worst = float(np.sqrt(max(0, 1 - singular.min() ** 2)))
                orth = float(np.linalg.norm(B.T @ B - np.eye(len(singular))))
                for field, value in (
                    ("edr", edr),
                    ("max_sin_angle", worst),
                    ("orthogonality_error", orth),
                ):
                    check(abs(row[field] - value) < 2e-8, f"metric:{fit}:{field}")
                check(orth < 1e-8, f"orthogonality:{fit}")
                if expected_j in initials:
                    check(
                        np.array_equal(initials[expected_j], saved["initial"]),
                        f"initial_pair:{fit}",
                    )
                initials[expected_j] = saved["initial"]
                check(
                    row["data_sha256"] == _hash(block / "data.npz"),
                    f"paired_data:{fit}",
                )
                effective = json.loads((fit / "trace.json").read_text())
                check(effective["N_J"] == expected_j, f"actual_N_J:{fit}")
                trace = effective["trace"]
                certified = 0
                for t, step in enumerate(trace):
                    d = step["solver"]
                    certified += d["converged"] is True
                    if method in {"mini", "mini_ema"}:
                        check(
                            d["certificate_scope"] == "full_frozen_profile",
                            f"scope:{fit}:{t}",
                        )
                        check(
                            not d["converged"] or d["stationarity_applicable"],
                            f"rank_certificate:{fit}:{t}",
                        )
                        h = d["profile_history"]
                        check(
                            h[-1] <= h[0] + 1e-12 * max(1, abs(h[0])),
                            f"guard:{fit}:{t}",
                        )
                check(certified == row["n_inner_certified"], f"certified:{fit}")
                check(len(trace) == row["n_inner_total"], f"trace_count:{fit}")
                for filename in (
                    "job.json",
                    "basis.npz",
                    "result.json",
                    "trace.json",
                    "profile.json",
                ):
                    path = fit / filename
                    artifacts[str(path.relative_to(root))] = _hash(path)
    return {
        "pass": not failures,
        "checks": checks,
        "failures": failures,
        "expected_fits": len(expected_keys),
        "valid_fits": valid,
        "failed_fits": sum(r["status"] != "ok" for r in rows),
        "artifact_hashes": artifacts,
        "limits": "Numerical read-back, not interval arithmetic; failed rows retained.",
    }
