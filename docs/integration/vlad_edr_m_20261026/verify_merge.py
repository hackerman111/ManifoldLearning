"""Проверить сохранность импорта и файлов EDR_code до объединения."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
ALLOWED_EDITS = {
    "PLAN.md",
    "agent-notes/STATE.md",
    "agent-notes/DECISIONS.md",
    "agent-notes/README.md",
    "agent-notes/tex/README.md",
    "agent-notes/ADP/cli-and-compat.md",
    "experiments/r_sdr/README.md",
    "experiments/README.md",
    "ADP/cli/main.py",
    "experiments/r_sdr/design.py",
    "experiments/r_sdr/__main__.py",
    "experiments/r_sdr/report.py",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, evidence: Path) -> dict[str, object]:
    manifest = json.loads((evidence / "IMPORT_MANIFEST.json").read_text())
    before = json.loads((evidence / "TARGET_BEFORE.json").read_text())
    errors = []
    changed = []
    source_changes = []
    source = root / manifest["source"]
    for entry in manifest["files"]:
        path = root / entry["target"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            errors.append(entry["target"])
        original = source / entry["source"]
        if source.exists() and (
            not original.is_file() or digest(original) != entry["sha256"]
        ):
            source_changes.append(entry["source"])
    for relative, expected in before.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"deleted original: {relative}")
        elif digest(path) != expected:
            changed.append(relative)
            if relative not in ALLOWED_EDITS:
                errors.append(f"unapproved change: {relative}")
    return {
        "source_files": len(manifest["files"]),
        "target_files_before": len(before),
        "changed_original_files": sorted(changed),
        "import_or_preservation_errors": errors,
        "source_changes_since_import": source_changes,
        "ok": not errors and not source_changes,
    }


if __name__ == "__main__":
    result = verify(ROOT, EVIDENCE)
    (EVIDENCE / "integrity-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
