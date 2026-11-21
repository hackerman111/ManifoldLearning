from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarks.clean_npz import delete_selected, scan, tracked_npz


def test_cleanup_only_removes_selected_unchanged_untracked_npz(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    output = repo / "old run"
    output.mkdir()
    contents = {
        name: output / name
        for name in ("remove.npz", "keep.npz", "tracked.NPZ", "edited.npz", "run.json")
    }
    for file in contents.values():
        file.write_bytes(b"data")
    subprocess.run(["git", "add", "--", "tracked.NPZ"], cwd=output, check=True)
    (output / "link.npz").symlink_to(contents["keep.npz"])
    (output / "outside").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "outside.npz").write_bytes(b"outside")

    snapshots = scan(output, tracked_npz(output))

    assert {path.name for path in snapshots} == {"remove.npz", "keep.npz", "edited.npz"}

    contents["edited.npz"].write_bytes(b"data updated by the runner")
    selected = {contents["remove.npz"], contents["edited.npz"]}
    deleted, skipped = delete_selected(selected, snapshots, output)

    assert deleted == 1
    assert len(skipped) == 1
    assert "edited.npz" in skipped[0]
    assert not contents["remove.npz"].exists()
    assert all(contents[name].exists() for name in contents if name != "remove.npz")
    assert (tmp_path / "outside.npz").read_bytes() == b"outside"


def test_cleanup_preserves_file_added_to_git_after_scan(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    file = tmp_path / "frozen.npz"
    file.write_bytes(b"raw experiment input")
    snapshots = scan(tmp_path, tracked_npz(tmp_path))
    subprocess.run(["git", "add", "--", file.name], cwd=tmp_path, check=True)

    deleted, skipped = delete_selected({file}, snapshots, tmp_path)

    assert deleted == 0
    assert len(skipped) == 1
    assert file.read_bytes() == b"raw experiment input"
