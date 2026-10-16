"""Output helpers: Part I's writers plus the run record Part II attaches to every file."""
from __future__ import annotations

import subprocess
from pathlib import Path

import part1_bridge  # noqa: F401  (Part I's src on the path)
from io_utils import environment_record, write_csv, write_json  # noqa: E402  (Part I)

__all__ = ["environment_record", "run_record", "write_csv", "write_json"]


def run_record() -> dict:
    """Interpreter, package versions and the commit the run was made from."""
    record = environment_record()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "."],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    record["git_commit"] = commit
    record["part2_tree_dirty"] = dirty
    return record
