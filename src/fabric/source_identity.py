"""Bind real artifacts and runs to the committed ``src/fabric`` tree."""

from __future__ import annotations

from pathlib import Path
import subprocess


def committed_source_identity(*, require_clean: bool) -> str:
    """Return the last commit that changed ``src/fabric``."""

    repository = Path(__file__).resolve().parents[2]
    if require_clean:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--", "src/fabric"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("full-cohort execution requires committed src/fabric code")
    return subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "log",
            "-1",
            "--format=%H",
            "--",
            "src/fabric",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
