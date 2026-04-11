"""Antipartisan enforcement (project rule).

The Empowered Vote platform never reads, logs, or persists party
affiliation. Phase 107's roster endpoint strips the field server-side,
and Phase 108's new client code must contain zero references to the
word in its source.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PHASE_108_FILES = [
    "src/essentials_client.py",
    "src/alias_gen.py",
    "refresh_roster.py",
]

_FORBIDDEN = "p" + "arty"  # split to avoid literal in this file's source


def test_no_forbidden_references_in_phase_108_sources():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["grep", "-n", _FORBIDDEN, *PHASE_108_FILES],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    # grep returns 1 when no matches (desired) and 0 when matches found (failure).
    assert result.returncode != 0, (
        f"Antipartisan violation — forbidden word found in Phase 108 sources:\n"
        f"{result.stdout}"
    )


def test_no_forbidden_references_in_roster_py_extension():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["grep", "-n", _FORBIDDEN, "src/roster.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Antipartisan violation in src/roster.py:\n{result.stdout}"
    )
