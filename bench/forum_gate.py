"""Pass/fail a candidate forum diarization against a handoff-derived reference.

Pure: no env loading, no Modal, no I/O beyond a path the caller hands in. The
CLI wrapper is `scripts/score_forum_diarization.py`.

The gate is asymmetric on purpose. Any conflation fails it; fragmentation never
does. An extra unnamed speaker costs a reviewer seconds at label level, while a
silent merge misattributes quotes to a candidate in a live race.
"""

from __future__ import annotations

import json
from pathlib import Path

from .identity_score import Turns

#: This repair's gate.
GATE_MIN_FRACTION = 0.05
#: `identity_score`'s own default, reported alongside so this meeting's numbers
#: stay comparable with every other diarization measurement in the repo.
COMPARABLE_MIN_FRACTION = 0.02


def load_turns(path: Path) -> Turns:
    """Read a JSON list of segment dicts into scoring turns."""
    segments = json.loads(Path(path).read_text())
    return [
        (float(s["start_time"]), float(s["end_time"]), str(s["speaker_label"]))
        for s in segments
    ]


def reference_half(windows: list[Turns], half: str) -> Turns:
    """All windows, the odd ones (tune) or the even ones (holdout).

    Halving by WINDOW, never by turn: the flat reference alternates moderator,
    person, moderator, person, so a parity slice of turns would hand one half a
    reference with no moderator in it — and the moderator is the label this
    repair exists to break apart.
    """
    if half == "all":
        chosen = windows
    elif half == "tune":
        chosen = windows[1::2]
    elif half == "holdout":
        chosen = windows[0::2]
    else:
        raise ValueError(f"half must be all/tune/holdout, got {half!r}")
    return [turn for window in chosen for turn in window]


def gate_verdict(
    report, max_minority: float, *, unattributed_label: str | None = None
) -> tuple[bool, list[str]]:
    """Pass unless some IDENTIFIED label holds two reference people above the floor.

    `max_minority` is the floor the caller already passed to `identity_report`;
    it is accepted here so the verdict line can state the bar it applied.

    `unattributed_label` names the bucket where turns with too little voice evidence
    are parked. That bucket holds slivers from many people by construction, so
    scoring it as a speaker identity would guarantee a failure and punish the design
    for being honest about what it does not know. It is excluded for the same reason
    `IdentityReport.unmapped_labels` is not an error: neither is a claim about who
    spoke.
    """
    reasons = [
        f"label {c.label} holds {len(c.people)} people: "
        + ", ".join(f"{p} {c.seconds[p]:.1f}s" for p in c.people)
        for c in report.conflation
        if unattributed_label is None or c.label != unattributed_label
    ]
    return (not reasons), reasons
