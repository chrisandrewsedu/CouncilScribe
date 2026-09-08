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


def _label_scored_seconds(report, label: str) -> float:
    """Total reference speech `label` overlaps, across every person it touches.

    `report.mapping[label]` (an `identity_score.LabelMapping`) carries only the
    DOMINANT person's seconds and the resulting purity
    (`dominant_seconds / total_seconds`). Recovering the total from those two
    numbers is the only way to get it without reopening the hypothesis/reference
    overlap that produced the report — `IdentityReport` exposes no per-label
    breakdown, and `identity_score.py` must not be modified to add one.
    """
    entry = report.mapping.get(label)
    if entry is None or entry.purity <= 0:
        return 0.0
    return entry.seconds / entry.purity


def scored_reference_seconds(report) -> float:
    """Total reference speech overlapped by ANY hypothesis label.

    This is the denominator `unattributed_bucket_share` measures against: how
    much of the reference speech that got matched to *some* label at all fell
    under the unattributed bucket specifically.
    """
    return sum(_label_scored_seconds(report, label) for label in report.mapping)


def unattributed_bucket_share(report, unattributed_label: str) -> tuple[float, float]:
    """(seconds, share) of scored reference speech held by `unattributed_label`."""
    total = scored_reference_seconds(report)
    bucket = _label_scored_seconds(report, unattributed_label)
    return bucket, (bucket / total if total else 0.0)


def gate_verdict(
    report, max_minority: float, *, unattributed_label: str | None = None
) -> tuple[bool, list[str]]:
    """Pass unless some IDENTIFIED label holds two reference people above the
    floor, or the excluded unattributed bucket has grown past its own bound.

    `max_minority` does double duty, by design: it is the floor the caller
    already passed to `identity_report` for the conflation check (accepted
    here so the verdict line can state the bar it applied), AND it is the
    bound this function enforces on the unattributed bucket's share of scored
    reference speech, below. One knob for both keeps them visibly the same
    policy choice instead of two independent ones that could silently drift
    apart.

    `unattributed_label` names the bucket where turns with too little voice
    evidence are parked. That bucket holds slivers from many people by
    construction, so scoring it as a speaker identity would guarantee a
    failure and punish the design for being honest about what it does not
    know — the same reason `IdentityReport.unmapped_labels` is not an error.
    But exempting it from the identity check cannot mean exempting it from
    every check: folding every turn into it — one label over the whole
    meeting, a worse form of the very defect this repair exists to fix —
    would otherwise PASS with zero reasons, and the same verdict backs
    `forum_recluster.calibrate`'s threshold selection, where an unbounded
    exclusion makes `conflated` fall monotonically as the sliver floor rises
    and the tie-break then picks the highest floor: the knob and the verdict
    would point the same way. So the bucket is exempt from being scored as an
    identity, but not from a SIZE bound: past `max_minority` of scored
    reference speech, it is reported and the gate fails.
    """
    reasons = [
        f"label {c.label} holds {len(c.people)} people: "
        + ", ".join(f"{p} {c.seconds[p]:.1f}s" for p in c.people)
        for c in report.conflation
        if unattributed_label is None or c.label != unattributed_label
    ]
    if unattributed_label is not None:
        bucket_seconds, share = unattributed_bucket_share(report, unattributed_label)
        if share > max_minority:
            reasons.append(
                f"unattributed bucket {unattributed_label} holds {bucket_seconds:.1f}s "
                f"({share:.1%}) of scored reference speech, over the "
                f"{max_minority:.1%} bound"
            )
    return (not reasons), reasons
