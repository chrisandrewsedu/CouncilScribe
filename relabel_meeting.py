#!/usr/bin/env python
"""Move a span of a meeting's transcript from one speaker label to another.

The capability `review.merge_speakers` cannot provide: it is all-or-nothing to
ONE target, so it can fold a split cluster back together but it cannot take a
span OUT of a label that wrongly holds it, and it cannot cut a named segment
whose own text spans two voices. Both are needed to repair what
`mismerge_scan.py` finds. Before this, each such repair was a hand-written
one-off script against live meeting data.

DRY RUN BY DEFAULT. Nothing is written without --apply.

Two ways to say what moves:

    # everything raw SPEAKER_03 said, split out to a label of its own
    .venv/bin/python relabel_meeting.py <meeting_id> --raw-label SPEAKER_03 \
        --to NEW --name "Brian Sterling"

    # one explicit span, onto a label that already exists
    .venv/bin/python relabel_meeting.py <meeting_id> --span 3029.6-3106.0 \
        --to SPEAKER_04

--to NEW mints the lowest unused SPEAKER_NN. --name is optional; without it the
new label is left unnamed and flagged needs_review, which is the honest state
for a voice nobody has identified yet.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config
from src.atomic_io import atomic_write_json
from src.mismerge import load_meeting_segments
from src.models import SpeakerMapping
from src.relabel import (
    apply_plan,
    next_free_label,
    plan_relabel,
    spans_for_raw_label,
)


def _clock(seconds: float) -> str:
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def _snip(text: str, n: int = 88) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def _parse_span(token: str) -> tuple[float, float]:
    try:
        start, end = token.split("-", 1)
        return (float(start), float(end))
    except ValueError:
        raise SystemExit(f"ERROR: --span wants START-END in seconds, got {token!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("meeting_id")
    parser.add_argument("--raw-label", help="move everything this ORIGINAL diarized "
                                            "label said (from transcript_raw.json)")
    parser.add_argument("--span", action="append", default=[],
                        help="explicit START-END in seconds; repeatable")
    parser.add_argument("--to", required=True,
                        help="destination label, or NEW to mint the next free one")
    parser.add_argument("--name", help="display name for the destination label")
    parser.add_argument("--apply", action="store_true",
                        help="write transcript_named.json (default: dry run)")
    args = parser.parse_args()

    if not args.raw_label and not args.span:
        raise SystemExit("ERROR: give --raw-label or at least one --span")

    directory = config.MEETINGS_DIR / args.meeting_id
    named_path = directory / "transcript_named.json"
    if not named_path.exists():
        raise SystemExit(f"ERROR: missing {named_path}")
    payload = json.loads(named_path.read_text())
    raw_segments, named_segments = load_meeting_segments(directory)
    speakers = payload.get("speakers") or {}

    spans = [_parse_span(s) for s in args.span]
    if args.raw_label:
        derived = spans_for_raw_label(raw_segments, named_segments, args.raw_label)
        if not derived:
            raise SystemExit(
                f"ERROR: raw label {args.raw_label} occupies no time in the named "
                f"transcript — nothing to move"
            )
        spans += derived

    # Raw labels are reserved: provenance depends on a raw and a named
    # SPEAKER_NN meaning the same person, so a new label must not reuse a
    # number transcript_raw.json already spent.
    raw_labels = {s.speaker_label for s in raw_segments}
    to_label = (next_free_label(named_segments, speakers, reserved=raw_labels)
                if args.to.upper() == "NEW" else args.to)
    plan = plan_relabel(named_segments, spans, to_label)

    print(f"Meeting: {args.meeting_id}")
    print(f"Source:  {'raw ' + args.raw_label if args.raw_label else 'explicit spans'}"
          f"  ({len(spans)} span(s) after joining)")
    print(f"Target:  {to_label}"
          + (f" (new)" if args.to.upper() == "NEW" else
             f" ({(speakers.get(to_label) or {}).get('speaker_name') or 'unnamed'})")
          + (f" -> name {args.name!r}" if args.name else ""))
    if not plan.moves:
        print("\nNothing to do: every covered segment is already on the target.")
        return 0

    from_labels = sorted({m.from_label for m in plan.moves})
    print(f"\n{len(plan.moves)} segment(s) move off {', '.join(from_labels)}; "
          f"{plan.cuts} need a cut; {plan.seconds:.0f}s total.\n")
    for move in plan.moves:
        mark = "CUT " if not move.whole else "whole"
        print(f"  [{_clock(move.start)}-{_clock(move.end)}] {mark} "
              f"{move.from_label} -> {move.to_label}  ({move.seconds:.1f}s)")
        print(f"        {_snip(move.text)!r}")

    # Apply on a deep copy so the loaded meeting is never disturbed, and so the
    # dry run can report the true post-merge segment count.
    probe = apply_plan(copy.deepcopy(named_segments), plan)
    from src.identify import merge_adjacent_segments
    remerged = merge_adjacent_segments(copy.deepcopy(probe))
    print(f"\nSegments: {len(named_segments)} -> {len(probe)} after cuts "
          f"-> {len(remerged)} after adjacent re-merge.")

    left_behind = {m.from_label for m in plan.moves} - {s.speaker_label for s in probe}
    if left_behind:
        print(f"NOTE: {', '.join(sorted(left_behind))} would be left with no segments.")

    if not args.apply:
        print("\n(dry run — nothing written; pass --apply to write)")
        return 0

    # --- write path -------------------------------------------------------
    import backfill_segment_merge as bsm

    segments = apply_plan(named_segments, plan)
    if to_label not in speakers:
        mapping = SpeakerMapping(speaker_label=to_label)
        mapping.needs_review = True
        if args.name:
            mapping.speaker_name = args.name
            mapping.id_method = "human_review"
            mapping.confidence = 1.0
            mapping.needs_review = False
        speakers[to_label] = mapping.to_dict()
    elif args.name:
        speakers[to_label]["speaker_name"] = args.name
        speakers[to_label]["id_method"] = "human_review"
        speakers[to_label]["confidence"] = 1.0
        speakers[to_label]["needs_review"] = False
    name_of = (speakers.get(to_label) or {}).get("speaker_name")
    for segment in segments:
        if segment.speaker_label == to_label:
            segment.speaker_name = name_of

    class _Meeting:
        pass

    meeting = _Meeting()
    meeting.segments = segments
    meeting.summary = None
    before, after, reindexed = bsm.remerge_meeting(meeting)
    segments = meeting.segments

    for label in sorted(left_behind):
        speakers.pop(label, None)
        print(f"Dropped empty label {label} from speakers.")

    backup = named_path.with_suffix(".json.prerelabel.bak")
    if not backup.exists():
        shutil.copy2(named_path, backup)
    payload["segments"] = [s.to_dict() for s in segments]
    payload["speakers"] = speakers
    atomic_write_json(named_path, payload)
    resynced = bsm.resync_summary_json(directory, segments)
    print(f"\nWrote {named_path} (backup: {backup.name}); "
          f"re-merge {before}->{after}, {reindexed} section(s) reindexed, "
          f"summary.json {resynced if resynced is not None else 'n/a'}.")
    print("Embeddings are NOT recomputed: the new label has no centroid, which "
          "reads as unmeasurable everywhere and never as a mismatch. Re-run "
          "review to name/enrol it, then REPUBLISH if this meeting is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
