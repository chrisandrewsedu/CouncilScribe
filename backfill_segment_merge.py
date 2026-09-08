#!/usr/bin/env python3
"""Re-merge adjacent same-speaker segments in already-processed meetings.

The pipeline's segment merge used to run only in memory at export time, so
transcript_named.json (read by the review UI and by GUI publish) kept the
un-merged, fragmented segments. This walks every meeting, merges its segments,
reindexes summary sections from their (stable) times, and rewrites
transcript_named.json + re-exports. It also resyncs the standalone summary.json
checkpoint, which holds a second copy of the same section boundaries.

A meeting needs rewriting when *either* its segments merge down *or* its section
boundaries move. Those are separate conditions: this script originally rewrote
only on a segment-count change, so a meeting whose segments had already been
merged by another path — leaving the summary alone with pre-merge indices — was
reindexed in memory and then skipped. That is how the three summaries whose
boundaries outran their segment count (2026-04-01-ca-courier-stevehiltoninterview,
2026-04-14-pod-save-america-nithya-raman, 2026-06-27-interview) survived the
original sweep.

It does NOT re-publish. It prints which affected meetings are currently live so
you can re-publish them (the fixed publish path will push the merged transcript).

Usage:
    .venv/bin/python backfill_segment_merge.py [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src import config
from src.identify import merge_adjacent_segments
from src.models import Meeting, SummarySection
from src.summary_sections import reindex_sections_from_times, sections_index_into


def reindex_summary_sections(meeting) -> int:
    """Recompute each summary section's start_segment/end_segment from its
    start_time/end_time against ``meeting.segments``. Returns the number of
    sections whose boundaries moved.

    Segment start/end *times* are stable across a segment merge, but the stored
    segment *indices* are not — so after re-merging segments, the section indices
    must be recomputed or they point at the wrong rows. (In a fresh pipeline run
    this isn't needed: the merge happens before summary, so section indices are
    computed against the merged segments natively.) No-op when there's no summary
    or no segments. Times are the source of truth — see src/summary_sections.py
    for how a time maps back onto a segment."""
    return reindex_sections_from_times(_sections_of(meeting),
                                       getattr(meeting, "segments", None))


def _sections_of(meeting):
    summary = getattr(meeting, "summary", None)
    return (getattr(summary, "sections", None) if summary else None) or []


def sections_are_stale(meeting) -> bool:
    """True when this meeting's section boundaries no longer name real segments.

    Measured against the transcript's own segment ids, deliberately: publish
    skips empty-text segments, so ``max(segment_index)`` in meetings.segments can
    sit *below* a perfectly correct boundary. Comparing against the DB reports
    healthy meetings as broken."""
    sections = _sections_of(meeting)
    return bool(sections) and not sections_index_into(sections, meeting.segments)


def merge_would_change(meeting) -> bool:
    """Whether merging this meeting's segments would collapse any of them.

    Asked without committing to the merge, so it must not disturb ``meeting``.
    merge_adjacent_segments mutates the Segment objects it is handed — it extends
    the surviving segment's end_time/text/words and renumbers every segment_id —
    so a copy of the *list* is not enough protection; the objects themselves have
    to be copied. Probing on the live objects silently renumbers them, which then
    makes a perfectly valid summary look stale."""
    probe = copy.deepcopy(meeting.segments)
    return len(merge_adjacent_segments(probe)) != len(meeting.segments)


def remerge_meeting(meeting) -> tuple[int, int, int]:
    """Merge segments + reindex sections where needed.
    Returns (before_count, after_count, sections_reindexed).

    Sections are reindexed when the merge renumbered the segments out from under
    them, or when their stored boundaries have already drifted off the current
    segments. Boundaries that still resolve are left alone: the summariser picked
    them against these segments, which makes them better evidence than anything
    re-derived from times."""
    before = len(meeting.segments)
    was_stale = sections_are_stale(meeting)
    meeting.segments = merge_adjacent_segments(meeting.segments)
    after = len(meeting.segments)
    reindexed = 0
    if after != before or was_stale:
        reindexed = reindex_summary_sections(meeting)
    return before, after, reindexed


def resync_summary_json(meeting_dir: Path, segments) -> int | None:
    """Re-derive the standalone summary.json checkpoint's section boundaries from
    the same times, in place. Returns the number changed, or None when there is
    no readable summary.json with sections.

    summary.json is a second copy of the section boundaries, read back into
    meeting.summary whenever run_local resumes past the SUMMARIZED stage. It is
    reindexed from times directly rather than copied from the embedded copy, so
    it stays correct even if the two copies have diverged."""
    sfile = meeting_dir / "summary.json"
    if not sfile.exists():
        return None
    try:
        summary = json.loads(sfile.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"    (summary.json unreadable for {meeting_dir.name}: {exc})")
        return None
    raw_sections = summary.get("sections") or []
    if not raw_sections:
        return None

    parsed = [SummarySection.from_dict(s) for s in raw_sections]
    changed = reindex_sections_from_times(parsed, segments)
    if changed:
        # Write back only the boundaries, leaving every other key as it was.
        for raw, sec in zip(raw_sections, parsed):
            raw["start_segment"], raw["end_segment"] = sec.start_segment, sec.end_segment
        from src.atomic_io import atomic_write_json
        atomic_write_json(sfile, summary)
    return changed


def republish_notice(touched, live) -> str:
    """What to tell the user about re-publishing the meetings this run changed.

    ``live`` is the set of live slugs, or None when it could not be determined —
    live_published_slugs() is best-effort and swallows any DB failure. That third
    case MUST read as a warning, never as silence: an unreachable DB once left a
    backlog of 13 changed meetings, every one of them live, stale on the public
    site because the run said nothing at all. Note the DB is unreachable by
    default here, since this script does not load .env.local.
    """
    if live is None:
        return ("\n⚠ Could not determine which changed meetings are live — no DB access, so "
                "this is UNKNOWN, not 'nothing to re-publish'. Any that are live are now "
                "stale on the site.\n"
                "  This script does not load .env.local; re-run as:\n"
                "      set -a; . ./.env.local; set +a\n"
                "  or check by hand with gui.publish_api.live_published_slugs().")
    to_republish = [s for s in touched if s in live]
    if to_republish:
        return ("\nRe-publish these (they are live and were changed here):\n"
                + "\n".join(f"    - {slug}" for slug in to_republish))
    return "\nNone of the changed meetings are live — nothing to re-publish."


def _load(meeting_dir: Path):
    named = meeting_dir / "transcript_named.json"
    if not named.exists():
        return None
    try:
        return Meeting.from_dict(json.loads(named.read_text(encoding="utf-8")))
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return None


def backfill(*, dry_run: bool = False, sections_only: bool = False) -> int:
    """Re-merge every meeting whose transcript_named.json has fragmented segments,
    and reindex every meeting whose summary section boundaries have drifted off
    its current segments. Returns the number of meetings changed.

    With ``sections_only``, segments are never rewritten: only meetings whose
    boundaries are stale *and* whose merge is already a no-op are repaired, so
    reindexing is the complete fix for each one. Meetings that still need merging
    are named and deferred — reindexing them now would only have to be redone
    once the merge renumbered their segments."""
    meetings_dir = config.MEETINGS_DIR
    if not meetings_dir.exists():
        print("No meetings directory — nothing to do.")
        return 0

    changed = 0
    deferred: list[str] = []
    touched: list[str] = []
    for mdir in sorted(p for p in meetings_dir.iterdir() if p.is_dir()):
        meeting = _load(mdir)
        if meeting is None or not meeting.segments:
            continue
        if sections_only:
            if not sections_are_stale(meeting):
                continue
            if merge_would_change(meeting):
                deferred.append(mdir.name)
                print(f"  DEFERRED {mdir.name}: needs the segment merge first — "
                      "run without --sections-only")
                continue
        before, after, reindexed = remerge_meeting(meeting)
        # Two independent reasons to rewrite. Gating on the segment count alone
        # is what let already-merged meetings keep stale section boundaries.
        if after == before and not reindexed:
            continue
        changed += 1
        touched.append(mdir.name)
        what = []
        if after != before:
            what.append(f"{before} -> {after} segments")
        if reindexed:
            what.append(f"{reindexed} section boundarie(s) reindexed")
        detail = ", ".join(what)
        if dry_run:
            print(f"  [dry-run] {mdir.name}: {detail}")
            continue
        from src.atomic_io import atomic_write_json
        atomic_write_json(mdir / "transcript_named.json", meeting.to_dict())
        resynced = resync_summary_json(mdir, meeting.segments)
        if resynced:
            detail += f", summary.json resynced ({resynced})"
        try:
            from src.export import export_all
            export_all(meeting, mdir / "exports")
        except Exception as exc:  # exports regenerate at publish; never block
            print(f"    (export refresh skipped for {mdir.name}: {exc})")
        print(f"  {mdir.name}: {detail}")

    if deferred:
        print(f"\n{len(deferred)} meeting(s) deferred — they need the segment merge, "
              "not just a reindex.")

    if not changed:
        print("No meetings needed re-merging.")
        return 0

    # Flag which *changed* meetings are live so the user can re-publish them.
    if not dry_run:
        try:
            from gui.publish_api import live_published_slugs
            live = live_published_slugs()
        except Exception:
            live = None
        print(republish_notice(touched, live))
    print(f"\nDone — {changed} meeting(s) changed.")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing.")
    ap.add_argument("--sections-only", action="store_true",
                    help="Never touch segments; only repair summary section "
                         "boundaries that have drifted off the current segments, "
                         "and only where the segment merge is already a no-op.")
    args = ap.parse_args()
    backfill(dry_run=args.dry_run, sections_only=args.sections_only)


if __name__ == "__main__":
    main()
