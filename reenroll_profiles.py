#!/usr/bin/env python
"""Rebuild the voice profile DB from existing human-reviewed identifications.

Reads each meeting's transcript_named.json, keeps only trusted identifications
(human_review + strong patterns by default), re-extracts per-speaker centroids
using the current EMBEDDING_MODEL, and enrolls them into the profile DB.

Use after changing EMBEDDING_MODEL (or after a schema-version bump) to rebuild
profiles without re-running the full pipeline on every meeting.

Usage:
  python reenroll_profiles.py                    # all meetings, default methods
  python reenroll_profiles.py 2026-02-04-council # specific meeting(s)
  python reenroll_profiles.py --methods human_review   # only human-reviewed
  python reenroll_profiles.py --dry-run          # preview without saving
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load HF token from .env.local if present (matches run_local.py convention)
_env_path = ROOT / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from src import config
from src.checkpoint import PipelineState
from src.diarize import extract_speaker_embeddings
from src.enroll import enroll_speakers, load_profiles, save_profiles
from src.models import Segment, SpeakerMapping
from src.roster import load_roster


TRUSTED_METHODS_DEFAULT = [
    "human_review",        # conf 1.0, user-confirmed
    "voice_profile",       # conf ~0.87+, prior run matched a profile
    "roll_call",           # conf 0.95, clerk-calling pattern
    "chair_recognition",   # conf 0.92, chair-recognizing pattern
    "self_identification", # conf 0.90, "I'm councilman X" pattern
]


def _load_named_transcript(path: Path) -> tuple[list[Segment], dict[str, SpeakerMapping]]:
    with open(path) as f:
        data = json.load(f)
    segments = [Segment.from_dict(s) for s in data.get("segments", [])]
    mappings = {
        label: SpeakerMapping.from_dict(m)
        for label, m in data.get("speakers", {}).items()
    }
    return segments, mappings


def _discover_meetings(requested: list[str]) -> list[Path]:
    if requested:
        return [config.MEETINGS_DIR / m for m in requested]
    return sorted(
        p for p in config.MEETINGS_DIR.iterdir()
        if p.is_dir() and (p / "transcript_named.json").exists()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "meetings", nargs="*",
        help="Meeting IDs (default: every meeting with transcript_named.json)",
    )
    ap.add_argument(
        "--methods", nargs="+", default=TRUSTED_METHODS_DEFAULT,
        help=f"id_methods to enroll from (default: {' '.join(TRUSTED_METHODS_DEFAULT)})",
    )
    ap.add_argument(
        "--min-conf", type=float, default=config.VOICE_MATCH_THRESHOLD,
        help=f"minimum confidence to enroll (default: {config.VOICE_MATCH_THRESHOLD})",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="show what would be enrolled without saving the profile DB",
    )
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set (check env or .env.local)", file=sys.stderr)
        return 1

    trusted = set(args.methods)
    meetings = _discover_meetings(args.meetings)

    print(f"Embedding model:    {config.EMBEDDING_MODEL}")
    print(f"Profile schema:     v{config.PROFILE_SCHEMA_VERSION}")
    print(f"Trusted methods:    {sorted(trusted)}")
    print(f"Min confidence:     {args.min_conf}")
    print(f"Meetings:           {[m.name for m in meetings]}")
    print()

    db = load_profiles()
    initial_count = len(db.profiles)
    print(f"Starting profile DB: {initial_count} profile(s)\n")

    for meeting_dir in meetings:
        meeting_id = meeting_dir.name
        print(f"=== {meeting_id} ===")

        # Load body_slug from pipeline state (Phase 109 — may be None for legacy meetings)
        try:
            state = PipelineState(meeting_dir)
            body_slug = state.body_slug
        except Exception:
            body_slug = None

        roster = None
        if body_slug:
            roster = load_roster(body_slug=body_slug)
            if roster:
                print(f"  Body: {body_slug} ({len(roster.members)} roster members)")
            else:
                print(f"  Body: {body_slug} (no cached roster — using local slugs)")

        named_path = meeting_dir / "transcript_named.json"
        wav_path = meeting_dir / "audio.wav"
        if not named_path.exists():
            print("  no transcript_named.json, skipping")
            continue
        if not wav_path.exists():
            print("  no audio.wav, skipping")
            continue

        segments, mappings = _load_named_transcript(named_path)

        eligible: dict[str, SpeakerMapping] = {}
        for label, m in mappings.items():
            if not m.speaker_name:
                continue
            # "Unknown" is a human-review marker for unidentifiable speakers;
            # enrolling it would create a centroid that false-matches any voice.
            if m.speaker_name.strip().lower() in {"unknown", "unidentified", "n/a"}:
                continue
            if m.id_method not in trusted:
                continue
            if m.confidence < args.min_conf:
                continue
            eligible[label] = m

        print(f"  {len(eligible)}/{len(mappings)} mappings eligible")
        if not eligible:
            continue

        # Show the names we're about to enroll
        for label in sorted(eligible):
            m = eligible[label]
            print(f"    {label} -> {m.speaker_name} ({m.id_method}, conf={m.confidence:.2f})")

        # Only extract embeddings for eligible speakers
        eligible_segs = [s for s in segments if s.speaker_label in eligible]
        print(f"  Extracting embeddings from {len(eligible_segs)} segments...")
        speaker_embeddings = extract_speaker_embeddings(
            wav_path, eligible_segs, hf_token
        )
        print(f"  Got centroids for {len(speaker_embeddings)} speaker(s)")

        before = len(db.profiles)
        db = enroll_speakers(db, speaker_embeddings, eligible, meeting_id, segments, roster=roster)
        delta = len(db.profiles) - before
        print(f"  Enrolled: +{delta} new, total now {len(db.profiles)}")
        print()

    print("=" * 60)
    print(f"Summary: {initial_count} → {len(db.profiles)} profiles")
    for pid, p in sorted(db.profiles.items()):
        print(
            f"  {pid}: {p.display_name}  "
            f"({len(p.meetings_seen)} meeting(s), {len(p.embeddings)} embedding(s))"
        )

    if args.dry_run:
        print("\n(dry run — DB not saved)")
    else:
        save_profiles(db)
        print(f"\nSaved to {config.PROFILES_DIR / config.PROFILE_DB_FILENAME}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
