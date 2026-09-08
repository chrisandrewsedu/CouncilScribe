#!/usr/bin/env python3
"""CouncilScribe — Local CLI runner for macOS / Linux.

Replaces the Colab notebook with a command-line interface.
All data is stored under ~/CouncilScribe (override with CS_DATA_DIR env var).

Usage:
    python run_local.py --input meeting.mp4 --city Bloomington --date 2026-02-10
    python run_local.py --input "https://catstv.net/..." --city Bloomington --date 2026-02-10
    python run_local.py --browse-catstv
    python run_local.py --resume 2026-02-10-regular
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import UUID

# Ensure src/ is importable when running from the repo root
_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

# Load .env.local if present (HF_TOKEN, CS_DATA_DIR, etc.)
_env_file = _REPO_DIR / ".env.local"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from src import config  # lightweight; must follow .env.local load (CS_DATA_DIR)
from src.atomic_io import atomic_write_json
from src.event_kinds import EVENT_KINDS, INTERVIEW_KINDS, validate_event_kind
from src.crec_identify import parse_crec_arg
from src.house_cdn import resolve_session

#: Diarizer backends Stage 2.5's centroid merge must never run against.
#: `api`/`vibevoice` produce their own well-separated clustering; `api-recluster`
#: exists SPECIFICALLY to carry a hand-repaired clustering forward unchanged —
#: running the same merge-by-similarity over it risks silently re-merging the
#: very people that repair pulled apart, while keeping the "+recluster"
#: provenance stamp as if nothing had touched it.
_MERGE_UNSUPPORTED_DIARIZERS = ("api", "vibevoice", "api-recluster")


def _merge_stage_skipped(diarizer: str) -> bool:
    """True when Stage 2.5's centroid merge must not run for this backend."""
    return diarizer in _MERGE_UNSUPPORTED_DIARIZERS


def _resolve_chunk_minutes(args, event_kind) -> int:
    """Chunk size for this run, after the meeting-kind gate.

    Chunking is only validated for single-room civic meetings and legislative
    floor sessions (config.DIARIZE_CHUNK_EVENT_KINDS). On dense, many-voice,
    fast-turn audio pyannote merges speakers INSIDE a window, which no
    cross-window threshold can repair, so those kinds take the single-pass path.
    An explicit --diarize-chunk-minutes still wins — it is the escape hatch for
    deliberate experiments — but it says so, because a silent override would
    make a bad speaker count hard to explain later.
    """
    from src.modal_compute import chunk_minutes_for_kind

    requested = getattr(args, "diarize_chunk_minutes", None)
    if requested is not None:
        if requested > 0 and chunk_minutes_for_kind(event_kind, requested) == 0:
            print(f"  ! --diarize-chunk-minutes={requested} overrides the "
                  f"meeting-kind gate for event_kind={event_kind!r}, which is "
                  "NOT a validated kind for chunking. Watch the speaker count "
                  "at review: chunking can merge speakers on dense audio.")
        return requested
    allowed = chunk_minutes_for_kind(event_kind, config.DIARIZE_CHUNK_MINUTES)
    if config.DIARIZE_CHUNK_MINUTES > 0 and allowed == 0:
        print(f"  Chunked diarization skipped: event_kind={event_kind!r} is not "
              "a validated kind for chunking, so this runs single-pass "
              "(slower, and the accuracy we know).")
    return allowed


def should_run_llm(skip_llm: bool, crec_request, event_kind=None) -> bool:
    """Whether to run Layer-3 LLM speaker identification.

    Off entirely when ``config.SPEAKER_ID_LLM_ENABLED`` is False (the shipped
    default as of 2026-08-06 — see src/config.py): an 88-interview eval measured
    ~13-16% correct-name coverage across five models, and in practice names come
    from human review. This is checked first and short-circuits every other
    condition below.

    When the switch is on, off when --skip-llm is set, off on a Congressional
    Record run (``crec_request`` truthy): CREC is authoritative for who spoke,
    and the local LLM hallucinates congressional names, so an unresolved floor
    speaker should go to review as 'unidentified' rather than get a garbage
    guess.

    Also off for interview-kind meetings (``event_kind in INTERVIEW_KINDS`` —
    news_clip/press_conference/podcast): the same eval found the anchor rule
    requires names spoken in the transcript and interview guests are rarely
    full-named on air. Names come from review instead. Civic kinds and
    debates/forums are unaffected.
    """
    if not config.SPEAKER_ID_LLM_ENABLED:
        return False
    return (not skip_llm) and (crec_request is None) and (event_kind not in INTERVIEW_KINDS)


def _validate_diarizer_compute(args) -> None:
    if args.diarizer == "vibevoice" and args.compute != "modal":
        raise ValueError("--diarizer vibevoice requires --compute modal")
    if args.diarizer == "api-recluster":
        _validate_recluster_provenance(args)


def _recluster_meeting_id(args) -> str | None:
    """Best-effort meeting id for an `api-recluster` run, before the normal
    --resume / metadata-resolve dance has run (this validation happens first,
    at argparse time). `api-recluster` only ever makes sense against an
    already-existing meeting, so --resume or an explicit --meeting-id is what
    every real invocation supplies; a date+meeting-type pair is included too
    since that is how a fresh id would otherwise be spelled."""
    meeting_id = getattr(args, "resume", None) or getattr(args, "meeting_id", None)
    if meeting_id:
        return meeting_id
    date = getattr(args, "date", None)
    meeting_type = getattr(args, "meeting_type", None)
    if date and meeting_type:
        return f"{date}-{meeting_type.lower().replace(' ', '-')}"
    return None


def _validate_recluster_provenance(args) -> None:
    """Refuse `--diarizer api-recluster` unless diarization is already on disk.

    `api-recluster` is provenance-only (see its --diarizer help text): it
    stamps `diarization_model = "pyannote/ai-precision-2+recluster"` without
    ever running Precision-2 itself, on the assumption that diarization.json/
    embeddings.json were already hand-replaced by an offline re-clustering. If
    they are not there, the Stage 2 diarizer dispatch falls through to plain
    OSS pyannote 3.1 and the run still stamps the recluster provenance string
    — a false claim about how the shipped labels were produced, exactly the
    error this value exists to prevent.
    """
    meeting_id = _recluster_meeting_id(args)
    if not meeting_id:
        raise ValueError(
            "--diarizer api-recluster requires an existing meeting "
            "(--resume MEETING_ID, or --meeting-id/--date+--meeting-type "
            "naming one) with diarization.json and embeddings.json already "
            "on disk — it replaces clustering only, never runs diarization "
            "from scratch."
        )
    meeting_dir = config.MEETINGS_DIR / meeting_id
    missing = [
        name for name in ("diarization.json", "embeddings.json")
        if not (meeting_dir / name).exists()
    ]
    if missing:
        raise ValueError(
            f"--diarizer api-recluster requires diarization.json and "
            f"embeddings.json to already exist for meeting {meeting_id!r} "
            f"(missing {', '.join(missing)} under {meeting_dir}). "
            "api-recluster is provenance-only: it never runs diarization "
            "from scratch."
        )


def _diarization_model_name(diarizer: str) -> str:
    if diarizer == "api":
        return "pyannote/ai-precision-2"
    if diarizer == "api-recluster":
        # Precision-2's segmentation kept, its clustering replaced by a
        # per-turn re-clustering. Distinct from "api" because the labels this
        # meeting ships did NOT come out of Precision-2.
        return "pyannote/ai-precision-2+recluster"
    if diarizer == "vibevoice":
        from src.vibevoice import VIBEVOICE_MODEL_ID, VIBEVOICE_MODEL_REVISION

        return f"{VIBEVOICE_MODEL_ID}@{VIBEVOICE_MODEL_REVISION}"
    return config.DIARIZATION_MODEL


# ---------------------------------------------------------------------------
# Phase 109: pre-Stage-1 fail-fast guard (CSMEETING-02, D-07/D-08/D-09/D-13)
# ---------------------------------------------------------------------------

_BODY_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def ensure_body_roster_cached(body_slug: Optional[str]) -> None:
    """Phase 109 fail-fast guard: verify {body_slug}.json exists in the roster cache.

    Implements CSMEETING-02:
      - D-05: if body_slug is None/empty, return silently (legacy path).
      - D-07: runs BEFORE Stage 1, after argparse + metadata resolve.
      - D-08: on missing cache, print 2-line stderr error + sys.exit(2).
      - D-09: stale cache (>30 days) is NOT a fail-fast — file must merely exist.
      - D-10: behaves identically on resume after cache delete.
      - T-109-03: validates slug shape before composing filesystem paths.
    """
    if not body_slug:
        return  # D-05 legacy path

    # T-109-03: reject path-traversal / shell metacharacters BEFORE filesystem join.
    if not _BODY_SLUG_RE.match(body_slug):
        print(
            f'ERROR: Invalid body slug "{body_slug}" — must match '
            f'[a-z0-9][a-z0-9_-]{{0,63}} (1-64 chars)',
            file=sys.stderr,
        )
        sys.exit(2)

    cache_path = config.CONFIG_DIR / "rosters" / f"{body_slug}.json"
    if not cache_path.exists():
        # D-08: exact 2-line error. D-13: literal ~-path string, do NOT expand CONFIG_DIR.
        print(
            f'ERROR: Body "{body_slug}" has no cached roster at '
            f'~/CouncilScribe/config/rosters/{body_slug}.json',
            file=sys.stderr,
        )
        print(
            f'Run: python refresh_roster.py --body {body_slug}',
            file=sys.stderr,
        )
        sys.exit(2)
    # D-09: staleness is checked later inside load_roster() — not our concern here.


def _reconcile_clip_window(
    state: "PipelineState",
    cli_start: Optional[float],
    cli_end: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """Resolve and persist the clip window, mirroring the --body persist pattern.

    - First run with --clip: persist to state.
    - Resume with no --clip: read from state.
    - Re-pass with the SAME window: no-op.
    - Re-pass with a DIFFERENT window BEFORE ingest: overwrite (no audio is cut
      yet, so a corrected window is harmless).
    - Re-pass with a DIFFERENT window AFTER ingest: hard error (audio.wav is
      already cut and cannot be re-clipped in place; the operator must use a
      fresh --meeting-id).
    """
    from src.checkpoint import PipelineStage

    persisted = (state.clip_start_seconds, state.clip_end_seconds)
    requested = (cli_start, cli_end)

    if cli_start is None and cli_end is None:
        return persisted

    if persisted == (None, None):
        state.clip_start_seconds, state.clip_end_seconds = requested
        state.save()
        return requested

    if requested == persisted:
        return persisted

    # Different window: harmless to overwrite until the audio has been cut.
    if state.completed_stage < PipelineStage.INGESTED:
        state.clip_start_seconds, state.clip_end_seconds = requested
        state.save()
        return requested

    s0, e0 = persisted
    print(
        f"ERROR: this meeting was already clipped to {s0}-{e0}s. The cut audio "
        f"cannot be re-clipped in place. To use a different window, process the "
        f"source into a new --meeting-id.",
        file=sys.stderr,
    )
    sys.exit(2)


def _load_summary_checkpoint(meeting, summary_path: Path) -> str:
    """Resolve the summary on a resumed run, preferring the embedded copy.

    The summary is stored twice: as the standalone summary.json stage checkpoint
    and inside transcript_named.json, from which the identification stage has
    already loaded it into `meeting`. The embedded copy is authoritative — the
    segment-merge and boundary-snap backfills renumber segments and rewrite only
    transcript_named.json, which leaves summary.json holding section
    start/end_segment values that point at the pre-backfill numbering. Preferring
    the checkpoint here published stale section boundaries for live meetings, so
    the embedded copy wins and a drifted checkpoint is re-synced from it.

    summary.json is still read when the transcript carries no embedded summary
    (meetings summarized before that copy existed, or a run whose transcript
    re-save didn't land), and is still the SUMMARIZED stage marker that
    checkpointing and --rewind-to key on, so it is never left absent.

    Returns a one-line description of the outcome for the caller to print.
    """
    from src.models import MeetingSummary

    if meeting.summary is not None:
        embedded = meeting.summary.to_dict()
        on_disk = None
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    # Round-trip through the model so an older on-disk spelling
                    # (`key_decisions` for `highlights`) isn't read as drift.
                    on_disk = MeetingSummary.from_dict(json.load(f)).to_dict()
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                on_disk = None  # unusable checkpoint: rewrite it from the transcript
        count = len(meeting.summary.sections)
        if on_disk == embedded:
            return f"Loaded summary ({count} sections)"
        atomic_write_json(summary_path, embedded)
        return (f"Loaded summary ({count} sections) from the transcript; "
                f"re-synced the out-of-date summary.json checkpoint")

    if summary_path.exists():
        with open(summary_path, "r") as f:
            meeting.summary = MeetingSummary.from_dict(json.load(f))
        return (f"Loaded summary ({len(meeting.summary.sections)} sections) from "
                f"summary.json (the transcript has no embedded summary)")

    return "No summary on disk — nothing to load."


def _list_cached_rosters() -> list[tuple[str, str]]:
    """Return [(body_slug, label), ...] for each cached per-body roster.

    Scans CONFIG_DIR/rosters/*.json, sorted by filename. label is
    "{body_key} ({N} members) [{slug}]", falling back to the slug if the
    file can't be parsed.
    """
    rosters_dir = config.CONFIG_DIR / "rosters"
    out: list[tuple[str, str]] = []
    if not rosters_dir.exists():
        return out
    for path in sorted(rosters_dir.glob("*.json")):
        slug = path.stem
        label = slug
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            body_key = data.get("body_key") or slug
            count = len(data.get("politicians", []))
            label = f"{body_key} ({count} members) [{slug}]"
        except Exception:
            pass
        out.append((slug, label))
    return out


def _should_prompt_roster(
    *,
    cli_body,
    persisted_body,
    roster_choice,
    identified: bool,
    isatty: bool,
) -> bool:
    """Decide whether to show the interactive roster chooser.

    Prompt only on a fresh interactive run where the operator hasn't already
    chosen a roster: TTY attached, no --body, no persisted body_slug, no prior
    roster_choice, and Stage 4 (identification) not already complete.
    """
    return (
        isatty
        and not cli_body
        and not persisted_body
        and roster_choice is None
        and not identified
    )


def _prompt_roster_choice() -> tuple[Optional[str], str]:
    """Interactive roster chooser. Returns (body_slug_or_None, marker).

    marker is the value to persist in state.roster_choice:
      - the slug itself for a cached roster (body_slug is also returned)
      - "__legacy__" for the legacy council_roster.json
      - "__none__" for no roster (also the bare-Enter default)

    Caller is responsible for only invoking this when interactive
    (see _should_prompt_roster).
    """
    cached = _list_cached_rosters()
    legacy_path = config.CONFIG_DIR / "council_roster.json"
    has_legacy = legacy_path.exists()

    print("=" * 60)
    print("ROSTER SELECTION")
    print("=" * 60)
    print("  Which council roster should guide speaker identification?")
    print()

    # options[i] = ("cached"|"legacy"|"none", slug_or_None)
    options: list[tuple[str, Optional[str]]] = []
    n = 0
    for slug, label in cached:
        n += 1
        print(f"  {n}. {label}")
        options.append(("cached", slug))

    if has_legacy:
        legacy_label = "legacy council_roster.json"
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = f"{data.get('city', '')} {data.get('body', '')}".strip()
            members = len(data.get("members", []))
            legacy_label = f"{name or 'council_roster.json'} (legacy, {members} members)"
        except Exception:
            pass
        n += 1
        print(f"  {n}. {legacy_label}")
        options.append(("legacy", None))

    n += 1
    print(f"  {n}. No roster (skip name correction)")
    options.append(("none", None))
    print()

    while True:
        choice = input(f"  Select [1-{n}] (default {n} = no roster): ").strip()
        if choice == "":
            kind, value = "none", None
            break
        try:
            idx = int(choice)
        except ValueError:
            print("  Please enter a number.")
            continue
        if 1 <= idx <= len(options):
            kind, value = options[idx - 1]
            break
        print(f"  Out of range. Enter 1-{n}.")

    if kind == "cached":
        return value, value
    if kind == "legacy":
        return None, "__legacy__"
    return None, "__none__"


def _resolve_roster(effective_body_slug: Optional[str], roster_choice: Optional[str]) -> Optional["Roster"]:
    """Resolve the Roster (or None) for Stage 4 given the meeting's state.

    - body_slug set      → load that body's cached roster.
    - roster_choice legacy → bare load_roster() (legacy council_roster.json).
    - "__none__" / unchosen → no roster (no name correction). This is the
      non-interactive default since the chooser only runs interactively.
    """
    # Local import so tests can patch src.roster.load_roster.
    from src.roster import load_roster

    if effective_body_slug:
        return load_roster(body_slug=effective_body_slug)
    if roster_choice == "__legacy__":
        return load_roster()
    return None


def _resolve_race_id(
    state: "PipelineState",
    cli_race_id: str | None,
) -> str | None:
    if cli_race_id is not None:
        try:
            cli_race_id = str(UUID(cli_race_id))
        except ValueError as exc:
            raise RuntimeError("--race-id must be a UUID") from exc

    if state.race_id and cli_race_id and state.race_id != cli_race_id:
        raise RuntimeError(
            f"Meeting already linked to race {state.race_id}; "
            "changing races requires editing the meeting metadata explicitly"
        )

    if cli_race_id and state.race_id is None:
        state.race_id = cli_race_id
        state.save()

    return state.race_id


def get_hf_token() -> str:
    """Resolve HuggingFace token from env, cached login, or prompt."""
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # 2. Cached token from `huggingface-cli login`
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            return token
    except Exception:
        pass

    # 3. Prompt user
    print("\nHuggingFace token required (for pyannote models).")
    print("Get one at: https://huggingface.co/settings/tokens")
    print("Accept the pyannote model agreements:")
    print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("  https://huggingface.co/pyannote/embedding")
    token = input("\nHF Token: ").strip()
    if not token:
        print("No token provided. Exiting.")
        sys.exit(1)
    return token


def browse_catstv(search_url: str | None = None, limit: int = 25) -> dict | None:
    """Interactive CATS TV meeting browser. Returns selected meeting dict or None."""
    from src.download import fetch_catstv_meetings, display_catstv_meetings

    print("Fetching CATS TV meeting archive...")
    meetings = fetch_catstv_meetings(search_url)
    print(f"Found {len(meetings)} meetings.\n")
    display_catstv_meetings(meetings, limit=limit)

    print()
    choice = input("Enter meeting number (or 'q' to cancel): ").strip()
    if choice.lower() == "q":
        return None

    try:
        idx = int(choice)
        if 0 <= idx < len(meetings):
            return meetings[idx]
        print(f"Invalid: must be 0-{len(meetings) - 1}")
        return None
    except ValueError:
        print("Invalid input.")
        return None


def human_review(mappings: dict) -> dict:
    """Interactive prompt for correcting speaker identifications."""
    from src.models import SpeakerMapping

    review_needed = [m for m in mappings.values() if m.needs_review]
    if not review_needed:
        return mappings

    print(f"\n  {len(review_needed)} speaker(s) flagged for review:")
    for m in review_needed:
        name = m.speaker_name or "(unidentified)"
        print(f"    {m.speaker_label} -> {name} (conf={m.confidence:.2f})")

    # Skip interactive prompt if stdin is not a terminal (e.g. background task)
    if not sys.stdin.isatty():
        print("  (non-interactive mode — skipping review)")
        return mappings

    print("\n  Enter corrections as: SPEAKER_00=Mayor Johnson, SPEAKER_03=Clerk Smith")
    print("  Or press Enter to skip:")
    corrections = input("  > ").strip()

    if corrections:
        for pair in corrections.split(","):
            pair = pair.strip()
            if "=" in pair:
                label, name = pair.split("=", 1)
                label, name = label.strip(), name.strip()
                if label in mappings:
                    mappings[label].speaker_name = name
                    mappings[label].confidence = 1.0
                    mappings[label].id_method = "human_review"
                    mappings[label].needs_review = False
                    print(f"    Updated: {label} -> {name}")
        print("  Corrections applied.")
    else:
        print("  No corrections. Continuing.")

    return mappings


# find_video_file and _attach_thumbnail live in src.thumbnail so the GUI publish
# path (gui/publish_api) can share the exact same thumbnail step.
from src.thumbnail import find_video_file, attach_thumbnail as _attach_thumbnail


def _format_delete_summary(result: dict) -> str:
    lines = [f"Meeting: {result['meeting_id']}  [{result['status']}]"]
    if result.get("db_deleted"):
        rows = ", ".join(f"{t}: {n}" for t, n in result.get("rows_deleted", {}).items())
        lines.append(f"  live DB rows deleted — {rows}")
    else:
        lines.append("  live DB: no rows deleted (unpublished or DB not configured)")
    lines.append(f"  local folder deleted: {'yes' if result.get('local_deleted') else 'no'}")
    quotes = result.get("quotes_found") or []
    lines.append(f"  derived quotes found (NOT deleted): {len(quotes)}")
    for q in quotes:
        lines.append(f"    - quote id={q['id']} politician={q['politician_id']} topic={q['topic_key']}")
    if result.get("profile_contamination"):
        lines.append("  ⚠ this meeting contributed to voice profiles; its embeddings remain in "
                     "speaker_profiles.pkl. Run reenroll_profiles.py to rebuild if needed.")
    return "\n".join(lines)


def _review_audio_path(meeting_dir: Path) -> str | None:
    """Clip-local audio for terminal review: audio.wav if present, else the
    compressed audio.opus left after media cleanup, else None."""
    for name in ("audio.wav", "audio.opus"):
        candidate = meeting_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def _review_seek(start_time: float, video_offset: float = 0.0) -> float:
    """Seek position for a review clip: 3s of lead-in, plus the clip offset.

    A clipped meeting's segment times are clip-local, but the local video file is
    the FULL source — so seeking the video must add clip_start (passed as
    video_offset). The clip audio (audio.wav) is already clip-local and passes 0.
    """
    return max(0.0, start_time - 3.0) + video_offset


def play_video_clip(
    video_path: str,
    start_time: float,
    duration: float = 15.0,
    title: str = "",
    video_offset: float = 0.0,
) -> None:
    """Play a video clip using ffplay starting at the given timestamp.

    Args:
        video_path: Path to the video file.
        start_time: Clip-local start time in seconds.
        duration: Duration to play in seconds.
        title: Window title.
        video_offset: Added to the seek for a clipped meeting whose video file is
            the full source (so clip-local times map to the right footage).
    """
    # Start a few seconds early to give visual context
    seek = _review_seek(start_time, video_offset)

    cmd = [
        "ffplay",
        "-ss", str(seek),
        "-t", str(duration),
        "-autoexit",
        "-loglevel", "quiet",
    ]
    if title:
        cmd += ["-window_title", title]
    cmd.append(video_path)

    print(f"    Playing video clip ({duration:.0f}s from {int(seek // 60):02d}:{int(seek % 60):02d})...")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("    ffplay not found — install ffmpeg to enable video playback")


def play_speaker_clip(
    video_path: str | None,
    audio_path: str | None,
    start_time: float,
    duration: float = 40.0,
    title: str = "",
    video_offset: float = 0.0,
):
    """Play a looping clip of a speaker WITHOUT blocking, returning the player handle.

    Video if available, else the audio segment (ffplay -nodisp). Loops (`-loop 0`)
    so the clip stays up while the operator types the name; the caller stops it via
    _stop_player. Returns the subprocess.Popen handle, or None if there's no media
    or ffplay isn't installed.

    video_offset (clip_start for a clipped meeting) is added ONLY when playing the
    full-source video; the clip-local audio path ignores it.
    """
    media = video_path or audio_path
    if not media:
        print("    No media to play (no video or audio found).")
        return None

    seek = _review_seek(start_time, video_offset if video_path else 0.0)
    cmd = ["ffplay", "-ss", str(seek), "-t", str(duration), "-loop", "0", "-loglevel", "quiet"]
    if not video_path:
        cmd.append("-nodisp")
    if title:
        cmd += ["-window_title", title]
    cmd.append(media)

    kind = "video" if video_path else "audio"
    print(f"    Playing {kind} clip ({duration:.0f}s from {int(seek // 60):02d}:{int(seek % 60):02d}) "
          f"— looping; press V for the next clip, R to replay, or type a name / Enter to move on...")
    try:
        # Detach from the terminal's stdin so the looping player can never
        # contend with the review prompt's input().
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        print("    ffplay not found — install ffmpeg to enable clip playback")
        return None


def _stop_player(proc) -> None:
    """Terminate a clip player started by play_speaker_clip (tolerant of None/exited)."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
    except Exception:
        pass


def _focus_terminal() -> None:
    """Best-effort: return keyboard focus to the terminal (macOS only).

    ffplay's video window grabs keyboard focus on launch, so review shortcut
    keys reach the player instead of the prompt. On macOS we re-activate the
    terminal app via osascript. This is a deliberately thin, best-effort stopgap
    (the durable fix is a non-terminal review UI): it no-ops off macOS or when
    the terminal app is unknown, and never raises.
    """
    if sys.platform != "darwin":
        return
    app = {
        "Apple_Terminal": "Terminal",
        "iTerm.app": "iTerm2",
        "vscode": "Code",
    }.get(os.environ.get("TERM_PROGRAM", ""))
    if not app:
        return
    try:
        subprocess.Popen(
            ["osascript", "-e", f'tell application "{app}" to activate'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _read_one_key() -> str | None:
    """Read a single keypress from stdin in cbreak mode (no Enter needed).

    Returns the character read, or None when raw mode is unavailable (no
    termios, or stdin is not a real TTY — e.g. under pytest or a pipe), so the
    caller can fall back to a line-based input().
    """
    try:
        import termios
        import tty
    except ImportError:
        return None  # non-Unix (e.g. Windows): no cbreak
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None  # not a real TTY (captured stdin, pipe, etc.)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_review_command(prompt: str = "", *, refocus: bool = False) -> str:
    """Read one review command, preferring instant single keypresses.

    On a real TTY, command keys fire immediately (no Enter): V/R/Y/M/Q, a digit
    jumps to that clip (returned as the existing ``v<n>`` token), Enter skips.
    Any other printable key begins a typed name and we fall back to a line read
    for the rest; ``/`` starts a name explicitly (for names that begin with a
    reserved command letter, e.g. "Maria"). When raw mode is unavailable the
    whole thing degrades to input(), so scripted tests and piped runs are
    unaffected.

    refocus: when a clip player is up, re-assert terminal focus first so the
    keypress lands in the terminal and not ffplay's video window.
    """
    if refocus:
        _focus_terminal()
    sys.stdout.write(prompt)
    sys.stdout.flush()

    key = _read_one_key()
    if key is None:
        return input()  # no raw mode — prompt already shown; read a line

    if key in ("\r", "\n", ""):
        sys.stdout.write("\n")
        return ""
    lower = key.lower()
    if lower in ("v", "r", "y", "m", "q", "u", "x", "b"):
        sys.stdout.write(key + "\n")
        return lower
    if key.isdigit():
        sys.stdout.write(key + "\n")
        return "v" + key  # reuse the loop's "v<n>" jump parser
    # Anything else begins a typed name. "/" is an explicit, char-less start.
    prefix = "" if key == "/" else key
    sys.stdout.write(prefix)
    sys.stdout.flush()
    rest = input()
    return prefix + rest


def free_gpu_memory():
    """Release GPU memory (CUDA or MPS)."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _expand_house_floor(args) -> None:
    """Resolve --house-floor DATE into the standard pipeline args + stash the source."""
    if getattr(args, "_house_source", None) is not None:
        return  # already expanded — idempotent across the main() and run_pipeline calls
    date = getattr(args, "house_floor", None)
    if not date:
        return
    source = resolve_session(date)
    if source is None:
        raise SystemExit(
            f"No House floor session resolvable for {date} "
            f"(not in session, or CDN has no HLS stream)."
        )
    args.input = source.manifest_url
    args.event_kind = "floor"
    args.meeting_type = "House Floor"
    args.date = date
    if not getattr(args, "title", None):
        args.title = source.title
    args.congressional_record = [date, "house"]  # matches --congressional-record nargs=2 (DATE CHAMBER)
    args._house_source = source


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full 6-stage pipeline."""
    _expand_house_floor(args)
    import numpy as np
    import torch

    from src import config
    from src.checkpoint import PipelineStage, PipelineState, ensure_drive_structure
    from src.models import Meeting, ProcessingMetadata

    print(f"Data directory: {config.DRIVE_ROOT}")

    # --- Resolve audio source ---
    audio_path = args.input
    if not audio_path:
        print("Error: --input is required (file path or URL). Use --browse-catstv to pick a meeting.")
        sys.exit(1)

    # --- HuggingFace token ---
    hf_token = get_hf_token()
    print(f"HuggingFace token: ...{hf_token[-4:]}")

    # --- Device info ---
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU: Apple Silicon (MPS)")
    else:
        print("GPU: None (CPU mode — slower, smaller model)")

    # --- Initialize meeting ---
    meeting_id = args.meeting_id or f"{args.date}-{args.meeting_type.lower().replace(' ', '-')}"
    # The meeting id IS the directory basename (ensure_drive_structure makes
    # MEETINGS_DIR/<meeting_id>). A separator would create a nested dir whose
    # name no longer matches meeting_id, drifting the calibration leave-one-out
    # key away from the stamped enrollment provenance. Reject it up front.
    if not _is_simple_meeting_id(meeting_id):
        print(
            f"ERROR: invalid meeting id {meeting_id!r} — it must be a single path "
            f"component (no '/' or '\\'). Pass a plain --meeting-id like "
            f"'2026-02-10-regular-session'.",
            file=sys.stderr,
        )
        sys.exit(2)
    meeting_dir = ensure_drive_structure(meeting_id)
    state = PipelineState(meeting_dir)

    # Resolve the clip window (parse the flag, reconcile against persisted state).
    from src.clip import parse_clip_time
    _cli_clip_start = _cli_clip_end = None
    if getattr(args, "clip", None):
        try:
            _cli_clip_start = parse_clip_time(args.clip[0])
            _cli_clip_end = parse_clip_time(args.clip[1])
        except ValueError as exc:
            print(f"ERROR: invalid --clip value: {exc}", file=sys.stderr)
            sys.exit(2)
        if _cli_clip_end <= _cli_clip_start:
            print("ERROR: --clip END must be greater than START", file=sys.stderr)
            sys.exit(2)
    clip_start, clip_end = _reconcile_clip_window(state, _cli_clip_start, _cli_clip_end)
    if clip_start is not None:
        print(f"Clip window: {clip_start}-{clip_end}s (transcribing this slice only)",
              file=sys.stderr)

    # Persist pipeline metadata so --resume can recover without transcript_named.json.
    _state_dirty = False
    if state.event_kind != args.event_kind and args.event_kind is not None:
        state.event_kind = args.event_kind
        _state_dirty = True
    if state.city != args.city and args.city is not None:
        state.city = args.city
        _state_dirty = True
    if state.date != args.date and args.date is not None:
        state.date = args.date
        _state_dirty = True
    if state.meeting_type != args.meeting_type and args.meeting_type is not None:
        state.meeting_type = args.meeting_type
        _state_dirty = True
    if getattr(args, "guest", None) is not None and state.guest != args.guest:
        state.guest = args.guest
        _state_dirty = True
    # Record the normalized source key so the GUI can detect duplicate grabs.
    if audio_path and not state.source_key:
        from src.source_key import source_key as _source_key
        _sk = _source_key(str(audio_path))
        if _sk and state.source_key != _sk:
            state.source_key = _sk
            _state_dirty = True
    if _state_dirty:
        state.save()

    # Apply --redo rewind now that meeting_dir is known (works with both
    # --resume and --input; the --resume branch may have already done this,
    # but rewind_to is idempotent so the double-call is harmless).
    if getattr(args, "redo", None):
        _redo_map = {
            "ingest":     PipelineStage.INGESTED,
            "diarize":    PipelineStage.DIARIZED,
            "transcribe": PipelineStage.TRANSCRIBED,
            "identify":   PipelineStage.IDENTIFIED,
            "summary":    PipelineStage.SUMMARIZED,
            "all":        PipelineStage.INGESTED,
        }
        state.rewind_to(_redo_map[args.redo])
        print(f"Re-running from stage: {args.redo}")

    # ── Phase 109: resolve effective body_slug (D-01..D-06, D-11) ──
    cli_body = getattr(args, "body", None)
    persisted_body = state.body_slug
    force_retag = getattr(args, "force_retag", False)

    if cli_body and persisted_body and cli_body != persisted_body and not force_retag:
        # D-02: hard error on mismatch
        print(
            f"ERROR: Meeting already tagged as \"{persisted_body}\". "
            f"Pass --body {persisted_body} to continue, or add --force-retag "
            f"to change the body (this will re-run Stages 4-7).",
            file=sys.stderr,
        )
        sys.exit(2)

    if cli_body and persisted_body and cli_body != persisted_body and force_retag:
        # D-03 + D-04 + D-11: overwrite, rewind, clear stale pre_ids
        print(f"  Force-retag: {persisted_body} → {cli_body}", file=sys.stderr)
        state.body_slug = cli_body
        state.rewind_for_retag()
    elif cli_body and not persisted_body:
        # D-01: first run persists
        if force_retag:
            print(
                f"  --force-retag on untagged meeting: behaving as first-run "
                f"persist of {cli_body}",
                file=sys.stderr,
            )
        state.body_slug = cli_body
        state.save()
    elif not cli_body and persisted_body and force_retag:
        # Should be unreachable: D-12 enforced at argparse (line 1835).
        raise AssertionError("--force-retag without --body bypassed D-12 guard")
    # else: D-05 (no flag, no persisted — legacy) or D-06 (no flag, persisted — silent read)

    # Roster chooser: on a fresh interactive run with no --body, ask which
    # roster should guide Stage 4 instead of silently using the legacy file.
    # Non-interactive runs (no TTY) fall through to no roster (handled in
    # _resolve_roster) unless --body was passed.
    if _should_prompt_roster(
        cli_body=cli_body,
        persisted_body=persisted_body,
        roster_choice=state.roster_choice,
        identified=state.is_complete(PipelineStage.IDENTIFIED),
        isatty=sys.stdin.isatty() and not getattr(args, "batch_mode", False),
    ):
        chosen_slug, marker = _prompt_roster_choice()
        state.roster_choice = marker
        if chosen_slug:
            state.body_slug = chosen_slug
        state.save()

    effective_body_slug = state.body_slug  # used by Plan 02 guard + Plan 03 Stage 4

    if effective_body_slug:
        # D-01 / D-06: single info line for operator visibility
        print(f"Body: {effective_body_slug}", file=sys.stderr)

    # Phase 109 D-07: fail fast if tagged meeting has no cached roster.
    # Must run before Stage 1 ingestion so operators don't burn GPU on a bad run.
    ensure_body_roster_cached(effective_body_slug)
    # Same fail-fast rationale: validate --congressional-record before Stage 1 so a
    # bad DATE/CHAMBER doesn't waste ingest/diarize/transcribe. Reused at Stage 4.
    crec_request = parse_crec_arg(getattr(args, "congressional_record", None))
    effective_race_id = _resolve_race_id(
        state,
        getattr(args, "race_id", None),
    )

    meeting = Meeting(
        meeting_id=meeting_id,
        city=args.city,
        date=args.date,
        meeting_type=args.meeting_type,
        title=args.title.strip() if args.title and args.title.strip() else None,
        event_kind=args.event_kind,
        event_orgs=getattr(args, "event_org", None) or [],
        race_id=effective_race_id,
        audio_source=str(audio_path),
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
    )

    _house_source = getattr(args, "_house_source", None)
    if _house_source is not None:
        meeting.audio_source = _house_source.citation_url            # DB source_url = live.house.gov page
        meeting.processing_metadata.source_audio_url = _house_source.manifest_url  # playback = HLS

    display_title = meeting.title or " ".join(
        part for part in (meeting.city, meeting.meeting_type) if part
    )
    print(f"\nMeeting: {display_title} ({meeting.date})")
    print(f"Meeting ID: {meeting_id}")
    print(f"Directory: {meeting_dir}")
    if state.completed_stage > PipelineStage.NOT_STARTED:
        print(f"Resuming from checkpoint: stage {state.completed_stage.name} ({state.completed_stage.value}/6)")
    print()

    num_speakers = args.num_speakers if args.num_speakers > 0 else None
    wav_path = meeting_dir / "audio.wav"

    # ======================================================================
    # Stage 1: Ingest
    # ======================================================================
    print("=" * 60)
    print("STAGE 1: Audio Ingestion")
    print("=" * 60)

    vtt_path = meeting_dir / "captions.vtt"

    if state.is_complete(PipelineStage.INGESTED):
        print("  Already complete. Skipping.")
        from src.audio_utils import get_audio_duration
        meeting.duration_seconds = get_audio_duration(wav_path)
    else:
        from src.ingest import normalize_audio

        t0 = time.time()
        metadata = normalize_audio(
            audio_path, wav_path,
            noise_reduce=args.noise_reduce,
            cookies_file=getattr(args, "cookies", None),
            clip_start=clip_start,
            clip_end=clip_end,
        )
        elapsed = time.time() - t0
        meeting.duration_seconds = metadata["duration_seconds"]
        if metadata.get("source_title"):
            meeting.processing_metadata.source_title = metadata["source_title"]
        if metadata.get("source_channel"):
            meeting.processing_metadata.source_channel = metadata["source_channel"]
        if metadata.get("source_chapters"):
            meeting.processing_metadata.source_chapters = metadata["source_chapters"]
        if metadata.get("source_image_url"):
            meeting.processing_metadata.source_image_url = metadata["source_image_url"]
        if metadata.get("source_description"):
            meeting.processing_metadata.source_description = metadata["source_description"]
        if metadata.get("source_audio_url"):
            meeting.processing_metadata.source_audio_url = metadata["source_audio_url"]
        state.mark_complete(PipelineStage.INGESTED)
        print(f"  Done in {elapsed:.1f}s")

        # Try to download VTT if input is a CATS TV URL
        if not vtt_path.exists() and isinstance(audio_path, str) and "catstv" in audio_path:
            from src.download import download_vtt, extract_catstv_vtt_url
            vtt_url = extract_catstv_vtt_url(audio_path)
            if vtt_url:
                result = download_vtt(vtt_url, vtt_path)
                if result:
                    print(f"  Downloaded VTT captions: {vtt_path.name} (will use instead of Whisper)")
                else:
                    print("  No VTT captions available from CATS TV")
            else:
                print("  No VTT captions available from CATS TV")

        # Try to download captions for YouTube / Facebook / other yt-dlp URLs
        elif not vtt_path.exists() and isinstance(audio_path, str):
            from src.download import download_captions_via_ytdlp, is_ytdlp_url
            if is_ytdlp_url(str(audio_path)):
                print("  Checking for closed captions...")
                result = download_captions_via_ytdlp(str(audio_path), vtt_path)
                if result:
                    print(f"  Downloaded captions: {vtt_path.name} (will use instead of Whisper)")
                else:
                    print("  No captions available — will transcribe with Whisper")

    duration_min = meeting.duration_seconds / 60
    print(f"  Audio duration: {duration_min:.1f} minutes\n")

    # ======================================================================
    # Stage 2: Diarization
    # ======================================================================
    print("=" * 60)
    print("STAGE 2: Speaker Diarization")
    print("=" * 60)

    from src.models import Segment

    diarization_path = meeting_dir / "diarization.json"
    embeddings_path = meeting_dir / "embeddings.json"

    if state.is_complete(PipelineStage.DIARIZED):
        print("  Already complete. Loading from checkpoint...")
        with open(diarization_path, "r") as f:
            segments = [Segment.from_dict(d) for d in json.load(f)]
        print(f"  Loaded {len(segments)} segments")
    else:
        _compute = getattr(args, "compute", "local")
        if _compute == "modal" and getattr(args, "diarizer", "oss") != "api":
            if args.diarizer == "vibevoice" and num_speakers is not None:
                print(
                    f"  ! --num-speakers={num_speakers} is ignored by VibeVoice "
                    "(the model does not accept a speaker-count hint)."
                )
            if not diarization_path.exists() or not embeddings_path.exists():
                from src.modal_compute import run_diarization as _modal_diarize
                t0 = time.time()
                _segs_data, _emb_data = _modal_diarize(
                    wav_path,
                    meeting_id,
                    use_merge=args.merge and args.diarizer == "oss",
                    diarizer=args.diarizer,
                    chunk_minutes=_resolve_chunk_minutes(args, state.event_kind),
                )
                elapsed = time.time() - t0
                segments = [Segment.from_dict(d) for d in _segs_data]
                with open(diarization_path, "w") as f:
                    json.dump([s.to_dict() for s in segments], f, indent=2)
                with open(embeddings_path, "w") as f:
                    json.dump(_emb_data, f)
                print(f"  Done in {elapsed:.1f}s (Modal)")
            else:
                print("  Diarization + embeddings found. Loading from checkpoint...")
                with open(diarization_path, "r") as f:
                    segments = [Segment.from_dict(d) for d in json.load(f)]
                print(f"  Loaded {len(segments)} segments")
        else:
            # Sub-step A: Diarization
            if diarization_path.exists():
                print("  Diarization file found. Loading instead of re-running...")
                with open(diarization_path, "r") as f:
                    segments = [Segment.from_dict(d) for d in json.load(f)]
                print(f"  Loaded {len(segments)} segments from previous run")
            else:
                diarizer = getattr(args, "diarizer", "oss")
                if diarizer == "api":
                    from src.diarize_api import run_diarization_via_api

                    api_key = os.environ.get("PYANNOTE_AI_KEY")
                    if not api_key:
                        raise RuntimeError(
                            "--diarizer api requires PYANNOTE_AI_KEY in env "
                            "(add it to .env.local)."
                        )
                    print("  Running speaker diarization (pyannote.ai Precision-2)...")
                    if num_speakers is not None:
                        print(
                            f"  ! --num-speakers={num_speakers} is ignored by the API "
                            "backend (Precision-2 does not accept a speaker-count hint)."
                        )
                    t0 = time.time()
                    segments = run_diarization_via_api(wav_path, api_key)
                    elapsed = time.time() - t0

                    with open(diarization_path, "w") as f:
                        json.dump([s.to_dict() for s in segments], f, indent=2)

                    print(f"  Diarization done in {elapsed:.1f}s")
                else:
                    from src.diarize import load_diarization_pipeline, run_diarization

                    print("  Running speaker diarization (pyannote OSS 3.1)...")
                    t0 = time.time()
                    pipeline = load_diarization_pipeline(hf_token)
                    segments = run_diarization(pipeline, wav_path, num_speakers=num_speakers)
                    elapsed = time.time() - t0

                    with open(diarization_path, "w") as f:
                        json.dump([s.to_dict() for s in segments], f, indent=2)

                    del pipeline
                    free_gpu_memory()
                    print(f"  Diarization done in {elapsed:.1f}s")

            # Sub-step B: Speaker embeddings
            if embeddings_path.exists():
                print("  Embeddings file found. Skipping extraction.")
            else:
                from src.diarize import extract_speaker_embeddings

                print("  Extracting speaker embeddings...")
                t0 = time.time()
                speaker_embeddings = extract_speaker_embeddings(wav_path, segments, hf_token)

                emb_data = {k: v.tolist() for k, v in speaker_embeddings.items()}
                with open(embeddings_path, "w") as f:
                    json.dump(emb_data, f)

                elapsed = time.time() - t0
                print(f"  Embeddings done in {elapsed:.1f}s")

        free_gpu_memory()
        state.mark_complete(PipelineStage.DIARIZED)

    unique_speakers = set(s.speaker_label for s in segments)
    print(f"  {len(segments)} segments, {len(unique_speakers)} speakers detected")
    meeting.processing_metadata.diarization_model = _diarization_model_name(
        getattr(args, "diarizer", "oss")
    )
    print()

    # ======================================================================
    # Stage 2.5: Auto-merge fragmented speakers (opt-in via --merge)
    # ======================================================================
    if args.merge and _merge_stage_skipped(getattr(args, "diarizer", "oss")):
        backend = getattr(args, "diarizer", "oss")
        print(
            f"  ! --merge requested with --diarizer {backend}: skipping merge stage. "
            "Speaker merge is only supported for OSS pyannote."
        )
    elif args.merge and getattr(args, "compute", "local") == "modal":
        print("  (--merge was applied inside Modal — skipping local merge step)")
    elif args.merge:
        if embeddings_path.exists():
            with open(embeddings_path, "r") as f:
                emb_data = json.load(f)
            speaker_embeddings = {k: np.array(v) for k, v in emb_data.items()}

            from src.merge import merge_similar_speakers

            before_count = len(set(s.speaker_label for s in segments))
            segments, speaker_embeddings, merge_log = merge_similar_speakers(
                segments, speaker_embeddings,
            )
            after_count = len(set(s.speaker_label for s in segments))

            if merge_log:
                print("Speaker merge:")
                for entry in merge_log:
                    print(f"  {entry}")
                print(f"  {before_count} speakers -> {after_count} speakers")

                # Update embeddings.json on disk
                emb_data = {k: v.tolist() for k, v in speaker_embeddings.items()}
                with open(embeddings_path, "w") as f:
                    json.dump(emb_data, f)

                # Update diarization.json with merged labels
                with open(diarization_path, "w") as f:
                    json.dump([s.to_dict() for s in segments], f, indent=2)

                print()

    # ======================================================================
    # Pre-identification (optional, between diarization and transcription)
    # ======================================================================
    pre_identifications = {}
    pre_id_path = meeting_dir / "pre_identifications.json"

    # Load existing pre-identifications if present (from --identify-speakers)
    if pre_id_path.exists():
        with open(pre_id_path, "r") as f:
            pre_data = json.load(f)
        from src.models import SpeakerMapping as SM
        for label, data in pre_data.items():
            pre_identifications[label] = SM(
                speaker_label=label,
                speaker_name=data["speaker_name"],
                confidence=data.get("confidence", 1.0),
                id_method=data.get("id_method", "human_review"),
            )
        print(f"  Loaded {len(pre_identifications)} pre-identification(s) from previous session")

    if args.pre_identify and sys.stdin.isatty():
        print("=" * 60)
        print("PRE-IDENTIFICATION: Identify speakers by video clip")
        print("=" * 60)

        video_path = find_video_file(meeting_dir, meeting.audio_source)

        import numpy as np
        from src.enroll import load_profiles as _load_profiles
        from src import review as _review
        if embeddings_path.exists():
            with open(embeddings_path, "r") as f:
                _emb = json.load(f)
            _pre_embeddings = {k: np.array(v) for k, v in _emb.items()}
        else:
            _pre_embeddings = {}
        _pre_profile_db = _load_profiles()

        from src.models import SpeakerMapping as SM
        temp_mappings = dict(pre_identifications)  # start with any existing
        _views = _review.build_review_state(segments, temp_mappings, _pre_embeddings, _pre_profile_db, show_text=False)
        for v in _views:
            if v.label not in temp_mappings:
                temp_mappings[v.label] = SM(speaker_label=v.label)

        if video_path:
            print(f"  Video: {Path(video_path).name}")
        else:
            print("  Video: not found")
        _hint_count = sum(1 for v in _views if v.soft_hints)
        if _hint_count:
            print(f"  Voice hints: {_hint_count} speaker(s) have possible profile matches")
        print(f"  Speakers: {len(_views)}")
        print()

        changes = _interactive_speaker_review(
            segments, temp_mappings, _pre_embeddings, _pre_profile_db,
            video_path, _review_audio_path(meeting_dir),
            body_slug=effective_body_slug, show_text=False,
            event_kind=args.event_kind,
            meeting_id=meeting_dir.name,
            clip_offset=clip_start or 0.0,
        )
        _persist_after_review(meeting_dir, segments, _pre_embeddings, changes)

        if changes:
            for label, mapping in temp_mappings.items():
                if isinstance(mapping, SM) and mapping.speaker_name:
                    pre_identifications[label] = mapping

            # Save pre-identifications
            pre_data = {}
            for label, m in pre_identifications.items():
                pre_data[label] = {
                    "speaker_name": m.speaker_name,
                    "confidence": m.confidence,
                    "id_method": m.id_method,
                }
            with open(pre_id_path, "w") as f:
                json.dump(pre_data, f, indent=2)

            print(f"\n  {len(changes)} identification(s) saved. These will be used in Stage 4.")

            # Offer enrollment
            _enroll_after_review(
                changes, temp_mappings, meeting_dir, segments,
                roster=_resolve_roster(effective_body_slug, state.roster_choice),
            )
        print()

    # ======================================================================
    # Stage 3: Transcription (Whisper or VTT alignment)
    # ======================================================================
    print("=" * 60)
    use_vtt = args.use_vtt or (vtt_path.exists() and not state.is_complete(PipelineStage.TRANSCRIBED))
    if use_vtt and vtt_path.exists():
        print("STAGE 3: VTT Alignment (skipping Whisper)")
    else:
        print("STAGE 3: Transcription")
        use_vtt = False  # force off if no VTT file
    print("=" * 60)

    from src.transcribe import (
        load_raw_transcript,
        load_whisper_model,
        remove_segment_overlaps,
        save_raw_transcript,
        transcribe_and_assign,
    )

    transcript_path = meeting_dir / "transcript_raw.json"

    if state.is_complete(PipelineStage.TRANSCRIBED):
        print("  Already complete. Loading from checkpoint...")
        segments = load_raw_transcript(transcript_path)
        print(f"  Loaded {len(segments)} transcribed segments")
    else:
        remove_segment_overlaps(segments)

    if not state.is_complete(PipelineStage.TRANSCRIBED) and use_vtt:
        from src.vtt_align import align_vtt_to_segments

        t0 = time.time()
        print(f"  Aligning VTT captions from {vtt_path.name}...")
        # Captions are downloaded for the full source; rebase to clip-local time
        # so a clipped meeting's diarized segments get the right text.
        segments = align_vtt_to_segments(vtt_path, segments, clip_offset=clip_start or 0.0)
        elapsed = time.time() - t0

        meeting.processing_metadata.transcription_model = "vtt_alignment"
        save_raw_transcript(segments, transcript_path)
        state.mark_complete(PipelineStage.TRANSCRIBED)
        print(f"  Done in {elapsed:.1f}s")
    elif (
        not state.is_complete(PipelineStage.TRANSCRIBED)
        and getattr(args, "compute", "local") == "modal"
    ):
        from src.modal_compute import run_transcription as _modal_transcribe, upload_audio as _modal_upload

        # When --diarizer api is used, run_diarization() on Modal was never called,
        # so the audio was never uploaded to the volume. Always ensure it's present.
        _modal_upload(wav_path, meeting_id)

        print("  Dispatching Whisper transcription to Modal GPU (large-v3)...")
        t0 = time.time()
        _segs_data = [s.to_dict() for s in segments]
        _updated_data = _modal_transcribe(meeting_id, _segs_data)
        elapsed = time.time() - t0
        segments = [Segment.from_dict(d) for d in _updated_data]

        meeting.processing_metadata.transcription_model = config.WHISPER_MODEL_GPU
        meeting.processing_metadata.gpu_used = True
        save_raw_transcript(segments, transcript_path)
        state.mark_complete(PipelineStage.TRANSCRIBED)
        print(f"  Done in {elapsed:.1f}s (Modal)")
    elif not state.is_complete(PipelineStage.TRANSCRIBED):
        t0 = time.time()
        whisper_model = load_whisper_model()

        model_name = config.WHISPER_MODEL_GPU if torch.cuda.is_available() else config.WHISPER_MODEL_CPU
        meeting.processing_metadata.transcription_model = model_name
        meeting.processing_metadata.gpu_used = torch.cuda.is_available()
        print(f"  Using model: {model_name} (whole-audio pass)")

        # Whole-audio transcription + timestamp-based word->turn assignment.
        # One pass over the full file yields drift-free word timestamps; per-turn
        # slicing is gone, so there is no per-segment checkpoint to resume from.
        segments = transcribe_and_assign(whisper_model, wav_path, segments)
        elapsed = time.time() - t0

        save_raw_transcript(segments, transcript_path)
        del whisper_model
        free_gpu_memory()
        state.mark_complete(PipelineStage.TRANSCRIBED)
        print(f"  Done in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Stage 3.5: Reconcile transcript against a source-provided reference
    # ------------------------------------------------------------------
    reference_path = meeting_dir / "reference_transcript.txt"
    reconciled_marker = meeting_dir / "reconciled.done"
    if reference_path.exists() and not reconciled_marker.exists() and segments:
        try:
            from src import config as _cfg
            from src.llm_providers import make_llm_client
            from src.reconcile import reconcile_segments

            client = make_llm_client()

            def _call_llm(prompt: str) -> str:
                msg = client.messages.create(
                    model=_cfg.SUMMARY_CLASSIFY_MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text

            reference_text = reference_path.read_text(encoding="utf-8")
            _, applied = reconcile_segments(segments, reference_text, call_llm=_call_llm)
            if applied:
                save_raw_transcript(segments, transcript_path)
                print("  Reconciled transcript against source reference.")
            else:
                print("  Reference transcript overlap too low; kept Whisper text.")
            reconciled_marker.write_text("done", encoding="utf-8")
        except Exception as exc:  # never fatal — timestamps/segments are intact
            print(f"  Transcript reconciliation skipped ({exc}).")

    # Show sample
    print("\n  Sample transcript:")
    for seg in segments[:5]:
        if seg.text:
            text = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
            print(f"    [{seg.speaker_label}] {text}")
    print()

    # ======================================================================
    # Stage 4: Speaker Identification
    # ======================================================================
    print("=" * 60)
    print("STAGE 4: Speaker Identification")
    print("=" * 60)

    from src.enroll import get_stored_centroids, load_profiles
    from src.identify import (
        apply_mappings_to_segments,
        flag_for_review,
        identify_speakers,
    )
    from src.roster import roster_names_for_prompt

    named_transcript_path = meeting_dir / "transcript_named.json"
    llm_partial_path = meeting_dir / "llm_partial_results.json"

    # Phase 109 CSMEETING-03 + roster chooser: resolve the Stage 4 roster from the
    # meeting's tagging/choice state. effective_body_slug comes from the resolve
    # block above; if it's set, Plan 02's guard has already verified the cache file
    # exists. The actual load_roster() calls live inside _resolve_roster(); this is
    # the only meeting-context roster resolution. The 3 offline utility sites
    # (_fix_transcripts, --show-roster, --fix-profiles) still call bare load_roster()
    # because they have no meeting context. See 109-RESEARCH.md §1.
    roster = _resolve_roster(effective_body_slug, state.roster_choice)
    if roster:
        # Roster dataclass may not have .city/.body when loaded from a body-keyed cache;
        # print whichever label is available without crashing the legacy path.
        label = f"{getattr(roster, 'city', '') or ''} {getattr(roster, 'body', '') or ''}".strip()
        if not label and effective_body_slug:
            label = effective_body_slug
        print(f"  Loaded council roster: {len(roster.members)} members ({label})")
    else:
        print("  No roster loaded — speaker names won't be corrected against a council roster.")
    roster_hint = roster_names_for_prompt(roster) if roster else ""

    if state.is_complete(PipelineStage.IDENTIFIED):
        print("  Already complete. Loading from checkpoint...")
        with open(named_transcript_path, "r") as f:
            meeting_data = json.load(f)
        meeting = Meeting.from_dict(meeting_data)
        if meeting.clip_start_seconds is None:
            meeting.clip_start_seconds = state.clip_start_seconds
        if meeting.clip_end_seconds is None:
            meeting.clip_end_seconds = state.clip_end_seconds
        segments = meeting.segments
    else:
        # Load embeddings
        if embeddings_path.exists():
            with open(embeddings_path, "r") as f:
                emb_data = json.load(f)
            speaker_embeddings = {k: np.array(v) for k, v in emb_data.items()}
        else:
            speaker_embeddings = {}

        # Layer 1: Voice profiles
        profile_db = load_profiles()
        stored_centroids = get_stored_centroids(profile_db)
        if stored_centroids:
            print(f"  Loaded {len(stored_centroids)} voice profiles")

        # Guard: re-extract embeddings if they don't match stored profile dimensions
        if stored_centroids and speaker_embeddings:
            emb_dim = next(iter(speaker_embeddings.values())).shape[0]
            prof_dim = next(iter(stored_centroids.values())).shape[0]
            if emb_dim != prof_dim:
                print(f"  ! Stale embeddings ({emb_dim}-dim) don't match profiles ({prof_dim}-dim). Re-extracting...")
                from src.diarize import extract_speaker_embeddings
                speaker_embeddings = extract_speaker_embeddings(wav_path, segments, hf_token)
                emb_data = {k: v.tolist() for k, v in speaker_embeddings.items()}
                with open(embeddings_path, "w") as f:
                    json.dump(emb_data, f)
                new_dim = next(iter(speaker_embeddings.values())).shape[0]
                print(f"  Re-extracted {len(speaker_embeddings)} embeddings ({new_dim}-dim)")

        # Layer 3: LLM (off by default — config.SPEAKER_ID_LLM_ENABLED; also
        # skipped on Congressional Record runs and on interview-kind meetings
        # when explicitly re-enabled)
        llm_fn = None
        if should_run_llm(args.skip_llm, crec_request, event_kind=state.event_kind):
            from src.llm_providers import get_provider
            from src.llm_utils import llm_identify_speakers

            provider = get_provider(config.SPEAKER_ID_ACTIVE)
            print(f"  LLM speaker ID via {config.SPEAKER_ID_ACTIVE} ({provider.model})")
            _llm_event_kind = state.event_kind
            llm_fn = lambda segs, maps: llm_identify_speakers(
                provider, segs, maps,
                event_kind=_llm_event_kind, roster=roster, roster_hint=roster_hint,
                partial_results_path=llm_partial_path,
            )
        elif not config.SPEAKER_ID_LLM_ENABLED:
            print("  Skipping LLM speaker ID (disabled — config.SPEAKER_ID_LLM_ENABLED; "
                  "layers 1-2 and review unaffected).")
        elif crec_request and not args.skip_llm:
            print("  Skipping LLM speaker ID (Congressional Record run — CREC is "
                  "authoritative; unresolved speakers go to review).")
        elif (not args.skip_llm and not crec_request
              and state.event_kind in INTERVIEW_KINDS):
            print("  Skipping LLM speaker ID (interview-kind meeting — eval "
                  "2026-08-05: ~13% name coverage; names come from review).")

        crec_mappings = None
        if crec_request:
            from src.crec_identify import crec_speaker_mappings
            _crec_date, _crec_chamber = crec_request
            crec_mappings = crec_speaker_mappings(_crec_date, _crec_chamber, segments)
            print(f"  Congressional Record: resolved {len(crec_mappings)} speaker label(s)")

        t0 = time.time()
        mappings = identify_speakers(
            segments, speaker_embeddings,
            stored_profiles=stored_centroids if stored_centroids else None,
            llm_identify_fn=llm_fn,
            roster=roster,
            profile_db=profile_db,
            crec_mappings=crec_mappings,
        )
        elapsed = time.time() - t0

        if llm_partial_path.exists():
            llm_partial_path.unlink()

        # Apply pre-identifications (override automated results)
        if pre_identifications:
            overrides = 0
            for label, pre_map in pre_identifications.items():
                if label in mappings:
                    if pre_map.confidence > mappings[label].confidence:
                        mappings[label] = pre_map
                        overrides += 1
                else:
                    mappings[label] = pre_map
                    overrides += 1
            if overrides:
                print(f"  Applied {overrides} pre-identification(s) as ground truth")

        print(f"  Done in {elapsed:.1f}s")
        for label, m in mappings.items():
            status = "REVIEW" if m.needs_review else "OK"
            name = m.speaker_name or "(unidentified)"
            print(f"    {label} -> {name} (conf={m.confidence:.2f}, method={m.id_method}, {status})")

        # Human review — rich interactive review on a terminal (clips, hints,
        # merge); fall back to the text-only quick review otherwise or when
        # --no-review is set.
        from src.relink import auto_link_confident

        if sys.stdin.isatty() and not getattr(args, "no_review", False):
            auto = auto_link_confident(mappings)
            if auto:
                print(f"  Auto-linked {len(auto)} confident speaker(s) before review: "
                      f"{', '.join(auto)}")
            review_video = find_video_file(meeting_dir, meeting.audio_source)
            review_changes = _interactive_speaker_review(
                segments, mappings, speaker_embeddings, profile_db,
                review_video, str(wav_path),
                roster=roster, body_slug=effective_body_slug, show_text=True,
                event_kind=meeting.event_kind,
                meeting_id=meeting_dir.name,
                clip_offset=clip_start or 0.0,
            )
            _persist_after_review(meeting_dir, segments, speaker_embeddings, review_changes)
        else:
            mappings = human_review(mappings)
            auto = auto_link_confident(mappings)
            if auto:
                print(f"  Auto-linked {len(auto)} confident speaker(s): {', '.join(auto)}")

        # Apply to segments
        segments = apply_mappings_to_segments(segments, mappings)
        meeting.segments = segments
        meeting.speakers = mappings

        # Merge adjacent same-speaker segments HERE, so transcript_named.json is
        # the canonical merged transcript that every downstream reader shares:
        # the review UI + GUI publish (which read this file), summary (whose
        # section indices are computed against these segments), and export. The
        # later export-time merge then becomes an idempotent no-op. Without this,
        # the merge only reached exports/ + a full run's in-memory publish, so the
        # review UI showed — and GUI publish pushed — the unmerged fragments.
        from src.identify import merge_adjacent_segments
        meeting.segments = merge_adjacent_segments(meeting.segments)

        # Federal floor structure: derive recorded votes + transcript timestamps
        # from the Congressional Record and persist them on the meeting artifact.
        # (Timestamps stay clip-local; absolutized at publish — a later slice.)
        # NON-FATAL: this runs after the (possibly human-reviewed) identification and
        # before the save, so a CREC fetch/parse failure must never discard that work.
        # Mirror the non-fatal-CREC contract (crec_speaker_mappings returns {} on
        # failure): on any error, warn, leave floor_votes empty, and continue to save.
        if crec_request:
            try:
                from src.crec_floor import extract_floor_structure, build_floor_votes
                _floor = extract_floor_structure(_crec_date, _crec_chamber)
                if _floor:
                    meeting.floor_votes = build_floor_votes(
                        _floor, [s.to_dict() for s in meeting.segments])
                    _stamped = sum(1 for v in meeting.floor_votes if v.matched)
                    print(f"  Floor structure: {len(meeting.floor_votes)} recorded vote(s), "
                          f"{_stamped} timestamped from the transcript")
            except Exception as exc:  # never lose the identification stage over this
                print(f"  Floor structure extraction skipped ({exc}); "
                      f"continuing without floor votes.")

        atomic_write_json(named_transcript_path, meeting.to_dict())
        state.mark_complete(PipelineStage.IDENTIFIED)

    print()

    # ======================================================================
    # Confidence gate (Phase A): score the meeting, route non-passing
    # non-interactive runs to the review queue BEFORE any paid summary or
    # voice enrollment (the latter would poison profiles with unreviewed guesses).
    # ======================================================================
    gate_report = _apply_gate(meeting, meeting_dir, state)
    _interactive = sys.stdin.isatty()
    _publish_anyway = getattr(args, "publish_anyway", False)
    if gate_report["verdict"] != "pass" and not _interactive and not _publish_anyway:
        print()
        print("=" * 60)
        print(f"QUEUED FOR REVIEW — verdict: {gate_report['verdict']}")
        print("=" * 60)
        print(f"  {gate_report['reason']}")
        print(f"  Review with: python run_local.py --review {meeting_id}")
        print(f"  Then publish with: python run_local.py --resume {meeting_id}")
        print("  (Summary, enrollment, and publish were skipped to save cost "
              "and protect voice profiles.)")
        return

    # ======================================================================
    # Stage 5: Summary Generation
    # ======================================================================
    print("=" * 60)
    print("STAGE 5: Summary Generation")
    print("=" * 60)

    summary_path = meeting_dir / "summary.json"

    if state.is_complete(PipelineStage.SUMMARIZED):
        print("  Already complete.")
        print(f"  {_load_summary_checkpoint(meeting, summary_path)}")
    elif args.skip_summary:
        print("  Skipped (--skip-summary).")
        state.mark_complete(PipelineStage.SUMMARIZED)
    else:
        from src.llm_providers import llm_client_env_key

        env_key = llm_client_env_key()
        if not os.environ.get(env_key):
            print(f"  No {env_key} found. Skipping summary generation.")
            print("  Set the environment variable or use --skip-summary to silence this.")
            state.mark_complete(PipelineStage.SUMMARIZED)
        else:
            from src.summarize import generate_summary

            def summary_progress(step, current=0, total=0):
                if total > 0:
                    print(f"    [{current}/{total}] {step}")
                else:
                    print(f"    {step}...")

            t0 = time.time()
            print("  Generating meeting summary via the configured LLM API...")
            try:
                meeting.summary = generate_summary(meeting, progress_callback=summary_progress)
            except Exception as e:
                # The summary stage is the only stage that needs the LLM API.
                # If it fails (e.g. out of credits, bad key, network), don't crash the
                # whole pipeline — report it clearly and continue. The stage is left
                # incomplete so it will be retried automatically on the next run.
                meeting.summary = None
                detail = str(e)
                if "credit balance is too low" in detail:
                    reason = "LLM API credit balance is too low."
                    hint = f"Check the active LLM backend's account balance ({env_key}), then re-run to generate the summary."
                elif "authentication" in detail.lower() or "invalid x-api-key" in detail.lower():
                    reason = "LLM API key was rejected."
                    hint = f"Check {env_key}, then re-run to generate the summary."
                else:
                    reason = f"Summary generation failed: {detail}"
                    hint = "Re-run once the issue is resolved to generate the summary."
                print(f"\n  ⚠ Skipping summary — {reason}")
                print(f"    {hint}")
                print("    (All other stages are complete; only the summary needs the API.)")
            else:
                elapsed = time.time() - t0

                # Save summary checkpoint
                with open(summary_path, "w") as f:
                    json.dump(meeting.summary.to_dict(), f, indent=2)

                # Re-save named transcript with summary included
                atomic_write_json(named_transcript_path, meeting.to_dict())

                state.mark_complete(PipelineStage.SUMMARIZED)

                print(f"\n  Summary generated in {elapsed:.1f}s")
                print(f"    Sections: {len(meeting.summary.sections)}")
                print(f"    Highlights: {len(meeting.summary.highlights)}")
                if meeting.summary.executive_summary:
                    # Show first 200 chars of executive summary
                    preview = meeting.summary.executive_summary[:200]
                    if len(meeting.summary.executive_summary) > 200:
                        preview += "..."
                    print(f"    Preview: {preview}")

    print()

    # --- Stage 5b: Topic classification (Phase 6) ---
    topics_path = meeting_dir / "topics.json"
    if topics_path.exists():
        from src.models import SectionTopic
        with open(topics_path, "r") as f:
            meeting.section_topics = [SectionTopic.from_dict(d) for d in json.load(f)]
        print(f"  Loaded topics ({len(meeting.section_topics)} sections)")
    elif args.skip_summary or meeting.summary is None:
        print("  Skipped topic classification (no summary).")
    else:
        db_url = os.environ.get("DATABASE_URL", "").strip()
        if not db_url:
            print("  No DATABASE_URL — skipping topic classification (vocabulary lives in the DB).")
        else:
            try:
                import psycopg2
                from src.llm_providers import make_llm_client
                from src.topics import fetch_live_topics, classify_sections

                conn = psycopg2.connect(db_url)
                try:
                    vocab = fetch_live_topics(conn)
                finally:
                    conn.close()
                print(f"  Classifying topics against {len(vocab)} live Compass topics...")
                client = make_llm_client()
                meeting.section_topics = classify_sections(client, meeting.summary.sections, vocab)
                with open(topics_path, "w") as f:
                    json.dump([st.to_dict() for st in meeting.section_topics], f, indent=2)
                tagged = sum(1 for st in meeting.section_topics if st.topic_keys)
                print(f"  Tagged {tagged}/{len(meeting.section_topics)} substantive sections")
            except Exception as e:
                print(f"  ⚠ Skipping topic classification — {e}")
                meeting.section_topics = []

    # ======================================================================
    # Stage 6: Voice Enrollment
    # ======================================================================
    print("=" * 60)
    print("STAGE 6: Voice Enrollment")
    print("=" * 60)

    from src.enroll import enroll_confirmed, enroll_speakers, get_borderline_speakers, save_profiles

    if state.is_complete(PipelineStage.ENROLLED):
        print("  Already complete. Skipping.")
    else:
        if embeddings_path.exists():
            with open(embeddings_path, "r") as f:
                emb_data = json.load(f)
            speaker_embeddings = {k: np.array(v) for k, v in emb_data.items()}
        else:
            speaker_embeddings = {}

        profile_db = load_profiles()
        before_count = len(profile_db.profiles)

        # Auto-enroll high-confidence speakers (>= 0.85)
        # Stamp the directory basename (not the meeting_id variable) so the
        # provenance key matches calibrate_gate's leave-one-out key exactly.
        profile_db = enroll_speakers(
            profile_db, speaker_embeddings, meeting.speakers,
            meeting_id=meeting_dir.name, segments=segments, roster=roster,
        )

        auto_count = len(profile_db.profiles) - before_count
        if auto_count > 0:
            print(f"  Auto-enrolled {auto_count} high-confidence speaker(s)")

        # Interactive enrollment for borderline speakers (0.70-0.85)
        if args.confirm_enroll and sys.stdin.isatty():
            borderline = get_borderline_speakers(
                meeting.speakers, speaker_embeddings, segments,
            )
            if borderline:
                video_path = find_video_file(meeting_dir, meeting.audio_source)
                if video_path:
                    print(f"\n  Video found: {Path(video_path).name}")
                else:
                    print("\n  No video file found (audio-only fallback with afplay)")

                print(f"  {len(borderline)} speaker(s) eligible for enrollment confirmation:")
                confirmed = []
                for info in borderline:
                    m = info["mapping"]
                    mins = info["total_speech_seconds"] / 60
                    print(f"\n  {m.speaker_label} identified as \"{m.speaker_name}\"")
                    print(f"    Method: {m.id_method} (confidence: {m.confidence:.2f})")
                    print(f"    Segments: {info['seg_count']} ({mins:.1f}m total speech)")
                    if info["sample_segment"]:
                        sample = info["sample_segment"]
                        text = sample.text[:100] + "..." if len(sample.text) > 100 else sample.text
                        ts_min = int(sample.start_time // 60)
                        ts_sec = int(sample.start_time % 60)
                        print(f"    Sample [{ts_min:02d}:{ts_sec:02d}]: \"{text}\"")

                    while True:
                        if video_path:
                            choice = input("\n    [V]iew clip / [E]nroll / [S]kip? ").strip().lower()
                        else:
                            choice = input("\n    [E]nroll / [S]kip? ").strip().lower()

                        if choice in ("v", "view") and video_path and info["sample_segment"]:
                            play_video_clip(
                                video_path,
                                start_time=info["sample_segment"].start_time,
                                duration=20.0,
                                title=f"{m.speaker_label} → {m.speaker_name or 'Unknown'}",
                                video_offset=clip_start or 0.0,
                            )
                            continue  # re-prompt after viewing
                        elif choice in ("e", "enroll", "y", "yes"):
                            confirmed.append(info["label"])
                            print(f"    -> Will enroll {m.speaker_name}")
                            break
                        else:
                            print(f"    -> Skipped")
                            break

                if confirmed:
                    profile_db = enroll_confirmed(
                        profile_db, speaker_embeddings, confirmed,
                        meeting.speakers, meeting_id=meeting_dir.name, segments=segments,
                        roster=roster,
                    )
                    print(f"\n  Enrolled {len(confirmed)} additional speaker(s) via confirmation")
            else:
                print("  No borderline speakers to confirm.")

        save_profiles(profile_db)

        after_count = len(profile_db.profiles)
        total_new = after_count - before_count
        state.mark_complete(PipelineStage.ENROLLED)

        print(f"\n  Total new profiles: {total_new}. Total stored: {after_count}")
        for pid, p in profile_db.profiles.items():
            print(f"    {pid}: {p.display_name} ({len(p.meetings_seen)} meetings, {p.total_segments_confirmed} segments)")

    print()

    # ======================================================================
    # Post-identification segment merging
    # ======================================================================
    from src.identify import merge_adjacent_segments

    before_count = len(meeting.segments)
    meeting.segments = merge_adjacent_segments(meeting.segments)
    after_count = len(meeting.segments)
    if before_count != after_count:
        print(f"  Segment merge: {before_count} -> {after_count} segments")

    # ======================================================================
    # Stage 7: Export
    # ======================================================================
    print("=" * 60)
    print("STAGE 7: Export")
    print("=" * 60)

    from src.export import export_all

    if state.is_complete(PipelineStage.EXPORTED):
        print("  Already complete.")
    else:
        export_dir = meeting_dir / "exports"
        results = export_all(meeting, export_dir)
        state.mark_complete(PipelineStage.EXPORTED)

        print("  Export complete:")
        for fmt, path in results.items():
            print(f"    {fmt}: {path}")

    if getattr(args, "publish", False):
        if not _may_publish(state.review_status, getattr(args, "publish_anyway", False)):
            print(f"  Not publishing — gate verdict is "
                  f"'{state.review_status}'. Review and re-run, or pass "
                  f"--publish-anyway to override.")
        else:
            try:
                from src.publish import publish_meeting

                _attach_thumbnail(meeting, meeting_dir)
                result = publish_meeting(meeting, state.body_slug)
                print(f"  Published to Supabase: {result.segments} segments, "
                      f"{result.speakers} speakers")
            except Exception as e:
                print(f"  WARNING: Supabase publish failed: {e}")
                print(f"  Retry later with: python run_local.py --publish-meeting {meeting.meeting_id}")

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    export_dir = meeting_dir / "exports"
    print(f"\nOutputs:")
    print(f"  Transcript: {export_dir / 'transcript.md'}")
    print(f"  JSON:       {export_dir / 'transcript.json'}")
    print(f"  Subtitles:  {export_dir / 'subtitles.srt'}")
    if meeting.summary:
        print(f"  Summary:    {export_dir / 'summary.md'}")


def _parse_batch_inputs(batch_path: str) -> list[dict]:
    """Parse batch input: a text file or a directory of video files.

    Returns list of dicts with keys: input, date, city, meeting_type.
    """
    p = Path(batch_path)

    if p.is_dir():
        # Directory of video files
        video_exts = {".m4v", ".mp4", ".mkv", ".webm", ".avi", ".mov"}
        entries = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in video_exts:
                # Try to extract date from filename (YYYY-MM-DD pattern)
                import re
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", f.stem)
                date = date_match.group(1) if date_match else ""
                entries.append({
                    "input": str(f),
                    "date": date,
                    "city": None,
                    "meeting_type": None,
                })
        return entries

    if p.is_file():
        # Text file with one entry per line
        # Format: PATH_OR_URL [DATE] [CITY] [TYPE]
        # or just: PATH_OR_URL
        entries = []
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=3)
                entry = {
                    "input": parts[0],
                    "date": parts[1] if len(parts) > 1 else "",
                    "city": parts[2] if len(parts) > 2 else None,
                    "meeting_type": parts[3] if len(parts) > 3 else None,
                }
                entries.append(entry)
        return entries

    print(f"Error: batch path '{batch_path}' is not a file or directory.")
    sys.exit(1)


def _run_batch(args: argparse.Namespace) -> None:
    """Run batch processing on multiple meetings.

    Runs Stages 1-3 (ingest, diarize, transcribe) + automated Stage 4
    (no interactive review) for each meeting. Skips pre-identify and
    human review in batch mode.
    """
    entries = _parse_batch_inputs(args.batch)
    if not entries:
        print("No inputs found for batch processing.")
        return

    print(f"Batch processing: {len(entries)} meeting(s)")
    print()

    results = []
    for i, entry in enumerate(entries):
        print(f"{'=' * 60}")
        print(f"BATCH [{i+1}/{len(entries)}]: {entry['input']}")
        print(f"{'=' * 60}")

        # Build args for run_pipeline
        batch_args = argparse.Namespace(
            input=entry["input"],
            date=entry["date"],
            city=entry["city"],
            meeting_type=entry["meeting_type"],
            meeting_id="",
            num_speakers=0,
            noise_reduce=False,
            skip_llm=args.skip_llm if hasattr(args, "skip_llm") else False,
            skip_summary=True,  # skip summary in batch mode
            confirm_enroll=False,
            merge=args.merge if hasattr(args, "merge") else False,
            pre_identify=False,  # skip interactive pre-identify
            use_vtt=args.use_vtt if hasattr(args, "use_vtt") else False,
            diarizer=getattr(args, "diarizer", "oss"),
            compute=getattr(args, "compute", "local"),
            body=getattr(args, "body", None),
            race_id=getattr(args, "race_id", None),
            force_retag=getattr(args, "force_retag", False),
            batch_mode=True,  # suppress the interactive roster chooser (D3: batch uses no roster unless --body)
            event_kind=getattr(args, "event_kind", None),
            default=getattr(args, "default", False),
            title=getattr(args, "title", None),
        )

        # Resolve metadata up front (batch is non-interactive): an
        # under-specified entry fails here and is recorded, not silently stamped.
        try:
            _resolve_metadata(batch_args)
        except ValueError as e:
            print(f"\n  ERROR: {e}")
            results.append({"input": entry["input"], "status": f"failed: {e}", "meeting_id": entry["input"]})
            print()
            continue

        mid = f"{batch_args.date}-{batch_args.meeting_type.lower().replace(' ', '-')}"

        # Check if already processed (for --batch-resume). Uses the RESOLVED
        # meeting_type/date so an entry that relies on --default to fill those
        # still short-circuits when the meeting is already complete.
        if args.batch_resume:
            from src import config
            from src.checkpoint import PipelineStage, PipelineState
            mdir = config.MEETINGS_DIR / mid
            if (mdir / "pipeline_state.json").exists():
                state = PipelineState(mdir)
                if state.is_complete(PipelineStage.IDENTIFIED):
                    print(f"  Already complete (stage {state.completed_stage.name}). Skipping.")
                    results.append({"input": entry["input"], "status": "skipped (complete)", "meeting_id": mid})
                    print()
                    continue

        try:
            run_pipeline(batch_args)
            results.append({"input": entry["input"], "status": "completed", "meeting_id": mid})
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({"input": entry["input"], "status": f"failed: {e}", "meeting_id": mid})

        print()

    # Print summary
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    completed = [r for r in results if r["status"] == "completed"]
    skipped = [r for r in results if r["status"].startswith("skipped")]
    failed = [r for r in results if r["status"].startswith("failed")]

    print(f"  Completed: {len(completed)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")

    if completed:
        print("\nCompleted:")
        for r in completed:
            print(f"  {r['meeting_id']}")

    if failed:
        print("\nFailed (need review):")
        for r in failed:
            print(f"  {r['meeting_id']}: {r['status']}")

    if completed or skipped:
        print(f"\nUse --review-meeting MEETING_ID to review speaker identifications.")


def _is_simple_meeting_id(meeting_id: str) -> bool:
    """True if meeting_id is a single path component (its own basename).

    The id IS the meeting directory basename, so it must contain no path
    separator and not be empty/'.'/'..'. A separator would nest the directory
    and drift its name away from the id stamped into enrollment provenance —
    breaking calibration's leave-one-out key.
    """
    return bool(meeting_id) and meeting_id not in {".", ".."} \
        and not Path(meeting_id).is_absolute() \
        and Path(meeting_id).name == meeting_id


def _meeting_dir_for_id(meeting_id: str) -> Path:
    """Resolve a simple meeting ID to a direct child of MEETINGS_DIR."""
    meeting_path = Path(meeting_id)
    if not _is_simple_meeting_id(meeting_id):
        raise ValueError("invalid meeting ID")

    meetings_root = config.MEETINGS_DIR.resolve(strict=False)
    candidate = (meetings_root / meeting_path).resolve(strict=False)
    if candidate.parent != meetings_root:
        raise ValueError("meeting ID resolves outside the meetings directory")
    return candidate


def _repair_transcript_standalone(meeting_id: str) -> None:
    """Repair transcript artifacts for one already-processed meeting."""
    try:
        meeting_dir = _meeting_dir_for_id(meeting_id)
    except ValueError as exc:
        print(f"Transcript repair failed: {exc}", file=sys.stderr)
        sys.exit(1)

    from src.repair import RepairError, repair_transcript

    try:
        result = repair_transcript(meeting_dir)
    except RepairError as exc:
        print(f"Transcript repair failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Transcript repair complete:")
    print(f"  Meeting: {result.meeting_id}")
    print(f"  Segments: {result.segment_count}")
    print(f"  Backup: {result.backup_dir}")
    print("  Exports:")
    for export_name, export_path in result.exports.items():
        print(f"    {export_name}: {export_path}")


def _option_supplied(argv: list[str], *options: str) -> bool:
    """Return whether argv explicitly contains any option or option=value."""
    return any(
        argument == option
        or argument.startswith(f"{option}=")
        or (
            option == "-i"
            and argument.startswith("-i")
            and not argument.startswith("--")
        )
        for argument in argv
        for option in options
    )


def _load_meeting_and_body(meeting_dir):
    """Load a meeting + its body_slug for (re)publishing. Assumes the
    transcript_named.json exists (caller checks). Carries topic tags and the
    pipeline-state body_slug, mirroring the standalone publish loader."""
    from src.checkpoint import PipelineState
    from src.models import Meeting, SectionTopic

    with open(meeting_dir / "transcript_named.json", "r", encoding="utf-8") as f:
        meeting = Meeting.from_dict(json.load(f))
    topics_path = meeting_dir / "topics.json"
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            meeting.section_topics = [SectionTopic.from_dict(d) for d in json.load(f)]
    state = PipelineState(meeting_dir)
    if meeting.race_id is None:
        meeting.race_id = state.race_id
    # transcript_named.json is authoritative for the clip window; fall back to
    # state only for transcripts written before the clip feature existed.
    if meeting.clip_start_seconds is None:
        meeting.clip_start_seconds = state.clip_start_seconds
    if meeting.clip_end_seconds is None:
        meeting.clip_end_seconds = state.clip_end_seconds
    return meeting, state.body_slug


def _publish_meeting_standalone(meeting_id: str, publish_anyway: bool = False) -> None:
    """Publish an already-processed meeting to Supabase (backfill workhorse)."""
    from src import config
    from src.checkpoint import PipelineState
    from src.publish import publish_meeting

    meeting_dir = config.MEETINGS_DIR / meeting_id
    named_path = meeting_dir / "transcript_named.json"
    if not named_path.exists():
        print(f"No transcript_named.json found for meeting ID: {meeting_id}")
        print(f"  Expected at: {named_path}")
        sys.exit(1)

    meeting, body_slug = _load_meeting_and_body(meeting_dir)

    state = PipelineState(meeting_dir)
    if not _may_publish(state.review_status, publish_anyway):
        print(f"Refusing to publish {meeting_id} — gate verdict is "
              f"'{state.review_status}'.")
        print("  Review it (python run_local.py --review "
              f"{meeting_id}) and re-run, or pass --publish-anyway to override.")
        sys.exit(2)

    print(f"Publishing {meeting_id} to Supabase...")
    _attach_thumbnail(meeting, meeting_dir)
    result = publish_meeting(meeting, body_slug)
    print(f"  Meeting:  {result.meeting_id}")
    print(f"  Segments: {result.segments}")
    print(f"  Speakers: {result.speakers}")


def _published_meeting_slugs() -> set[str]:
    """Slugs already present in meetings.meetings (the resync target set)."""
    import psycopg2

    from src.publish import _require_db_url

    conn = psycopg2.connect(_require_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM meetings.meetings WHERE slug IS NOT NULL")
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def _republish_all(args) -> None:
    """Re-publish every already-published meeting (resync), optionally reenroll,
    one deploy at the end. Continue-on-error; non-zero exit if any failed."""
    from src import config
    from src.publish import publish_meeting

    dirs = sorted(
        d for d in config.MEETINGS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (d / "transcript_named.json").exists()
    )
    published = _published_meeting_slugs()
    to_publish = [d for d in dirs if d.name in published]
    skipped = [d for d in dirs if d.name not in published]
    local_names = {d.name for d in dirs}
    missing = sorted(s for s in published if s not in local_names)

    print(f"republish-all: {len(to_publish)} published meeting(s) to resync, "
          f"{len(skipped)} unpublished (skip), "
          f"{len(missing)} published with no local transcript.")

    if args.dry_run:
        print("\n(dry run — nothing published, reenrolled, or deployed)")
        for d in to_publish:
            print(f"  would re-publish: {d.name}")
        if skipped:
            print("  would skip (not published): " + ", ".join(d.name for d in skipped))
        print(f"  would reenroll: {'yes' if args.reenroll else 'no'}")
        print(f"  would deploy: {'no' if args.no_deploy else 'yes'}")
        return

    failed = []
    for d in to_publish:
        try:
            meeting, body_slug = _load_meeting_and_body(d)
            publish_meeting(meeting, body_slug, trigger_deploy=False)
            print(f"  ✅ {d.name}")
        except Exception as exc:  # noqa: BLE001 - continue-on-error; report + collect
            print(f"  ❌ {d.name}: {exc}")
            failed.append(d.name)

    if args.reenroll:
        print("Reenrolling voice profiles...")
        result = subprocess.run([sys.executable, "reenroll_profiles.py"], cwd=_REPO_DIR)
        if result.returncode != 0:
            print("  reenroll failed (see output above) — publishes are unaffected.")

    if not args.no_deploy:
        _trigger_render_deploy()

    print(f"\nDone: {len(to_publish) - len(failed)} published, {len(failed)} failed.")
    if skipped:
        print("  skipped (not published): " + ", ".join(d.name for d in skipped))
    if missing:
        print("  published but no local transcript (not re-published): " + ", ".join(missing))
    if failed:
        print("  FAILED: " + ", ".join(failed))
        sys.exit(1)


def _trigger_render_deploy() -> None:
    """POST the Render deploy hook (RENDER_DEPLOY_HOOK_URL), loading .env.local if needed."""
    import urllib.request

    hook = os.environ.get("RENDER_DEPLOY_HOOK_URL")
    if not hook:
        env_path = _REPO_DIR / ".env.local"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("RENDER_DEPLOY_HOOK_URL="):
                    hook = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not hook:
        print("  --deploy: RENDER_DEPLOY_HOOK_URL not set; skipping deploy.")
        return
    try:
        with urllib.request.urlopen(hook, data=b"", timeout=30) as resp:
            print(f"  Render deploy triggered (HTTP {resp.status}).")
    except Exception as exc:  # noqa: BLE001 - best-effort deploy ping
        print(f"  --deploy failed: {exc}")


def _bulk_relink_scan(args) -> None:
    """Enumerate unlinked named speakers into an editable YAML review file."""
    import yaml

    from src import config
    from src.bulk_relink import (
        DECISION_LINK, build_review_doc, enumerate_unlinked, suggest_link,
    )
    from src.enroll import load_profiles
    from src.essentials_client import EssentialsClientError
    from src.models import Meeting

    db = load_profiles()
    meetings = []
    dirs = sorted(d for d in config.MEETINGS_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith("."))
    for mdir in dirs:
        named = mdir / "transcript_named.json"
        if not named.exists():
            continue
        with open(named, "r", encoding="utf-8") as f:
            meetings.append(Meeting.from_dict(json.load(f)))

    speakers = enumerate_unlinked(meetings, db)
    speakers.sort(key=lambda s: (-s.meeting_count, s.normalized_name))

    try:
        for s in speakers:
            s.decision, s.candidates = suggest_link(s)
    except EssentialsClientError as exc:
        print(f"Essentials search failed ({exc}); aborting scan (no file written).")
        sys.exit(2)

    doc = build_review_doc(speakers)
    header = (
        "# Bulk relink review — edit then apply with:\n"
        f"#   python run_local.py --bulk-relink-apply {args.out}\n"
        "# For rows marked `review`: pick a candidate, set politician_id, and\n"
        "# change decision to `link`. Use `skip` to leave a speaker unlinked\n"
        "# (e.g. a moderator or a non-essentials local person).\n"
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    linked = sum(1 for s in speakers if s.decision == DECISION_LINK)
    print(f"Wrote {len(speakers)} unlinked speaker(s) to {args.out}")
    print(f"  {linked} auto-approved (link), {len(speakers) - linked} need review")



def _bulk_relink_apply(args) -> None:
    """Apply approved links from a bulk-relink review file through the engine."""
    import yaml

    from src import config
    from src.bulk_relink import (
        DECISION_LINK, DECISION_REVIEW, DECISION_SKIP,
        BulkRelinkParseError, parse_review_doc,
    )
    from src.checkpoint import PipelineState
    from src.enroll import load_profiles, save_profiles
    from src.models import Meeting
    from src.relink import relink_in_meeting, rekey_profile_for_link, resolve_link_target

    with open(args.bulk_relink_apply, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        decisions = parse_review_doc(data)
    except BulkRelinkParseError as exc:
        print(f"Review file invalid: {exc}")
        sys.exit(2)

    links = [d for d in decisions if d.decision == DECISION_LINK]
    review = [d for d in decisions if d.decision == DECISION_REVIEW]
    skipped = [d for d in decisions if d.decision == DECISION_SKIP]

    print(f"Plan: {len(links)} to link, {len(review)} still need review (skipped), "
          f"{len(skipped)} marked skip.")
    if not links:
        print("Nothing approved to link. Edit the file (set decision: link + politician_id) and re-run.")
        if review:
            print("Unresolved review rows: " + ", ".join(d.name for d in review))
        return

    # Resolve each approved name to id/slug/full_name (display-only; id is authoritative).
    targets = {}
    for d in links:
        t = resolve_link_target(d.name, explicit_id=d.politician_id)
        targets[d.name] = t

    # Load every meeting once; compute relinks in memory.
    dirs = sorted(p for p in config.MEETINGS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("."))
    loaded = []  # (mdir, Meeting)
    for mdir in dirs:
        named = mdir / "transcript_named.json"
        if not named.exists():
            continue
        with open(named, "r", encoding="utf-8") as f:
            loaded.append((mdir, Meeting.from_dict(json.load(f))))

    touched = {}      # mdir -> Meeting (transcript changed this run; needs a write)
    to_publish = {}   # mdir -> Meeting (contains an approved person linked to target)
    for d in links:
        t = targets[d.name]
        want = d.name.strip().lower()
        for mdir, meeting in loaded:
            changed = relink_in_meeting(meeting, d.name, t.politician_id, t.politician_slug)
            if changed:
                touched[mdir] = meeting
            # Publish any meeting where this approved person is now linked to the
            # target — even if nothing changed this run. Covers the
            # "linked-in-transcript-but-not-in-DB" case (e.g. a prior gated or
            # race-blocked publish). Publishing is an idempotent upsert, so
            # re-publishing an already-correct meeting is safe.
            if any((m.speaker_name or "").strip().lower() == want
                   and m.politician_id == t.politician_id
                   for m in meeting.speakers.values()):
                to_publish[mdir] = meeting

    if not to_publish:
        print("No meetings contain an approved speaker (nothing to publish).")
    print(f"Will relink {len(touched)} transcript(s); publish {len(to_publish)} meeting(s); "
          f"fold {len(links)} voice profile(s).")
    if review:
        print("Skipping unresolved review rows: " + ", ".join(d.name for d in review))

    if args.dry_run:
        print("\n(dry run — no transcript, profile, publish, or deploy writes)")
        for mdir in sorted(to_publish, key=lambda p: p.name):
            tag = "relink + publish" if mdir in touched else "publish (already linked)"
            print(f"  would {tag}: {mdir.name}")
        print(f"  would fold profiles: {', '.join(d.name for d in links)}")
        print(f"  would deploy: {'yes' if args.deploy else 'no'}")
        if review:
            print("  To finish the review rows: edit the file, then re-run "
                  f"--bulk-relink-apply {args.bulk_relink_apply}")
        return

    # Persist changed transcripts.
    for mdir, meeting in touched.items():
        atomic_write_json(mdir / "transcript_named.json", meeting.to_dict())

    # Fold each approved person's voice profile once.
    db = load_profiles()
    for d in links:
        t = targets[d.name]
        rekey_profile_for_link(db, d.name, politician_id=t.politician_id,
                               politician_slug=t.politician_slug, full_name=t.full_name)
    save_profiles(db)
    print(f"  Folded voice profiles for {len(links)} person(s).")

    # Publish each affected meeting.
    blocked = []
    for mdir in sorted(to_publish, key=lambda p: p.name):
        state = PipelineState(mdir)
        if not _may_publish(state.review_status, args.publish_anyway):
            print(f"  skip publish {mdir.name}: gate verdict '{state.review_status}' "
                  f"(re-run with --publish-anyway)")
            blocked.append(mdir.name)
            continue
        _publish_meeting_standalone(mdir.name, args.publish_anyway)

    if args.deploy:
        _trigger_render_deploy()

    # Closing summary.
    print(f"\nDone: linked {len(links)} person(s); published "
          f"{len(to_publish) - len(blocked)} meeting(s).")
    if blocked:
        print(f"  {len(blocked)} meeting(s) not published: {', '.join(blocked)}")
        print("    Resolve the blocker (race_id / chamber / gate), then re-run apply "
              "(it retries every affected meeting) or publish directly:")
        for mid in blocked:
            print(f"      python run_local.py --publish-meeting {mid} --publish-anyway")
    if review:
        print(f"  {len(review)} row(s) still need review: {', '.join(d.name for d in review)}")
        print(f"  Finish them: edit {args.bulk_relink_apply} (or re-run "
              f"--bulk-relink-scan for a fresh narrowed list), then --bulk-relink-apply again.")


def _relink_person(args) -> None:
    """Link a person to an essentials politician across all their meetings."""
    from src import config
    from src.checkpoint import PipelineState
    from src.enroll import load_profiles, save_profiles
    from src.models import Meeting
    from src.relink import RelinkAmbiguous, relink_in_meeting, rekey_profile_for_link, resolve_link_target

    name = args.relink_person

    # 1. Resolve the target politician (refuse on ambiguity).
    try:
        target = resolve_link_target(args.to_name or name, explicit_id=args.to_id)
    except RelinkAmbiguous as exc:
        print(f"Could not resolve '{exc.query}' to a single politician.")
        if exc.candidates:
            print("Candidates — re-run with --to-id <uuid>:")
            for c in exc.candidates:
                line = f"  {c.get('politician_id')}  {c.get('full_name')}"
                extra = " ".join(x for x in (c.get("office_title"), c.get("district_label")) if x)
                print(f"{line} — {extra}" if extra else line)
        else:
            print("  No essentials matches. Pass --to-id <uuid> explicitly.")
        sys.exit(2)

    print(f"Target: {target.full_name}  id={target.politician_id}  "
          f"slug={target.politician_slug or '(none)'}")

    # 2. Find appearances and compute the relink (in memory).
    changed: list[tuple] = []  # (meeting_dir, Meeting, labels)
    dirs = sorted(d for d in config.MEETINGS_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith("."))
    for mdir in dirs:
        if args.meeting and mdir.name != args.meeting:
            continue
        named = mdir / "transcript_named.json"
        if not named.exists():
            continue
        with open(named, "r", encoding="utf-8") as f:
            meeting = Meeting.from_dict(json.load(f))
        labels = relink_in_meeting(meeting, name, target.politician_id, target.politician_slug)
        if labels:
            changed.append((mdir, meeting, labels))

    if not changed:
        print(f"No meetings have an unlinked '{name}' speaker (nothing to do).")
        return

    print(f"\nWill relink '{name}' in {len(changed)} meeting(s):")
    for mdir, _meeting, labels in changed:
        print(f"  {mdir.name}: {', '.join(labels)}")

    if args.dry_run:
        print("\n(dry run — no transcript, profile, publish, or deploy writes)")
        print(f"  would fold the '{name}' voice profile into essentials:{target.politician_id}")
        print(f"  would re-publish: {', '.join(m.name for m, _x, _y in changed)}")
        print(f"  would deploy: {'yes' if args.deploy else 'no'}")
        return

    # 3. Persist the edited transcripts.
    for mdir, meeting, _labels in changed:
        atomic_write_json(mdir / "transcript_named.json", meeting.to_dict())
    print(f"  Saved {len(changed)} transcript(s).")

    # 4. Re-key the voice profile (cheap fold, no audio).
    db = load_profiles()
    key = rekey_profile_for_link(
        db, name,
        politician_id=target.politician_id, politician_slug=target.politician_slug,
        full_name=target.full_name,
    )
    if key:
        save_profiles(db)
        print(f"  Profile re-keyed -> {key}")
    else:
        print(f"  No existing voice profile for '{name}' — skipped (DB link still published).")

    # 5. Re-publish each changed meeting (respect the gate; skip blocked ones).
    for mdir, _meeting, _labels in changed:
        state = PipelineState(mdir)
        if not _may_publish(state.review_status, args.publish_anyway):
            print(f"  skip publish {mdir.name}: gate verdict '{state.review_status}' "
                  f"(re-run with --publish-anyway)")
            continue
        _publish_meeting_standalone(mdir.name, args.publish_anyway)

    # 6. Optional web redeploy.
    if args.deploy:
        _trigger_render_deploy()


def _fix_transcripts() -> None:
    """Re-correct speaker names in all existing transcripts using the roster.

    Walks through every meeting directory, loads transcript_named.json,
    applies roster corrections to speaker mappings and segments, saves
    the corrected transcript, and re-exports markdown/json/srt.
    """
    from src import config
    from src.export import export_all
    from src.models import Meeting
    from src.roster import add_alias, correct_speaker_name, load_roster

    roster = load_roster()
    if not roster:
        print("No council roster found. Cannot fix transcripts.")
        print(f"  Create one at: {config.CONFIG_DIR / 'council_roster.json'}")
        sys.exit(1)

    meetings_dir = config.MEETINGS_DIR
    if not meetings_dir.exists():
        print("No meetings directory found.")
        return

    meeting_dirs = sorted(
        d for d in meetings_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not meeting_dirs:
        print("No meetings found.")
        return

    print(f"Fixing transcripts using roster ({len(roster.members)} members)...")
    print()

    total_corrections = 0
    total_aliases = 0

    for mdir in meeting_dirs:
        named_path = mdir / "transcript_named.json"
        if not named_path.exists():
            continue

        with open(named_path, "r") as f:
            meeting = Meeting.from_dict(json.load(f))

        corrections = []

        # Fix speaker mappings
        for label, mapping in meeting.speakers.items():
            if mapping.speaker_name:
                corrected = correct_speaker_name(mapping.speaker_name, roster)
                if corrected != mapping.speaker_name:
                    corrections.append({
                        "label": label,
                        "original": mapping.speaker_name,
                        "corrected": corrected,
                    })
                    mapping.speaker_name = corrected

        # Fix segment speaker names
        for seg in meeting.segments:
            if seg.speaker_name:
                corrected = correct_speaker_name(seg.speaker_name, roster)
                if corrected != seg.speaker_name:
                    seg.speaker_name = corrected

        if corrections:
            # Save corrected transcript
            atomic_write_json(named_path, meeting.to_dict())

            # Re-export
            export_dir = mdir / "exports"
            export_all(meeting, export_dir)

            print(f"  {mdir.name}: {len(corrections)} correction(s)")
            for c in corrections:
                print(f"    {c['label']}: {c['original']} -> {c['corrected']}")
                # Auto-learn: add original name as alias for the corrected name
                if add_alias(None, c["corrected"], c["original"]):
                    print(f"      -> Added '{c['original']}' as alias for '{c['corrected']}'")
                    total_aliases += 1

            total_corrections += len(corrections)
        else:
            print(f"  {mdir.name}: no corrections needed")

    print(f"\nDone. {total_corrections} total correction(s) across {len(meeting_dirs)} meeting(s).")
    if total_aliases:
        print(f"  {total_aliases} new alias(es) auto-added to roster.")


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _persist_after_review(meeting_dir: Path, segments, embeddings, changes) -> None:
    """If the review performed any merges, rewrite diarization.json + embeddings.json."""
    if not any("merged_into" in c for c in changes):
        return
    diar_path = meeting_dir / "diarization.json"
    emb_path = meeting_dir / "embeddings.json"
    with open(diar_path, "w") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2)
    emb_out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in embeddings.items()}
    with open(emb_path, "w") as f:
        json.dump(emb_out, f)


def _apply_gate(meeting, meeting_dir: Path, state) -> dict:
    """Evaluate the confidence gate, write quality.json, mirror headline to state.

    Returns the full quality report dict. Pure-ish: only writes quality.json
    and the two state fields. Recomputed on every run that reaches Stage 4 so
    the verdict always reflects current attributions (incl. human review).
    """
    from src import quality

    report = quality.evaluate_meeting(meeting)
    with open(meeting_dir / "quality.json", "w") as f:
        json.dump(report, f, indent=2)

    state.review_status = report["verdict"]
    state.trusted_coverage = report["trusted_coverage"]
    state.save()

    print(
        f"  Gate: {report['verdict'].upper()} "
        f"(trusted={report['trusted_coverage']:.0%}, "
        f"effective={report['effective_coverage']:.0%}) — {report['reason']}"
    )
    return report


def _may_publish(review_status: str | None, publish_anyway: bool) -> bool:
    """Publishing is allowed only on a 'pass' verdict, unless forced by a human."""
    return publish_anyway or review_status == "pass"


def _meeting_body_slug(meeting_dir: Path) -> str | None:
    """Read the persisted body_slug from a meeting's pipeline_state.json, if any."""
    state_file = meeting_dir / "pipeline_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r") as f:
            return json.load(f).get("body_slug")
    except Exception:
        return None


def _review_queue() -> None:
    """List meetings awaiting review, grouped by verdict and ranked by coverage."""
    from src import config
    from src.checkpoint import PipelineState

    meetings_dir = config.MEETINGS_DIR
    if not meetings_dir.exists():
        print("No meetings directory found.")
        return

    review, failed, passed, unscored = [], [], 0, 0
    for mdir in sorted(d for d in meetings_dir.iterdir()
                       if d.is_dir() and not d.name.startswith(".")):
        if not (mdir / "pipeline_state.json").exists():
            continue
        state = PipelineState(mdir)
        cov = state.trusted_coverage if state.trusted_coverage is not None else 0.0
        row = (mdir.name, cov)
        if state.review_status == "review":
            review.append(row)
        elif state.review_status == "failed":
            failed.append(row)
        elif state.review_status == "pass":
            passed += 1
        else:
            unscored += 1

    review.sort(key=lambda r: r[1], reverse=True)
    failed.sort(key=lambda r: r[1], reverse=True)

    if not review and not failed:
        print(f"Review queue empty. ({passed} passing, {unscored} unscored)")
        return

    if review:
        print(f"NEEDS REVIEW ({len(review)}):")
        for name, cov in review:
            print(f"  {cov:5.0%}  {name}")
    if failed:
        print(f"\nLOW YIELD / FAILED ({len(failed)}) "
              f"— may need a roster or voice profiles before review is productive:")
        for name, cov in failed:
            print(f"  {cov:5.0%}  {name}")
    print(f"\nSummary: {len(review)} review, {len(failed)} failed, "
          f"{passed} passing, {unscored} unscored.")
    print("Review one with: python run_local.py --review <MEETING_ID>")


# ---------------------------------------------------------------------------
# Metadata defaults / interactive prompt helpers
# ---------------------------------------------------------------------------

CITY_DEFAULT = "Bloomington"
MEETING_TYPE_DEFAULT = "Regular Session"
EVENT_KIND_DEFAULT = "council"


def _resolve_metadata(args, *, allow_prompt: bool = True) -> None:
    """Fill args.city/date/meeting_type/event_kind for a new run.

    Per field, three modes:
      * supplied on the CLI            -> kept (event_kind validated vs the enum)
      * --default                      -> civic defaults applied silently for
                                          city/meeting_type/event_kind; date is
                                          NEVER defaulted and is always required
      * interactive TTY (no --default) -> prompt for each unset field
    Non-interactive (no TTY, --default, or batch_mode) with neither a CLI value
    nor an applicable default raises ValueError naming every missing field, so a
    run never silently guesses metadata. Prompt order: event_kind, city,
    meeting_type, date, title.

    Pass allow_prompt=False (resume) to suppress prompting entirely: metadata is
    expected to have been restored from saved state, so a still-unset field is a
    genuine gap that hard-fails (or is filled by --default) rather than guessed.
    """
    use_defaults = bool(getattr(args, "default", False))
    interactive = (
        allow_prompt
        and sys.stdin.isatty()
        and not use_defaults
        and not getattr(args, "batch_mode", False)
    )

    # event_kind first: it decides whether a city is required.
    if args.event_kind is not None:
        args.event_kind = validate_event_kind(args.event_kind)
    elif interactive:
        ans = input(
            f"  Event kind [{EVENT_KIND_DEFAULT}] ({'/'.join(EVENT_KINDS)}): "
        ).strip()
        args.event_kind = validate_event_kind(ans or EVENT_KIND_DEFAULT)
    elif use_defaults:
        args.event_kind = EVENT_KIND_DEFAULT

    requires_city = args.event_kind in ("council", "school_board")

    if args.city is None and requires_city:
        if interactive:
            ans = input(f"  City [{CITY_DEFAULT}]: ").strip()
            args.city = ans or CITY_DEFAULT
        elif use_defaults:
            args.city = CITY_DEFAULT

    if args.meeting_type is None:
        if interactive:
            ans = input(f"  Meeting type [{MEETING_TYPE_DEFAULT}]: ").strip()
            args.meeting_type = ans or MEETING_TYPE_DEFAULT
        elif use_defaults:
            args.meeting_type = MEETING_TYPE_DEFAULT

    if not args.date and interactive:
        while not args.date:
            args.date = input("  Date YYYY-MM-DD (required): ").strip()

    # Anything still unset means we could neither get it explicitly nor were
    # told to default it (or it has no default, like date). Fail loudly.
    missing = []
    if args.event_kind is None:
        missing.append("--event-kind")
    if requires_city and args.city is None:
        missing.append("--city")
    if args.meeting_type is None:
        missing.append("--meeting-type")
    if not args.date:
        missing.append("--date")
    if missing:
        raise ValueError(
            "Refusing to guess meeting metadata: missing "
            + ", ".join(missing)
            + f". Pass them explicitly, or pass --default for {CITY_DEFAULT} / "
            f"{MEETING_TYPE_DEFAULT} / {EVENT_KIND_DEFAULT} "
            "(--date is always required)."
        )

    if args.event_kind in INTERVIEW_KINDS and not args.title and interactive:
        ans = input("  Title (required for interview/media events): ").strip()
        args.title = ans or None


def _prompt_link_politician(mappings: dict, label: str, query: str) -> None:
    """Offer to link a just-named speaker to an essentials politician/candidate.

    No-op when the speaker is already linked (e.g. roster auto-match) or when
    not attached to a TTY. Best-effort: any search failure degrades to a manual
    slug paste or a skip — never blocks or crashes review.
    """
    from src import review
    from src.essentials_client import EssentialsClientError, search_politicians

    mapping = mappings.get(label)
    if mapping is None or mapping.politician_slug or mapping.politician_id:
        return
    if not sys.stdin.isatty():
        return

    def _do_search(q: str):
        try:
            return search_politicians(q)
        except EssentialsClientError as e:
            print(f"  (politician search unavailable: {e})")
            return None

    matches = _do_search(query)
    while True:
        if matches:
            print("  Link to a politician/candidate? (Enter = leave unlinked)")
            for i, mt in enumerate(matches):
                print(review.format_match_line(mt, i))
            prompt = "  [number] pick · [m] search again · [p] paste slug · [Enter/s] skip · [n] none/unlink: "
        else:
            prompt = "  Link politician? [m] search · [p] paste slug · [Enter/s] skip: "

        choice = input(prompt).strip()
        action, idx = review.parse_link_selection(choice, len(matches or []))

        if action == "skip":
            return
        if action == "none":
            review.link_speaker(mappings, label, None, None)
            print("  Left unlinked.")
            return
        if action == "search":
            q = input("    Search name: ").strip()
            if q:
                matches = _do_search(q)
            continue
        if action == "pick":
            mt = matches[idx]
            review.link_speaker(mappings, label, mt["politician_slug"], mt["politician_id"])
            print(f"  Linked → {mt['full_name']} ({mt['politician_slug']})")
            return
        # 'invalid' — allow a manual slug paste (handy when there are no matches).
        if choice.lower() == "p":
            slug = input("    politician_slug: ").strip()
            if slug:
                review.link_speaker(mappings, label, slug, None)
                print(f"  Linked → {slug} (id unknown)")
            return
        print("  Not understood.")


def _prompt_create_local_person(
    mappings: dict, label: str, name: str, event_kind: str | None = None
) -> None:
    """Offer to create a local person record for a non-essentials speaker.

    No-op when not attached to a TTY, when the mapping is missing, or when the
    speaker already has an essentials identity (politician_slug OR politician_id —
    essentials link wins). Slug validation is delegated to assign_local_person.
    """
    if not sys.stdin.isatty():
        return
    mapping = mappings.get(label)
    if mapping is None or mapping.politician_slug or mapping.politician_id:
        return

    print("  Not in essentials. Create as local person?")
    choice = input("  [L]ocal person / [Enter] leave unlinked: ").strip().lower()
    if choice != "l":
        return

    from src.event_kinds import local_roles_for, resolve_local_role
    from src.review import LOCAL_SLUG_RE, assign_local_person, default_local_slug

    default_slug = default_local_slug(name, label)
    slug_raw = input(f"  Slug [{default_slug}]: ").strip()
    slug = slug_raw or default_slug

    if not LOCAL_SLUG_RE.fullmatch(slug):
        print(f"  Invalid slug {slug!r} — left unlinked.")
        return

    roles = local_roles_for(event_kind)
    options = "  ".join(f"{i + 1}) {r}" for i, r in enumerate(roles))
    print(f"  Role: {options}")
    print("  (pick a number, or type a custom role)")
    role = resolve_local_role(input(f"  [1={roles[0]}]: "), event_kind)

    try:
        assign_local_person(mappings, label, slug, role)
    except ValueError as exc:
        print(f"  {exc} — left unlinked.")
        return
    print(f"  Local person: {slug} ({role})")


def _interactive_speaker_review(
    segments,
    mappings: dict,
    embeddings: dict,
    profile_db,
    video_path: str | None,
    audio_path: str | None,
    *,
    roster=None,
    body_slug: str | None = None,
    show_text: bool = True,
    event_kind: str | None = None,
    meeting_id: str | None = None,
    clip_offset: float = 0.0,
) -> list[dict]:
    """Interactive review loop built on the pure src/review.py core.

    Per speaker: play clips ([V] cycles through up to 8 candidates, [V3] jumps
    to clip 3, [R] replays the current one), accept the top voice hint ([Y]),
    merge this speaker into another ([M]), skip ([Enter]), quit ([Q]), or type
    a name.
    Mutates segments/mappings/embeddings in place via review.py; the CALLER
    persists (diarization.json / embeddings.json / transcript).

    Returns change dicts: {"label","old_name","new_name"} for renames and
    {"label","merged_into"} for merges.
    """
    from src import review

    if not sys.stdin.isatty():
        print("(non-interactive mode — cannot review)")
        return []

    changes: list[dict] = []
    # Undo stack: each entry is {"index": i, "snap": <snapshot_mapping result>},
    # pushed BEFORE a speaker's change is committed. [B]ack pops the last one,
    # restores the mapping + segment names, drops the matching tail change, and
    # re-presents the restored speaker.
    history: list[dict] = []
    views = review.build_review_state(segments, mappings, embeddings, profile_db, show_text=show_text)

    i = 0
    quit_requested = False
    while i < len(views):
        view = views[i]
        label = view.label
        name = view.current_name or "(unidentified)"
        mins = view.total_speech_seconds / 60

        # One snapshot per speaker visit, taken before any branch mutates. Undo
        # for THIS visit reverts to exactly this state.
        visit_snapshot = review.snapshot_mapping(mappings, segments, label)

        print(f"\n[{i+1}/{len(views)}] {label}: {name}")
        print(f"  Segments: {view.seg_count}, Speech: {mins:.1f}m", end="")
        if view.current_confidence > 0:
            print(f", Confidence: {view.current_confidence:.2f}, Method: {view.current_method or 'none'}", end="")
        print()

        top_hint = None
        if view.soft_hints and not (view.current_name and view.current_confidence >= 0.85):
            for hint_name, hint_score, _hint_pid in view.soft_hints[:3]:
                marker = "*" if hint_score >= 0.85 else "?"
                print(f"  {marker} Voice match: {hint_name} ({hint_score:.2f})")
            top_hint = view.soft_hints[0]

        if view.sample_text:
            preview = view.sample_text[:120] + "..." if len(view.sample_text) > 120 else view.sample_text
            print(f"  Sample [{_format_ts(view.clip_start or 0)}]: \"{preview}\"")
        elif view.clip_start is not None:
            print(f"  Clip at [{_format_ts(view.clip_start)}]")

        advance = True
        undo_requested = False
        clip_idx = 0        # index of the clip [V] plays next
        last_clip = None    # index of the most recently played clip
        current_player = None
        has_clip = bool((video_path or audio_path) and view.clip_candidates)
        n_clips = len(view.clip_candidates)

        def _push_undo() -> None:
            """Record this visit's pre-change state so [B]ack can revert it."""
            history.append({"index": i, "snap": visit_snapshot})

        def _play_clip(n: int) -> None:
            nonlocal current_player, last_clip, clip_idx
            _stop_player(current_player)
            start = view.clip_candidates[n]
            print(f"  Clip {n + 1}/{n_clips} at [{_format_ts(start)}]")
            current_player = play_speaker_clip(
                video_path, audio_path, start,
                title=f"{label} → {name} (clip {n + 1}/{n_clips})",
                video_offset=clip_offset,
            )
            last_clip = n
            clip_idx = n + 1

        try:
            while True:
                parts = ["  "]
                if has_clip:
                    if last_clip is None:
                        parts.append(f"[V]iew clip (1/{n_clips})")
                    else:
                        parts.append(f"[V]=next clip ({clip_idx % n_clips + 1}/{n_clips}) [R]eplay")
                if top_hint:
                    _top_pid = top_hint[2] if len(top_hint) > 2 else ""
                    if _top_pid.startswith("local:unidentified-"):
                        parts.append(f"[Y=returning unidentified: {top_hint[0]}]")
                    else:
                        parts.append(f"[Y=accept {top_hint[0]}]")
                if len(views) > 1:
                    parts.append("[M]erge")
                parts.append("[U]nidentified [X]=not a speaker")
                if history:
                    parts.append("[B]ack")
                parts.append("[Enter=skip] [Q=quit] or type a name (/ first for V/R/Y/M/Q/U/X/B names): ")
                # Single keypresses fire instantly (cbreak); when a clip is
                # playing, re-assert terminal focus first so keys don't land in
                # ffplay's video window. Falls back to line input off-TTY.
                choice = _read_review_command(
                    " ".join(parts), refocus=current_player is not None
                ).strip()
                lower = choice.lower()

                if lower in ("v", "view") and has_clip:
                    _play_clip(clip_idx % n_clips)
                    continue
                elif lower.startswith("v") and lower[1:].isdigit() and has_clip:
                    n = int(lower[1:]) - 1
                    if not 0 <= n < n_clips:
                        print(f"  No clip {lower[1:]} — this speaker has {n_clips} clip(s).")
                        continue
                    _play_clip(n)
                    continue
                elif lower in ("r", "replay") and has_clip:
                    _play_clip(last_clip if last_clip is not None else 0)
                    continue
                elif choice.lower() == "b":
                    if not history:
                        print("  (nothing to undo)")
                        continue
                    entry = history.pop()
                    if entry.get("kind") == "merge":
                        # Merge undo is intentionally unsupported: snapshot/restore
                        # is name-based and cannot revert the relabeled segments or
                        # the combined embeddings, so reverting would leave a
                        # half-merged state. Refuse cleanly — the popped entry stays
                        # popped, so a further [B] undoes the action before the merge.
                        print("  (can't undo a merge — speakers are already combined; re-run review if needed)")
                        continue
                    undo_label = entry["snap"]["label"]
                    review.restore_mapping(mappings, segments, undo_label, entry["snap"])
                    # Drop the most recent change recorded for that label (the one
                    # this snapshot was taken before committing).
                    for j in range(len(changes) - 1, -1, -1):
                        if changes[j].get("label") == undo_label:
                            del changes[j]
                            break
                    print(f"  Undid last change to {undo_label}.")
                    undo_requested = True
                    advance = False
                    break
                elif choice.lower() == "q":
                    print("  Quitting review.")
                    quit_requested = True
                    break
                elif choice == "":
                    # Warn if keeping this name would duplicate another speaker's name —
                    # easy to miss when the LLM guesses the same person for multiple labels.
                    kept_name = view.current_name
                    if kept_name:
                        dupes = [
                            v for v in views
                            if v.label != label and v.current_name == kept_name
                        ]
                        if dupes:
                            dupe_labels = ", ".join(v.label for v in dupes)
                            print(f"  ⚠  '{kept_name}' is also assigned to {dupe_labels}. "
                                  f"Type a different name or [M]erge if they're the same person.")
                            continue
                    break  # skip
                elif choice.lower() == "m" and len(views) > 1:
                    others = [v for v in views if v.label != label]
                    print("  Merge THIS speaker into which?")
                    for k, ov in enumerate(others):
                        print(f"    {k+1}. {ov.label}: {ov.current_name or '(unidentified)'}")
                    sel = input("    Number (or Enter to cancel): ").strip()
                    if not sel:
                        continue
                    try:
                        target = others[int(sel) - 1]
                    except (ValueError, IndexError):
                        print("    Invalid selection.")
                        continue
                    _push_undo()
                    # Tag this entry as a merge: undo is name-based and cannot
                    # revert the relabeled segments/embeddings, so [B] refuses it
                    # rather than half-reverting (see test_review_ux.py).
                    history[-1]["kind"] = "merge"
                    try:
                        res = review.merge_speakers(segments, embeddings, mappings, label, target.label)
                    except ValueError as e:
                        print(f"    {e}")
                        history.pop()  # merge failed — discard the undo entry
                        continue
                    changes.append({"label": label, "merged_into": target.label})
                    print(f"  Merged {label} → {target.label} ({res.combined_name or 'unidentified'})")
                    views = review.build_review_state(segments, mappings, embeddings, profile_db, show_text=show_text)
                    advance = False
                    break
                elif choice.lower() in ("y", "yes") and top_hint:
                    _push_undo()
                    top_pid = top_hint[2] if len(top_hint) > 2 else ""
                    if top_pid.startswith("local:unidentified-"):
                        # Returning unknown: confirm-link to the SAME existing
                        # handle so the recurring speaker enrolls into one profile.
                        old_name = mappings.get(label).speaker_name if mappings.get(label) else name
                        review.link_to_unidentified_handle(
                            mappings, segments, label,
                            handle_key=top_pid, display_name=top_hint[0],
                        )
                        changes.append({"label": label, "old_name": old_name, "new_name": mappings[label].speaker_name})
                        print(f"  Confirmed returning unidentified speaker: {mappings[label].speaker_name}")
                        break
                    res = review.rename_speaker(mappings, segments, label, top_hint[0], roster=roster)
                    mappings[label].id_method = "human_confirmed"
                    changes.append({"label": label, "old_name": res.old_name, "new_name": res.new_name})
                    print(f"  Confirmed: {label} -> {res.new_name}")
                    _prompt_link_politician(mappings, label, res.new_name)
                    # Offer local person creation when essentials link was skipped or unavailable.
                    _prompt_create_local_person(mappings, label, res.new_name, event_kind=event_kind)
                    break
                elif choice.lower() == "u":
                    lbl = input("    Optional label (Enter for 'Unidentified Speaker'): ").strip()
                    _push_undo()
                    review.mark_unidentified(mappings, segments, label, meeting_id or "", display_label=lbl or None)
                    changes.append({"label": label, "old_name": name, "new_name": mappings[label].speaker_name})
                    print(f"  Unidentified: {label} -> {mappings[label].speaker_name} (handle {mappings[label].local_slug})")
                    break
                elif choice.lower() == "x":
                    lbl = input("    Optional label (Enter for 'Non-speaker'): ").strip()
                    _push_undo()
                    review.mark_non_speaker(mappings, segments, label, display_label=lbl or None)
                    changes.append({"label": label, "old_name": name, "new_name": mappings[label].speaker_name})
                    print(f"  Marked non-speaker: {label} -> {mappings[label].speaker_name}")
                    break
                else:
                    _push_undo()
                    res = review.rename_speaker(mappings, segments, label, choice, roster=roster)
                    changes.append({"label": label, "old_name": res.old_name, "new_name": res.new_name})
                    print(f"  Updated: {label} -> {res.new_name}")
                    if res.alias_suggestion:
                        from src.roster import add_alias
                        if add_alias(None, res.new_name, res.alias_suggestion, body_slug=body_slug):
                            target_label = body_slug or "council_roster.json"
                            print(f"  Auto-added alias: '{res.alias_suggestion}' -> '{res.new_name}' ({target_label})")
                    _prompt_link_politician(mappings, label, res.new_name)
                    # Offer local person creation when essentials link was skipped or unavailable.
                    _prompt_create_local_person(mappings, label, res.new_name, event_kind=event_kind)
                    break
        finally:
            _stop_player(current_player)

        if quit_requested:
            break
        if undo_requested:
            # Rebuild views (an undone merge may restore a label that had dropped
            # out of the table) and re-present the restored speaker. Labels are
            # stable across rebuilds, so locate it by label rather than trusting
            # the old positional index (sort order can shift).
            undo_label = entry["snap"]["label"]
            views = review.build_review_state(segments, mappings, embeddings, profile_db, show_text=show_text)
            i = next((k for k, v in enumerate(views) if v.label == undo_label), entry["index"])
            i = min(i, max(len(views) - 1, 0))
            continue
        if advance:
            i += 1

    return changes


def _enroll_after_review(
    changes: list[dict],
    current_mappings: dict,
    meeting_dir: Path,
    segments,
    roster=None,
) -> None:
    """Offer to enroll speakers that were identified or corrected during review.

    Only runs if embeddings are available on disk.

    The meeting id stamped into each EmbeddingRecord is ALWAYS the meeting
    directory basename (``meeting_dir.name``), never a caller-supplied or
    persisted ``meeting.meeting_id`` value. Calibration's leave-one-out
    (``calibrate_gate._decontaminated_centroids``) keys on the same
    ``meeting_dir.name``; if enrollment stamped anything else the scored meeting
    could fail to exclude its own embeddings and silently grade itself.
    """
    import numpy as np

    meeting_id = meeting_dir.name

    from src import review
    from src.enroll import (
        _enroll_mapping,
        load_profiles,
        resolve_mapping_enrollment,
        save_profiles,
    )
    from src.models import SpeakerMapping

    embeddings_path = meeting_dir / "embeddings.json"
    if not embeddings_path.exists():
        return

    if not changes:
        return

    if not sys.stdin.isatty():
        return

    with open(embeddings_path, "r") as f:
        emb_data = json.load(f)
    speaker_embeddings = {k: np.array(v) for k, v in emb_data.items()}

    # Find which changed speakers have embeddings available
    enrollable = []
    profile_db = load_profiles()

    for change in changes:
        if "merged_into" in change:
            continue  # merges aren't renames — nothing to enroll here
        label = change["label"]
        new_name = change["new_name"]
        if not new_name or label not in speaker_embeddings:
            continue
        # Show the exact key the speaker will enroll under — same resolver
        # _enroll_mapping uses, so the NEW/UPDATE tag can't drift from reality.
        mapping = current_mappings.get(label) or SpeakerMapping(
            speaker_label=label, speaker_name=new_name)
        slug, _, _ = resolve_mapping_enrollment(mapping)
        is_new = slug not in profile_db.profiles
        enrollable.append({
            "label": label,
            "name": new_name,
            "slug": slug,
            "is_new": is_new,
        })

    if not enrollable:
        return

    print(f"\n{len(enrollable)} speaker(s) can be enrolled/updated in voice profiles:")
    for e in enrollable:
        tag = "NEW" if e["is_new"] else "UPDATE"
        print(f"  {e['label']}: {e['name']} [{tag}]")

    # Pre-enroll safety check — surface suspicious states (name/slug mismatch,
    # duplicate names, unlinked roster matches) as non-blocking warnings before
    # the confirmation. This is the backstop that catches contaminated links
    # before they are written to a voice profile.
    warns = review.enrollment_warnings(current_mappings, roster)
    for w in warns:
        print(f"  ⚠ [{w['label']}] {w['detail']}")

    choice = input("\nEnroll these speakers? [Y/n] ").strip().lower()
    if choice in ("", "y", "yes"):
        for e in enrollable:
            mapping = current_mappings.get(e["label"]) or SpeakerMapping(
                speaker_label=e["label"], speaker_name=e["name"])
            seg_count = sum(1 for s in segments if s.speaker_label == e["label"])
            # roster=None on purpose: identity here comes from the mapping itself
            # (set when the speaker was linked during review), not a roster lookup.
            _enroll_mapping(
                profile_db, mapping,
                speaker_embeddings[e["label"]],
                meeting_id, seg_count, None,
            )
            tag = "NEW" if e["is_new"] else "UPDATE"
            print(f"  Enrolled: {e['name']} ({e['slug']}) [{tag}]")

        save_profiles(profile_db)
        print(f"  Voice profiles saved ({len(profile_db.profiles)} total)")
    else:
        print("  Skipped enrollment.")


def _review_meeting(meeting_id: str) -> None:
    """Interactively review and correct all speakers in an existing meeting."""
    from src import config
    from src.export import export_all
    from src.models import Meeting, SpeakerMapping

    meeting_dir = config.MEETINGS_DIR / meeting_id
    named_path = meeting_dir / "transcript_named.json"

    if not named_path.exists():
        print(f"No transcript found for meeting: {meeting_id}")
        print(f"  Expected at: {named_path}")
        available = sorted(
            d.name for d in config.MEETINGS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ) if config.MEETINGS_DIR.exists() else []
        if available:
            print(f"  Available meetings: {', '.join(available)}")
        sys.exit(1)

    with open(named_path, "r") as f:
        meeting = Meeting.from_dict(json.load(f))

    video_path = find_video_file(meeting_dir, meeting.audio_source)
    embeddings_path = meeting_dir / "embeddings.json"

    import numpy as np
    from src.enroll import load_profiles
    from src import review as _review

    if embeddings_path.exists():
        with open(embeddings_path, "r") as f:
            _emb = json.load(f)
        embeddings = {k: np.array(v) for k, v in _emb.items()}
    else:
        embeddings = {}
    profile_db = load_profiles()
    body_slug = _meeting_body_slug(meeting_dir)

    views = _review.build_review_state(meeting.segments, meeting.speakers, embeddings, profile_db, show_text=True)

    print(f"\nReviewing: {meeting.city} {meeting.meeting_type} ({meeting.date})")
    print(f"Meeting ID: {meeting_id}")
    if video_path:
        print(f"Video: {Path(video_path).name}")
    else:
        print("Video: not found (no clip playback available)")
    print(f"Speakers: {len(views)}")
    print()

    # Show overview table
    print("  #  Label         Current Name                  Identity                Segs  Speech  Conf   Method")
    print("  " + "-" * 98)
    for i, v in enumerate(views):
        name = v.current_name or "(unidentified)"
        method = v.current_method or ""
        mins = v.total_speech_seconds / 60
        identity = _review.identity_label(meeting.speakers.get(v.label))
        if len(identity) > 22:
            identity = identity[:21] + "…"
        hint = ""
        if v.soft_hints and not (v.current_name and v.current_confidence >= 0.85):
            top = v.soft_hints[0]
            hint = f"  ~ {top[0]} ({top[1]:.2f})"
        print(f"  {i+1:>2}  {v.label:<13} {name:<30} {identity:<22}  {v.seg_count:>4}  {mins:>5.1f}m  {v.current_confidence:.2f}  {method}{hint}")

    print()
    print("Commands for each speaker:")
    print("  [Enter]  Skip (keep current name)")
    print("  [V]      View video clip of this speaker")
    print("  [Y]      Accept suggested voice match (if shown)")
    print("  [M]      Merge this speaker into another")
    print("  [name]   Type a new name to assign")
    print("  [U]      Mark unidentified (distinct person, unnamed)")
    print("  [X]      Mark not a speaker (music/pledge/station ID)")
    print("  [Q]      Quit review (save changes so far)")
    print()

    changes = _interactive_speaker_review(
        meeting.segments, meeting.speakers, embeddings, profile_db,
        video_path, _review_audio_path(meeting_dir),
        body_slug=body_slug, show_text=True,
        event_kind=meeting.event_kind,
        meeting_id=meeting_id,
        clip_offset=meeting.clip_start_seconds or 0.0,
    )
    _persist_after_review(meeting_dir, meeting.segments, embeddings, changes)

    # Apply corrections to segments and save
    if changes:
        for seg in meeting.segments:
            m = meeting.speakers.get(seg.speaker_label)
            if m and m.speaker_name:
                seg.speaker_name = m.speaker_name
                seg.confidence = m.confidence
                seg.id_method = m.id_method

        atomic_write_json(named_path, meeting.to_dict())

        export_dir = meeting_dir / "exports"
        export_all(meeting, export_dir)

        print(f"\n{len(changes)} correction(s) saved:")
        for c in changes:
            if "merged_into" in c:
                print(f"  {c['label']}: merged into {c['merged_into']}")
                continue
            old = c["old_name"] or "(unidentified)"
            print(f"  {c['label']}: {old} -> {c['new_name']}")
        print(f"Exports updated: {export_dir}")

        # Offer enrollment
        from src.roster import load_roster
        _enroll_after_review(
            changes, meeting.speakers, meeting_dir, meeting.segments,
            roster=load_roster(body_slug=body_slug) if body_slug else None,
        )
    else:
        print("\nNo changes made.")

    # Recompute the confidence gate so the persisted verdict reflects this
    # review's corrections. Keeps a direct --publish-meeting honest if the
    # operator publishes without going back through --resume (which would
    # otherwise re-enter Stage 4 and recompute). Runs even with no changes so
    # meetings reviewed before the gate existed get scored on first open.
    from src.checkpoint import PipelineState
    _apply_gate(meeting, meeting_dir, PipelineState(meeting_dir))


def _identify_speakers_standalone(meeting_id: str) -> None:
    """Standalone pre-identification for an existing meeting.

    Works on any meeting that has diarization + embeddings.
    Does not require transcription to be complete.
    """
    from src import config
    from src.models import Meeting, SpeakerMapping

    meeting_dir = config.MEETINGS_DIR / meeting_id
    diarization_path = meeting_dir / "diarization.json"
    embeddings_path = meeting_dir / "embeddings.json"

    if not diarization_path.exists():
        print(f"No diarization found for meeting: {meeting_id}")
        print(f"  Expected at: {diarization_path}")
        available = sorted(
            d.name for d in config.MEETINGS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ) if config.MEETINGS_DIR.exists() else []
        if available:
            print(f"  Available meetings: {', '.join(available)}")
        sys.exit(1)

    # Load segments (prefer transcribed, fall back to diarization-only)
    from src.models import Segment

    transcript_path = meeting_dir / "transcript_raw.json"
    named_path = meeting_dir / "transcript_named.json"
    has_text = False

    # event_kind / clip window come from the named transcript when present, else
    # from pipeline state — this command runs before naming is complete, so
    # `meeting` may be None (diarization- or transcript-only paths below).
    from src.checkpoint import PipelineState
    _state = PipelineState(meeting_dir)
    meeting = None

    if named_path.exists():
        with open(named_path, "r") as f:
            meeting = Meeting.from_dict(json.load(f))
        segments = meeting.segments
        current_mappings = meeting.speakers
        has_text = any(s.text for s in segments)
    elif transcript_path.exists():
        with open(transcript_path, "r") as f:
            segments = [Segment.from_dict(d) for d in json.load(f)]
        current_mappings = {}
        has_text = any(s.text for s in segments)
    else:
        with open(diarization_path, "r") as f:
            segments = [Segment.from_dict(d) for d in json.load(f)]
        current_mappings = {}

    video_path = find_video_file(meeting_dir, "")

    import numpy as np
    from src.enroll import load_profiles
    from src import review as _review

    if embeddings_path.exists():
        with open(embeddings_path, "r") as f:
            _emb = json.load(f)
        embeddings = {k: np.array(v) for k, v in _emb.items()}
    else:
        embeddings = {}
    profile_db = load_profiles()
    body_slug = _meeting_body_slug(meeting_dir)

    views = _review.build_review_state(segments, current_mappings, embeddings, profile_db, show_text=has_text)

    print(f"\nSpeaker Identification: {meeting_id}")
    if video_path:
        print(f"Video: {Path(video_path).name}")
    else:
        print("Video: not found")
    if has_text:
        print("Transcript: available (text samples shown)")
    else:
        print("Transcript: not yet available (video clips only)")
    print(f"Speakers: {len(views)}")
    _hint_count = sum(1 for v in views if v.soft_hints)
    if _hint_count:
        print(f"Voice hints: {_hint_count} speaker(s) have possible profile matches")
    print()

    # Show overview table
    print("  #  Label         Current Name                  Segs  Speech  Voice Hint")
    print("  " + "-" * 85)
    for i, v in enumerate(views):
        name = v.current_name or "(unidentified)"
        mins = v.total_speech_seconds / 60
        hint = ""
        if v.soft_hints:
            top = v.soft_hints[0]
            if top[1] >= 0.85:
                hint = f"* {top[0]} ({top[1]:.2f})"
            else:
                hint = f"? {top[0]} ({top[1]:.2f})"
        print(f"  {i+1:>2}  {v.label:<13} {name:<30} {v.seg_count:>4}  {mins:>5.1f}m  {hint}")

    print()
    print("Commands for each speaker:")
    print("  [Enter]  Skip")
    print("  [V]      View video clip of this speaker")
    print("  [Y]      Accept suggested voice match (if shown)")
    print("  [M]      Merge this speaker into another")
    print("  [name]   Type a name to assign")
    print("  [U]      Mark unidentified (distinct person, unnamed)")
    print("  [X]      Mark not a speaker (music/pledge/station ID)")
    print("  [Q]      Quit (save changes so far)")
    print()

    changes = _interactive_speaker_review(
        segments, current_mappings, embeddings, profile_db,
        video_path, _review_audio_path(meeting_dir),
        body_slug=body_slug, show_text=has_text,
        event_kind=meeting.event_kind if meeting else _state.event_kind,
        meeting_id=meeting_id,
        clip_offset=(meeting.clip_start_seconds if meeting else _state.clip_start_seconds) or 0.0,
    )
    _persist_after_review(meeting_dir, segments, embeddings, changes)

    if changes:
        # Save identifications as pre_identifications.json
        pre_id_path = meeting_dir / "pre_identifications.json"
        pre_ids = {}
        for label, mapping in current_mappings.items():
            if isinstance(mapping, SpeakerMapping) and mapping.speaker_name:
                pre_ids[label] = {
                    "speaker_name": mapping.speaker_name,
                    "confidence": mapping.confidence,
                    "id_method": mapping.id_method,
                }
        with open(pre_id_path, "w") as f:
            json.dump(pre_ids, f, indent=2)

        print(f"\n{len(changes)} identification(s) saved to {pre_id_path.name}")
        for c in changes:
            if "merged_into" in c:
                print(f"  {c['label']}: merged into {c['merged_into']}")
                continue
            old = c["old_name"] or "(unidentified)"
            print(f"  {c['label']}: {old} -> {c['new_name']}")

        # If named transcript exists, update it too. Reuse the in-memory
        # `meeting` (loaded at the top of this function and mutated in place by
        # the review loop) — re-loading from disk here would discard merges.
        if named_path.exists():
            for seg in meeting.segments:
                m = meeting.speakers.get(seg.speaker_label)
                if m and m.speaker_name:
                    seg.speaker_name = m.speaker_name
                    seg.confidence = m.confidence
                    seg.id_method = m.id_method
            atomic_write_json(named_path, meeting.to_dict())
            from src.export import export_all
            export_dir = meeting_dir / "exports"
            export_all(meeting, export_dir)
            print(f"Transcript and exports updated.")

        # Offer enrollment
        from src.roster import load_roster
        _enroll_after_review(
            changes, current_mappings, meeting_dir, segments,
            roster=load_roster(body_slug=body_slug) if body_slug else None,
        )

        print("\nThese identifications will be used as ground truth in Stage 4")
        print("(overriding LLM/pattern matching for identified speakers).")
    else:
        print("\nNo identifications made.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CouncilScribe — Automated City Council Meeting Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  %(prog)s --input meeting.mp4 --city Bloomington --date 2026-02-10
  %(prog)s --input "https://catstv.net/..." --city Bloomington --date 2026-02-10
  %(prog)s --browse-catstv --city Bloomington
  %(prog)s --resume 2026-02-10-regular-session

Environment Variables:
  CS_DATA_DIR          Override data directory (default: ~/CouncilScribe)
  HF_TOKEN             HuggingFace API token (for pyannote model access)
""",
    )

    # Audio source
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", "-i",
        help="Path to audio/video file or URL (direct or CATS TV page)",
    )
    source.add_argument(
        "--browse-catstv",
        action="store_true",
        help="Browse CATS TV archive and select a meeting interactively",
    )
    source.add_argument(
        "--resume",
        metavar="MEETING_ID",
        help="Resume a previous meeting by its ID",
    )
    source.add_argument(
        "--house-floor", metavar="YYYY-MM-DD",
        help="Ingest a US House floor session by date from the House Clerk CDN",
    )

    # Meeting metadata
    parser.add_argument("--city", default=None,
                        help=f"City name (default: {CITY_DEFAULT}; prompted if omitted)")
    parser.add_argument("--date", default="", help="Meeting date (YYYY-MM-DD; prompted if omitted)")
    parser.add_argument("--meeting-type", default=None,
                        help="Meeting type or name, free text "
                             "(e.g. \"Regular Session\", \"Plan Commission\"; prompted if omitted)")
    parser.add_argument(
        "--title",
        default=None,
        help="Optional human display title; blank/omitted uses city + meeting type",
    )
    parser.add_argument(
        "--guest",
        default=None,
        help="Interview subject/guest name; recorded in pipeline_state for the "
             "library context line (display only).",
    )
    parser.add_argument(
        "--event-org",
        action="append",
        default=None,
        metavar="ORG",
        help="Producing/hosting organization; repeatable (e.g. --event-org CBS "
             "--event-org NBC). Published as 'Produced by ...'.",
    )
    parser.add_argument(
        "--event-kind",
        choices=EVENT_KINDS,
        default=None,
        help="Event format (default: council)",
    )
    parser.add_argument("--meeting-id", default="", help="Custom meeting ID (auto-generated if omitted)")

    # Processing options
    parser.add_argument("--num-speakers", type=int, default=0,
                        help="Expected number of speakers (0 = auto-detect)")
    parser.add_argument(
        "--clip",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Transcribe only the contiguous window START..END of the source "
             "(seconds, MM:SS, or HH:MM:SS). The site still plays/links the full "
             "source. Example: --clip 23:00 48:00",
    )
    parser.add_argument("--noise-reduce", action="store_true",
                        help="Apply spectral noise reduction to audio")
    parser.add_argument("--cookies", metavar="FILE",
                        help="Netscape-format cookies file for authenticated downloads "
                             "(e.g. private Facebook videos). Export from browser with "
                             "a 'Get cookies.txt' extension.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based speaker identification (Layer 3)")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Skip meeting summary generation (requires the active LLM backend's API key)")
    parser.add_argument("--confirm-enroll", action="store_true",
                        help="Interactively confirm enrollment for borderline speakers (0.70-0.85 confidence)")
    parser.add_argument("--merge", action="store_true",
                        help="Opt-in: collapse speakers whose voice embeddings exceed "
                             "SPEAKER_MERGE_THRESHOLD. Disabled by default — current "
                             "pyannote 3.1 doesn't fragment Bloomington audio in practice, "
                             "and embeddings have known NaN issues. See bench/diagnose_merge.py.")
    parser.add_argument("--use-vtt", action="store_true",
                        help="Use VTT subtitles instead of Whisper (auto-detected if captions.vtt exists)")
    parser.add_argument("--diarizer", choices=["oss", "api", "vibevoice", "api-recluster"],
                        default="oss",
                        help="Diarization backend. 'oss' uses local pyannote 3.1 "
                             "(default, free, ~50min/3hr meeting on L4). 'api' uses "
                             "pyannote.ai Precision-2 (cleaner segmentation, ~3min "
                             "for the same meeting, ~$0.45 per audio hour). "
                             "Requires PYANNOTE_AI_KEY in env. 'vibevoice' uses "
                             "Microsoft VibeVoice-ASR on Modal and requires "
                             "--compute modal. 'api-recluster' is not a diarization "
                             "backend to run from scratch: it is provenance-only, for "
                             "--redo transcribe on a meeting whose diarization.json/"
                             "embeddings.json were already hand-replaced by an offline "
                             "re-clustering of Precision-2's turns (see "
                             "bench/diagnose_merge.py-style recluster workflows). "
                             "Requires diarization to already be complete on disk. "
                             "Recommended per bench/FINDINGS.md.")
    parser.add_argument("--compute", choices=["local", "modal"], default="local",
                        help="Compute backend for GPU-intensive stages (diarization "
                             "with --diarizer oss, and Whisper transcription). "
                             "'local' runs on this machine (default). "
                             "'modal' offloads to Modal cloud GPUs — requires modal "
                             "installed and authenticated (modal token new). "
                             "Has no effect when --diarizer api is used (pyannote.ai "
                             "is always remote).")
    parser.add_argument("--diarize-chunk-minutes", type=int, metavar="N",
                        help="Diarize in N-minute windows fanned out across Modal "
                             "containers and stitch speakers globally (0 = single "
                             "pass; default from config.DIARIZE_CHUNK_MINUTES). "
                             "Diarization cost is ~quadratic in window length, so "
                             "this is the main speedup for long meetings.")
    parser.add_argument(
        "--congressional-record", nargs=2, metavar=("DATE", "CHAMBER"), default=None,
        help="Resolve speakers from the Congressional Record for a floor session: "
             "DATE (YYYY-MM-DD) and CHAMBER (house|senate).")
    parser.add_argument("--default", action="store_true",
                        help="Skip metadata prompts and use civic defaults "
                             f"({CITY_DEFAULT} / {MEETING_TYPE_DEFAULT} / "
                             f"{EVENT_KIND_DEFAULT}); --date is still required")

    # Utilities
    parser.add_argument("--list-profiles", action="store_true",
                        help="List stored voice profiles and exit")
    parser.add_argument("--fix-profiles", action="store_true",
                        help="Rename stored voice profiles using the council roster and exit")
    parser.add_argument("--fix-transcripts", action="store_true",
                        help="Re-correct speaker names in all existing transcripts using the roster and re-export")
    parser.add_argument(
        "--repair-transcript",
        metavar="MEETING_ID",
        help="Rebuild one processed caption-backed transcript/exports without "
             "rerunning diarization or speaker identification.",
    )
    parser.add_argument("--publish", action="store_true",
                        help="After the pipeline completes, publish the meeting to Supabase for the web site")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip publishing even when resuming (overrides the auto-publish default on --resume)")
    parser.add_argument("--publish-anyway", action="store_true",
                        help="Force publishing even when the confidence gate "
                             "verdict is 'review' or 'failed' (human override)")
    parser.add_argument("--publish-meeting", metavar="MEETING_ID",
                        help="Publish an already-processed meeting to Supabase and exit")
    parser.add_argument("--align-agenda", metavar="MEETING_ID",
                        help="Align a published meeting's agenda items to its video "
                             "and flip them to 'happened' in place, then exit "
                             "(run after --publish-meeting; MEETING_ID must be the "
                             "scheduled slug, e.g. bloomington-city-council-2026-07-29)")
    parser.add_argument("--reconcile-memo", metavar="MEETING_ID",
                        help="Reconcile a published meeting's item outcomes and "
                             "votes from the clerk's post-meeting Memorandum, "
                             "then exit (MEETING_ID is the meeting slug; re-run "
                             "only when the clerk re-posts the memo)")
    parser.add_argument("--check-memo", metavar="MEETING_ID",
                        help="Read-only: recompute the memo reconcile plan and "
                             "report drift against the DB, then exit non-zero "
                             "on drift")
    parser.add_argument("--merge-profiles", nargs=2, metavar=("SOURCE", "DEST"),
                        help="Merge SOURCE profile into DEST profile and exit (use slugs from --list-profiles)")
    parser.add_argument("--relink-person", metavar="NAME",
                        help="Link a speaker (by transcript name) to an essentials politician across "
                             "every meeting they appear in, re-key their voice profile, and re-publish")
    parser.add_argument("--to-id", metavar="POLITICIAN_ID",
                        help="Target essentials politician_id for --relink-person (skips name search)")
    parser.add_argument("--to-name", metavar="NAME",
                        help="Search essentials by this name instead of --relink-person's value")
    parser.add_argument("--meeting", metavar="MEETING_ID",
                        help="Restrict --relink-person to a single meeting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview --relink-person changes without writing anything")
    parser.add_argument("--deploy", action="store_true",
                        help="After --relink-person publishes, trigger the Render web rebuild")
    parser.add_argument("--bulk-relink-scan", action="store_true",
                        help="Enumerate every unlinked named speaker across all meetings into "
                             "an editable YAML review file with suggested essentials matches")
    parser.add_argument("--bulk-relink-apply", metavar="REVIEW_FILE",
                        help="Apply approved links from a bulk-relink review file: relink "
                             "transcripts, fold voice profiles, re-publish (auto-resolving "
                             "debate race_id), and optionally redeploy")
    parser.add_argument("--out", metavar="PATH", default="bulk_relink_review.yaml",
                        help="Output path for --bulk-relink-scan (default: ./bulk_relink_review.yaml)")
    parser.add_argument("--republish-all", action="store_true",
                        help="Re-publish every already-published meeting (resync live data), "
                             "then trigger one web deploy. Add --reenroll to also rebuild the "
                             "voice-profile DB; --no-deploy to skip the rebuild.")
    parser.add_argument("--reenroll", action="store_true",
                        help="With --republish-all: also rebuild the voice-profile DB "
                             "(runs reenroll_profiles.py before the deploy)")
    parser.add_argument("--no-deploy", action="store_true",
                        help="With --republish-all: skip the single Render rebuild at the end")
    parser.add_argument("--show-roster", action="store_true",
                        help="Display the current council roster and exit")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip the interactive speaker review at the end of a run")
    parser.add_argument("--review", metavar="MEETING_ID",
                        help="Review/correct/merge speakers in an existing meeting "
                             "(canonical; --review-meeting and --identify-speakers are aliases)")
    parser.add_argument("--review-meeting", metavar="MEETING_ID",
                        help="Interactively review and correct all speakers in an existing meeting (alias of --review)")
    parser.add_argument("--identify-speakers", metavar="MEETING_ID",
                        help="Standalone speaker identification with video clips and voice hints (works pre-transcription) (alias of --review)")
    parser.add_argument("--delete-meeting", metavar="MEETING_ID",
                        help="HARD-DELETE a meeting: remove its local folder and all meetings.* "
                             "rows from the live DB. Reports (never deletes) derived quotes.")
    parser.add_argument("--confirm", metavar="MEETING_ID", default="",
                        help="Non-interactive confirmation for --delete-meeting (must equal the id).")
    parser.add_argument("--cleanup-media", metavar="MEETING_ID",
                        help="Compress audio to Opus and delete the source video + WAV for one finalized meeting.")
    parser.add_argument("--cleanup-all", action="store_true",
                        help="Run media cleanup across every meeting (backfill). Skips unfinalized meetings.")
    parser.add_argument("--pre-identify", action="store_true",
                        help="Interactive speaker identification after diarization, before transcription (pipeline mode)")
    parser.add_argument("--batch", metavar="FILE_OR_DIR",
                        help="Batch mode: text file with one input per line (path or 'URL DATE'), or directory of videos")
    parser.add_argument("--batch-resume", action="store_true",
                        help="Resume an interrupted batch run (skip already-completed meetings)")
    parser.add_argument("--review-queue", action="store_true",
                        help="List meetings awaiting review (grouped by gate verdict) and exit")
    parser.add_argument(
        "--body",
        type=str,
        default=None,
        help="Governing body slug (e.g. bloomington-common-council). "
             "Persisted to pipeline_state.json on first run; omit on re-invocation.",
    )
    parser.add_argument(
        "--race-id",
        default=None,
        help="essentials.races UUID for a debate/forum",
    )
    parser.add_argument(
        "--force-retag",
        action="store_true",
        default=False,
        help="Overwrite a meeting's persisted body_slug. Rewinds stages 4-7. "
             "Requires --body.",
    )
    parser.add_argument(
        "--redo",
        choices=["ingest", "diarize", "transcribe", "identify", "summary", "all"],
        default=None,
        help="Re-run from this stage onward. Use with --resume or --input. "
             "'ingest' re-downloads and re-checks for captions. "
             "'all' re-runs everything from ingest.",
    )

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    if args.repair_transcript:
        cli_argv = sys.argv[1:]
        repair_conflict_map = {
            "--input": _option_supplied(cli_argv, "--input", "-i"),
            "--browse-catstv": _option_supplied(cli_argv, "--browse-catstv"),
            "--resume": _option_supplied(cli_argv, "--resume"),
            "--city": _option_supplied(cli_argv, "--city"),
            "--date": _option_supplied(cli_argv, "--date"),
            "--meeting-type": _option_supplied(cli_argv, "--meeting-type"),
            "--meeting-id": _option_supplied(cli_argv, "--meeting-id"),
            "--num-speakers": _option_supplied(cli_argv, "--num-speakers"),
            "--noise-reduce": _option_supplied(cli_argv, "--noise-reduce"),
            "--cookies": _option_supplied(cli_argv, "--cookies"),
            "--skip-llm": _option_supplied(cli_argv, "--skip-llm"),
            "--skip-summary": _option_supplied(cli_argv, "--skip-summary"),
            "--confirm-enroll": _option_supplied(cli_argv, "--confirm-enroll"),
            "--merge": _option_supplied(cli_argv, "--merge"),
            "--use-vtt": _option_supplied(cli_argv, "--use-vtt"),
            "--diarizer": _option_supplied(cli_argv, "--diarizer"),
            "--compute": _option_supplied(cli_argv, "--compute"),
            "--default": _option_supplied(cli_argv, "--default"),
            "--list-profiles": _option_supplied(cli_argv, "--list-profiles"),
            "--fix-profiles": _option_supplied(cli_argv, "--fix-profiles"),
            "--fix-transcripts": _option_supplied(cli_argv, "--fix-transcripts"),
            "--publish": _option_supplied(cli_argv, "--publish"),
            "--publish-meeting": _option_supplied(cli_argv, "--publish-meeting"),
            "--align-agenda": _option_supplied(cli_argv, "--align-agenda"),
            "--reconcile-memo": _option_supplied(cli_argv, "--reconcile-memo"),
            "--check-memo": _option_supplied(cli_argv, "--check-memo"),
            "--merge-profiles": _option_supplied(cli_argv, "--merge-profiles"),
            "--show-roster": _option_supplied(cli_argv, "--show-roster"),
            "--no-review": _option_supplied(cli_argv, "--no-review"),
            "--review": _option_supplied(cli_argv, "--review"),
            "--review-meeting": _option_supplied(cli_argv, "--review-meeting"),
            "--identify-speakers": _option_supplied(cli_argv, "--identify-speakers"),
            "--pre-identify": _option_supplied(cli_argv, "--pre-identify"),
            "--batch": _option_supplied(cli_argv, "--batch"),
            "--batch-resume": _option_supplied(cli_argv, "--batch-resume"),
            "--body": _option_supplied(cli_argv, "--body"),
            "--race-id": _option_supplied(cli_argv, "--race-id"),
            "--force-retag": _option_supplied(cli_argv, "--force-retag"),
            "--redo": _option_supplied(cli_argv, "--redo"),
            "--title": _option_supplied(cli_argv, "--title"),
            "--event-kind": _option_supplied(cli_argv, "--event-kind"),
        }
        repair_conflicts = [
            flag
            for flag, supplied in repair_conflict_map.items()
            if supplied
        ]
        if repair_conflicts:
            parser.error(
                "--repair-transcript cannot be combined with "
                + ", ".join(repair_conflicts)
            )
        _repair_transcript_standalone(args.repair_transcript)
        return

    try:
        _validate_diarizer_compute(args)
    except ValueError as exc:
        parser.error(str(exc))

    # D-12: --force-retag requires --body
    if args.force_retag and not args.body:
        parser.error("--force-retag requires --body <slug>")

    if args.redo and not args.resume and not args.input:
        parser.error("--redo requires --resume <MEETING_ID> or --input <URL/FILE>")

    # --- Utility commands ---
    if args.show_roster:
        from src import config
        from src.roster import load_roster
        roster = load_roster()
        if not roster:
            print("No council roster found.")
            print(f"  Create one at: {config.CONFIG_DIR / 'council_roster.json'}")
        else:
            print(f"Council Roster: {roster.city} {roster.body}")
            print(f"  {len(roster.members)} member(s):\n")
            for m in roster.members:
                print(f"  {m.name}")
                if m.aliases:
                    print(f"    Aliases: {', '.join(m.aliases)}")
        return

    if args.review_queue:
        _review_queue()
        return

    if args.list_profiles:
        from src.enroll import load_profiles
        db = load_profiles()
        if not db.profiles:
            print("No voice profiles stored yet.")
        else:
            print(f"Stored profiles ({len(db.profiles)}):")
            for pid, p in db.profiles.items():
                print(f"  {pid}: {p.display_name}")
                print(f"    Meetings: {', '.join(p.meetings_seen)}")
                print(f"    Confirmed segments: {p.total_segments_confirmed}")
                print(f"    Embeddings: {len(p.embeddings)}")
        return

    if args.fix_profiles:
        from src.enroll import fix_profiles_with_roster, load_profiles, save_profiles
        from src.roster import load_roster
        roster = load_roster()
        if not roster:
            print("No council roster found. Cannot fix profiles.")
            sys.exit(1)
        db = load_profiles()
        if not db.profiles:
            print("No voice profiles stored yet.")
            return
        print(f"Checking {len(db.profiles)} profile(s) against roster...")
        changes = fix_profiles_with_roster(db, roster)
        if changes:
            save_profiles(db)
            print(f"\nRenamed {len(changes)} profile(s):")
            for c in changes:
                print(f"  {c}")
            print(f"\nTotal profiles: {len(db.profiles)}")
            for pid, p in db.profiles.items():
                print(f"  {pid}: {p.display_name}")
        else:
            print("All profiles already match the roster. No changes needed.")
        return

    if args.merge_profiles:
        from src.enroll import load_profiles, merge_profiles, save_profiles
        source, dest = args.merge_profiles
        db = load_profiles()
        if source not in db.profiles:
            print(f"Source profile '{source}' not found.")
            print(f"Available: {', '.join(db.profiles.keys())}")
            sys.exit(1)
        if dest not in db.profiles:
            print(f"Destination profile '{dest}' not found.")
            print(f"Available: {', '.join(db.profiles.keys())}")
            sys.exit(1)
        src_p = db.profiles[source]
        dst_p = db.profiles[dest]
        print(f"Merging '{source}' ({src_p.display_name}) into '{dest}' ({dst_p.display_name})...")
        merge_profiles(db, source, dest)
        save_profiles(db)
        merged = db.profiles[dest]
        print(f"  Done. '{dest}' now has {len(merged.embeddings)} embeddings, "
              f"{merged.total_segments_confirmed} segments, "
              f"{len(merged.meetings_seen)} meetings")
        return

    if args.fix_transcripts:
        _fix_transcripts()
        return

    if args.cleanup_media:
        from src.cleanup import cleanup_meeting

        result = cleanup_meeting(args.cleanup_media)
        mb = result["reclaimed_bytes"] / 1_000_000
        print(f"[{result['status']}] {result['meeting_id']} — reclaimed {mb:.1f} MB")
        return 0

    if args.cleanup_all:
        from src.cleanup import backfill_all

        results = backfill_all()
        total = sum(r["reclaimed_bytes"] for r in results)
        cleaned = sum(1 for r in results if r["status"] == "cleaned")
        for r in results:
            if r["status"] not in ("already_clean", "not_finalized"):
                print(f"  [{r['status']}] {r['meeting_id']}")
        print(f"Cleaned {cleaned}/{len(results)} meetings — reclaimed {total / 1_000_000:.1f} MB total")
        return 0

    if args.delete_meeting:
        from src import purge

        slug = args.delete_meeting
        preview = purge.purge_meeting(slug, delete_db=False, delete_local=False)
        if preview["status"] == "invalid":
            print(f"Invalid meeting id: {slug!r}")
            return 1
        if preview["status"] == "not_found":
            print(f"No such meeting (no local folder and no live DB row): {slug}")
            return 1
        print("About to HARD-DELETE (irreversible):")
        print(_format_delete_summary(preview))
        confirmed = args.confirm == slug
        if not confirmed and sys.stdin.isatty():
            typed = input(f"\nType the meeting id to confirm deletion ({slug}): ").strip()
            confirmed = typed == slug
        if not confirmed:
            print("Aborted — confirmation did not match.")
            return 1
        result = purge.purge_meeting(slug)
        print("\nDeleted:")
        print(_format_delete_summary(result))
        return 0

    if args.publish_meeting:
        _publish_meeting_standalone(args.publish_meeting, getattr(args, "publish_anyway", False))
        return

    if args.align_agenda:
        from src.publish import align_and_flip

        align_and_flip(args.align_agenda)
        return

    if args.reconcile_memo:
        from src.publish import reconcile_memo

        reconcile_memo(args.reconcile_memo)
        return

    if args.check_memo:
        from src.publish import reconcile_memo

        result = reconcile_memo(args.check_memo, check=True)
        sys.exit(1 if result.get("drift") else 0)

    if args.relink_person:
        _relink_person(args)
        return

    if args.bulk_relink_scan:
        _bulk_relink_scan(args)
        return

    if args.bulk_relink_apply:
        _bulk_relink_apply(args)
        return

    if args.republish_all:
        _republish_all(args)
        return

    if args.review:
        # Canonical review: full post-transcription review when a named
        # transcript exists, else diarization-only identification.
        from src import config as _config
        if (_config.MEETINGS_DIR / args.review / "transcript_named.json").exists():
            _review_meeting(args.review)
        else:
            _identify_speakers_standalone(args.review)
        return

    if args.review_meeting:
        _review_meeting(args.review_meeting)
        return

    if args.identify_speakers:
        _identify_speakers_standalone(args.identify_speakers)
        return

    # --- Batch mode ---
    if args.batch:
        _run_batch(args)
        return

    # --- House floor: resolve the CDN source into --input BEFORE the source-required
    # validation below (populates args.input/event_kind/date/CREC from the date). ---
    _expand_house_floor(args)

    # --- CATS TV browser ---
    if args.browse_catstv:
        selected = browse_catstv()
        if selected is None:
            print("No meeting selected. Exiting.")
            return
        args.input = selected["video_url"]
        if selected["date"] and not args.date:
            args.date = selected["date"]
        if selected["name"]:
            args.meeting_type = selected["name"]
        print(f"\nSelected: {selected['name']} ({selected['date']})")
        print(f"  URL: {args.input}\n")

    named_transcript_loaded = False

    # --- Resume mode ---
    if args.resume:
        from src import config
        meeting_dir = config.MEETINGS_DIR / args.resume
        state_file = meeting_dir / "pipeline_state.json"
        if not state_file.exists():
            print(f"No checkpoint found for meeting ID: {args.resume}")
            print(f"  Expected at: {state_file}")
            sys.exit(1)

        # Load meeting metadata from named transcript or reconstruct
        named_path = meeting_dir / "transcript_named.json"
        named_transcript_loaded = named_path.exists()
        if named_transcript_loaded:
            with open(named_path, "r") as f:
                data = json.load(f)
            args.input = data.get("audio_source", "")
            args.city = data.get("city", args.city)
            args.date = data.get("date", args.date)
            args.meeting_type = data.get("meeting_type", args.meeting_type)
            args.title = data.get("title", args.title)
            args.event_kind = data.get("event_kind", args.event_kind)
            args.race_id = data.get("race_id", args.race_id)
        else:
            # Use the WAV file as input since audio is already ingested
            wav = meeting_dir / "audio.wav"
            if wav.exists():
                args.input = str(wav)
            else:
                print(f"Cannot resume: no audio.wav found in {meeting_dir}")
                sys.exit(1)
            # Load all persisted metadata from pipeline_state.json.
            from src.checkpoint import PipelineState as _PS
            _ps = _PS(meeting_dir)
            if args.event_kind is None and _ps.event_kind is not None:
                args.event_kind = _ps.event_kind
            if args.city is None and _ps.city is not None:
                args.city = _ps.city
            if not args.date and _ps.date is not None:
                args.date = _ps.date
            if args.meeting_type is None and _ps.meeting_type is not None:
                args.meeting_type = _ps.meeting_type

        args.meeting_id = args.resume
        # Auto-publish on resume unless the caller explicitly passed --no-publish.
        if not getattr(args, "no_publish", False):
            args.publish = True
        print(f"Resuming meeting: {args.resume}")


    # --- Validate ---
    if not args.input:
        parser.print_help()
        print("\nError: --input, --browse-catstv, or --resume is required.")
        sys.exit(1)

    # Resolve city/date/meeting-type (prompt unless --default / non-interactive).
    # Resume never prompts (metadata is restored from the existing meeting's
    # transcript_named.json / pipeline_state.json above); a field that state
    # could not restore hard-fails rather than being silently stamped with a
    # civic default, unless --default was passed.
    try:
        _resolve_metadata(args, allow_prompt=not args.resume)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    # --- Run ---
    run_pipeline(args)


if __name__ == "__main__":
    main()
