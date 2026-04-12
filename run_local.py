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
import sys
import time
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Phase 109: pre-Stage-1 fail-fast guard (CSMEETING-02, D-07/D-08/D-09/D-13)
# ---------------------------------------------------------------------------

_BODY_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
            f'[a-z0-9][a-z0-9_-]*',
            file=sys.stderr,
        )
        sys.exit(2)

    from src import config
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


def find_video_file(meeting_dir: Path, original_input: str) -> str | None:
    """Find the video file for a meeting, checking the meeting directory first.

    Returns path to video file, or None if not found.
    """
    # Check for downloaded source video in meeting directory (source.m4v, source.mp4, etc.)
    for ext in (".m4v", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
        candidate = meeting_dir / f"source{ext}"
        if candidate.exists():
            return str(candidate)

    # Check if original input is a local video file that still exists
    if original_input and not original_input.startswith(("http://", "https://")):
        p = Path(original_input)
        if p.exists() and p.suffix.lower() in (".m4v", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
            return str(p)

    return None


def play_video_clip(video_path: str, start_time: float, duration: float = 15.0, title: str = "") -> None:
    """Play a video clip using ffplay starting at the given timestamp.

    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds.
        duration: Duration to play in seconds.
        title: Window title.
    """
    import subprocess

    # Start a few seconds early to give visual context
    seek = max(0, start_time - 3.0)

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


def free_gpu_memory():
    """Release GPU memory (CUDA or MPS)."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full 6-stage pipeline."""
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
    meeting_dir = ensure_drive_structure(meeting_id)
    state = PipelineState(meeting_dir)

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
        state.body_slug = cli_body
        state.save()
    elif not cli_body and persisted_body and force_retag:
        # Should be unreachable: D-12 enforced at argparse (line 1835).
        raise AssertionError("--force-retag without --body bypassed D-12 guard")
    # else: D-05 (no flag, no persisted — legacy) or D-06 (no flag, persisted — silent read)

    effective_body_slug = state.body_slug  # used by Plan 02 guard + Plan 03 Stage 4

    if effective_body_slug:
        # D-01 / D-06: single info line for operator visibility
        print(f"Body: {effective_body_slug}")

    # Phase 109 D-07: fail fast if tagged meeting has no cached roster.
    # Must run before Stage 1 ingestion so operators don't burn GPU on a bad run.
    ensure_body_roster_cached(effective_body_slug)

    meeting = Meeting(
        meeting_id=meeting_id,
        city=args.city,
        date=args.date,
        meeting_type=args.meeting_type,
        audio_source=str(audio_path),
    )

    print(f"\nMeeting: {args.city} {args.meeting_type} ({args.date})")
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
        metadata = normalize_audio(audio_path, wav_path, noise_reduce=args.noise_reduce)
        elapsed = time.time() - t0
        meeting.duration_seconds = metadata["duration_seconds"]
        state.mark_complete(PipelineStage.INGESTED)
        print(f"  Done in {elapsed:.1f}s")

        # Try to download VTT if input is a CATS TV URL
        if not vtt_path.exists() and isinstance(audio_path, str) and "catstv" in audio_path:
            from src.download import download_vtt
            # Derive VTT URL from video URL
            m4v_base = audio_path.rsplit(".", 1)[0] if "." in audio_path else audio_path
            vtt_url = m4v_base + ".vtt"
            result = download_vtt(vtt_url, vtt_path)
            if result:
                print(f"  Downloaded VTT captions: {vtt_path.name}")
            else:
                print("  No VTT captions available from CATS TV")

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
        # Sub-step A: Diarization
        if diarization_path.exists():
            print("  Diarization file found. Loading instead of re-running...")
            with open(diarization_path, "r") as f:
                segments = [Segment.from_dict(d) for d in json.load(f)]
            print(f"  Loaded {len(segments)} segments from previous run")
        else:
            from src.diarize import load_diarization_pipeline, run_diarization

            print("  Running speaker diarization...")
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
    meeting.processing_metadata.diarization_model = config.DIARIZATION_MODEL
    print()

    # ======================================================================
    # Stage 2.5: Auto-merge fragmented speakers
    # ======================================================================
    if not args.no_merge:
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
        speaker_stats = _build_speaker_stats(segments)
        soft_matches = _load_soft_matches(embeddings_path)

        sorted_labels = sorted(
            speaker_stats.keys(),
            key=lambda l: speaker_stats[l]["total_speech"],
            reverse=True,
        )

        if video_path:
            print(f"  Video: {Path(video_path).name}")
        else:
            print("  Video: not found")
        if soft_matches:
            matched = sum(1 for l in sorted_labels if l in soft_matches)
            print(f"  Voice hints: {matched} speaker(s) have possible profile matches")
        print(f"  Speakers: {len(sorted_labels)}")
        print()

        # Build temporary mappings dict for the review
        from src.models import SpeakerMapping as SM
        temp_mappings = dict(pre_identifications)  # start with any existing
        for label in sorted_labels:
            if label not in temp_mappings:
                temp_mappings[label] = SM(speaker_label=label)

        changes = _interactive_speaker_review(
            sorted_labels, speaker_stats, temp_mappings,
            video_path, soft_matches=soft_matches, show_text=False,
        )

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
                changes, temp_mappings, meeting_dir,
                meeting_id, segments,
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
        save_raw_transcript,
        transcribe_segments,
    )

    transcript_path = meeting_dir / "transcript_raw.json"

    if state.is_complete(PipelineStage.TRANSCRIBED):
        print("  Already complete. Loading from checkpoint...")
        segments = load_raw_transcript(transcript_path)
        print(f"  Loaded {len(segments)} transcribed segments")
    elif use_vtt:
        from src.vtt_align import align_vtt_to_segments

        t0 = time.time()
        print(f"  Aligning VTT captions from {vtt_path.name}...")
        segments = align_vtt_to_segments(vtt_path, segments)
        elapsed = time.time() - t0

        meeting.processing_metadata.transcription_model = "vtt_alignment"
        save_raw_transcript(segments, transcript_path)
        state.mark_complete(PipelineStage.TRANSCRIBED)
        print(f"  Done in {elapsed:.1f}s")
    else:
        resume_from = state.transcription_progress
        if resume_from > 0:
            print(f"  Resuming from segment {resume_from}/{len(segments)}")
            if transcript_path.exists():
                segments = load_raw_transcript(transcript_path)

        t0 = time.time()
        whisper_model = load_whisper_model()

        model_name = config.WHISPER_MODEL_GPU if torch.cuda.is_available() else config.WHISPER_MODEL_CPU
        meeting.processing_metadata.transcription_model = model_name
        meeting.processing_metadata.gpu_used = torch.cuda.is_available()
        print(f"  Using model: {model_name}")

        def checkpoint_fn(current, total):
            save_raw_transcript(segments, transcript_path)
            state.update_transcription_progress(current, total)
            pct = (current / total) * 100
            print(f"  Checkpoint: {current}/{total} segments ({pct:.0f}%)")

        segments = transcribe_segments(
            whisper_model, wav_path, segments,
            checkpoint_callback=checkpoint_fn,
            resume_from=resume_from,
        )
        elapsed = time.time() - t0

        save_raw_transcript(segments, transcript_path)
        del whisper_model
        free_gpu_memory()
        state.mark_complete(PipelineStage.TRANSCRIBED)
        print(f"  Done in {elapsed:.1f}s")

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
    from src.roster import load_roster, roster_names_for_prompt

    named_transcript_path = meeting_dir / "transcript_named.json"
    llm_partial_path = meeting_dir / "llm_partial_results.json"

    # Phase 109 CSMEETING-03: load body-specific roster when meeting is tagged.
    # effective_body_slug comes from the Plan 01 resolve block; Plan 02's guard has
    # already verified the cache file exists if effective_body_slug is set.
    # NOTE: this is the ONLY load_roster() site Phase 109 updates. The 3 offline
    # utility sites (~line 1021 _fix_transcripts, ~line 1719 --show-roster,
    # ~line 1749 --fix-profiles) remain on bare load_roster() because they have
    # no meeting context. See 109-RESEARCH.md §1. Phase 110/111 will revisit them.
    if effective_body_slug:
        roster = load_roster(body_slug=effective_body_slug)
    else:
        roster = load_roster()  # D-05 legacy fallback
    if roster:
        # Roster dataclass may not have .city/.body when loaded from a body-keyed cache;
        # print whichever label is available without crashing the legacy path.
        label = f"{getattr(roster, 'city', '') or ''} {getattr(roster, 'body', '') or ''}".strip()
        if not label and effective_body_slug:
            label = effective_body_slug
        print(f"  Loaded council roster: {len(roster.members)} members ({label})")
    roster_hint = roster_names_for_prompt(roster) if roster else ""

    if state.is_complete(PipelineStage.IDENTIFIED):
        print("  Already complete. Loading from checkpoint...")
        with open(named_transcript_path, "r") as f:
            meeting_data = json.load(f)
        meeting = Meeting.from_dict(meeting_data)
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

        # Layer 3: LLM (optional)
        llm_fn = None
        llm = None
        if not args.skip_llm:
            print("  Loading LLM for speaker identification...")
            from src.llm_utils import llm_identify_speakers, load_llm, unload_llm

            llm = load_llm()
            llm_fn = lambda segs, maps: llm_identify_speakers(
                llm, segs, maps, partial_results_path=llm_partial_path,
                roster_hint=roster_hint,
            )

        t0 = time.time()
        mappings = identify_speakers(
            segments, speaker_embeddings,
            stored_profiles=stored_centroids if stored_centroids else None,
            llm_identify_fn=llm_fn,
            roster=roster,
            profile_db=profile_db,
        )
        elapsed = time.time() - t0

        if llm is not None:
            unload_llm(llm)
            del llm
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

        # Human review
        mappings = human_review(mappings)

        # Apply to segments
        segments = apply_mappings_to_segments(segments, mappings)
        meeting.segments = segments
        meeting.speakers = mappings

        with open(named_transcript_path, "w") as f:
            json.dump(meeting.to_dict(), f, indent=2)
        state.mark_complete(PipelineStage.IDENTIFIED)

    print()

    # ======================================================================
    # Stage 5: Summary Generation
    # ======================================================================
    print("=" * 60)
    print("STAGE 5: Summary Generation")
    print("=" * 60)

    summary_path = meeting_dir / "summary.json"

    if state.is_complete(PipelineStage.SUMMARIZED):
        print("  Already complete. Loading from checkpoint...")
        if summary_path.exists():
            from src.models import MeetingSummary
            with open(summary_path, "r") as f:
                meeting.summary = MeetingSummary.from_dict(json.load(f))
            print(f"  Loaded summary ({len(meeting.summary.sections)} sections)")
    elif args.skip_summary:
        print("  Skipped (--skip-summary).")
        state.mark_complete(PipelineStage.SUMMARIZED)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("  No ANTHROPIC_API_KEY found. Skipping summary generation.")
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
            print("  Generating meeting summary via Anthropic API...")
            meeting.summary = generate_summary(meeting, progress_callback=summary_progress)
            elapsed = time.time() - t0

            # Save summary checkpoint
            with open(summary_path, "w") as f:
                json.dump(meeting.summary.to_dict(), f, indent=2)

            # Re-save named transcript with summary included
            with open(named_transcript_path, "w") as f:
                json.dump(meeting.to_dict(), f, indent=2)

            state.mark_complete(PipelineStage.SUMMARIZED)

            print(f"\n  Summary generated in {elapsed:.1f}s")
            print(f"    Sections: {len(meeting.summary.sections)}")
            print(f"    Key decisions: {len(meeting.summary.key_decisions)}")
            if meeting.summary.executive_summary:
                # Show first 200 chars of executive summary
                preview = meeting.summary.executive_summary[:200]
                if len(meeting.summary.executive_summary) > 200:
                    preview += "..."
                print(f"    Preview: {preview}")

    print()

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
        profile_db = enroll_speakers(
            profile_db, speaker_embeddings, meeting.speakers,
            meeting_id=meeting_id, segments=segments,
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
                        meeting.speakers, meeting_id=meeting_id, segments=segments,
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
                    "city": "Bloomington",
                    "meeting_type": "Regular Session",
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
                    "city": parts[2] if len(parts) > 2 else "Bloomington",
                    "meeting_type": parts[3] if len(parts) > 3 else "Regular Session",
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

        # Check if already processed (for --batch-resume)
        if args.batch_resume and entry["date"]:
            from src import config
            from src.checkpoint import PipelineStage, PipelineState
            mid = f"{entry['date']}-{entry['meeting_type'].lower().replace(' ', '-')}"
            mdir = config.MEETINGS_DIR / mid
            state_file = mdir / "pipeline_state.json"
            if state_file.exists():
                state = PipelineState(mdir)
                if state.is_complete(PipelineStage.IDENTIFIED):
                    print(f"  Already complete (stage {state.completed_stage.name}). Skipping.")
                    results.append({"input": entry["input"], "status": "skipped (complete)", "meeting_id": mid})
                    print()
                    continue

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
            no_merge=args.no_merge if hasattr(args, "no_merge") else False,
            pre_identify=False,  # skip interactive pre-identify
            use_vtt=args.use_vtt if hasattr(args, "use_vtt") else False,
            body=getattr(args, "body", None),
            force_retag=getattr(args, "force_retag", False),
        )

        # Auto-generate date if missing
        if not batch_args.date:
            from datetime import date
            batch_args.date = date.today().isoformat()
            print(f"  No date provided, using today: {batch_args.date}")

        mid = f"{batch_args.date}-{batch_args.meeting_type.lower().replace(' ', '-')}"

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
            with open(named_path, "w") as f:
                json.dump(meeting.to_dict(), f, indent=2)

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


def _build_speaker_stats(segments) -> dict:
    """Build per-speaker statistics from segments.

    Returns dict of label -> {seg_count, total_speech, first_seg, sample_seg, segments}.
    """
    stats = {}
    for seg in segments:
        label = seg.speaker_label
        if label not in stats:
            stats[label] = {
                "seg_count": 0,
                "total_speech": 0.0,
                "first_seg": seg,
                "sample_seg": None,
                "segments": [],
            }
        stats[label]["seg_count"] += 1
        stats[label]["total_speech"] += seg.end_time - seg.start_time
        stats[label]["segments"].append(seg)

    # Pick representative sample for each speaker (near 1/3 point)
    for label, s in stats.items():
        text_segs = [seg for seg in s["segments"] if seg.text and seg.text.strip()]
        if text_segs:
            idx = max(0, len(text_segs) // 3 - 1)
            s["sample_seg"] = text_segs[idx]
        elif s["segments"]:
            # No text segments — use a segment near 1/3 for video clip
            idx = max(0, len(s["segments"]) // 3 - 1)
            s["sample_seg"] = s["segments"][idx]

    return stats


def _load_soft_matches(embeddings_path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load embeddings and compute soft voice profile matches.

    Returns dict of label -> [(display_name, similarity), ...] or empty dict
    if embeddings or profiles aren't available.
    """
    import numpy as np

    if not embeddings_path.exists():
        return {}

    try:
        from src.enroll import get_stored_centroids, load_profiles
        from src.identify import soft_match_voice_profiles
    except ImportError:
        return {}

    with open(embeddings_path, "r") as f:
        emb_data = json.load(f)
    speaker_embeddings = {k: np.array(v) for k, v in emb_data.items()}

    profile_db = load_profiles()
    stored_centroids = get_stored_centroids(profile_db)
    if not stored_centroids:
        return {}

    display_names = {pid: p.display_name for pid, p in profile_db.profiles.items()}
    return soft_match_voice_profiles(speaker_embeddings, stored_centroids, display_names)


def _interactive_speaker_review(
    sorted_labels: list[str],
    speaker_stats: dict,
    current_mappings: dict,
    video_path: str | None,
    soft_matches: dict[str, list[tuple[str, float]]] | None = None,
    show_text: bool = True,
) -> list[dict]:
    """Core interactive loop for reviewing/identifying speakers.

    Args:
        sorted_labels: Speaker labels in display order.
        speaker_stats: Per-speaker stats from _build_speaker_stats().
        current_mappings: label -> SpeakerMapping dict (modified in place).
        video_path: Path to video file for clip playback, or None.
        soft_matches: label -> [(name, score), ...] from soft voice matching.
        show_text: Whether to show transcript text samples (False if pre-transcription).

    Returns:
        List of change dicts: [{label, old_name, new_name}, ...].
    """
    from src.models import SpeakerMapping

    if not sys.stdin.isatty():
        print("(non-interactive mode — cannot review)")
        return []

    changes = []

    for i, label in enumerate(sorted_labels):
        stats = speaker_stats[label]
        mapping = current_mappings.get(label, SpeakerMapping(speaker_label=label))
        name = mapping.speaker_name or "(unidentified)"
        mins = stats["total_speech"] / 60
        sample = stats["sample_seg"]

        # Header
        print(f"\n[{i+1}/{len(sorted_labels)}] {label}: {name}")
        print(f"  Segments: {stats['seg_count']}, Speech: {mins:.1f}m", end="")
        if mapping.confidence > 0:
            print(f", Confidence: {mapping.confidence:.2f}, Method: {mapping.id_method or 'none'}", end="")
        print()

        # Soft match hints
        if soft_matches and label in soft_matches:
            hints = soft_matches[label]
            # Don't show hint if already identified with high confidence as this name
            if not (mapping.speaker_name and mapping.confidence >= 0.85):
                for hint_name, hint_score in hints[:3]:  # show top 3
                    marker = "*" if hint_score >= 0.85 else "?"
                    print(f"  {marker} Voice match: {hint_name} ({hint_score:.2f})")

        # Text sample
        if show_text and sample and sample.text and sample.text.strip():
            text_preview = sample.text[:120] + "..." if len(sample.text) > 120 else sample.text
            print(f"  Sample [{_format_ts(sample.start_time)}]: \"{text_preview}\"")
        elif sample:
            print(f"  Clip at [{_format_ts(sample.start_time)}]")

        # Accept shortcut: if there's exactly one high soft match, allow [Y] to confirm
        top_hint = None
        if soft_matches and label in soft_matches:
            hints = soft_matches[label]
            if hints and not (mapping.speaker_name and mapping.confidence >= 0.85):
                top_hint = hints[0]

        while True:
            prompt_parts = ["  "]
            if video_path and sample:
                prompt_parts.append("[V]iew")
            if top_hint:
                prompt_parts.append(f"[Y=accept {top_hint[0]}]")
            prompt_parts.append("[Enter=skip] [Q=quit] or type name: ")
            choice = input(" ".join(prompt_parts)).strip()

            if choice.lower() in ("v", "view") and video_path and sample:
                play_video_clip(
                    video_path,
                    start_time=sample.start_time,
                    duration=20.0,
                    title=f"{label} → {name}",
                )
                continue  # re-prompt after viewing
            elif choice.lower() == "q":
                print("  Quitting review.")
                break
            elif choice == "":
                # Skip — keep current name
                break
            elif choice.lower() in ("y", "yes") and top_hint:
                # Accept top soft match
                old_name = mapping.speaker_name
                new_name = top_hint[0]
                mapping.speaker_name = new_name
                mapping.confidence = 1.0
                mapping.id_method = "human_confirmed"
                mapping.needs_review = False
                current_mappings[label] = mapping
                changes.append({"label": label, "old_name": old_name, "new_name": new_name})
                print(f"  Confirmed: {label} -> {new_name}")
                break
            else:
                # User typed a new name
                old_name = mapping.speaker_name
                mapping.speaker_name = choice
                mapping.confidence = 1.0
                mapping.id_method = "human_review"
                mapping.needs_review = False
                current_mappings[label] = mapping
                changes.append({"label": label, "old_name": old_name, "new_name": choice})
                print(f"  Updated: {label} -> {choice}")

                # Roster auto-learning: offer to add old wrong name as alias
                if old_name and old_name != choice:
                    from src.roster import add_alias
                    if add_alias(None, choice, old_name):
                        print(f"  Auto-added alias: '{old_name}' -> '{choice}'")

                break
        else:
            continue

        if choice.lower() == "q":
            break

    return changes


def _enroll_after_review(
    changes: list[dict],
    current_mappings: dict,
    meeting_dir: Path,
    meeting_id: str,
    segments,
) -> None:
    """Offer to enroll speakers that were identified or corrected during review.

    Only runs if embeddings are available on disk.
    """
    import numpy as np

    from src.enroll import _enroll_one, _name_to_slug, load_profiles, save_profiles

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
        label = change["label"]
        new_name = change["new_name"]
        if not new_name or label not in speaker_embeddings:
            continue
        slug = _name_to_slug(new_name)
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

    choice = input("\nEnroll these speakers? [Y/n] ").strip().lower()
    if choice in ("", "y", "yes"):
        for e in enrollable:
            mapping = current_mappings.get(e["label"])
            seg_count = sum(1 for s in segments if s.speaker_label == e["label"])
            _enroll_one(
                profile_db, e["slug"], e["name"],
                speaker_embeddings[e["label"]],
                meeting_id, seg_count,
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
    speaker_stats = _build_speaker_stats(meeting.segments)
    embeddings_path = meeting_dir / "embeddings.json"
    soft_matches = _load_soft_matches(embeddings_path)

    sorted_labels = sorted(
        speaker_stats.keys(),
        key=lambda l: speaker_stats[l]["total_speech"],
        reverse=True,
    )

    print(f"\nReviewing: {meeting.city} {meeting.meeting_type} ({meeting.date})")
    print(f"Meeting ID: {meeting_id}")
    if video_path:
        print(f"Video: {Path(video_path).name}")
    else:
        print("Video: not found (no clip playback available)")
    print(f"Speakers: {len(sorted_labels)}")
    print()

    # Show overview table
    print("  #  Label         Current Name                  Segs  Speech  Conf   Method")
    print("  " + "-" * 90)
    for i, label in enumerate(sorted_labels):
        stats = speaker_stats[label]
        mapping = meeting.speakers.get(label)
        name = mapping.speaker_name if mapping and mapping.speaker_name else "(unidentified)"
        conf = mapping.confidence if mapping else 0.0
        method = mapping.id_method or ""
        mins = stats["total_speech"] / 60
        hint = ""
        if soft_matches and label in soft_matches:
            top = soft_matches[label][0]
            if not (mapping and mapping.speaker_name and mapping.confidence >= 0.85):
                hint = f"  ~ {top[0]} ({top[1]:.2f})"
        print(f"  {i+1:>2}  {label:<13} {name:<30} {stats['seg_count']:>4}  {mins:>5.1f}m  {conf:.2f}  {method}{hint}")

    print()
    print("Commands for each speaker:")
    print("  [Enter]  Skip (keep current name)")
    print("  [V]      View video clip of this speaker")
    print("  [Y]      Accept suggested voice match (if shown)")
    print("  [name]   Type a new name to assign")
    print("  [Q]      Quit review (save changes so far)")
    print()

    changes = _interactive_speaker_review(
        sorted_labels, speaker_stats, meeting.speakers,
        video_path, soft_matches=soft_matches, show_text=True,
    )

    # Apply corrections to segments and save
    if changes:
        for seg in meeting.segments:
            m = meeting.speakers.get(seg.speaker_label)
            if m and m.speaker_name:
                seg.speaker_name = m.speaker_name
                seg.confidence = m.confidence
                seg.id_method = m.id_method

        with open(named_path, "w") as f:
            json.dump(meeting.to_dict(), f, indent=2)

        export_dir = meeting_dir / "exports"
        export_all(meeting, export_dir)

        print(f"\n{len(changes)} correction(s) saved:")
        for c in changes:
            old = c["old_name"] or "(unidentified)"
            print(f"  {c['label']}: {old} -> {c['new_name']}")
        print(f"Exports updated: {export_dir}")

        # Offer enrollment
        _enroll_after_review(
            changes, meeting.speakers, meeting_dir,
            meeting.meeting_id, meeting.segments,
        )
    else:
        print("\nNo changes made.")


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
    speaker_stats = _build_speaker_stats(segments)
    soft_matches = _load_soft_matches(embeddings_path)

    sorted_labels = sorted(
        speaker_stats.keys(),
        key=lambda l: speaker_stats[l]["total_speech"],
        reverse=True,
    )

    print(f"\nSpeaker Identification: {meeting_id}")
    if video_path:
        print(f"Video: {Path(video_path).name}")
    else:
        print("Video: not found")
    if has_text:
        print("Transcript: available (text samples shown)")
    else:
        print("Transcript: not yet available (video clips only)")
    print(f"Speakers: {len(sorted_labels)}")
    if soft_matches:
        matched = sum(1 for l in sorted_labels if l in soft_matches)
        print(f"Voice hints: {matched} speaker(s) have possible profile matches")
    print()

    # Show overview table
    print("  #  Label         Current Name                  Segs  Speech  Voice Hint")
    print("  " + "-" * 85)
    for i, label in enumerate(sorted_labels):
        stats = speaker_stats[label]
        mapping = current_mappings.get(label)
        name = mapping.speaker_name if mapping and mapping.speaker_name else "(unidentified)"
        mins = stats["total_speech"] / 60
        hint = ""
        if soft_matches and label in soft_matches:
            top = soft_matches[label][0]
            if top[1] >= 0.85:
                hint = f"* {top[0]} ({top[1]:.2f})"
            else:
                hint = f"? {top[0]} ({top[1]:.2f})"
        print(f"  {i+1:>2}  {label:<13} {name:<30} {stats['seg_count']:>4}  {mins:>5.1f}m  {hint}")

    print()
    print("Commands for each speaker:")
    print("  [Enter]  Skip")
    print("  [V]      View video clip of this speaker")
    print("  [Y]      Accept suggested voice match (if shown)")
    print("  [name]   Type a name to assign")
    print("  [Q]      Quit (save changes so far)")
    print()

    changes = _interactive_speaker_review(
        sorted_labels, speaker_stats, current_mappings,
        video_path, soft_matches=soft_matches, show_text=has_text,
    )

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
            old = c["old_name"] or "(unidentified)"
            print(f"  {c['label']}: {old} -> {c['new_name']}")

        # If named transcript exists, update it too
        if named_path.exists():
            with open(named_path, "r") as f:
                meeting = Meeting.from_dict(json.load(f))
            for label, mapping in current_mappings.items():
                if isinstance(mapping, SpeakerMapping):
                    meeting.speakers[label] = mapping
            for seg in meeting.segments:
                m = meeting.speakers.get(seg.speaker_label)
                if m and m.speaker_name:
                    seg.speaker_name = m.speaker_name
                    seg.confidence = m.confidence
                    seg.id_method = m.id_method
            with open(named_path, "w") as f:
                json.dump(meeting.to_dict(), f, indent=2)
            from src.export import export_all
            export_dir = meeting_dir / "exports"
            export_all(meeting, export_dir)
            print(f"Transcript and exports updated.")

        # Offer enrollment
        _enroll_after_review(
            changes, current_mappings, meeting_dir,
            meeting_id, segments,
        )

        print("\nThese identifications will be used as ground truth in Stage 4")
        print("(overriding LLM/pattern matching for identified speakers).")
    else:
        print("\nNo identifications made.")


def main():
    parser = argparse.ArgumentParser(
        description="CouncilScribe — Automated City Council Meeting Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # Meeting metadata
    parser.add_argument("--city", default="Bloomington", help="City name (default: Bloomington)")
    parser.add_argument("--date", default="", help="Meeting date (YYYY-MM-DD)")
    parser.add_argument("--meeting-type", default="Regular Session",
                        choices=["Regular Session", "Special Session", "Work Session", "Committee Meeting"],
                        help="Meeting type (default: Regular Session)")
    parser.add_argument("--meeting-id", default="", help="Custom meeting ID (auto-generated if omitted)")

    # Processing options
    parser.add_argument("--num-speakers", type=int, default=0,
                        help="Expected number of speakers (0 = auto-detect)")
    parser.add_argument("--noise-reduce", action="store_true",
                        help="Apply spectral noise reduction to audio")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based speaker identification (Layer 3)")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Skip meeting summary generation (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--confirm-enroll", action="store_true",
                        help="Interactively confirm enrollment for borderline speakers (0.70-0.85 confidence)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip auto-merging of fragmented speakers after diarization")
    parser.add_argument("--use-vtt", action="store_true",
                        help="Use VTT subtitles instead of Whisper (auto-detected if captions.vtt exists)")

    # Utilities
    parser.add_argument("--list-profiles", action="store_true",
                        help="List stored voice profiles and exit")
    parser.add_argument("--fix-profiles", action="store_true",
                        help="Rename stored voice profiles using the council roster and exit")
    parser.add_argument("--fix-transcripts", action="store_true",
                        help="Re-correct speaker names in all existing transcripts using the roster and re-export")
    parser.add_argument("--merge-profiles", nargs=2, metavar=("SOURCE", "DEST"),
                        help="Merge SOURCE profile into DEST profile and exit (use slugs from --list-profiles)")
    parser.add_argument("--show-roster", action="store_true",
                        help="Display the current council roster and exit")
    parser.add_argument("--review-meeting", metavar="MEETING_ID",
                        help="Interactively review and correct all speakers in an existing meeting")
    parser.add_argument("--identify-speakers", metavar="MEETING_ID",
                        help="Standalone speaker identification with video clips and voice hints (works pre-transcription)")
    parser.add_argument("--pre-identify", action="store_true",
                        help="Interactive speaker identification after diarization, before transcription (pipeline mode)")
    parser.add_argument("--batch", metavar="FILE_OR_DIR",
                        help="Batch mode: text file with one input per line (path or 'URL DATE'), or directory of videos")
    parser.add_argument("--batch-resume", action="store_true",
                        help="Resume an interrupted batch run (skip already-completed meetings)")
    parser.add_argument(
        "--body",
        type=str,
        default=None,
        help="Governing body slug (e.g. bloomington-common-council). "
             "Persisted to pipeline_state.json on first run; omit on re-invocation.",
    )
    parser.add_argument(
        "--force-retag",
        action="store_true",
        default=False,
        help="Overwrite a meeting's persisted body_slug. Rewinds stages 4-7. "
             "Requires --body.",
    )

    args = parser.parse_args()

    # D-12: --force-retag requires --body
    if args.force_retag and not args.body:
        parser.error("--force-retag requires --body <slug>")

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
        if named_path.exists():
            with open(named_path, "r") as f:
                data = json.load(f)
            args.input = data.get("audio_source", "")
            args.city = data.get("city", args.city)
            args.date = data.get("date", args.date)
            args.meeting_type = data.get("meeting_type", args.meeting_type)
        else:
            # Use the WAV file as input since audio is already ingested
            wav = meeting_dir / "audio.wav"
            if wav.exists():
                args.input = str(wav)
            else:
                print(f"Cannot resume: no audio.wav found in {meeting_dir}")
                sys.exit(1)

        args.meeting_id = args.resume
        print(f"Resuming meeting: {args.resume}")

    # --- Validate ---
    if not args.input:
        parser.print_help()
        print("\nError: --input, --browse-catstv, or --resume is required.")
        sys.exit(1)

    if not args.date:
        from datetime import date
        args.date = date.today().isoformat()
        print(f"No --date provided, using today: {args.date}")

    # --- Run ---
    run_pipeline(args)


if __name__ == "__main__":
    main()
