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
import sys
import time
from pathlib import Path

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
    # Stage 3: Transcription
    # ======================================================================
    print("=" * 60)
    print("STAGE 3: Transcription")
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

    # Load roster for name correction
    roster = load_roster()
    if roster:
        print(f"  Loaded council roster: {len(roster.members)} members ({roster.city} {roster.body})")
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
        )
        elapsed = time.time() - t0

        if llm is not None:
            unload_llm(llm)
            del llm
        if llm_partial_path.exists():
            llm_partial_path.unlink()

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


def _fix_transcripts() -> None:
    """Re-correct speaker names in all existing transcripts using the roster.

    Walks through every meeting directory, loads transcript_named.json,
    applies roster corrections to speaker mappings and segments, saves
    the corrected transcript, and re-exports markdown/json/srt.
    """
    from src import config
    from src.export import export_all
    from src.models import Meeting
    from src.roster import correct_speaker_name, load_roster

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
                    corrections.append(f"    {label}: {mapping.speaker_name} -> {corrected}")
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
                print(c)
            total_corrections += len(corrections)
        else:
            print(f"  {mdir.name}: no corrections needed")

    print(f"\nDone. {total_corrections} total correction(s) across {len(meeting_dirs)} meeting(s).")


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

    args = parser.parse_args()

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
