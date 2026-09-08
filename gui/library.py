"""Read the meetings directory into a sorted list of MeetingSummary rows.

Pure filesystem reads — no HTTP. Reuses src.checkpoint.PipelineState so the
GUI and the pipeline agree on how pipeline_state.json is parsed (and tolerate
older state files missing the newer metadata keys)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.checkpoint import PipelineState

from gui.models import MeetingSummary


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _speaker_count(meeting_dir: Path, named: Optional[dict]) -> Optional[int]:
    """Prefer identified/merged speakers (transcript_named 'speakers'); else unique
    raw diarization labels; else None."""
    if isinstance(named, dict) and isinstance(named.get("speakers"), dict) and named["speakers"]:
        return len(named["speakers"])  # Meeting.to_dict() keys speakers BY LABEL (a dict)
    diar = _read_json(meeting_dir / "diarization.json")
    if isinstance(diar, list):
        labels = {s.get("speaker_label") for s in diar if isinstance(s, dict)}
        labels.discard(None)
        return len(labels) if labels else None
    return None


def _duration_seconds(meeting_dir: Path, named: Optional[dict]) -> Optional[float]:
    """transcript_named duration_seconds; else read the audio.wav header (cheap)."""
    if isinstance(named, dict) and isinstance(named.get("duration_seconds"), (int, float)):
        return float(named["duration_seconds"])
    wav = meeting_dir / "audio.wav"
    if wav.exists():
        try:
            import soundfile as sf
            return float(sf.info(str(wav)).duration)
        except Exception:
            return None
    return None


def _summarize(meeting_dir: Path) -> Optional[MeetingSummary]:
    if not (meeting_dir / "pipeline_state.json").exists():
        return None
    try:
        state = PipelineState(meeting_dir)
    except Exception:
        return None  # malformed/incompatible state file — skip, don't 500
    named = _read_json(meeting_dir / "transcript_named.json")
    title = None
    if isinstance(named, dict):
        t = named.get("title")
        title = t if isinstance(t, str) and t.strip() else None
    event_orgs = []
    if isinstance(named, dict) and isinstance(named.get("event_orgs"), list):
        event_orgs = [o for o in named["event_orgs"] if isinstance(o, str) and o.strip()]
    try:
        processed_at = (meeting_dir / "pipeline_state.json").stat().st_mtime
    except OSError:
        processed_at = None
    return MeetingSummary(
        meeting_id=meeting_dir.name,
        title=title,
        city=state.city,
        meeting_type=state.meeting_type,
        date=state.date,
        event_kind=state.event_kind,
        completed_stage=int(state.completed_stage),
        speaker_count=_speaker_count(meeting_dir, named),
        duration_seconds=_duration_seconds(meeting_dir, named),
        review_status=state.review_status,
        trusted_coverage=state.trusted_coverage,
        has_thumbnail=(meeting_dir / "thumbnail.jpg").exists(),
        event_orgs=event_orgs,
        body_slug=state.body_slug,
        race_id=state.race_id,
        guest=state.guest,
        processed_at=processed_at,
    )


def scan_meetings(
    meetings_dir: Path,
    live_slugs: Optional[set] = None,
    live_speaker_counts: Optional[dict] = None,
) -> list[MeetingSummary]:
    """All meetings under meetings_dir, newest date first (missing dates last).

    ``live_slugs`` is the set of slugs currently live on the site (from the DB).
    When provided, each row's ``is_live`` is set True/False accordingly; when
    None (DB not checked), ``is_live`` stays None and no live badge is shown.

    ``live_speaker_counts`` is {slug: prod speaker_count} from the same query.
    It populates ``live_speaker_count`` so the Speakers column can show a
    divergence from the local transcript — the symptom of an orphan speaker
    row, which this scan (local files only) cannot otherwise see."""
    if not meetings_dir.exists():
        return []
    summaries: list[MeetingSummary] = []
    for child in sorted(meetings_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = _summarize(child)
        if summary is not None:
            if live_slugs is not None:
                summary.is_live = summary.meeting_id in live_slugs
            if live_speaker_counts is not None:
                summary.live_speaker_count = live_speaker_counts.get(
                    summary.meeting_id
                )
            summaries.append(summary)
    # Sort by most-recent processing activity (state-file mtime) so running and
    # just-finished meetings float to the top; fall back to clip date / id.
    summaries.sort(key=lambda s: (s.processed_at or 0.0, s.date or "", s.meeting_id),
                   reverse=True)
    return summaries
