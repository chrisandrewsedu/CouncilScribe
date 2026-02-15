"""Stage 6: Export transcripts to JSON, Markdown, and SRT formats."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Meeting, Segment


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _speaker_display(seg: Segment) -> str:
    """Get display name for a segment's speaker."""
    return seg.speaker_name or seg.speaker_label


def export_markdown(meeting: Meeting, output_path: str | Path) -> Path:
    """Export transcript as readable Markdown.

    Format: **[HH:MM:SS] Speaker Name:**\\nText
    """
    output_path = Path(output_path)
    lines = []

    # Header
    title = f"{meeting.city} {meeting.meeting_type}"
    lines.append(f"# {title} — {meeting.date}")
    lines.append("")

    for seg in meeting.segments:
        if not seg.text:
            continue
        ts = _format_timestamp(seg.start_time)
        speaker = _speaker_display(seg)
        lines.append(f"**[{ts}] {speaker}:**")
        lines.append(seg.text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_json(meeting: Meeting, output_path: str | Path) -> Path:
    """Export full meeting data as structured JSON."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meeting.to_dict(), f, indent=2, ensure_ascii=False)
    return output_path


def _split_into_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for SRT subtitle chunking."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def export_srt(meeting: Meeting, output_path: str | Path, max_chars: int = 80) -> Path:
    """Export transcript as SRT subtitles.

    Long segments are split at sentence boundaries. Each subtitle
    is prefixed with [Speaker Name].
    """
    output_path = Path(output_path)
    entries = []
    counter = 1

    for seg in meeting.segments:
        if not seg.text:
            continue

        speaker = _speaker_display(seg)
        duration = seg.end_time - seg.start_time

        sentences = _split_into_sentences(seg.text)
        if not sentences:
            continue

        # If the segment is short enough, emit as a single subtitle
        if len(seg.text) <= max_chars or len(sentences) == 1:
            entries.append(
                _srt_entry(counter, seg.start_time, seg.end_time, speaker, seg.text)
            )
            counter += 1
        else:
            # Split across sentences with proportional timing
            total_len = sum(len(s) for s in sentences)
            current_time = seg.start_time
            for sentence in sentences:
                frac = len(sentence) / total_len if total_len > 0 else 1.0
                end_time = current_time + duration * frac
                entries.append(
                    _srt_entry(counter, current_time, end_time, speaker, sentence)
                )
                counter += 1
                current_time = end_time

    output_path.write_text("\n".join(entries), encoding="utf-8")
    return output_path


def _srt_entry(
    index: int, start: float, end: float, speaker: str, text: str
) -> str:
    """Format a single SRT subtitle entry."""
    return (
        f"{index}\n"
        f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
        f"[{speaker}] {text}\n"
    )


def export_all(meeting: Meeting, export_dir: str | Path) -> dict[str, Path]:
    """Run all three exporters. Returns dict of format -> filepath."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "markdown": export_markdown(meeting, export_dir / "transcript.md"),
        "json": export_json(meeting, export_dir / "transcript.json"),
        "srt": export_srt(meeting, export_dir / "subtitles.srt"),
    }
    return results
