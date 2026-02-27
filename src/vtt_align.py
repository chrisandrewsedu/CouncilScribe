"""Align VTT subtitle cues to diarized speaker segments.

Replaces Whisper transcription by mapping pre-existing VTT captions
(from CATS TV) onto diarized segments based on timestamp overlap.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Segment, Word


def parse_vtt(vtt_path: str | Path) -> list[dict]:
    """Parse a WebVTT file into a list of cue dicts.

    Each cue dict has: start (float), end (float), text (str).
    """
    content = Path(vtt_path).read_text(encoding="utf-8")
    cues = []

    # Split on blank lines to get blocks
    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        lines = block.strip().split("\n")
        # Find the timestamp line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                # Everything after timestamp is text
                text_lines.append(line)

        if ts_line and text_lines:
            start, end = _parse_timestamp_line(ts_line)
            if start is not None:
                # Strip HTML tags and speaker labels like <v Speaker>
                text = " ".join(text_lines)
                text = re.sub(r"<[^>]+>", "", text).strip()
                if text:
                    cues.append({"start": start, "end": end, "text": text})

    return cues


def _parse_timestamp_line(line: str) -> tuple[float | None, float | None]:
    """Parse a VTT timestamp line like '00:01:23.456 --> 00:01:25.789'."""
    match = re.search(
        r"(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})",
        line,
    )
    if not match:
        return None, None

    def to_seconds(h, m, s, ms):
        h = int(h.rstrip(":")) if h else 0
        return h * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    start = to_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
    end = to_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
    return start, end


def _overlap(seg_start: float, seg_end: float, cue_start: float, cue_end: float) -> float:
    """Compute overlap duration between a segment and a cue."""
    overlap_start = max(seg_start, cue_start)
    overlap_end = min(seg_end, cue_end)
    return max(0.0, overlap_end - overlap_start)


def align_vtt_to_segments(
    vtt_path: str | Path,
    diarized_segments: list[Segment],
) -> list[Segment]:
    """Align VTT cues to diarized segments by timestamp overlap.

    For each diarized segment, finds overlapping VTT cues and assigns
    the text proportionally. This replaces Whisper transcription.

    Args:
        vtt_path: Path to the VTT subtitle file.
        diarized_segments: Segments from diarization (no text yet).

    Returns:
        The same segments list, now with text populated from VTT.
    """
    cues = parse_vtt(vtt_path)
    if not cues:
        print("  Warning: VTT file contains no cues")
        return diarized_segments

    for seg in diarized_segments:
        matched_texts = []

        for cue in cues:
            overlap_dur = _overlap(seg.start_time, seg.end_time, cue["start"], cue["end"])
            if overlap_dur <= 0:
                continue

            cue_dur = cue["end"] - cue["start"]
            if cue_dur <= 0:
                continue

            # If the overlap covers most of the cue, take the full text
            overlap_fraction = overlap_dur / cue_dur
            if overlap_fraction >= 0.5:
                matched_texts.append(cue["text"])
            elif overlap_fraction >= 0.2:
                # Partial overlap: take a proportional slice of words
                words = cue["text"].split()
                n_words = max(1, int(len(words) * overlap_fraction))
                # Determine which portion of the cue overlaps
                if seg.start_time <= cue["start"]:
                    # Segment covers the beginning of the cue
                    matched_texts.append(" ".join(words[:n_words]))
                else:
                    # Segment covers the end of the cue
                    matched_texts.append(" ".join(words[-n_words:]))

        if matched_texts:
            seg.text = " ".join(matched_texts)
            # Create simple word-level entries (without precise per-word timestamps)
            seg_dur = seg.end_time - seg.start_time
            words = seg.text.split()
            if words and seg_dur > 0:
                word_dur = seg_dur / len(words)
                seg.words = [
                    Word(
                        word=w,
                        start=round(seg.start_time + i * word_dur, 3),
                        end=round(seg.start_time + (i + 1) * word_dur, 3),
                    )
                    for i, w in enumerate(words)
                ]

    aligned_count = sum(1 for s in diarized_segments if s.text)
    total = len(diarized_segments)
    print(f"  VTT alignment: {aligned_count}/{total} segments received text")

    return diarized_segments
