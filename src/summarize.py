"""Stage 5: Meeting summary generation using Anthropic API.

Two-pass pipeline:
  Pass 1 (Haiku)  — Classify transcript into sections (roll call, discussion, vote, etc.)
  Pass 2 (Sonnet) — Summarize each substantive section + generate executive summary
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from . import config
from .models import Meeting, MeetingSummary, Segment, SummarySection


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Transcript formatting helpers
# ---------------------------------------------------------------------------

def _condensed_transcript(segments: list[Segment], max_chars_per_seg: int = 120) -> str:
    """Build a condensed transcript for section classification.

    Shows timestamp, speaker, and truncated text for each segment.
    This keeps the input small enough for efficient Haiku classification.
    """
    lines = []
    for seg in segments:
        if not seg.text:
            continue
        speaker = seg.speaker_name or seg.speaker_label
        ts = _format_ts(seg.start_time)
        text = seg.text[:max_chars_per_seg]
        if len(seg.text) > max_chars_per_seg:
            text += "..."
        lines.append(f"[{ts}] (seg {seg.segment_id}) {speaker}: {text}")
    return "\n".join(lines)


def _full_section_transcript(segments: list[Segment], start: int, end: int) -> str:
    """Build full transcript text for a range of segments."""
    lines = []
    for seg in segments:
        if seg.segment_id < start or seg.segment_id > end:
            continue
        if not seg.text:
            continue
        speaker = seg.speaker_name or seg.speaker_label
        ts = _format_ts(seg.start_time)
        lines.append(f"[{ts}] {speaker}: {seg.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass 1: Section classification (Haiku)
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are an expert at analyzing city council meeting transcripts.
Your job is to identify the structural sections of the meeting from the transcript.

Section types:
- opening: Call to order, procedural opening
- roll_call: Roll call, attendance taking
- consent_agenda: Consent agenda items (usually passed as a batch)
- discussion: Substantive discussion on an agenda item (the most important sections)
- public_comment: Public comment period
- vote: A distinct voting moment (if embedded in a discussion, include it in the discussion section)
- procedural: Motions, seconds, parliamentary procedure, announcements
- closing: Adjournment, closing remarks

Rules:
- Every segment must belong to exactly one section
- Sections are contiguous ranges of segments
- A long discussion on one topic = one section; a new topic = new section
- Give each section a descriptive title (e.g., "Discussion: Affordable Housing Ordinance")
- For roll_call sections, title should be "Roll Call"
- For consent_agenda, title should describe what was on the consent agenda
- Keep the number of sections reasonable (typically 5-20 for a full meeting)

Respond with ONLY valid JSON in this format:
{
  "sections": [
    {"type": "opening", "start_segment": 0, "end_segment": 5, "title": "Call to Order"},
    {"type": "roll_call", "start_segment": 6, "end_segment": 18, "title": "Roll Call"},
    ...
  ]
}"""


def _classify_sections_chunk(
    client,
    condensed: str,
    seg_offset: int = 0,
) -> list[dict]:
    """Classify one chunk of transcript into sections using Haiku."""
    message = client.messages.create(
        model=config.SUMMARY_CLASSIFY_MODEL,
        max_tokens=config.SUMMARY_MAX_TOKENS_CLASSIFY,
        system=_CLASSIFY_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Classify this council meeting transcript into sections:\n\n{condensed}",
        }],
    )

    text = message.content[0].text
    # Extract JSON from response (handle markdown code fences)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
        return data.get("sections", [])
    except json.JSONDecodeError:
        return []


def classify_sections(
    client,
    segments: list[Segment],
) -> list[dict]:
    """Classify the full transcript into sections, chunking if needed."""
    chunk_size = config.SUMMARY_CHUNK_SIZE

    if len(segments) <= chunk_size:
        condensed = _condensed_transcript(segments)
        return _classify_sections_chunk(client, condensed)

    # Chunk large transcripts with overlap for context
    all_sections = []
    for i in range(0, len(segments), chunk_size):
        chunk = segments[i : i + chunk_size]
        condensed = _condensed_transcript(chunk)
        chunk_sections = _classify_sections_chunk(client, condensed, seg_offset=i)
        all_sections.extend(chunk_sections)

    # Merge adjacent sections of the same type at chunk boundaries
    merged = []
    for sec in all_sections:
        if (
            merged
            and merged[-1]["type"] == sec["type"]
            and sec["start_segment"] <= merged[-1]["end_segment"] + 2
        ):
            merged[-1]["end_segment"] = sec["end_segment"]
            # Keep the more descriptive title
            if len(sec.get("title", "")) > len(merged[-1].get("title", "")):
                merged[-1]["title"] = sec["title"]
        else:
            merged.append(sec)

    return merged


# ---------------------------------------------------------------------------
# Pass 2: Section summaries (Sonnet) + structured extraction (Haiku)
# ---------------------------------------------------------------------------

_SUMMARIZE_DISCUSSION_SYSTEM = """You are summarizing a section of a city council meeting for citizens who want to understand what happened.

Write a clear, informative summary in markdown. Include:
- What topic/item was being discussed
- Key points raised by each speaker (attribute quotes and positions to specific people)
- Any motions or amendments proposed
- The outcome (if a vote occurred, include the result and who voted how)
- Relevant context that helps a citizen understand why this matters

Keep it concise but substantive — aim for 3-8 bullet points or short paragraphs.
Use > blockquote format for notable direct quotes with speaker attribution.
Include timestamps in [MM:SS] format when referencing specific moments.

Respond with ONLY the markdown summary content. Do NOT include a title/heading — just the summary body.
Do not start with "## ..." or "# ..." — the title is added separately."""


_EXTRACT_ROLL_CALL_SYSTEM = """You are extracting roll call information from a city council meeting transcript.

Respond with ONLY valid JSON:
{
  "present": ["Name1", "Name2"],
  "absent": ["Name3"],
  "notes": "any relevant notes (e.g., 'arrived late', 'left early')"
}

If you can't determine attendance clearly, include your best interpretation and note uncertainty."""


_EXTRACT_VOTE_SYSTEM = """You are extracting vote information from a city council meeting transcript section.

Respond with ONLY valid JSON:
{
  "votes": [
    {
      "resolution": "Resolution 26-04 or descriptive name",
      "description": "Brief description of what was voted on",
      "result": "passed" or "failed" or "tabled",
      "vote_type": "unanimous" or "roll_call" or "voice",
      "yea": ["Name1", "Name2"],
      "nay": ["Name3"],
      "abstain": [],
      "absent": []
    }
  ]
}

If vote details aren't clear, include what you can determine and note uncertainty in the description."""


def _summarize_discussion(client, section_transcript: str, title: str) -> str:
    """Generate a rich summary of a discussion section using Sonnet."""
    message = client.messages.create(
        model=config.SUMMARY_SYNTHESIZE_MODEL,
        max_tokens=config.SUMMARY_MAX_TOKENS_SYNTHESIZE,
        system=_SUMMARIZE_DISCUSSION_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Meeting section: \"{title}\"\n\nTranscript:\n{section_transcript}",
        }],
    )
    return message.content[0].text.strip()


def _extract_roll_call(client, section_transcript: str) -> str:
    """Extract roll call data and format as markdown."""
    message = client.messages.create(
        model=config.SUMMARY_CLASSIFY_MODEL,  # Haiku — structured extraction
        max_tokens=1024,
        system=_EXTRACT_ROLL_CALL_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Extract roll call from:\n\n{section_transcript}",
        }],
    )

    text = message.content[0].text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return "_Roll call data could not be extracted._"

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return "_Roll call data could not be parsed._"

    lines = []
    if data.get("present"):
        lines.append(f"**Present:** {', '.join(data['present'])}")
    if data.get("absent"):
        lines.append(f"**Absent:** {', '.join(data['absent'])}")
    else:
        lines.append("**Absent:** None")
    if data.get("notes"):
        lines.append(f"\n_{data['notes']}_")
    return "\n".join(lines)


def _extract_votes(client, section_transcript: str) -> tuple[str, list[dict]]:
    """Extract vote data and format as markdown. Returns (markdown, raw_votes)."""
    message = client.messages.create(
        model=config.SUMMARY_CLASSIFY_MODEL,  # Haiku — structured extraction
        max_tokens=2048,
        system=_EXTRACT_VOTE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Extract votes from:\n\n{section_transcript}",
        }],
    )

    text = message.content[0].text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return "_Vote data could not be extracted._", []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return "_Vote data could not be parsed._", []

    votes = data.get("votes", [])
    lines = []
    for v in votes:
        result = v.get("result", "unknown").upper()
        lines.append(f"**{v.get('resolution', 'Motion')}** — {result}")
        if v.get("description"):
            lines.append(f"_{v['description']}_")
        if v.get("vote_type") == "unanimous":
            lines.append("Vote: Unanimous")
        else:
            if v.get("yea"):
                lines.append(f"Yea: {', '.join(v['yea'])}")
            if v.get("nay"):
                lines.append(f"Nay: {', '.join(v['nay'])}")
            if v.get("abstain"):
                lines.append(f"Abstain: {', '.join(v['abstain'])}")
        lines.append("")

    return "\n".join(lines), votes


# ---------------------------------------------------------------------------
# Pass 3: Executive summary (Sonnet)
# ---------------------------------------------------------------------------

_EXECUTIVE_SYSTEM = """You are writing an executive summary of a city council meeting for citizens.

You will receive section summaries from the meeting. Write:

1. An "executive_summary": 3-5 sentences capturing the most important things that happened.
   Write for a citizen who wants to know "what happened at last night's meeting" in 60 seconds.

2. A "key_decisions" list: bullet points of concrete outcomes (votes passed/failed, actions directed, etc.)

Respond with ONLY valid JSON:
{
  "executive_summary": "The council...",
  "key_decisions": [
    "Approved Resolution 26-04 renewing the street repair contract (unanimous)",
    "Passed Affordable Housing Ordinance 7-2 with rental property amendment",
    ...
  ]
}"""


def _generate_executive_summary(
    client,
    sections: list[SummarySection],
    meeting: Meeting,
) -> tuple[str, list[str]]:
    """Generate executive summary from section summaries using Sonnet."""
    # Build a condensed view of all sections for the executive summary prompt
    section_text = []
    for sec in sections:
        section_text.append(f"### {sec.title} ({sec.section_type}) [{_format_ts(sec.start_time)}]")
        section_text.append(sec.content)
        section_text.append("")

    meeting_header = f"{meeting.city} {meeting.meeting_type} — {meeting.date}"

    message = client.messages.create(
        model=config.SUMMARY_SYNTHESIZE_MODEL,
        max_tokens=2048,
        system=_EXECUTIVE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Meeting: {meeting_header}\n"
                f"Duration: {_format_ts(meeting.duration_seconds)}\n\n"
                f"Section summaries:\n\n{''.join(section_text)}"
            ),
        }],
    )

    text = message.content[0].text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return "Executive summary could not be generated.", []

    try:
        data = json.loads(json_match.group())
        return data.get("executive_summary", ""), data.get("key_decisions", [])
    except json.JSONDecodeError:
        return "Executive summary could not be parsed.", []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_summary(
    meeting: Meeting,
    progress_callback: Optional[callable] = None,
) -> MeetingSummary:
    """Generate a structured meeting summary using the Anthropic API.

    Pipeline:
      1. Classify transcript into sections (Haiku)
      2. Summarize each section (Sonnet for discussions, Haiku for extraction)
      3. Generate executive summary (Sonnet)

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        meeting: Meeting object with named segments.
        progress_callback: Optional function(step_name, current, total) for progress.

    Returns:
        MeetingSummary with executive summary, key decisions, and section summaries.
    """
    import anthropic

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    segments = [s for s in meeting.segments if s.text]
    if not segments:
        return MeetingSummary(
            executive_summary="No transcript segments available.",
            model=config.SUMMARY_CLASSIFY_MODEL,
            generated_at=datetime.now().isoformat(),
        )

    def _progress(step: str, current: int = 0, total: int = 0):
        if progress_callback:
            progress_callback(step, current, total)

    # --- Pass 1: Classify sections ---
    _progress("classifying sections")
    raw_sections = classify_sections(client, segments)

    if not raw_sections:
        return MeetingSummary(
            executive_summary="Could not identify meeting sections from transcript.",
            model=config.SUMMARY_CLASSIFY_MODEL,
            generated_at=datetime.now().isoformat(),
        )

    print(f"    Identified {len(raw_sections)} sections")

    # --- Pass 2: Summarize each section ---
    summary_sections = []
    total_sections = len(raw_sections)

    # Build timestamp lookup from ALL meeting segments (including empty-text ones)
    all_segments = meeting.segments
    seg_start_map = {s.segment_id: s.start_time for s in all_segments}
    seg_end_map = {s.segment_id: s.end_time for s in all_segments}

    for i, sec in enumerate(raw_sections):
        sec_type = sec.get("type", "procedural")
        title = sec.get("title", f"Section {i + 1}")
        start_seg = sec.get("start_segment", 0)
        end_seg = sec.get("end_segment", start_seg)

        # Resolve timestamps from segment data
        start_time = seg_start_map.get(start_seg, 0.0)
        end_time = seg_end_map.get(end_seg, 0.0)

        section_transcript = _full_section_transcript(segments, start_seg, end_seg)

        _progress(f"summarizing: {title}", i + 1, total_sections)

        if sec_type == "roll_call":
            content = _extract_roll_call(client, section_transcript)
        elif sec_type in ("discussion", "public_comment"):
            content = _summarize_discussion(client, section_transcript, title)
        elif sec_type == "consent_agenda":
            # Use Sonnet for consent agenda since it benefits from richer summary
            content = _summarize_discussion(client, section_transcript, title)
        elif sec_type == "vote":
            content, _ = _extract_votes(client, section_transcript)
        else:
            # Opening, closing, procedural — brief Haiku summary
            if section_transcript.strip():
                msg = client.messages.create(
                    model=config.SUMMARY_CLASSIFY_MODEL,
                    max_tokens=512,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Briefly summarize this {sec_type} section of a council meeting "
                            f"in 1-2 sentences:\n\n{section_transcript}"
                        ),
                    }],
                )
                content = msg.content[0].text.strip()
            else:
                content = ""

        summary_sections.append(SummarySection(
            section_type=sec_type,
            title=title,
            content=content,
            start_time=start_time,
            end_time=end_time,
            start_segment=start_seg,
            end_segment=end_seg,
        ))

    # --- Pass 3: Executive summary ---
    _progress("generating executive summary")
    executive, decisions = _generate_executive_summary(client, summary_sections, meeting)

    return MeetingSummary(
        executive_summary=executive,
        key_decisions=decisions,
        sections=summary_sections,
        model=f"{config.SUMMARY_CLASSIFY_MODEL}+{config.SUMMARY_SYNTHESIZE_MODEL}",
        generated_at=datetime.now().isoformat(),
    )
