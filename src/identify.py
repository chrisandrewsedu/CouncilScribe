"""Stage 4: Speaker identification (Layers 1-3)."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine

from . import config
from .models import Segment, SpeakerMapping


# ---------------------------------------------------------------------------
# Layer 1: Voice Profile Matching
# ---------------------------------------------------------------------------

def match_voice_profiles(
    speaker_embeddings: dict[str, np.ndarray],
    stored_profiles: dict[str, np.ndarray],
) -> dict[str, SpeakerMapping]:
    """Compare speaker embeddings against stored profile centroids.

    Returns mappings for speakers that exceed VOICE_MATCH_THRESHOLD.
    """
    mappings: dict[str, SpeakerMapping] = {}

    for label, embedding in speaker_embeddings.items():
        best_score = 0.0
        best_name = None

        for profile_id, centroid in stored_profiles.items():
            similarity = 1.0 - cosine(embedding, centroid)
            if similarity > best_score:
                best_score = similarity
                best_name = profile_id

        if best_name and best_score >= config.VOICE_MATCH_THRESHOLD:
            mappings[label] = SpeakerMapping(
                speaker_label=label,
                speaker_name=best_name,
                confidence=round(best_score, 3),
                id_method="voice_profile",
            )

    return mappings


# ---------------------------------------------------------------------------
# Layer 2: Rule-Based Pattern Matching
# ---------------------------------------------------------------------------

# Title patterns that precede names in council meetings
_TITLES = r"(?:Councilmember|Council\s*Member|Councilwoman|Councilman|" \
          r"Alderman|Alderwoman|Commissioner|Mayor|Vice[\s-]?Mayor|" \
          r"President|Vice[\s-]?President|Clerk|Secretary|Treasurer|" \
          r"Supervisor|Representative)"

# Name pattern: one or more capitalized words, handling hyphens/apostrophes
_NAME = r"([A-Z][a-zA-Z'\u2019-]+(?:\s+[A-Z][a-zA-Z'\u2019-]+)*)"

_PATTERNS: list[tuple[str, re.Pattern, float, str]] = [
    # Roll call: "Councilmember X?" -> next speaker says "Present"/"Here"
    (
        "roll_call",
        re.compile(
            rf"{_TITLES}\s+{_NAME}\s*\?",
            re.IGNORECASE,
        ),
        0.95,
        "roll_call",
    ),
    # Chair recognition: "The chair recognizes Councilmember X"
    (
        "chair_recognition",
        re.compile(
            rf"(?:the\s+)?chair\s+recognizes\s+{_TITLES}\s+{_NAME}",
            re.IGNORECASE,
        ),
        0.92,
        "chair_recognition",
    ),
    # Self-identification: "This is Councilmember X"
    (
        "self_identification",
        re.compile(
            rf"(?:this\s+is|I\s+am|I'm)\s+{_TITLES}\s+{_NAME}",
            re.IGNORECASE,
        ),
        0.90,
        "self_identification",
    ),
    # Name addressing: "Councilmember X, would you..."
    (
        "name_addressing",
        re.compile(
            rf"{_TITLES}\s+{_NAME}\s*,",
            re.IGNORECASE,
        ),
        0.80,
        "name_addressing",
    ),
    # Title context: "As city attorney..." / "As the mayor..."
    (
        "title_context",
        re.compile(
            r"[Aa]s\s+(?:the\s+)?(?:city\s+)?"
            r"(attorney|clerk|mayor|treasurer|manager|administrator|auditor|comptroller)",
            re.IGNORECASE,
        ),
        0.75,
        "title_context",
    ),
]


def apply_pattern_matching(
    segments: list[Segment],
) -> dict[str, list[SpeakerMapping]]:
    """Scan transcript for name-revealing patterns.

    Returns dict mapping speaker_label -> list of candidate SpeakerMappings
    (multiple patterns may fire for the same speaker).
    """
    candidates: dict[str, list[SpeakerMapping]] = {}

    for i, seg in enumerate(segments):
        text = seg.text

        for pattern_name, regex, conf, method in _PATTERNS:
            match = regex.search(text)
            if not match:
                continue

            # Extract the identified name from the match
            name = match.group(match.lastindex) if match.lastindex else None
            if not name:
                continue

            # Determine which speaker the name applies to
            if pattern_name == "roll_call":
                # The name is in the current segment (clerk calling).
                # The *next* speaker who says "present"/"here" is the named person.
                target_label = _find_roll_call_responder(segments, i, name)
                if not target_label:
                    continue
            elif pattern_name == "self_identification":
                target_label = seg.speaker_label
            elif pattern_name == "title_context":
                target_label = seg.speaker_label
                name = name.title()  # normalize "attorney" -> "Attorney"
            elif pattern_name in ("chair_recognition", "name_addressing"):
                # Name mentioned refers to someone else; try to find who speaks next
                target_label = _find_next_speaker(segments, i, seg.speaker_label)
                if not target_label:
                    continue
            else:
                continue

            mapping = SpeakerMapping(
                speaker_label=target_label,
                speaker_name=name,
                confidence=conf,
                id_method=method,
            )

            if target_label not in candidates:
                candidates[target_label] = []
            candidates[target_label].append(mapping)

    return candidates


def _find_roll_call_responder(
    segments: list[Segment], current_idx: int, name: str
) -> Optional[str]:
    """After a roll call question, find the next speaker who responds."""
    response_pattern = re.compile(
        r"\b(?:present|here|aye|yes)\b", re.IGNORECASE
    )
    for j in range(current_idx + 1, min(current_idx + 5, len(segments))):
        if segments[j].speaker_label != segments[current_idx].speaker_label:
            if response_pattern.search(segments[j].text):
                return segments[j].speaker_label
    return None


def _find_next_speaker(
    segments: list[Segment], current_idx: int, current_label: str
) -> Optional[str]:
    """Find the next different speaker after current segment."""
    for j in range(current_idx + 1, min(current_idx + 5, len(segments))):
        if segments[j].speaker_label != current_label:
            return segments[j].speaker_label
    return None


# ---------------------------------------------------------------------------
# Combined identification orchestrator
# ---------------------------------------------------------------------------

def identify_speakers(
    segments: list[Segment],
    speaker_embeddings: dict[str, np.ndarray],
    stored_profiles: Optional[dict[str, np.ndarray]] = None,
    llm_identify_fn=None,
) -> dict[str, SpeakerMapping]:
    """Orchestrate all identification layers. Higher confidence wins.

    Args:
        segments: Transcribed segments.
        speaker_embeddings: Per-speaker centroid embeddings from diarization.
        stored_profiles: Existing voice profile centroids (Layer 1).
        llm_identify_fn: Optional callable for Layer 3 LLM identification.
            Signature: (segments, current_mappings) -> dict[str, SpeakerMapping]

    Returns:
        Final speaker_label -> SpeakerMapping dict.
    """
    mappings: dict[str, SpeakerMapping] = {}

    # Layer 1: Voice profiles
    if stored_profiles:
        voice_matches = match_voice_profiles(speaker_embeddings, stored_profiles)
        for label, mapping in voice_matches.items():
            mappings[label] = mapping

    # Layer 2: Pattern matching
    pattern_candidates = apply_pattern_matching(segments)
    for label, candidates in pattern_candidates.items():
        best = max(candidates, key=lambda c: c.confidence)
        if label not in mappings or best.confidence > mappings[label].confidence:
            mappings[label] = best

    # Layer 3: LLM-assisted (optional)
    if llm_identify_fn:
        llm_results = llm_identify_fn(segments, mappings)
        for label, mapping in llm_results.items():
            if label not in mappings or mapping.confidence > mappings[label].confidence:
                mappings[label] = mapping

    # Flag low-confidence speakers for review
    all_labels = {seg.speaker_label for seg in segments}
    for label in all_labels:
        if label not in mappings:
            mappings[label] = SpeakerMapping(
                speaker_label=label,
                needs_review=True,
            )
        elif mappings[label].confidence < config.CONFIDENCE_REVIEW_THRESHOLD:
            mappings[label].needs_review = True

    return mappings


def apply_mappings_to_segments(
    segments: list[Segment],
    mappings: dict[str, SpeakerMapping],
) -> list[Segment]:
    """Apply speaker name mappings to all segments."""
    for seg in segments:
        mapping = mappings.get(seg.speaker_label)
        if mapping and mapping.speaker_name:
            seg.speaker_name = mapping.speaker_name
            seg.confidence = mapping.confidence
            seg.id_method = mapping.id_method
    return segments


def flag_for_review(
    mappings: dict[str, SpeakerMapping],
) -> list[SpeakerMapping]:
    """Return speakers that need human review."""
    return [m for m in mappings.values() if m.needs_review]
