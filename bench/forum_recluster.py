"""Re-cluster an existing diarization's turns over per-turn voice embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

Pure: no env loading, no Modal, no torch. The Modal fetch and the CLI live in
`scripts/recluster_forum_turns.py`.
"""

from __future__ import annotations

import copy


def turn_label(index: int) -> str:
    """Unique per-turn label. Zero-padded so lexical order is turn order."""
    return f"TURN_{index:04d}"


def as_unique_label_segments(segments: list[dict]) -> list[dict]:
    """Copy `segments`, relabelling each with its own unique speaker label.

    `bench/modal_app.py:1305 pipeline_extract_embeddings` averages wespeaker
    embeddings per `speaker_label` over arbitrary supplied segments, so one
    label per turn makes each returned "centroid" that turn's own embedding.
    That is why Experiment B needs no new Modal code.

    Deep-copies rather than editing in place: `merge_adjacent_segments`
    renumbering its own inputs has already cost this repo a false staleness
    diagnosis, so a probe must never mutate what it probes.
    """
    unique = []
    for index, segment in enumerate(segments):
        clone = copy.deepcopy(segment)
        clone["speaker_label"] = turn_label(index)
        unique.append(clone)
    return unique
