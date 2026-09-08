#!/usr/bin/env python
"""CLI for re-clustering an existing diarization over per-turn embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

The clustering itself is pure and lives in `bench.forum_recluster`; this file
holds the one Modal call and the command-line surface.

Loads no env: Modal authenticates from ~/.modal.toml, and the worker gets
HF_TOKEN from a Modal secret. No DB writes, no LLM. Usage:
  .venv/bin/python scripts/recluster_forum_turns.py <meeting-id> --out turns.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.forum_recluster import (  # noqa: E402
    as_unique_label_segments,
    turn_label,
)


def fetch_turn_embeddings(
    meeting_id: str, segments: list[dict]
) -> dict[int, list[float]]:
    """One wespeaker vector per turn, from an L4 on Modal.

    Turns shorter than the worker's 0.3s floor, and any whose embedding is
    non-finite, come back absent rather than zeroed — the caller decides what
    to do with a turn that has no voice evidence.
    """
    from src.modal_compute import _modal_app

    app = _modal_app()
    payload = json.dumps(as_unique_label_segments(segments))
    with app.app.run():
        raw = app.pipeline_extract_embeddings.remote(meeting_id, payload)
    by_label = json.loads(raw)
    return {
        index: by_label[turn_label(index)]
        for index in range(len(segments))
        if turn_label(index) in by_label
    }
