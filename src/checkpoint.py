"""Pipeline state machine with checkpoint/resume logic."""

from __future__ import annotations

import json
import os
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Optional

from . import config


class PipelineStage(IntEnum):
    NOT_STARTED = 0
    INGESTED = 1
    DIARIZED = 2
    TRANSCRIBED = 3
    IDENTIFIED = 4
    SUMMARIZED = 5
    ENROLLED = 6
    EXPORTED = 7


class PipelineState:
    """Tracks pipeline progress for a single meeting, persisted as JSON."""

    def __init__(self, meeting_dir: Path) -> None:
        self.meeting_dir = meeting_dir
        self._state_file = meeting_dir / "pipeline_state.json"
        self.completed_stage: PipelineStage = PipelineStage.NOT_STARTED
        self.transcription_progress: int = 0  # last completed segment index
        self.total_segments: int = 0
        self.body_slug: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if self._state_file.exists():
            with open(self._state_file, "r") as f:
                data = json.load(f)
            self.completed_stage = PipelineStage(data.get("completed_stage", 0))
            self.transcription_progress = data.get("transcription_progress", 0)
            self.total_segments = data.get("total_segments", 0)
            self.body_slug = data.get("body_slug")  # None if legacy/untagged — D-05 compat

    def save(self) -> None:
        """Atomic write: write to temp file then rename."""
        data = {
            "completed_stage": int(self.completed_stage),
            "transcription_progress": self.transcription_progress,
            "total_segments": self.total_segments,
            "body_slug": self.body_slug,
        }
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.meeting_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self._state_file))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def mark_complete(self, stage: PipelineStage) -> None:
        self.completed_stage = stage
        self.save()

    def rewind_for_retag(self) -> None:
        """D-04: Invalidate downstream stages when body_slug changes.

        Rewinds completed_stage to TRANSCRIBED (stage 3) so Stages 4-7 re-run,
        deletes stale Stage 4+ artifacts produced against the old roster
        (pre_identifications.json per D-11, plus llm_partial_results.json and
        transcript_named.json), and atomically persists the rewound state.
        Caller must set self.body_slug to the new slug BEFORE calling this.

        Files are deleted BEFORE save() so that if the process crashes between
        the unlinks and the save, pipeline_state.json still reflects the old
        stage and a resume will regenerate the missing artifacts rather than
        trust a stale state file.
        """
        self.completed_stage = PipelineStage.TRANSCRIBED
        stale = (
            "pre_identifications.json",   # D-11: stale roster-specific guesses
            "llm_partial_results.json",   # Stage 4 LLM resume cache
            "transcript_named.json",      # Stage 4 output with old-roster names
        )
        for name in stale:
            p = self.meeting_dir / name
            if p.exists():
                p.unlink()
        self.save()

    def is_complete(self, stage: PipelineStage) -> bool:
        return self.completed_stage >= stage

    def update_transcription_progress(
        self, segment_index: int, total: int
    ) -> None:
        self.transcription_progress = segment_index
        self.total_segments = total
        self.save()


def ensure_drive_structure(meeting_id: Optional[str] = None) -> Path:
    """Create the CouncilScribe directory tree on Google Drive.

    Returns the meeting directory if meeting_id is provided, else the root.
    """
    for d in (config.MEETINGS_DIR, config.PROFILES_DIR, config.CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if meeting_id:
        meeting_dir = config.MEETINGS_DIR / meeting_id
        meeting_dir.mkdir(parents=True, exist_ok=True)
        (meeting_dir / "exports").mkdir(exist_ok=True)
        return meeting_dir

    return config.DRIVE_ROOT
