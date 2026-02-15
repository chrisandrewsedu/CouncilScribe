"""Data classes for CouncilScribe pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    word: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"word": self.word, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict) -> Word:
        return cls(word=d["word"], start=d["start"], end=d["end"])


@dataclass
class Segment:
    segment_id: int
    start_time: float
    end_time: float
    speaker_label: str
    text: str = ""
    words: list[Word] = field(default_factory=list)
    speaker_name: Optional[str] = None
    confidence: Optional[float] = None
    id_method: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_label": self.speaker_label,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }
        if self.speaker_name is not None:
            d["speaker_name"] = self.speaker_name
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.id_method is not None:
            d["id_method"] = self.id_method
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            segment_id=d["segment_id"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            speaker_label=d["speaker_label"],
            text=d.get("text", ""),
            words=[Word.from_dict(w) for w in d.get("words", [])],
            speaker_name=d.get("speaker_name"),
            confidence=d.get("confidence"),
            id_method=d.get("id_method"),
        )


@dataclass
class SpeakerMapping:
    speaker_label: str
    speaker_name: Optional[str] = None
    confidence: float = 0.0
    id_method: Optional[str] = None
    needs_review: bool = False

    def to_dict(self) -> dict:
        return {
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
            "confidence": self.confidence,
            "id_method": self.id_method,
            "needs_review": self.needs_review,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpeakerMapping:
        return cls(
            speaker_label=d["speaker_label"],
            speaker_name=d.get("speaker_name"),
            confidence=d.get("confidence", 0.0),
            id_method=d.get("id_method"),
            needs_review=d.get("needs_review", False),
        )


@dataclass
class ProcessingMetadata:
    pipeline_version: str = "1.0.0"
    diarization_model: str = ""
    transcription_model: str = ""
    gpu_used: bool = False
    processing_time_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "pipeline_version": self.pipeline_version,
            "diarization_model": self.diarization_model,
            "transcription_model": self.transcription_model,
            "gpu_used": self.gpu_used,
        }
        if self.processing_time_seconds is not None:
            d["processing_time_seconds"] = self.processing_time_seconds
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProcessingMetadata:
        return cls(
            pipeline_version=d.get("pipeline_version", "1.0.0"),
            diarization_model=d.get("diarization_model", ""),
            transcription_model=d.get("transcription_model", ""),
            gpu_used=d.get("gpu_used", False),
            processing_time_seconds=d.get("processing_time_seconds"),
        )


@dataclass
class Meeting:
    meeting_id: str
    city: str
    date: str
    meeting_type: str = "Regular Session"
    audio_source: str = ""
    duration_seconds: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    speakers: dict[str, SpeakerMapping] = field(default_factory=dict)
    processing_metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "city": self.city,
            "date": self.date,
            "meeting_type": self.meeting_type,
            "audio_source": self.audio_source,
            "duration_seconds": self.duration_seconds,
            "segments": [s.to_dict() for s in self.segments],
            "speakers": {k: v.to_dict() for k, v in self.speakers.items()},
            "processing_metadata": self.processing_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Meeting:
        return cls(
            meeting_id=d["meeting_id"],
            city=d["city"],
            date=d["date"],
            meeting_type=d.get("meeting_type", "Regular Session"),
            audio_source=d.get("audio_source", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            speakers={
                k: SpeakerMapping.from_dict(v)
                for k, v in d.get("speakers", {}).items()
            },
            processing_metadata=ProcessingMetadata.from_dict(
                d.get("processing_metadata", {})
            ),
        )
