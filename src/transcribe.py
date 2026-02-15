"""Stage 3: Segment-level transcription using faster-whisper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import config
from .audio_utils import load_wav, slice_audio
from .models import Segment, Word


def load_whisper_model():
    """Load faster-whisper model. GPU: large-v3 float16, CPU: medium int8."""
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():
        model = WhisperModel(
            config.WHISPER_MODEL_GPU,
            device="cuda",
            compute_type=config.WHISPER_COMPUTE_GPU,
        )
    else:
        model = WhisperModel(
            config.WHISPER_MODEL_CPU,
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_CPU,
        )
    return model


def transcribe_segments(
    model,
    wav_path: str | Path,
    segments: list[Segment],
    checkpoint_callback: Optional[Callable[[int, int], None]] = None,
    resume_from: int = 0,
) -> list[Segment]:
    """Transcribe each diarized segment with word-level timestamps.

    Args:
        model: faster-whisper WhisperModel instance.
        wav_path: Path to the normalized WAV file.
        segments: List of diarized segments (modified in-place).
        checkpoint_callback: Called every CHECKPOINT_EVERY_N_SEGMENTS with
            (current_index, total) to allow saving progress.
        resume_from: Segment index to resume from (for checkpoint recovery).

    Returns:
        The same segments list with text and words populated.
    """
    samples, sr = load_wav(wav_path)
    total = len(segments)

    for i in range(resume_from, total):
        seg = segments[i]
        audio_slice = slice_audio(samples, sr, seg.start_time, seg.end_time)

        if len(audio_slice) < sr * 0.1:  # skip segments shorter than 0.1s
            seg.text = ""
            seg.words = []
            continue

        result_segments, _ = model.transcribe(
            audio_slice,
            word_timestamps=True,
            language="en",
        )

        words = []
        text_parts = []
        for rs in result_segments:
            if rs.words:
                for w in rs.words:
                    words.append(
                        Word(
                            word=w.word.strip(),
                            start=round(seg.start_time + w.start, 3),
                            end=round(seg.start_time + w.end, 3),
                        )
                    )
            text_parts.append(rs.text.strip())

        seg.text = " ".join(text_parts).strip()
        seg.words = words

        if (
            checkpoint_callback
            and (i + 1) % config.CHECKPOINT_EVERY_N_SEGMENTS == 0
        ):
            checkpoint_callback(i + 1, total)

    return segments


def save_raw_transcript(segments: list[Segment], output_path: str | Path) -> None:
    """Save transcript segments to JSON for checkpoint recovery."""
    data = [seg.to_dict() for seg in segments]
    with open(str(output_path), "w") as f:
        json.dump(data, f, indent=2)


def load_raw_transcript(input_path: str | Path) -> list[Segment]:
    """Load transcript segments from a checkpoint JSON file."""
    with open(str(input_path), "r") as f:
        data = json.load(f)
    return [Segment.from_dict(d) for d in data]
