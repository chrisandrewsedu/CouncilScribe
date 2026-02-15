"""Stage 2: Speaker diarization using pyannote.audio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config
from .audio_utils import load_wav, slice_audio
from .models import Segment


def load_diarization_pipeline(hf_token: str):
    """Load pyannote speaker-diarization-3.1 pipeline.

    Places model on CUDA if available.
    """
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        config.DIARIZATION_MODEL, token=hf_token
    )
    pipeline.to(device)
    return pipeline


def run_diarization(
    pipeline,
    wav_path: str | Path,
    num_speakers: Optional[int] = None,
) -> list[Segment]:
    """Run diarization and return merged speaker segments.

    Adjacent same-speaker segments with gaps < MERGE_GAP_SECONDS are merged.
    """
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(str(wav_path), **kwargs)

    raw_segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append((turn.start, turn.end, speaker))

    merged = _merge_segments(raw_segments)

    segments = []
    for i, (start, end, label) in enumerate(merged):
        segments.append(
            Segment(
                segment_id=i,
                start_time=round(start, 3),
                end_time=round(end, 3),
                speaker_label=label,
            )
        )
    return segments


def _merge_segments(
    raw: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge adjacent same-speaker segments with small gaps."""
    if not raw:
        return []

    merged = [raw[0]]
    for start, end, label in raw[1:]:
        prev_start, prev_end, prev_label = merged[-1]
        if label == prev_label and (start - prev_end) < config.MERGE_GAP_SECONDS:
            merged[-1] = (prev_start, end, label)
        else:
            merged.append((start, end, label))
    return merged


def extract_speaker_embeddings(
    wav_path: str | Path,
    segments: list[Segment],
    hf_token: str,
) -> dict[str, np.ndarray]:
    """Extract average 512-dim d-vector embedding per speaker.

    Returns dict mapping speaker_label -> centroid embedding (numpy array).
    """
    from pyannote.audio import Model, Inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.from_pretrained(config.EMBEDDING_MODEL, token=hf_token)
    inference = Inference(model, window="whole", device=device)

    samples, sr = load_wav(wav_path)

    speaker_embeddings: dict[str, list[np.ndarray]] = {}

    for seg in segments:
        audio_slice = slice_audio(samples, sr, seg.start_time, seg.end_time)
        if len(audio_slice) < sr * 0.3:  # skip very short segments (<0.3s)
            continue

        waveform = torch.tensor(audio_slice).unsqueeze(0).to(device)
        embedding = inference({"waveform": waveform, "sample_rate": sr})

        label = seg.speaker_label
        if label not in speaker_embeddings:
            speaker_embeddings[label] = []
        speaker_embeddings[label].append(embedding)

    centroids = {}
    for label, embs in speaker_embeddings.items():
        centroids[label] = np.mean(embs, axis=0)

    return centroids
