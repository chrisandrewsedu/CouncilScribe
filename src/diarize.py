"""Stage 2: Speaker diarization using pyannote.audio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import config
from .audio_utils import load_wav, slice_audio
from .models import Segment


def _get_torch_device() -> torch.device:
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_diarization_pipeline(hf_token: str):
    """Load pyannote speaker-diarization-3.1 pipeline.

    Places model on CUDA or MPS if available, else CPU.
    """
    from pyannote.audio import Pipeline

    device = _get_torch_device()
    pipeline = Pipeline.from_pretrained(
        config.DIARIZATION_MODEL, token=hf_token
    )
    pipeline.to(device)
    print(f"  Diarization device: {device}")
    return pipeline


class _LineProgressHook:
    """Progress hook that prints line-by-line instead of tqdm carriage returns."""

    def __init__(self):
        self._current_step = None
        self._last_pct = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._current_step:
            print(flush=True)

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name != self._current_step:
            if self._current_step:
                print(flush=True)
            self._current_step = step_name
            self._last_pct = -1
            print(f"  [{step_name}]", end="", flush=True)

        if total and completed:
            pct = int((completed / total) * 100)
            if pct >= self._last_pct + 10:
                print(f" {pct}%", end="", flush=True)
                self._last_pct = pct


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

    with _LineProgressHook() as hook:
        diarization = pipeline(str(wav_path), hook=hook, **kwargs)

    # Handle both old Annotation API (itertracks) and new DiarizeOutput API
    raw_segments: list[tuple[float, float, str]] = []
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append((turn.start, turn.end, speaker))
    elif hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            label = f"SPEAKER_{int(speaker):02d}" if str(speaker).isdigit() else str(speaker)
            raw_segments.append((turn.start, turn.end, label))
    else:
        raise RuntimeError(
            f"Unexpected diarization output type: {type(diarization)}. "
            "Check pyannote.audio version compatibility."
        )

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

    device = _get_torch_device()
    model = Model.from_pretrained(config.EMBEDDING_MODEL, token=hf_token)
    inference = Inference(model, window="whole", device=device)

    samples, sr = load_wav(wav_path)

    speaker_embeddings: dict[str, list[np.ndarray]] = {}

    total = len(segments)
    for i, seg in enumerate(segments):
        audio_slice = slice_audio(samples, sr, seg.start_time, seg.end_time)
        if len(audio_slice) < sr * 0.3:  # skip very short segments (<0.3s)
            continue

        waveform = torch.tensor(audio_slice).unsqueeze(0).to(device)
        embedding = inference({"waveform": waveform, "sample_rate": sr})

        label = seg.speaker_label
        if label not in speaker_embeddings:
            speaker_embeddings[label] = []
        speaker_embeddings[label].append(embedding)

        if (i + 1) % 50 == 0:
            pct = ((i + 1) / total) * 100
            print(f"  Embeddings: {i + 1}/{total} ({pct:.0f}%)", flush=True)

    centroids = {}
    for label, embs in speaker_embeddings.items():
        centroids[label] = np.mean(embs, axis=0)

    return centroids
