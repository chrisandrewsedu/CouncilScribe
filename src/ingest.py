"""Stage 1: Audio ingestion and normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import config
from .audio_utils import (
    apply_noise_reduction,
    check_ffmpeg_installed,
    get_audio_duration,
    load_wav,
)


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    noise_reduce: bool = False,
) -> dict:
    """Normalize audio to 16kHz mono WAV via ffmpeg.

    Args:
        input_path: Source audio/video file.
        output_path: Destination WAV file path.
        noise_reduce: If True, apply spectral-gating noise reduction after conversion.

    Returns:
        Metadata dict with source, output path, duration, and whether noise reduction was applied.
    """
    if not check_ffmpeg_installed():
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ac", str(config.CHANNELS),
            "-ar", str(config.SAMPLE_RATE),
            "-vn",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    if noise_reduce:
        import soundfile as sf

        samples, sr = load_wav(output_path)
        cleaned = apply_noise_reduction(samples, sr)
        sf.write(str(output_path), cleaned, sr)

    duration = get_audio_duration(output_path)

    return {
        "source": str(input_path),
        "output": str(output_path),
        "duration_seconds": duration,
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.CHANNELS,
        "noise_reduced": noise_reduce,
    }
