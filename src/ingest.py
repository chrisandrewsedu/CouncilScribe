"""Stage 1: Audio ingestion and normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .audio_utils import (
    apply_noise_reduction,
    check_ffmpeg_installed,
    get_audio_duration,
    load_wav,
)


def _is_url(path: str) -> bool:
    """Check if a string looks like a URL."""
    try:
        parsed = urlparse(str(path))
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    noise_reduce: bool = False,
) -> dict:
    """Normalize audio to 16kHz mono WAV via ffmpeg.

    Accepts a local file path or a URL. If a URL is provided, the video is
    downloaded first, then normalized.

    Args:
        input_path: Source audio/video file path or URL (direct video URL or CATS TV page URL).
        output_path: Destination WAV file path.
        noise_reduce: If True, apply spectral-gating noise reduction after conversion.

    Returns:
        Metadata dict with source, output path, duration, and whether noise reduction was applied.
    """
    if not check_ffmpeg_installed():
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_str = str(input_path)

    # Download from URL if needed
    if _is_url(source_str):
        from .download import download_from_url

        # Save downloaded video next to the output WAV
        parsed = urlparse(source_str)
        ext = Path(parsed.path).suffix or ".m4v"
        download_path = output_path.parent / f"source{ext}"
        print(f"  Downloading from URL...")
        download_from_url(source_str, download_path)
        ffmpeg_input = str(download_path)
    else:
        ffmpeg_input = str(Path(input_path))

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", ffmpeg_input,
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
        "source": source_str,
        "output": str(output_path),
        "duration_seconds": duration,
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.CHANNELS,
        "noise_reduced": noise_reduce,
    }
