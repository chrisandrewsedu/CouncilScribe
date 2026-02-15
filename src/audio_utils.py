"""Shared audio helper functions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import config


def check_ffmpeg_installed() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def get_audio_duration(wav_path: str | Path) -> float:
    """Return duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def load_wav(wav_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a numpy array. Returns (samples, sample_rate)."""
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")
    return data, sr


def slice_audio(
    samples: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Extract a time slice from audio samples."""
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    return samples[start_idx:end_idx]


def apply_noise_reduction(
    samples: np.ndarray, sr: int
) -> np.ndarray:
    """Apply spectral-gating noise reduction."""
    import noisereduce as nr

    return nr.reduce_noise(y=samples, sr=sr)
