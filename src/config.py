"""Configuration constants and paths for CouncilScribe."""

import os
from pathlib import Path

# --- Data root (auto-detect Colab vs local) ---
_DEFAULT_LOCAL = Path.home() / "CouncilScribe"
_DEFAULT_COLAB = Path("/content/drive/MyDrive/CouncilScribe")

def _detect_root() -> Path:
    """Resolve data root: CS_DATA_DIR env var > Colab Drive > ~/CouncilScribe."""
    env = os.environ.get("CS_DATA_DIR")
    if env:
        return Path(env)
    if _DEFAULT_COLAB.exists():
        return _DEFAULT_COLAB
    return _DEFAULT_LOCAL

DRIVE_ROOT = _detect_root()
MEETINGS_DIR = DRIVE_ROOT / "meetings"
PROFILES_DIR = DRIVE_ROOT / "profiles"
CONFIG_DIR = DRIVE_ROOT / "config"

# --- Audio parameters ---
SAMPLE_RATE = 16000
CHANNELS = 1  # mono

# --- Model identifiers ---
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# WeSpeaker ResNet34 — higher-quality embeddings than pyannote/embedding (256-dim).
# NOTE: changing this invalidates stored voice profiles (different dimension).
# PROFILE_SCHEMA_VERSION is bumped whenever this model changes so load_profiles()
# can detect and discard stale embeddings instead of silently mis-matching.
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
WHISPER_MODEL_GPU = "large-v3"
WHISPER_MODEL_CPU = "medium"
WHISPER_COMPUTE_GPU = "float16"
WHISPER_COMPUTE_CPU = "int8"

# --- LLM (Layer 3 speaker identification) ---
LLM_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
LLM_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLM_CONTEXT_TOKENS = 8192

# --- Summary generation (Anthropic API) ---
SUMMARY_CLASSIFY_MODEL = "claude-haiku-4-5-20251001"    # Section classification
SUMMARY_SYNTHESIZE_MODEL = "claude-sonnet-4-5"  # Discussion summaries & executive summary
SUMMARY_MAX_TOKENS_CLASSIFY = 4096
SUMMARY_MAX_TOKENS_SYNTHESIZE = 4096
SUMMARY_CHUNK_SIZE = 150  # Max segments per classification chunk

# --- Thresholds ---
VOICE_MATCH_THRESHOLD = 0.85          # Auto-enroll: voice match or high-confidence ID
SOFT_MATCH_THRESHOLD = 0.50           # Show as hint during pre-identification
ENROLLMENT_PROMPT_THRESHOLD = 0.70    # Prompt for enrollment confirmation (interactive mode)
CONFIDENCE_REVIEW_THRESHOLD = 0.70    # Flag for speaker ID review below this
RETURNING_SPEAKER_THRESHOLD_2 = 0.78  # Lowered match threshold for profiles seen in 2 meetings
RETURNING_SPEAKER_THRESHOLD_3 = 0.70  # Lowered match threshold for profiles seen in 3+ meetings

# --- Diarization tuning ---
MERGE_GAP_SECONDS = 0.5  # merge adjacent same-speaker segments closer than this
SPEAKER_MERGE_THRESHOLD = 0.80  # merge diarized speakers with embedding similarity above this

# --- Post-identification segment merging ---
SEGMENT_MERGE_GAP = 2.0  # merge adjacent same-speaker segments with gap < this (seconds)

# --- Checkpoint ---
CHECKPOINT_EVERY_N_SEGMENTS = 50

# --- Profile DB ---
PROFILE_DB_FILENAME = "speaker_profiles.pkl"
# Bumped to 2 on 2026-04-10 when EMBEDDING_MODEL switched from pyannote/embedding
# (512-dim) to pyannote/wespeaker-voxceleb-resnet34-LM (256-dim). Profiles with
# older schema versions are discarded on load and must be re-enrolled.
# Bumped to 3 on 2026-04-12: added politician_slug and politician_id identity
# fields to StoredProfile for essentials-keyed enrollment.
PROFILE_SCHEMA_VERSION = 3
