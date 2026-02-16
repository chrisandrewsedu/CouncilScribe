"""Configuration constants and paths for CouncilScribe."""

from pathlib import Path

# --- Google Drive paths ---
DRIVE_ROOT = Path("/content/drive/MyDrive/CouncilScribe")
MEETINGS_DIR = DRIVE_ROOT / "meetings"
PROFILES_DIR = DRIVE_ROOT / "profiles"
CONFIG_DIR = DRIVE_ROOT / "config"

# --- Audio parameters ---
SAMPLE_RATE = 16000
CHANNELS = 1  # mono

# --- Model identifiers ---
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
EMBEDDING_MODEL = "pyannote/embedding"
WHISPER_MODEL_GPU = "large-v3"
WHISPER_MODEL_CPU = "medium"
WHISPER_COMPUTE_GPU = "float16"
WHISPER_COMPUTE_CPU = "int8"

# --- LLM (Layer 3 speaker identification) ---
LLM_REPO = "bartowski/Phi-3.5-mini-instruct-GGUF"
LLM_FILENAME = "Phi-3.5-mini-instruct-Q4_K_M.gguf"
LLM_CONTEXT_TOKENS = 4096

# --- Thresholds ---
VOICE_MATCH_THRESHOLD = 0.85
CONFIDENCE_REVIEW_THRESHOLD = 0.70

# --- Diarization tuning ---
MERGE_GAP_SECONDS = 0.5  # merge adjacent same-speaker segments closer than this

# --- Checkpoint ---
CHECKPOINT_EVERY_N_SEGMENTS = 50

# --- Profile DB ---
PROFILE_DB_FILENAME = "speaker_profiles.pkl"
PROFILE_SCHEMA_VERSION = 1
