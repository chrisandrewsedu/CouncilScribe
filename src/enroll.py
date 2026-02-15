"""Stage 5: Voice profile enrollment to persistent database."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .models import Segment, SpeakerMapping


@dataclass
class StoredProfile:
    speaker_id: str  # slug, e.g. "adams_jane"
    display_name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    meetings_seen: list[str] = field(default_factory=list)
    total_segments_confirmed: int = 0

    def recompute_centroid(self) -> None:
        if self.embeddings:
            self.centroid = np.mean(self.embeddings, axis=0)


@dataclass
class ProfileDB:
    schema_version: int = config.PROFILE_SCHEMA_VERSION
    profiles: dict[str, StoredProfile] = field(default_factory=dict)


def _db_path() -> Path:
    return config.PROFILES_DIR / config.PROFILE_DB_FILENAME


def load_profiles() -> ProfileDB:
    """Load profile database from pickle file on Drive."""
    path = _db_path()
    if path.exists():
        with open(path, "rb") as f:
            db = pickle.load(f)
        if not isinstance(db, ProfileDB):
            return ProfileDB()
        return db
    return ProfileDB()


def save_profiles(db: ProfileDB) -> None:
    """Save profile database to pickle file on Drive."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)


def get_stored_centroids(db: ProfileDB) -> dict[str, np.ndarray]:
    """Extract speaker_id -> centroid mapping for Layer 1 matching."""
    return {
        pid: p.centroid
        for pid, p in db.profiles.items()
        if p.centroid is not None
    }


def _name_to_slug(name: str) -> str:
    """Convert display name to a slug ID. e.g. 'Jane Adams' -> 'adams_jane'."""
    parts = name.strip().split()
    # Remove titles
    titles = {
        "councilmember", "councilwoman", "councilman", "alderman",
        "alderwoman", "commissioner", "mayor", "vice-mayor",
        "president", "vice-president", "clerk", "secretary",
        "treasurer", "supervisor", "representative",
    }
    filtered = [p for p in parts if p.lower() not in titles]
    if not filtered:
        filtered = parts

    if len(filtered) >= 2:
        return f"{filtered[-1]}_{filtered[0]}".lower()
    return filtered[0].lower()


def enroll_speakers(
    db: ProfileDB,
    speaker_embeddings: dict[str, np.ndarray],
    mappings: dict[str, SpeakerMapping],
    meeting_id: str,
    segments: list[Segment],
) -> ProfileDB:
    """Enroll confirmed speakers into the profile database.

    Only enrolls speakers with confidence >= VOICE_MATCH_THRESHOLD.
    """
    for label, mapping in mappings.items():
        if not mapping.speaker_name:
            continue
        if mapping.confidence < config.VOICE_MATCH_THRESHOLD:
            continue
        if label not in speaker_embeddings:
            continue

        slug = _name_to_slug(mapping.speaker_name)
        embedding = speaker_embeddings[label]

        # Count segments for this speaker
        seg_count = sum(
            1 for s in segments if s.speaker_label == label
        )

        if slug in db.profiles:
            profile = db.profiles[slug]
            profile.embeddings.append(embedding)
            if meeting_id not in profile.meetings_seen:
                profile.meetings_seen.append(meeting_id)
            profile.total_segments_confirmed += seg_count
            profile.recompute_centroid()
        else:
            profile = StoredProfile(
                speaker_id=slug,
                display_name=mapping.speaker_name,
                embeddings=[embedding],
                meetings_seen=[meeting_id],
                total_segments_confirmed=seg_count,
            )
            profile.recompute_centroid()
            db.profiles[slug] = profile

    return db
