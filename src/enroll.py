"""Stage 5: Voice profile enrollment to persistent database."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import config
from .models import Segment, SpeakerMapping

if TYPE_CHECKING:
    from .roster import Roster


@dataclass
class StoredProfile:
    speaker_id: str  # slug, e.g. "adams_jane"
    display_name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    meetings_seen: list[str] = field(default_factory=list)
    total_segments_confirmed: int = 0
    politician_slug: Optional[str] = None   # essentials identifier
    politician_id: Optional[str] = None     # essentials UUID

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
        stored_version = getattr(db, "schema_version", 1)
        if stored_version != config.PROFILE_SCHEMA_VERSION:
            print(
                f"  [enroll] Profile DB schema v{stored_version} incompatible with "
                f"current v{config.PROFILE_SCHEMA_VERSION} (embedding model changed). "
                f"Discarding {len(db.profiles)} stale profile(s); re-enroll from fresh meetings."
            )
            backup = path.with_suffix(f".v{stored_version}.pkl.bak")
            try:
                path.rename(backup)
                print(f"  [enroll] Previous DB backed up to {backup.name}")
            except OSError:
                pass
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


def resolve_enrollment_key(
    display_name: str,
    roster: Optional["Roster"] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Return (profile_key, politician_slug, politician_id).

    If display_name matches a roster member (via correct_speaker_name),
    key = 'essentials:<politician_slug>', identity fields from roster.
    Otherwise: key = _name_to_slug(display_name), both identity fields None.
    """
    if roster is not None:
        from .roster import correct_speaker_name

        corrected = correct_speaker_name(display_name, roster)
        for member in roster.members:
            if corrected == member.name:
                if member.politician_slug:
                    return (
                        f"essentials:{member.politician_slug}",
                        member.politician_slug,
                        member.politician_id,
                    )
                break
    return (_name_to_slug(display_name), None, None)


def _enroll_one(
    db: ProfileDB,
    slug: str,
    display_name: str,
    embedding: np.ndarray,
    meeting_id: str,
    seg_count: int,
    politician_slug: Optional[str] = None,
    politician_id: Optional[str] = None,
) -> None:
    """Add or update a single speaker profile in the database."""
    if slug in db.profiles:
        profile = db.profiles[slug]
        profile.embeddings.append(embedding)
        if meeting_id not in profile.meetings_seen:
            profile.meetings_seen.append(meeting_id)
        profile.total_segments_confirmed += seg_count
        if politician_slug and not profile.politician_slug:
            profile.politician_slug = politician_slug
            profile.politician_id = politician_id
        profile.recompute_centroid()
    else:
        profile = StoredProfile(
            speaker_id=slug,
            display_name=display_name,
            embeddings=[embedding],
            meetings_seen=[meeting_id],
            total_segments_confirmed=seg_count,
            politician_slug=politician_slug,
            politician_id=politician_id,
        )
        profile.recompute_centroid()
        db.profiles[slug] = profile


def enroll_speakers(
    db: ProfileDB,
    speaker_embeddings: dict[str, np.ndarray],
    mappings: dict[str, SpeakerMapping],
    meeting_id: str,
    segments: list[Segment],
    roster: Optional["Roster"] = None,
) -> ProfileDB:
    """Enroll confirmed speakers into the profile database.

    Only enrolls speakers with confidence >= VOICE_MATCH_THRESHOLD.
    When a roster is provided, roster-matched speakers are keyed under
    ``essentials:<politician_slug>`` with identity fields populated.
    """
    for label, mapping in mappings.items():
        if not mapping.speaker_name:
            continue
        if mapping.confidence < config.VOICE_MATCH_THRESHOLD:
            continue
        if label not in speaker_embeddings:
            continue

        slug, pol_slug, pol_id = resolve_enrollment_key(mapping.speaker_name, roster)
        seg_count = sum(1 for s in segments if s.speaker_label == label)
        _enroll_one(
            db, slug, mapping.speaker_name, speaker_embeddings[label],
            meeting_id, seg_count,
            politician_slug=pol_slug, politician_id=pol_id,
        )

    return db


def get_borderline_speakers(
    mappings: dict[str, SpeakerMapping],
    speaker_embeddings: dict[str, np.ndarray],
    segments: list[Segment],
) -> list[dict]:
    """Find speakers eligible for interactive enrollment confirmation.

    Returns speakers with:
    - A name assigned
    - Confidence between ENROLLMENT_PROMPT_THRESHOLD and VOICE_MATCH_THRESHOLD
    - An embedding available
    """
    borderline = []
    for label, mapping in mappings.items():
        if not mapping.speaker_name:
            continue
        if label not in speaker_embeddings:
            continue
        if mapping.confidence >= config.VOICE_MATCH_THRESHOLD:
            continue  # already auto-enrolled
        if mapping.confidence < config.ENROLLMENT_PROMPT_THRESHOLD:
            continue  # too low to consider

        seg_count = sum(1 for s in segments if s.speaker_label == label)
        total_speech = sum(
            s.end_time - s.start_time
            for s in segments if s.speaker_label == label
        )

        # Find a representative segment (near 1/3 point for context)
        speaker_segs = [s for s in segments if s.speaker_label == label and s.text]
        sample_seg = None
        if speaker_segs:
            idx = max(0, len(speaker_segs) // 3 - 1)
            sample_seg = speaker_segs[idx]

        borderline.append({
            "label": label,
            "mapping": mapping,
            "seg_count": seg_count,
            "total_speech_seconds": total_speech,
            "sample_segment": sample_seg,
        })

    # Sort by confidence descending (most likely correct first)
    borderline.sort(key=lambda x: x["mapping"].confidence, reverse=True)
    return borderline


def rename_profile(
    db: ProfileDB,
    old_slug: str,
    new_display_name: str,
) -> bool:
    """Rename a profile's display_name and re-key it under the new slug.

    Returns True if the rename was performed, False if old_slug not found.
    If the new slug already exists, the profiles are merged instead.
    """
    if old_slug not in db.profiles:
        return False

    new_slug = _name_to_slug(new_display_name)
    profile = db.profiles.pop(old_slug)

    if new_slug in db.profiles:
        # Merge into existing profile
        target = db.profiles[new_slug]
        target.embeddings.extend(profile.embeddings)
        for mid in profile.meetings_seen:
            if mid not in target.meetings_seen:
                target.meetings_seen.append(mid)
        target.total_segments_confirmed += profile.total_segments_confirmed
        target.recompute_centroid()
    else:
        profile.speaker_id = new_slug
        profile.display_name = new_display_name
        db.profiles[new_slug] = profile

    return True


def merge_profiles(
    db: ProfileDB,
    source_slug: str,
    dest_slug: str,
) -> bool:
    """Merge source profile into destination profile.

    All embeddings, meetings, and segment counts from source are added
    to dest. Source profile is then removed.

    Returns True if merge was performed, False if either slug not found.
    """
    if source_slug not in db.profiles or dest_slug not in db.profiles:
        return False
    if source_slug == dest_slug:
        return False

    source = db.profiles.pop(source_slug)
    dest = db.profiles[dest_slug]

    dest.embeddings.extend(source.embeddings)
    for mid in source.meetings_seen:
        if mid not in dest.meetings_seen:
            dest.meetings_seen.append(mid)
    dest.total_segments_confirmed += source.total_segments_confirmed
    dest.recompute_centroid()

    return True


def fix_profiles_with_roster(db: ProfileDB, roster) -> list[str]:
    """Rename all profiles whose display_name matches a roster alias.

    Returns list of change descriptions for logging.
    """
    from .roster import correct_speaker_name

    changes = []
    # Collect renames first to avoid mutating dict during iteration
    renames = []
    for slug, profile in list(db.profiles.items()):
        corrected = correct_speaker_name(profile.display_name, roster)
        if corrected != profile.display_name:
            renames.append((slug, corrected, profile.display_name))

    for old_slug, new_name, old_name in renames:
        new_slug = _name_to_slug(new_name)
        rename_profile(db, old_slug, new_name)
        changes.append(f"{old_slug} ({old_name}) -> {new_slug} ({new_name})")

    return changes


def enroll_confirmed(
    db: ProfileDB,
    speaker_embeddings: dict[str, np.ndarray],
    confirmed_labels: list[str],
    mappings: dict[str, SpeakerMapping],
    meeting_id: str,
    segments: list[Segment],
    roster: Optional["Roster"] = None,
) -> ProfileDB:
    """Enroll specific speakers that were confirmed interactively."""
    for label in confirmed_labels:
        mapping = mappings.get(label)
        if not mapping or not mapping.speaker_name:
            continue
        if label not in speaker_embeddings:
            continue

        slug, pol_slug, pol_id = resolve_enrollment_key(mapping.speaker_name, roster)
        seg_count = sum(1 for s in segments if s.speaker_label == label)
        _enroll_one(
            db, slug, mapping.speaker_name, speaker_embeddings[label],
            meeting_id, seg_count,
            politician_slug=pol_slug, politician_id=pol_id,
        )

    return db
