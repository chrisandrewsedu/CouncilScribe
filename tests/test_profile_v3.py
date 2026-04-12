"""Tests for profile schema v3 — essentials identity fields and enrollment keying."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.enroll import ProfileDB, StoredProfile, load_profiles
from src.roster import RosterMember, load_roster


# ---------------------------------------------------------------------------
# Task 1: Schema bump + dataclass enrichment
# ---------------------------------------------------------------------------


def test_profile_db_schema_version():
    """ProfileDB defaults to schema version 3."""
    db = ProfileDB()
    assert db.schema_version == 3


def test_stored_profile_v3_fields():
    """StoredProfile has politician_slug and politician_id defaulting to None."""
    profile = StoredProfile(speaker_id="test", display_name="Test Person")
    assert profile.politician_slug is None
    assert profile.politician_id is None


def test_v2_auto_discard(tmp_path, monkeypatch):
    """Loading a v2 profile DB auto-discards it, creates backup, returns empty v3 DB."""
    # Create a fake v2 ProfileDB pickle
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    db_path = profiles_dir / "speaker_profiles.pkl"

    db = ProfileDB()
    db.schema_version = 2
    db.profiles["old_speaker"] = StoredProfile(
        speaker_id="old_speaker",
        display_name="Old Speaker",
    )
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    # Monkeypatch _db_path to return our temp file
    monkeypatch.setattr("src.enroll._db_path", lambda: db_path)

    result = load_profiles()
    assert result.schema_version == 3
    assert len(result.profiles) == 0
    assert (profiles_dir / "speaker_profiles.v2.pkl.bak").exists()


def test_roster_member_has_identity_fields():
    """RosterMember can carry politician_slug and politician_id."""
    member = RosterMember(
        name="X",
        politician_slug="x-slug",
        politician_id="uuid-x",
    )
    assert member.politician_slug == "x-slug"
    assert member.politician_id == "uuid-x"


def test_load_roster_populates_identity_fields(fake_roster_cache):
    """load_roster with body_slug populates politician_slug and politician_id from cache."""
    fake_roster_cache("bloomington-common-council")
    roster = load_roster(body_slug="bloomington-common-council")
    assert roster is not None
    assert len(roster.members) >= 1
    member = roster.members[0]
    assert member.politician_slug == "isabel-piedmont-smith"
    assert member.politician_id == "uuid-ips"


# ---------------------------------------------------------------------------
# Task 2: Enrollment keying — resolve_enrollment_key + wiring
# ---------------------------------------------------------------------------

from src.enroll import enroll_speakers, enroll_confirmed
from src.models import Segment, SpeakerMapping
from src.roster import Roster


def _make_roster():
    """Build a test roster with one politician member."""
    return Roster(
        city="",
        body="Bloomington Common Council",
        members=[
            RosterMember(
                name="Councilmember Piedmont-Smith",
                aliases=["Isabel Piedmont-Smith", "Piedmont-Smith", "Isabel"],
                politician_slug="isabel-piedmont-smith",
                politician_id="uuid-ips",
            ),
        ],
    )


def _make_embedding():
    return np.random.randn(256).astype(np.float32)


def _make_mapping(speaker_name, label="SPEAKER_01"):
    return {
        label: SpeakerMapping(
            speaker_label=label,
            speaker_name=speaker_name,
            confidence=0.90,
            id_method="human_review",
        )
    }


def _make_segments(label="SPEAKER_01"):
    return [
        Segment(
            segment_id=0,
            start_time=0.0,
            end_time=5.0,
            speaker_label=label,
            text="Hello world",
        )
    ]


def test_enroll_roster_member_uses_essentials_key():
    """Enrolling a roster-matched speaker uses essentials:<slug> key."""
    roster = _make_roster()
    embeddings = {"SPEAKER_01": _make_embedding()}
    mappings = _make_mapping("Councilmember Piedmont-Smith")
    segments = _make_segments()

    db = enroll_speakers(ProfileDB(), embeddings, mappings, "m1", segments, roster=roster)
    assert "essentials:isabel-piedmont-smith" in db.profiles


def test_essentials_profile_identity_fields():
    """After enrolling a roster member, profile carries identity fields."""
    roster = _make_roster()
    embeddings = {"SPEAKER_01": _make_embedding()}
    mappings = _make_mapping("Councilmember Piedmont-Smith")
    segments = _make_segments()

    db = enroll_speakers(ProfileDB(), embeddings, mappings, "m1", segments, roster=roster)
    profile = db.profiles["essentials:isabel-piedmont-smith"]
    assert profile.politician_slug == "isabel-piedmont-smith"
    assert profile.politician_id == "uuid-ips"


def test_non_roster_speaker_local_slug():
    """Non-roster speaker enrolls under local slug with no identity fields."""
    roster = _make_roster()
    embeddings = {"SPEAKER_01": _make_embedding()}
    mappings = _make_mapping("John Public")
    segments = _make_segments()

    db = enroll_speakers(ProfileDB(), embeddings, mappings, "m1", segments, roster=roster)
    assert "public_john" in db.profiles
    assert db.profiles["public_john"].politician_slug is None


def test_mixed_profiles_coexist():
    """Essentials-keyed and local-keyed profiles coexist in the same DB."""
    roster = _make_roster()
    embeddings = {
        "SPEAKER_01": _make_embedding(),
        "SPEAKER_02": _make_embedding(),
    }
    mappings = {
        "SPEAKER_01": SpeakerMapping(
            speaker_label="SPEAKER_01",
            speaker_name="Councilmember Piedmont-Smith",
            confidence=0.90,
            id_method="human_review",
        ),
        "SPEAKER_02": SpeakerMapping(
            speaker_label="SPEAKER_02",
            speaker_name="John Public",
            confidence=0.90,
            id_method="human_review",
        ),
    }
    segments = [
        Segment(segment_id=0, start_time=0.0, end_time=5.0, speaker_label="SPEAKER_01", text="A"),
        Segment(segment_id=1, start_time=5.0, end_time=10.0, speaker_label="SPEAKER_02", text="B"),
    ]

    db = enroll_speakers(ProfileDB(), embeddings, mappings, "m1", segments, roster=roster)
    assert "essentials:isabel-piedmont-smith" in db.profiles
    assert "public_john" in db.profiles
    assert len(db.profiles) == 2


def test_essentials_profile_accumulates_across_meetings():
    """Re-enrolling same politician from second meeting adds to existing profile."""
    roster = _make_roster()
    mappings = _make_mapping("Councilmember Piedmont-Smith")
    segments = _make_segments()

    emb1 = {"SPEAKER_01": _make_embedding()}
    emb2 = {"SPEAKER_01": _make_embedding()}

    db = enroll_speakers(ProfileDB(), emb1, mappings, "m1", segments, roster=roster)
    db = enroll_speakers(db, emb2, mappings, "m2", segments, roster=roster)

    profile = db.profiles["essentials:isabel-piedmont-smith"]
    assert len(profile.embeddings) == 2
    assert profile.meetings_seen == ["m1", "m2"]


def test_enroll_confirmed_roster_member():
    """enroll_confirmed with a roster uses essentials key for roster members."""
    roster = _make_roster()
    embeddings = {"SPEAKER_01": _make_embedding()}
    mappings = _make_mapping("Councilmember Piedmont-Smith")
    segments = _make_segments()

    db = enroll_confirmed(
        ProfileDB(), embeddings, ["SPEAKER_01"], mappings, "m1", segments, roster=roster
    )
    assert "essentials:isabel-piedmont-smith" in db.profiles


# ---------------------------------------------------------------------------
# Task 3 (Plan 02): reenroll_profiles.py body-slug-aware re-enrollment
# ---------------------------------------------------------------------------

import json
import sys


def _write_transcript_named(meeting_dir: Path, speaker_name: str = "Councilmember Piedmont-Smith"):
    """Write a minimal transcript_named.json into a meeting directory."""
    data = {
        "segments": [
            {
                "segment_id": 0,
                "start_time": 0,
                "end_time": 1,
                "speaker_label": "SPEAKER_01",
                "text": "test",
            }
        ],
        "speakers": {
            "SPEAKER_01": {
                "speaker_label": "SPEAKER_01",
                "speaker_name": speaker_name,
                "confidence": 0.95,
                "id_method": "human_review",
            }
        },
    }
    (meeting_dir / "transcript_named.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def test_reenroll_reads_body_slug(
    tagged_meeting_dir, fake_roster_cache, tmp_meetings_dir, monkeypatch
):
    """reenroll main() reads body_slug from PipelineState and loads the roster,
    producing an essentials-keyed profile for a roster-matched speaker."""
    # Set up meeting dir with body_slug and transcript
    mdir = tagged_meeting_dir("bloomington-common-council", "2026-test-meeting", completed_stage=4)
    _write_transcript_named(mdir)
    (mdir / "audio.wav").touch()

    # Set up roster cache
    fake_roster_cache("bloomington-common-council")

    # Mock embedding extraction (avoid loading real model)
    monkeypatch.setattr(
        "src.diarize.extract_speaker_embeddings",
        lambda wav, segs, token: {"SPEAKER_01": np.random.randn(256).astype(np.float32)},
    )

    # Mock load_profiles to return empty DB
    monkeypatch.setattr("src.enroll.load_profiles", lambda: ProfileDB())

    # Capture saved DB
    saved = {}

    def _capture_save(db):
        saved["db"] = db

    monkeypatch.setattr("src.enroll.save_profiles", _capture_save)

    # Mock HF token
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    # Run reenroll for just this meeting
    monkeypatch.setattr(sys, "argv", ["reenroll_profiles.py", "2026-test-meeting"])

    import reenroll_profiles

    rc = reenroll_profiles.main()
    assert rc == 0
    assert "db" in saved
    assert "essentials:isabel-piedmont-smith" in saved["db"].profiles


def test_reenroll_promotes_to_essentials_key(
    tagged_meeting_dir, fake_roster_cache, tmp_meetings_dir, monkeypatch
):
    """Re-enrollment of a body-tagged meeting promotes roster-matched speakers
    to essentials:<politician_slug> keys with identity fields populated."""
    mdir = tagged_meeting_dir("bloomington-common-council", "2026-test-promote", completed_stage=4)
    _write_transcript_named(mdir)
    (mdir / "audio.wav").touch()

    fake_roster_cache("bloomington-common-council")

    monkeypatch.setattr(
        "src.diarize.extract_speaker_embeddings",
        lambda wav, segs, token: {"SPEAKER_01": np.random.randn(256).astype(np.float32)},
    )
    monkeypatch.setattr("src.enroll.load_profiles", lambda: ProfileDB())

    saved = {}
    monkeypatch.setattr("src.enroll.save_profiles", lambda db: saved.update(db=db))
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(sys, "argv", ["reenroll_profiles.py", "2026-test-promote"])

    import reenroll_profiles

    rc = reenroll_profiles.main()
    assert rc == 0

    profile = saved["db"].profiles["essentials:isabel-piedmont-smith"]
    assert profile.politician_slug == "isabel-piedmont-smith"
    assert profile.politician_id == "uuid-ips"
    assert profile.display_name == "Councilmember Piedmont-Smith"


def test_reenroll_untagged_meeting_local_slug(
    tmp_meetings_dir, monkeypatch
):
    """Re-enrollment of an untagged legacy meeting enrolls under local slugs,
    not essentials keys."""
    # Create meeting dir WITHOUT body_slug in pipeline_state.json
    mdir = tmp_meetings_dir / "2026-test-untagged"
    mdir.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_stage": 4,
        "transcription_progress": 0,
        "total_segments": 0,
    }
    (mdir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    _write_transcript_named(mdir)
    (mdir / "audio.wav").touch()

    monkeypatch.setattr(
        "src.diarize.extract_speaker_embeddings",
        lambda wav, segs, token: {"SPEAKER_01": np.random.randn(256).astype(np.float32)},
    )
    monkeypatch.setattr("src.enroll.load_profiles", lambda: ProfileDB())

    saved = {}
    monkeypatch.setattr("src.enroll.save_profiles", lambda db: saved.update(db=db))
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(sys, "argv", ["reenroll_profiles.py", "2026-test-untagged"])

    import reenroll_profiles

    rc = reenroll_profiles.main()
    assert rc == 0
    assert "db" in saved

    keys = list(saved["db"].profiles.keys())
    # Should use local slug, not essentials: prefix
    assert not any(k.startswith("essentials:") for k in keys)
    # _name_to_slug("Councilmember Piedmont-Smith") strips "councilmember" -> "piedmont-smith"
    assert "piedmont-smith" in keys
