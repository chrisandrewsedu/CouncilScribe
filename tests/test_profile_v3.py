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
# Task 2: Enrollment keying (placeholders — will be un-skipped in Task 2)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Task 2")
def test_enroll_roster_member_uses_essentials_key():
    pass


@pytest.mark.skip(reason="Task 2")
def test_essentials_profile_identity_fields():
    pass


@pytest.mark.skip(reason="Task 2")
def test_non_roster_speaker_local_slug():
    pass


@pytest.mark.skip(reason="Task 2")
def test_mixed_profiles_coexist():
    pass


@pytest.mark.skip(reason="Task 2")
def test_essentials_profile_accumulates_across_meetings():
    pass


@pytest.mark.skip(reason="Task 2")
def test_enroll_confirmed_roster_member():
    pass
