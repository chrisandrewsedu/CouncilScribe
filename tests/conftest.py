"""Shared pytest fixtures for Phase 108 roster client tests and Phase 109 body-tagging tests.

Provides:
- tmp_config_dir: monkeypatches src.config.CONFIG_DIR to a tmp path + creates rosters/ subdir
- tmp_meetings_dir: monkeypatches src.config.MEETINGS_DIR to a tmp path
- fake_roster_cache: factory that writes a minimal valid {slug}.json to the rosters dir
- tagged_meeting_dir: factory that creates a meeting dir with pipeline_state.json pre-populated

Plus sample politician dicts mirroring the Phase 107 RosterResponse member shape and a loader
for the bloomington roster fixture JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path, monkeypatch):
    """Redirect src.config.CONFIG_DIR to a tmp directory.

    Also ensures `tmp_path/config/rosters` exists for tests that write caches.
    """
    cfg = tmp_path / "config"
    (cfg / "rosters").mkdir(parents=True, exist_ok=True)
    # roster.py imports `from . import config` so attribute access via
    # monkeypatching the CONFIG_DIR attribute on the module is sufficient.
    monkeypatch.setattr("src.config.CONFIG_DIR", cfg)
    yield cfg


@pytest.fixture
def tmp_meetings_dir(tmp_path, monkeypatch):
    """Redirect src.config.MEETINGS_DIR to a tmp directory."""
    meetings = tmp_path / "meetings"
    meetings.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.config.MEETINGS_DIR", meetings)
    yield meetings


@pytest.fixture
def fake_roster_cache(tmp_config_dir):
    """Factory: write a minimal valid {slug}.json to tmp_config_dir/rosters/{slug}.json.

    Usage::

        def test_something(fake_roster_cache):
            path = fake_roster_cache("bloomington-common-council")
            # optionally with custom fetched_at for staleness tests:
            path = fake_roster_cache("bloomington-common-council", fetched_at="2020-01-01T00:00:00Z")

    Returns the Path to the written cache file.
    """

    def _write(slug: str, *, fetched_at: str | None = None) -> Path:
        if fetched_at is None:
            fetched_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "body_slug": slug,
            "body_key": slug,
            "fetched_at": fetched_at,
            "politicians": [
                {
                    "politician_slug": "isabel-piedmont-smith",
                    "politician_id": "uuid-ips",
                    "full_name": "Isabel Piedmont-Smith",
                    "preferred_name": "Isabel",
                    "title": "Councilmember",
                    "chamber_name": "Bloomington Common Council",
                    "district_label": "District 5",
                    "photo_url": None,
                }
            ],
        }
        cache_path = tmp_config_dir / "rosters" / f"{slug}.json"
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cache_path

    return _write


@pytest.fixture
def tagged_meeting_dir(tmp_meetings_dir):
    """Factory: create a meeting dir with pipeline_state.json pre-populated with body_slug + completed_stage.

    Usage::

        def test_something(tagged_meeting_dir):
            mdir = tagged_meeting_dir("bloomington-common-council", completed_stage=3)

    Returns the Path to the meeting directory.
    """

    def _create(
        slug: str,
        meeting_id: str = "2026-02-04-regular-session",
        *,
        completed_stage: int = 3,
    ) -> Path:
        mdir = tmp_meetings_dir / meeting_id
        mdir.mkdir(parents=True, exist_ok=True)
        state = {
            "completed_stage": completed_stage,
            "transcription_progress": 0,
            "total_segments": 0,
            "body_slug": slug,
        }
        (mdir / "pipeline_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        return mdir

    return _create


@pytest.fixture
def sample_politician_piedmont_smith():
    return {
        "politician_slug": "isabel-piedmont-smith",
        "politician_id": "uuid-ips",
        "full_name": "Isabel Piedmont-Smith",
        "preferred_name": "Isabel",
        "title": "Councilmember",
        "chamber_name": "Bloomington Common Council",
        "district_label": "District 5",
        "photo_url": None,
    }


@pytest.fixture
def sample_politician_asare():
    return {
        "politician_slug": "sydney-asare",
        "politician_id": "uuid-a",
        "full_name": "Sydney Asare",
        "preferred_name": "Sydney",
        "title": "Council President",
        "chamber_name": "Bloomington Common Council",
        "district_label": "At-Large",
        "photo_url": None,
    }


@pytest.fixture
def sample_politician_garcia():
    return {
        "politician_slug": "maria-garcia",
        "politician_id": "uuid-g",
        "full_name": "María García",
        "preferred_name": "María",
        "title": "Councilmember",
        "chamber_name": "Bloomington Common Council",
        "district_label": "District 2",
        "photo_url": None,
    }


@pytest.fixture
def sample_politician_bolden():
    return {
        "politician_slug": "nicole-bolden",
        "politician_id": "uuid-b",
        "full_name": "Nicole Bolden",
        "preferred_name": "Nicole",
        "title": "City Clerk",
        "chamber_name": "Bloomington Common Council",
        "district_label": "",
        "photo_url": None,
    }


@pytest.fixture
def sample_roster_response():
    """Load the Bloomington roster response fixture."""
    path = Path(__file__).parent / "fixtures" / "bloomington_roster_response.json"
    return json.loads(path.read_text(encoding="utf-8"))
