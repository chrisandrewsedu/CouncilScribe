"""Shared pytest fixtures for Phase 108 roster client tests.

Provides a tmp_config_dir fixture that monkeypatches src.config.CONFIG_DIR
so tests never touch the real ~/CouncilScribe/config, plus sample politician
dicts mirroring the Phase 107 RosterResponse member shape and a loader for
the bloomington roster fixture JSON.
"""
from __future__ import annotations

import json
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
