"""Tests for load_roster() slug path, legacy fallback, and staleness warning."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import refresh_roster
from src.roster import correct_speaker_name, load_roster


def _write_cache(
    tmp_config_dir, body_slug, sample_roster_response, *, fetched_at=None
):
    if fetched_at is not None:
        sample_roster_response = dict(sample_roster_response)
        sample_roster_response["fetched_at"] = fetched_at
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        refresh_roster.main(["--body", body_slug])
    return tmp_config_dir / "rosters" / f"{body_slug}.json"


def test_load_legacy_fallback(tmp_config_dir):
    legacy = tmp_config_dir / "council_roster.json"
    legacy.write_text(
        json.dumps(
            {
                "city": "Bloomington",
                "body": "Common Council",
                "members": [
                    {
                        "name": "Councilmember Foo",
                        "aliases": ["Foo", "Councilmember Foo"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    roster = load_roster()
    assert roster is not None
    assert roster.city == "Bloomington"
    assert len(roster.members) == 1
    assert roster.members[0].name == "Councilmember Foo"


def test_load_legacy_returns_none_when_missing(tmp_config_dir):
    assert load_roster() is None


def test_load_by_slug_success(tmp_config_dir, sample_roster_response):
    _write_cache(tmp_config_dir, "bloomington-common-council", sample_roster_response)
    roster = load_roster(body_slug="bloomington-common-council")
    assert roster is not None
    names = {m.name for m in roster.members}
    assert "Councilmember Piedmont-Smith" in names
    assert "Council President Asare" in names
    assert "City Clerk Bolden" in names


def test_load_by_slug_missing_returns_none(tmp_config_dir):
    assert load_roster(body_slug="does-not-exist") is None


def test_slug_roster_drives_correct_speaker_name(tmp_config_dir, sample_roster_response):
    _write_cache(tmp_config_dir, "bloomington-common-council", sample_roster_response)
    roster = load_roster(body_slug="bloomington-common-council")
    assert correct_speaker_name("Piedmont Smith", roster) == "Councilmember Piedmont-Smith"
    assert correct_speaker_name("President Asare", roster) == "Council President Asare"


def test_staleness_warning_fires_over_30_days(
    tmp_config_dir, sample_roster_response, caplog
):
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    _write_cache(
        tmp_config_dir,
        "bloomington-common-council",
        sample_roster_response,
        fetched_at=old,
    )
    with caplog.at_level(logging.WARNING):
        roster = load_roster(body_slug="bloomington-common-council")
    assert roster is not None
    assert any(
        "bloomington-common-council" in rec.message and "days old" in rec.message
        for rec in caplog.records
    )


def test_staleness_warning_not_fired_when_fresh(
    tmp_config_dir, sample_roster_response, caplog
):
    fresh = datetime.now(timezone.utc).isoformat()
    _write_cache(
        tmp_config_dir,
        "bloomington-common-council",
        sample_roster_response,
        fetched_at=fresh,
    )
    with caplog.at_level(logging.WARNING):
        load_roster(body_slug="bloomington-common-council")
    for rec in caplog.records:
        assert "days old" not in rec.message


def test_staleness_warning_not_fired_on_legacy_path(tmp_config_dir, caplog):
    legacy = tmp_config_dir / "council_roster.json"
    legacy.write_text(
        json.dumps(
            {
                "city": "x",
                "body": "y",
                "members": [{"name": "A", "aliases": []}],
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        load_roster()
    for rec in caplog.records:
        assert "days old" not in rec.message


def test_z_suffix_isoformat_parsed(tmp_config_dir, sample_roster_response):
    _write_cache(
        tmp_config_dir,
        "bloomington-common-council",
        sample_roster_response,
        fetched_at="2026-04-11T12:00:00Z",
    )
    roster = load_roster(body_slug="bloomington-common-council")
    assert roster is not None


def test_run_local_compat(tmp_config_dir):
    # No args — matches run_local.py call sites. Must not raise TypeError.
    result = load_roster()
    assert result is None  # no legacy file present in tmp
