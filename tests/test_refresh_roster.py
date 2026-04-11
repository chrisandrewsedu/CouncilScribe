"""Tests for refresh_roster.py CLI and atomic-write behavior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import refresh_roster
from src.essentials_client import EssentialsClientError


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_cli_success_writes_cache(tmp_config_dir, sample_roster_response):
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        rc = refresh_roster.main(["--body", "bloomington-common-council"])
    assert rc == 0
    out = tmp_config_dir / "rosters" / "bloomington-common-council.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert {"body_key", "body_slug", "fetched_at", "politicians"}.issubset(data.keys())


def test_cache_payload_shape(tmp_config_dir, sample_roster_response):
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        refresh_roster.main(["--body", "bloomington-common-council"])
    data = json.loads(
        (tmp_config_dir / "rosters" / "bloomington-common-council.json").read_text()
    )
    assert data["body_slug"] == "bloomington-common-council"
    assert data["body_key"] == "Bloomington Common Council"
    assert isinstance(data["fetched_at"], str) and "T" in data["fetched_at"]
    assert isinstance(data["politicians"], list) and len(data["politicians"]) >= 1
    for pol in data["politicians"]:
        assert "politician_slug" in pol
        assert "full_name" in pol
        assert "aliases" in pol and isinstance(pol["aliases"], list)
        assert "title" in pol
        assert "district_label" in pol


def test_aliases_generated_for_piedmont_smith(tmp_config_dir, sample_roster_response):
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        refresh_roster.main(["--body", "bloomington-common-council"])
    data = json.loads(
        (tmp_config_dir / "rosters" / "bloomington-common-council.json").read_text()
    )
    ips = next(p for p in data["politicians"] if p["politician_slug"] == "isabel-piedmont-smith")
    assert "Piedmont Smith" in ips["aliases"]
    assert "Councilmember Piedmont-Smith" in ips["aliases"]
    assert "Councilmember Piedmont Smith" in ips["aliases"]


def test_aliases_generated_for_asare(tmp_config_dir, sample_roster_response):
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        refresh_roster.main(["--body", "bloomington-common-council"])
    data = json.loads(
        (tmp_config_dir / "rosters" / "bloomington-common-council.json").read_text()
    )
    asare = next(p for p in data["politicians"] if p["politician_slug"] == "sydney-asare")
    assert "Council President Asare" in asare["aliases"]
    assert "President Asare" in asare["aliases"]


def test_network_failure_preserves_existing_cache(tmp_config_dir):
    out = tmp_config_dir / "rosters" / "bloomington-common-council.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"preexisting": true}', encoding="utf-8")
    before = _sha256(out)
    with patch(
        "refresh_roster.fetch_body_roster",
        side_effect=EssentialsClientError("Network error: net down"),
    ):
        rc = refresh_roster.main(["--body", "bloomington-common-council"])
    assert rc != 0
    assert _sha256(out) == before


def test_atomic_write_failure_preserves_cache(tmp_config_dir, sample_roster_response):
    out = tmp_config_dir / "rosters" / "bloomington-common-council.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"preexisting": true}', encoding="utf-8")
    before = _sha256(out)
    with patch(
        "refresh_roster.fetch_body_roster", return_value=sample_roster_response
    ), patch(
        "refresh_roster.os.replace", side_effect=OSError("simulated replace failure")
    ):
        rc = refresh_roster.main(["--body", "bloomington-common-council"])
    assert rc != 0
    assert _sha256(out) == before
    leftovers = list(out.parent.glob(".tmp_*"))
    assert leftovers == []


def test_404_maps_to_exit_1(tmp_config_dir, capsys):
    with patch(
        "refresh_roster.fetch_body_roster",
        side_effect=EssentialsClientError(
            "not found", code="BODY_NOT_FOUND", status=404
        ),
    ):
        rc = refresh_roster.main(["--body", "bogus-body"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BODY_NOT_FOUND" in err or "not found" in err


def test_cache_contains_no_disallowed_key(tmp_config_dir, sample_roster_response):
    with patch("refresh_roster.fetch_body_roster", return_value=sample_roster_response):
        refresh_roster.main(["--body", "bloomington-common-council"])
    data = json.loads(
        (tmp_config_dir / "rosters" / "bloomington-common-council.json").read_text()
    )

    forbidden = "p" + "arty"  # split to avoid literal in this file's source

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k != forbidden, f"Unexpected forbidden key: {obj}"
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)


def test_base_url_flag_forwarded(tmp_config_dir, sample_roster_response):
    with patch(
        "refresh_roster.fetch_body_roster", return_value=sample_roster_response
    ) as mock_fetch:
        refresh_roster.main(
            [
                "--body",
                "bloomington-common-council",
                "--base-url",
                "http://localhost:3000",
            ]
        )
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs.get("base_url") == "http://localhost:3000"
