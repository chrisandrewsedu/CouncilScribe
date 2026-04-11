"""Tests for src.essentials_client.fetch_body_roster."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.essentials_client import EssentialsClientError, fetch_body_roster


def _mock_response(status=200, json_body=None, content_length=None):
    m = MagicMock()
    m.status_code = status
    m.headers = {}
    if content_length is not None:
        m.headers["Content-Length"] = str(content_length)
    m.json.return_value = json_body or {}
    return m


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_success(mock_get, sample_roster_response):
    mock_get.return_value = _mock_response(200, sample_roster_response)
    out = fetch_body_roster("bloomington-common-council")
    assert out == sample_roster_response
    called_url = mock_get.call_args[0][0]
    assert called_url == (
        "https://accounts.empowered.vote"
        "/api/essentials/bodies/bloomington-common-council/roster"
    )


@patch("src.essentials_client.requests.get")
def test_base_url_from_env(mock_get, sample_roster_response, monkeypatch):
    monkeypatch.setenv("EV_ACCOUNTS_URL", "http://localhost:3000")
    mock_get.return_value = _mock_response(200, sample_roster_response)
    fetch_body_roster("bloomington-common-council")
    assert mock_get.call_args[0][0].startswith("http://localhost:3000/")


@patch("src.essentials_client.requests.get")
def test_base_url_arg_overrides_env(mock_get, sample_roster_response, monkeypatch):
    monkeypatch.setenv("EV_ACCOUNTS_URL", "http://env.example")
    mock_get.return_value = _mock_response(200, sample_roster_response)
    fetch_body_roster("bloomington-common-council", base_url="http://arg.example")
    assert mock_get.call_args[0][0].startswith("http://arg.example/")


@patch("src.essentials_client.requests.get")
def test_base_url_trailing_slash_stripped(mock_get, sample_roster_response):
    mock_get.return_value = _mock_response(200, sample_roster_response)
    fetch_body_roster(
        "bloomington-common-council", base_url="http://localhost:3000/"
    )
    url = mock_get.call_args[0][0]
    assert "///" not in url
    assert url == (
        "http://localhost:3000"
        "/api/essentials/bodies/bloomington-common-council/roster"
    )


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_404(mock_get):
    mock_get.return_value = _mock_response(
        404, {"code": "BODY_NOT_FOUND", "message": "nope"}
    )
    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("unknown-body")
    assert exc_info.value.status == 404
    assert exc_info.value.code == "BODY_NOT_FOUND"


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_422(mock_get):
    mock_get.return_value = _mock_response(
        422, {"code": "VALIDATION_ERROR", "message": "bad"}
    )
    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("bad-slug-format-but-valid-regex")
    assert exc_info.value.status == 422
    assert exc_info.value.code == "VALIDATION_ERROR"


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_500(mock_get):
    mock_get.return_value = _mock_response(500, {})
    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("bloomington-common-council")
    assert exc_info.value.status == 500


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_network_error(mock_get):
    import requests as _r

    mock_get.side_effect = _r.exceptions.ConnectionError("net down")
    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("bloomington-common-council")
    assert "Network error" in str(exc_info.value)


@patch("src.essentials_client.requests.get")
def test_path_traversal_rejected(mock_get):
    with pytest.raises(EssentialsClientError):
        fetch_body_roster("../etc/passwd")
    mock_get.assert_not_called()


@patch("src.essentials_client.requests.get")
def test_slug_empty_rejected(mock_get):
    with pytest.raises(EssentialsClientError):
        fetch_body_roster("")
    mock_get.assert_not_called()


@patch("src.essentials_client.requests.get")
def test_response_size_cap(mock_get):
    mock_get.return_value = _mock_response(200, {}, content_length=99_999_999)
    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("bloomington-common-council")
    assert "too large" in str(exc_info.value)
    mock_get.return_value.json.assert_not_called()


@patch("src.essentials_client.requests.get")
def test_fetch_body_roster_non_json_200_wrapped(mock_get):
    """HTTP 200 with non-JSON body must raise EssentialsClientError, not
    let requests.exceptions.JSONDecodeError escape. Regression for
    Phase 108.1 / CSROSTER-03 / G2 gap from 108-HUMAN-UAT."""
    import requests as _r

    html_body = (
        "<!doctype html><html><head><title>Admin</title></head>"
        "<body><div id='root'></div></body></html>"
    )
    m = MagicMock()
    m.status_code = 200
    m.headers = {"Content-Type": "text/html; charset=utf-8"}
    m.text = html_body
    m.json.side_effect = _r.exceptions.JSONDecodeError(
        "Expecting value", "doc", 0
    )
    mock_get.return_value = m

    with pytest.raises(EssentialsClientError) as exc_info:
        fetch_body_roster("bloomington-common-council")

    msg = str(exc_info.value)
    assert "Non-JSON response" in msg
    assert "<!doctype html" in msg
