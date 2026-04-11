"""HTTP client for the ev-accounts essentials bodies endpoint (Phase 107).

CouncilScribe-side HTTP wrapper. Never reads, logs, or persists any
affiliation fields — the upstream endpoint already strips them
(Phase 107 D-15), and the Phase 108 test suite enforces this with a
grep assertion.
"""
from __future__ import annotations

import os
import re

import requests

DEFAULT_BASE_URL = "https://accounts.empowered.vote"
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB cap — T-108-01 mitigation
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 15


class EssentialsClientError(Exception):
    """Raised for any failure in fetch_body_roster.

    Attributes:
        code: Upstream error envelope code (e.g. "BODY_NOT_FOUND").
        status: HTTP status code, or None for transport failures.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status = status


def _resolve_base_url(base_url: str | None) -> str:
    base = base_url or os.environ.get("EV_ACCOUNTS_URL") or DEFAULT_BASE_URL
    return base.rstrip("/")


def _validate_slug(body_slug: str) -> None:
    if not body_slug or not _SLUG_RE.match(body_slug):
        raise EssentialsClientError(
            f"Invalid body slug format: {body_slug!r} (must match ^[a-z0-9-]+$)",
            code="INVALID_SLUG",
            status=None,
        )


def fetch_body_roster(body_slug: str, base_url: str | None = None) -> dict:
    """Fetch the roster JSON for a body slug from ev-accounts.

    Raises EssentialsClientError on any non-200 status, transport failure,
    invalid slug, or oversized response.
    """
    _validate_slug(body_slug)
    base = _resolve_base_url(base_url)
    url = f"{base}/api/essentials/bodies/{body_slug}/roster"

    try:
        resp = requests.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    except requests.exceptions.RequestException as exc:
        raise EssentialsClientError(f"Network error: {exc}") from exc

    # T-108-01: reject oversized responses before parsing.
    content_length = resp.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_RESPONSE_BYTES:
                raise EssentialsClientError(
                    f"Response too large: {content_length} bytes "
                    f"(cap {_MAX_RESPONSE_BYTES})",
                    status=resp.status_code,
                )
        except ValueError:
            pass

    if resp.status_code == 404:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        raise EssentialsClientError(
            body.get("message", "Body not found"),
            code=body.get("code", "BODY_NOT_FOUND"),
            status=404,
        )
    if resp.status_code == 422:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        raise EssentialsClientError(
            body.get("message", "Validation error"),
            code=body.get("code", "VALIDATION_ERROR"),
            status=422,
        )
    if resp.status_code >= 500:
        raise EssentialsClientError(
            f"Server error ({resp.status_code})",
            status=resp.status_code,
        )
    if resp.status_code != 200:
        raise EssentialsClientError(
            f"Unexpected status {resp.status_code}",
            status=resp.status_code,
        )

    return resp.json()
