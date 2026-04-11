"""CLI to fetch and cache a governing body's roster from ev-accounts.

Usage:
    python refresh_roster.py --body bloomington-common-council
    python refresh_roster.py --body <slug> --base-url http://localhost:3000

Writes to ~/CouncilScribe/config/rosters/{body_slug}.json using an
atomic tempfile + os.replace pattern so a crashed/partial write never
corrupts an existing cached roster (CSROSTER-03, T-108-03).

This module never reads, logs, or persists affiliation fields. The
Phase 108 test suite enforces this with a grep assertion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.alias_gen import generate_aliases
from src.essentials_client import EssentialsClientError, fetch_body_roster


def _build_cache_payload(body_slug: str, response: dict) -> dict:
    """Transform an API RosterResponse into the on-disk cache payload.

    Persisted keys per CSROSTER-02: body_key, body_slug, fetched_at,
    politicians[] with politician_slug, full_name, aliases[], title,
    district_label. politician_id and preferred_name are also persisted
    for Phase 110/111 use.
    """
    politicians_out: list[dict] = []
    for member in response.get("members", []):
        politician = {
            "politician_slug": member.get("politician_slug"),
            "politician_id": member.get("politician_id"),
            "full_name": member.get("full_name", ""),
            "preferred_name": member.get("preferred_name", ""),
            "title": member.get("title", ""),
            "district_label": member.get("district_label", ""),
            "aliases": generate_aliases(
                {
                    "full_name": member.get("full_name", ""),
                    "preferred_name": member.get("preferred_name", ""),
                    "title": member.get("title", ""),
                }
            ),
        }
        politicians_out.append(politician)

    return {
        "body_key": response.get("name_formal", ""),
        "body_slug": body_slug,
        "fetched_at": response.get("fetched_at")
        or datetime.now(timezone.utc).isoformat(),
        "politicians": politicians_out,
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomically write JSON to `path` via tempfile + os.replace.

    Uses dir=path.parent to guarantee same-filesystem placement so
    os.replace is truly atomic on macOS/Linux. On any exception after
    tempfile creation, the tempfile is unlinked and the original
    `path` is never modified (T-108-03).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _cache_path_for(body_slug: str) -> Path:
    return config.CONFIG_DIR / "rosters" / f"{body_slug}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="refresh_roster.py",
        description="Fetch and cache a governing body's roster from ev-accounts.",
    )
    parser.add_argument(
        "--body",
        required=True,
        help="Body slug (e.g. bloomington-common-council)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Override API base URL "
            "(default: $EV_ACCOUNTS_URL or https://accounts.empowered.vote)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Override output path "
            "(default: ~/CouncilScribe/config/rosters/{body}.json)"
        ),
    )
    args = parser.parse_args(argv)

    body_slug = args.body
    output_path = (
        Path(args.output) if args.output else _cache_path_for(body_slug)
    )

    print(f"Fetching roster for {body_slug}...", flush=True)

    try:
        response = fetch_body_roster(body_slug, base_url=args.base_url)
    except EssentialsClientError as exc:
        code = exc.code or "ERROR"
        print(f"ERROR ({code}): {exc}", file=sys.stderr)
        return 1

    payload = _build_cache_payload(body_slug, response)
    n_members = len(payload["politicians"])
    if n_members == 0:
        print(f"Warning: {body_slug} has 0 active members", file=sys.stdout)

    try:
        _write_json_atomic(output_path, payload)
    except OSError as exc:
        print(f"ERROR: failed to write cache: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path} ({n_members} members)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
