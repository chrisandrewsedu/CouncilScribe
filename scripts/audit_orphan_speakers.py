#!/usr/bin/env python3
"""Audit prod for orphan meetings.speakers rows — READ ONLY, deletes nothing.

An orphan is a speaker row whose label no longer exists in the meeting's local
transcript_named.json. Publish upserts speakers by (meeting_id, label) and only
sweeps vanished labels as a side effect of a full republish, so nothing counts
these and a republish is currently the only thing that removes them.

Why it is not merely cosmetic: memo_reconcile.match_speaker resolves a clerk-memo
member's last name by suffix-matching display_name across the meeting's speaker
rows and skips that member's vote record on 2+ hits. A stale orphan is a second
hit that publish's duplicate-name gate cannot see — it reads the local
transcript's mappings, never the DB rows. See src/speaker_orphans.py.

Usage (standalone scripts do NOT auto-load .env.local):

    set -a; . ./.env.local; set +a
    .venv/bin/python scripts/audit_orphan_speakers.py
    .venv/bin/python scripts/audit_orphan_speakers.py --json > orphans.json
    .venv/bin/python scripts/audit_orphan_speakers.py --slug <slug> [--slug ...]

DATABASE_URL must use the IPv4 pooler host with the tenant-qualified username;
the direct host is IPv6-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402

from src import config  # noqa: E402
from src.speaker_orphans import (  # noqa: E402
    audit_meeting,
    audit_query,
    rows_by_meeting,
)


def _load_local_speakers(slug: str) -> dict | None:
    """{"speakers": ...} from the local transcript, or None when there is none.

    None means "cannot judge", and the caller must never turn it into orphans.
    A transcript that exists but cannot be parsed is also None: a corrupt
    artifact is no more evidence about prod than a missing one.
    """
    path = config.MEETINGS_DIR / slug / "transcript_named.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"speakers": (json.load(f) or {}).get("speakers") or {}}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _fetch(slugs: list[str] | None) -> list[tuple]:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(
            "DATABASE_URL is not set. Standalone scripts do not load .env.local:\n"
            "    set -a; . ./.env.local; set +a"
        )
    conn = psycopg2.connect(url)
    try:
        # Belt and braces: this audit must not be able to write, whatever it does.
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            if slugs:
                cur.execute(audit_query(by_slug=True), (slugs,))
            else:
                cur.execute(audit_query())
            return cur.fetchall()
    finally:
        conn.close()


def _report(audits) -> int:
    """Print the human report. Returns the number of meetings with orphans."""
    judgeable = [a for a in audits if a.judgeable]
    unjudged = [a for a in audits if not a.judgeable]
    dirty = [a for a in judgeable if a.orphans]
    risky = [a for a in dirty if a.surname_risks]
    stale = [a for a in dirty if a.orphans_serving_segments]

    print(f"Meetings in meetings.meetings: {len(audits)}")
    print(f"  judged against a local transcript: {len(judgeable)}")
    print(f"  not judgeable (no usable local transcript): {len(unjudged)}"
          f"  [{sum(1 for a in unjudged if a.at_stake)} with speaker rows at stake]")
    print(f"  with orphan speaker rows: {len(dirty)}")
    print(f"  with a memo-matching surname collision: {len(risky)}")
    print()

    if risky:
        print("=" * 72)
        print("AMBIGUOUS — a memo member's vote record can be silently skipped here")
        print("=" * 72)
        for a in risky:
            print(f"\n{a.slug}")
            for r in a.surname_risks:
                print(f"  last name {r.surname.title()!r} claimed by "
                      f"{len(r.labels)} rows: {', '.join(r.labels)}")
                print(f"    names:  {'; '.join(r.names)}")
                print(f"    stale:  {', '.join(r.orphan_labels)}")
        print()

    plain = [a for a in dirty if not a.surname_risks]
    if plain:
        print("=" * 72)
        print("ORPHANS, no surname collision — inflated speaker_count only")
        print("=" * 72)
        for a in plain:
            print(f"\n{a.slug}  (speaker_count {a.stored_speaker_count} -> "
                  f"{a.kept_label_count} after republish)")
            for o in a.orphans:
                extra = []
                if o.politician_slug:
                    extra.append(f"politician={o.politician_slug}")
                if o.local_slug:
                    extra.append(f"local={o.local_slug}")
                if o.segment_count:
                    extra.append(f"{o.segment_count} segments STILL REFERENCE IT")
                tail = f"  [{', '.join(extra)}]" if extra else ""
                print(f"  {o.label:<16} {o.display_name!r}{tail}")
        print()

    if stale:
        print("=" * 72)
        print("PROD IS SERVING SEGMENTS UNDER A LABEL THAT NO LONGER EXISTS")
        print("=" * 72)
        print("Published segments still point at these rows, which means no publish")
        print("has run since the label was merged away locally. The whole meeting is")
        print("out of date on the site, not just its speaker_count. A republish")
        print("rebuilds the segments and removes the row.")
        for a in stale:
            for o in a.orphans_serving_segments:
                print(f"  {a.slug}  {o.label}  {o.display_name!r}  "
                      f"{o.segment_count} segments")
        print()

    blind = [a for a in unjudged if a.at_stake]
    if blind:
        print("=" * 72)
        print("BLIND SPOTS — published speakers, no local transcript to judge them")
        print("=" * 72)
        print("Never counted as orphans: absent evidence is not evidence.")
        for a in blind:
            print(f"  {a.slug:<60} {a.reason} "
                  f"({a.db_row_count} speaker rows in prod)")
        print()

    quiet = [a for a in unjudged if not a.at_stake]
    if quiet:
        print(f"Also not judgeable, nothing at stake ({len(quiet)}): no speaker")
        print("rows in prod either, so there is nothing that could be stale.")
        for a in quiet:
            print(f"  {a.slug}")
        print()

    if dirty:
        print(f"Republishing these {len(dirty)} meeting(s) clears every orphan above:")
        for a in dirty:
            print(f"  {a.slug}")
    else:
        print("No orphan speaker rows found in any judgeable meeting.")
    return len(dirty)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", action="append", dest="slugs",
                    help="limit to this meeting slug (repeatable)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable findings instead of the report")
    args = ap.parse_args()

    grouped = rows_by_meeting(_fetch(args.slugs))
    audits = [
        audit_meeting(slug, rows, _load_local_speakers(slug),
                      stored_speaker_count=stored)
        for slug, (stored, rows) in sorted(grouped.items())
    ]

    if args.json:
        json.dump([asdict(a) | {"orphans_serving_segments":
                                [asdict(o) for o in a.orphans_serving_segments]}
                   for a in audits], sys.stdout, indent=2)
        print()
        return 0

    _report(audits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
