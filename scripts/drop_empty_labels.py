#!/usr/bin/env python
"""Remove speaker labels that no published segment carries.

The blind spot the orphan audit cannot see. `speaker_orphans.keep_labels` reads
the local speakers dict, so a label that is PRESENT but whose every segment
publish drops (`if not seg.text: continue`) is not stale and the orphan audit
rightly passes it. Publish still writes the speaker ROW, so the live site gets
a row that serves nothing and a `speaker_count` that overstates the
participants. Measured on prod: two press conferences each reported 2 speakers
where there was 1, while the orphan audit reported zero problems.

DRY RUN BY DEFAULT.

Usage:
    .venv/bin/python scripts/drop_empty_labels.py
    .venv/bin/python scripts/drop_empty_labels.py --apply
    .venv/bin/python scripts/drop_empty_labels.py <meeting_id> [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.atomic_io import atomic_write_json
from src.speaker_orphans import drop_empty_labels, empty_published_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("meeting_id", nargs="?", help="one meeting (default: all)")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite transcript_named.json (default: dry run)")
    args = parser.parse_args()

    directories = (
        [config.MEETINGS_DIR / args.meeting_id] if args.meeting_id
        else sorted(d for d in config.MEETINGS_DIR.iterdir() if d.is_dir())
    )

    touched: list[str] = []
    scanned = 0
    for directory in directories:
        path = directory / "transcript_named.json"
        if not path.exists():
            continue
        scanned += 1
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            print(f"  UNREADABLE  {directory.name}: {error}")
            continue
        labels = empty_published_labels(data)
        if not labels:
            continue
        touched.append(directory.name)
        segments = {
            label: [s for s in data.get("segments") or []
                    if s.get("speaker_label") == label]
            for label in labels
        }
        print(f"  {directory.name}")
        for label in labels:
            mapping = (data.get("speakers") or {}).get(label) or {}
            rows = segments[label]
            span = (f"{min(s['start_time'] for s in rows):.1f}-"
                    f"{max(s['end_time'] for s in rows):.1f}s" if rows else "no segments")
            print(f"    {label}  name={mapping.get('speaker_name')!r}  "
                  f"{len(rows)} empty segment(s)  {span}")
        if args.apply:
            new_data, dropped = drop_empty_labels(data)
            backup = path.with_suffix(".json.preempty.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            atomic_write_json(path, new_data)
            print(f"    -> dropped {', '.join(dropped)} (backup: {backup.name})")

    print(f"\nScanned {scanned} transcript(s); {len(touched)} with empty labels.")
    if touched and not args.apply:
        print("(dry run — nothing written; pass --apply to write)")
    if touched and args.apply:
        print("REPUBLISH any of these that are live: publish removes the stale "
              "speaker row only as a side effect of a full republish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
