#!/usr/bin/env python
"""CLI for re-clustering an existing diarization over per-turn embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

The clustering itself is pure and lives in `bench.forum_recluster`; this file
holds the one Modal call and the command-line surface.

Loads no env: Modal authenticates from ~/.modal.toml, and the worker gets
HF_TOKEN from a Modal secret. No DB writes, no LLM. Usage:
  .venv/bin/python scripts/recluster_forum_turns.py <meeting-id> --out turns.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.forum_recluster import (  # noqa: E402
    as_unique_label_segments,
    turn_label,
)


def fetch_turn_embeddings(
    meeting_id: str, segments: list[dict]
) -> dict[int, list[float]]:
    """One wespeaker vector per turn, from an L4 on Modal.

    Turns shorter than the worker's 0.3s floor, and any whose embedding is
    non-finite, come back absent rather than zeroed — the caller decides what
    to do with a turn that has no voice evidence.
    """
    from src.modal_compute import _modal_app

    app = _modal_app()
    payload = json.dumps(as_unique_label_segments(segments))
    with app.app.run():
        raw = app.pipeline_extract_embeddings.remote(meeting_id, payload)
    by_label = json.loads(raw)
    return {
        index: by_label[turn_label(index)]
        for index in range(len(segments))
        if turn_label(index) in by_label
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from bench.forum_anchor_reference import (
        LWV_AUDITOR_FORUM_END,
        LWV_AUDITOR_SPEAKERS,
        anchor_reference_windows,
    )
    from bench.forum_gate import reference_half
    from bench.forum_recluster import (
        calibrate,
        cluster_turns,
        fold_slivers,
        relabel_segments,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Cached turn-embeddings JSON. Fetched from Modal if absent.")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--sliver-floor", type=float, default=20.0,
                        help="Fold labels holding less than this many seconds of "
                             "speech into the shared unclustered bucket.")
    args = parser.parse_args(argv)

    from src import config

    meeting_dir = config.MEETINGS_DIR / args.meeting_id
    segments = json.loads((meeting_dir / "transcript_raw.json").read_text())

    if args.embeddings and args.embeddings.exists():
        vectors = {int(k): v for k, v in json.loads(args.embeddings.read_text()).items()}
    else:
        vectors = fetch_turn_embeddings(args.meeting_id, segments)
        if args.embeddings:
            args.embeddings.write_text(json.dumps(vectors))
    print(f"{len(vectors)} of {len(segments)} turns embedded")

    windows = anchor_reference_windows(
        segments, LWV_AUDITOR_SPEAKERS, end_time=LWV_AUDITOR_FORUM_END
    )
    tune = reference_half(windows, "tune")
    print(f"{len(windows)} anchor windows; calibrating on the {len(tune)}-turn "
          f"tuning half, scoring later on the holdout half")

    best, grid = calibrate(
        segments, vectors, tune, args.thresholds, sliver_floor=args.sliver_floor
    )
    print(f"\nthreshold  labels  conflated  fragmented   (tuning half only, "
          f"{args.sliver_floor:.0f}s sliver floor)")
    for row in grid:
        mark = " <-- chosen" if row["threshold"] == best else ""
        print(f"  {row['threshold']:.2f}      {row['labels']:3d}       "
              f"{row['conflated']:2d}         {row['fragmented']:2d}{mark}")

    labels = cluster_turns(vectors, len(segments), best)
    labels = fold_slivers(labels, segments, args.sliver_floor)
    args.out.write_text(json.dumps(relabel_segments(segments, labels)))
    print(f"\nwrote {args.out} at threshold {best:.2f} "
          f"({len(set(labels))} labels, {args.sliver_floor:.0f}s sliver floor)")
    print("Now score it on the held-out half:")
    print(f"  .venv/bin/python scripts/score_forum_diarization.py {args.out} \\")
    print(f"      --raw-json {meeting_dir / 'transcript_raw.json'} \\")
    print("      --half holdout --label 'experiment B'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
