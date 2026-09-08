#!/usr/bin/env python
"""Score a candidate forum diarization against the moderator's own handoffs.

The label under repair swallowed three people, so the reviewed transcript
inherits the error and cannot referee its own fix. `bench.forum_anchor_reference`
builds an independent reference from the moderator's named handoffs, and
`bench.identity_score.identity_report` turns that into fragmentation and
conflation counts.

Reports at two floors, always both: `COMPARABLE_MIN_FRACTION` for continuity
with the repo's other diarization measurements, and `GATE_MIN_FRACTION` for this
repair's verdict.

Loads no env: it touches no database and no service. Usage:
  .venv/bin/python scripts/score_forum_diarization.py \
      ~/CouncilScribe/meetings/<id>/transcript_raw.json --label incumbent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.forum_anchor_reference import (  # noqa: E402
    LWV_AUDITOR_FORUM_END,
    LWV_AUDITOR_SPEAKERS,
    anchor_reference_windows,
)
from bench.forum_gate import (  # noqa: E402
    COMPARABLE_MIN_FRACTION,
    GATE_MAX_UNATTRIBUTED_SHARE,
    GATE_MIN_FRACTION,
    gate_verdict,
    load_turns,
    reference_half,
    unattributed_bucket_share,
)
from bench.forum_recluster import UNCLUSTERED_LABEL  # noqa: E402
from bench.identity_score import identity_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turns_json", type=Path,
                        help="JSON list of segments with start_time/end_time/speaker_label")
    parser.add_argument("--raw-json", type=Path, default=None,
                        help="Segments the reference is built from. Defaults to "
                             "turns_json; pass the ORIGINAL transcript_raw.json "
                             "when scoring a turn set that carries no text.")
    parser.add_argument("--label", default="candidate", help="Name for this run in the output")
    parser.add_argument("--forum-end", type=float, default=LWV_AUDITOR_FORUM_END)
    parser.add_argument("--half", choices=("all", "tune", "holdout"), default="all",
                        help="Which anchor windows to score against. Use 'holdout' "
                             "to score a clustering whose threshold was tuned on "
                             "'tune', so the reported number is not the tuned one.")
    args = parser.parse_args(argv)

    reference_source = json.loads((args.raw_json or args.turns_json).read_text())
    windows = anchor_reference_windows(
        reference_source, LWV_AUDITOR_SPEAKERS, end_time=args.forum_end
    )
    reference = reference_half(windows, args.half)
    if not reference:
        print("! reference is empty — no handoffs matched. Refusing to score.")
        return 2

    hypothesis = load_turns(args.turns_json)
    covered = sum(end - start for start, end, _ in reference)
    print(f"== {args.label} ==")
    print(f"reference half: {args.half} ({len(windows)} anchor windows total)")
    print(f"reference: {len(reference)} turns, {covered:.0f}s, "
          f"{len({p for _, _, p in reference})} people")
    print(f"hypothesis: {len(hypothesis)} turns, "
          f"{len({l for _, _, l in hypothesis})} labels")

    for floor in (COMPARABLE_MIN_FRACTION, GATE_MIN_FRACTION):
        report = identity_report(hypothesis, reference, min_fraction=floor)
        bucket_bound = GATE_MAX_UNATTRIBUTED_SHARE(report.reference_people)
        passed, reasons = gate_verdict(
            report, unattributed_label=UNCLUSTERED_LABEL,
            max_unattributed_share=bucket_bound,
        )
        tag = "GATE" if floor == GATE_MIN_FRACTION else "comparable"
        print(f"\n-- min_fraction {floor:.2f} ({tag}) --")
        print(f"  conflation:    {report.conflation_summary}")
        print(f"  fragmentation: {report.fragmentation_summary}")
        if report.unmapped_labels:
            print(f"  unmapped labels (reference gap, not an error): "
                  f"{', '.join(report.unmapped_labels)}")
        if floor == GATE_MIN_FRACTION:
            bucket_seconds, bucket_share = unattributed_bucket_share(
                report, UNCLUSTERED_LABEL
            )
            print(f"  unattributed bucket: {bucket_seconds:.1f}s "
                  f"({bucket_share:.1%} of scored reference speech, "
                  f"bound {bucket_bound:.1%})")
            print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
            for reason in reasons:
                print(f"    - {reason}")
            return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
