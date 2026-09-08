#!/usr/bin/env python
"""Scan for the INVERSE speaker-identity error: one label holding two people.

`review.duplicate_named_speakers` / `ambiguous_speaker_surnames` are name-based,
so a wrong review-time MERGE — one label, one name, two people — is invisible to
them and a clean scan is false reassurance. PR #162's cosine guard prevents new
mis-merges; this finds the ones that already happened.

Two stages. The cheap gate diffs transcript_raw.json (original diarized labels)
against transcript_named.json to find every merge that actually occurred: no
audio, no model, exact. The acoustic stage then re-embeds both sides of each
merge PER TURN from the audio and bands their cosine with review's calibrated
0.42 / 0.60 thresholds — per-turn because embeddings.json keeps one centroid per
label, which already averaged both voices together.

Reports only. A mismatch still splits into case (a) a split cluster and case (b)
two real people, and that needs a human reading the transcript.

Usage:
    .venv/bin/python mismerge_scan.py                 # whole corpus
    .venv/bin/python mismerge_scan.py <meeting_id>    # one meeting
    .venv/bin/python mismerge_scan.py --gate-only     # cheap stage, no audio
    .venv/bin/python mismerge_scan.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load HF token from .env.local (matches run_local.py / reverify_meeting.py).
_env_path = ROOT / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from src import config
from src.mismerge import (
    MAX_SLICE_SECONDS,
    MIN_SPLIT_SIDE_TURNS,
    assess_candidate,
    bimodal_split,
    label_source_turns,
    load_meeting_segments,
    merge_candidates,
    misattributed_floor_seconds,
    rank_assessments,
    scan_corpus_candidates,
    select_turns,
    slice_bounds,
    straddling_named_turns,
)
from src.review import merge_voice_verdict


def _make_embedder(wav_path: Path, hf_token: str, min_dur: float):
    """Return embed_fn(start, end) -> np.ndarray | None, seeking within the wav.

    Deliberately NOT reverify_meeting's load_wav: a 5-hour audio.wav is 572 MB
    on disk and ~1.2 GB decoded, and a corpus scan touches 172 of them. A
    seeking reader holds one slice at a time.
    """
    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model

    from src.diarize import _get_torch_device

    device = _get_torch_device()
    model = Model.from_pretrained(config.EMBEDDING_MODEL, token=hf_token)
    inference = Inference(model, window="whole", device=device)
    handle = sf.SoundFile(str(wav_path))
    sample_rate = handle.samplerate

    def embed_fn(start: float, end: float):
        if (end - start) < min_dur:
            return None
        handle.seek(int(start * sample_rate))
        block = handle.read(int((end - start) * sample_rate), dtype="float32")
        if block.ndim > 1:
            block = block.mean(axis=1)
        if len(block) < sample_rate * min_dur:
            return None
        waveform = torch.tensor(block).unsqueeze(0).to(device)
        try:
            return np.asarray(inference({"waveform": waveform, "sample_rate": sample_rate}))
        except Exception:
            # A bad slice must not abort a corpus scan; it reads as unmeasurable.
            return None

    return embed_fn, handle


def _clock(seconds: float) -> str:
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def _scan_all_labels(args) -> int:
    """Test every substantial label for two voices, merge or no merge.

    Provenance is blind to conflation that diarization itself created: raw and
    named agree, because the two people were one label before review ever
    started. This pass asks the question directly — split each label's turn
    vectors in two and see how far apart the halves are — at the cost of
    embedding audio for every label that could hold two people.

    Deliberately reported separately from the merge scan and NOT calibrated
    against a labelled reference: the 0.42/0.60 bands were measured on
    BETWEEN-label pairs, and a single speaker's own turns scatter more than two
    labels' centroids do, so a low similarity here is a lead, not a verdict.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set (check .env.local)", file=sys.stderr)
        return 1

    meetings = ([config.MEETINGS_DIR / args.meeting_id] if args.meeting_id else
                sorted(d for d in config.MEETINGS_DIR.iterdir() if d.is_dir()))
    rows: list[dict] = []
    unchecked: list[tuple[str, str]] = []
    started = time.time()
    for position, directory in enumerate(meetings, start=1):
        if not (directory / "transcript_named.json").exists():
            continue
        if not (directory / "transcript_raw.json").exists():
            unchecked.append((directory.name, "no transcript_raw.json"))
            continue
        if not (directory / "audio.wav").exists():
            unchecked.append((directory.name, "no audio.wav"))
            continue
        raw, named = load_meeting_segments(directory)
        groups = label_source_turns(raw, named)
        work = {}
        for label, indices in groups.items():
            chosen = select_turns(raw, indices, max_turns=args.max_turns,
                                  max_seconds=2 * args.max_seconds)
            slices = [slice_bounds(raw[i], max_slice_seconds=args.max_slice_seconds)
                      for i in chosen]
            seconds = sum(end - start for start, end in slices)
            if len(chosen) < 2 * MIN_SPLIT_SIDE_TURNS or seconds < 2 * args.min_side_seconds:
                continue
            work[label] = chosen
        if not work:
            continue
        embed_fn, handle = _make_embedder(directory / "audio.wav", hf_token, 0.5)
        try:
            for label, chosen in sorted(work.items()):
                vectors = [
                    embed_fn(*slice_bounds(
                        raw[i], max_slice_seconds=args.max_slice_seconds))
                    for i in chosen
                ]
                split = bimodal_split(vectors)
                if split is None:
                    continue
                rows.append({
                    "meeting_id": directory.name,
                    "label": label,
                    "similarity": split.similarity,
                    "verdict": merge_voice_verdict(split.similarity),
                    "turns_embedded": sum(v is not None for v in vectors),
                    "side_a_turns": len(split.side_a),
                    "side_b_turns": len(split.side_b),
                    "side_a_start": raw[chosen[min(split.side_a)]].start_time,
                    "side_b_start": raw[chosen[min(split.side_b)]].start_time,
                })
        finally:
            handle.close()
        print(f"  [{position}/{len(meetings)}] {directory.name}: {len(work)} label(s) "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    rows.sort(key=lambda r: r["similarity"])
    print(f"\nTested {len(rows)} label(s) in {time.time() - started:.0f}s.")
    if unchecked:
        print(f"\nNOT CHECKED ({len(unchecked)}):")
        for meeting, reason in unchecked:
            print(f"  {meeting}: {reason}")
    print("\nMost-separated labels first (low similarity = the label's own "
          "turns split into two unlike voices):")
    for row in rows[:40]:
        print(f"  cos {row['similarity']:+.3f}  {row['meeting_id']}  {row['label']}"
              f"  {row['side_a_turns']}v{row['side_b_turns']} turns"
              f"  first at {_clock(row['side_a_start'])} / {_clock(row['side_b_start'])}")
    bands: dict[str, int] = {}
    for row in rows:
        bands[row["verdict"]] = bands.get(row["verdict"], 0) + 1
    print("\nBands: " + ", ".join(f"{bands.get(v, 0)} {v}" for v in
                                  ("mismatch", "uncertain", "match")))
    print("These are LEADS, not verdicts: the 0.42/0.60 bands were calibrated on "
          "between-label pairs, and one speaker's own turns scatter more than "
          "two labels' centroids do. Cross-check a lead against the merge scan "
          "and the transcript.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("meeting_id", nargs="?", help="one meeting (default: all)")
    parser.add_argument("--gate-only", action="store_true",
                        help="cheap provenance stage only; no audio, no model")
    parser.add_argument("--min-side-seconds", type=float, default=3.0)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--max-slice-seconds", type=float, default=MAX_SLICE_SECONDS,
                        help="longest slice embedded from any one turn")
    parser.add_argument("--json", dest="json_out", help="write findings to this file")
    parser.add_argument("--all-labels", action="store_true",
                        help="also test EVERY substantial label for bimodality, "
                             "not just the ones a merge touched (covers "
                             "conflation that diarization itself created, which "
                             "provenance cannot see). ~47 min for the corpus.")
    args = parser.parse_args()

    if args.all_labels:
        return _scan_all_labels(args)

    if args.meeting_id:
        directory = config.MEETINGS_DIR / args.meeting_id
        if not (directory / "transcript_raw.json").exists():
            print(f"ERROR: {directory}/transcript_raw.json missing — provenance "
                  f"cannot check this meeting", file=sys.stderr)
            return 1
        raw, named = load_meeting_segments(directory)
        candidates = [(args.meeting_id, c) for c in merge_candidates(raw, named)]
        unchecked: list[tuple[str, str]] = []
    else:
        candidates, unchecked = scan_corpus_candidates(config.MEETINGS_DIR)

    audio_seconds = sum(c.host_seconds + c.absorbed_seconds for _, c in candidates)
    print(f"Cheap gate: {len(candidates)} merge(s) across "
          f"{len({m for m, _ in candidates})} meeting(s); "
          f"{audio_seconds / 60:.0f} min of speech is in scope before capping.")
    if unchecked:
        print(f"\nNOT CHECKED ({len(unchecked)}) — no provenance available:")
        for meeting, reason in unchecked:
            print(f"  {meeting}: {reason}")
    if not candidates:
        return 0

    if args.gate_only:
        print()
        for meeting, candidate in candidates:
            print(f"  {meeting}  {candidate.label}: "
                  f"host {candidate.host_raw} {candidate.host_seconds:.0f}s/"
                  f"{len(candidate.host_indices)}t  +  absorbed "
                  f"{candidate.absorbed_raw} {candidate.absorbed_seconds:.0f}s/"
                  f"{len(candidate.absorbed_indices)}t")
        print("\n(gate only — no voices compared; run without --gate-only to judge)")
        return 0

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set (check .env.local)", file=sys.stderr)
        return 1

    by_meeting: dict[str, list] = {}
    for meeting, candidate in candidates:
        by_meeting.setdefault(meeting, []).append(candidate)

    findings: list[tuple[str, object, int]] = []
    started = time.time()
    print()
    for position, (meeting, group) in enumerate(sorted(by_meeting.items()), start=1):
        directory = config.MEETINGS_DIR / meeting
        wav_path = directory / "audio.wav"
        if not wav_path.exists():
            print(f"  [{position}/{len(by_meeting)}] {meeting}: SKIPPED (no audio.wav)")
            unchecked.append((meeting, "no audio.wav"))
            continue
        raw, named = load_meeting_segments(directory)
        embed_fn, handle = _make_embedder(wav_path, hf_token, 0.5)
        try:
            for candidate in group:
                assessment = assess_candidate(
                    candidate, raw, embed_fn,
                    min_side_seconds=args.min_side_seconds,
                    max_turns=args.max_turns,
                    max_seconds=args.max_seconds,
                    max_slice_seconds=args.max_slice_seconds,
                )
                straddle = straddling_named_turns(candidate, raw, named)
                findings.append((meeting, assessment, straddle))
        finally:
            handle.close()
        print(f"  [{position}/{len(by_meeting)}] {meeting}: {len(group)} judged "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    print(f"\nEmbedded in {time.time() - started:.0f}s.\n")
    order = {id(a): i for i, a in enumerate(rank_assessments([a for _, a, _ in findings]))}
    findings.sort(key=lambda row: order[id(row[1])])

    name_of = {}
    for meeting in by_meeting:
        payload = json.loads(
            (config.MEETINGS_DIR / meeting / "transcript_named.json").read_text()
        )
        for label, speaker in (payload.get("speakers") or {}).items():
            name_of[(meeting, label)] = speaker.get("speaker_name")

    counts: dict[str, int] = {}
    for meeting, assessment, straddle in findings:
        counts[assessment.verdict] = counts.get(assessment.verdict, 0) + 1
        similarity = ("cos %+.3f" % assessment.similarity
                      if assessment.similarity is not None else "cos    n/a")
        print(f"  {assessment.verdict.upper():10} {similarity}  {meeting}")
        print(f"             {assessment.label} "
              f"({name_of.get((meeting, assessment.label)) or 'unnamed'!r})  "
              f"host {assessment.host_raw} {assessment.host_seconds:.0f}s  vs  "
              f"absorbed {assessment.absorbed_raw} {assessment.absorbed_seconds:.0f}s")
        detail = [
            f"≥{misattributed_floor_seconds(assessment):.0f}s under one name",
            f"judged on {assessment.host_turns_embedded} turns/"
            f"{assessment.host_seconds_embedded:.0f}s vs "
            f"{assessment.absorbed_turns_embedded} turns/"
            f"{assessment.absorbed_seconds_embedded:.0f}s",
        ]
        if straddle:
            detail.append(f"{straddle} named turn(s) span BOTH voices")
        if assessment.reason:
            detail.append(assessment.reason)
        print(f"             {'; '.join(detail)}")

    print("\nSummary: " + ", ".join(
        f"{counts.get(v, 0)} {v}" for v in ("mismatch", "uncertain", "unknown", "match")
    ))
    print("A mismatch is NOT a diagnosis: case (a) a split cluster (merge was "
          "right, labels differ acoustically) and case (b) two real people "
          "(merge corrupted the record) both look like this. Read the "
          "transcript for the two spans before changing anything.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([
            {"meeting_id": meeting, "straddling_named_turns": straddle,
             "speaker_name": name_of.get((meeting, assessment.label)),
             **asdict(assessment)}
            for meeting, assessment, straddle in findings
        ], indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
