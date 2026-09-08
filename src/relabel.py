"""Per-segment relabelling: hand one span back to its real owner.

`review.merge_speakers` is all-or-nothing to ONE target. That is enough to fold
a split cluster back together, but it cannot do either of the repairs a
mis-merge needs: take a span OUT of a label that wrongly holds it, or cut a
named segment whose own text spans two voices. Both were done by hand-written
one-off scripts before this module existed (see the Tanner Branham triage),
which is why there was no tested path for it.

What `mismerge_scan.py` finds needs exactly these two primitives:

* 2026-07-08 SPEAKER_05 "Haley Stevens" holds 262s of a television presenter
  who names himself on air. The presenter's turns must move OUT to a label of
  his own — 65 spans interleaved with her real answers.
* 2026-07-09 publishes ONE 99s segment holding an audience question AND the
  candidate's entire answer. No label operation can touch that: the segment
  itself has to be cut first.

Pure: dataclasses only. Loading, merging, reindexing and writing live in the
caller (`relabel_meeting.py`), so every decision rule here is unit-testable.
Nothing in this module writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def split_segment(segment, at_time: float):
    """Cut one segment in two at ``at_time``. Returns (before, after).

    Words are divided by their own start times, so the split is what the ASR
    actually heard rather than a guess. Both halves keep the original speaker;
    the caller relabels whichever half it means to move.

    Refuses rather than guesses in three cases: a cut outside the segment, a
    segment with no word timings (there is no honest place to divide the text,
    and inventing one fabricates who said what — the very failure being
    repaired), and a cut with every word on one side, which means the caller's
    span is wrong and should be reported instead of silently doing nothing.
    """
    if not (segment.start_time < at_time < segment.end_time):
        raise ValueError(
            f"cut at {at_time} is outside segment "
            f"{segment.start_time}-{segment.end_time}"
        )
    if not segment.words:
        raise ValueError(
            f"segment {segment.segment_id} has no word timings; it cannot be cut "
            f"faithfully"
        )
    before_words = [w for w in segment.words if w.start < at_time]
    after_words = [w for w in segment.words if w.start >= at_time]
    if not before_words or not after_words:
        raise ValueError(
            f"cut at {at_time} leaves no words on one side of segment "
            f"{segment.segment_id}"
        )
    return (
        replace(
            segment,
            end_time=at_time,
            words=before_words,
            text=" ".join(w.word for w in before_words),
        ),
        replace(
            segment,
            start_time=at_time,
            words=after_words,
            text=" ".join(w.word for w in after_words),
        ),
    )


#: A residual shorter than this is absorbed rather than cut off. Spans come
#: from RAW turn boundaries, which sit near but not exactly on the named ones
#: (boundary-snap moves words across them), so a fraction of a second of
#: mismatch is expected and is not a piece of another person.
MIN_PIECE_SECONDS = 0.25

#: Spans closer together than this are one run. mismerge_scan reports one span
#: per raw turn, so a presenter's uninterrupted stretch arrives as dozens of
#: near-contiguous spans; planning a cut per span would shred the segment.
SPAN_JOIN_GAP = 2.0


@dataclass
class Move:
    """One span of a segment moving from one label to another.

    ``index`` points into the segment list AS PASSED, so a plan can be printed
    against the loaded meeting before anything is applied. ``whole`` is True
    when the entire segment moves and no cut is needed.
    """

    index: int
    from_label: str
    to_label: str
    start: float
    end: float
    whole: bool
    seconds: float
    text: str  #: only the words that MOVE, so a diff cannot mislead


@dataclass
class RelabelPlan:
    to_label: str
    moves: list[Move]

    @property
    def seconds(self) -> float:
        return sum(m.seconds for m in self.moves)

    @property
    def cuts(self) -> int:
        return sum(1 for m in self.moves if not m.whole)


def join_spans(spans, gap: float = SPAN_JOIN_GAP) -> list[tuple[float, float]]:
    """Collapse near-contiguous spans into runs, in time order."""
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    joined: list[list[float]] = []
    for start, end in ordered:
        if joined and start - joined[-1][1] <= gap:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(a, b) for a, b in joined]


def plan_relabel(
    segments,
    spans,
    to_label: str,
    *,
    min_piece_seconds: float = MIN_PIECE_SECONDS,
    span_join_gap: float = SPAN_JOIN_GAP,
) -> RelabelPlan:
    """Work out what moving ``spans`` onto ``to_label`` would do. No mutation.

    A segment wholly inside a span moves as it is. A segment a span covers only
    partly is cut at the span's boundary and only the covered piece moves. A
    segment already on ``to_label`` is skipped, so re-running a repair is a
    no-op rather than a second round of cuts.

    Wordless zero-length segments are skipped too. They are publish-era
    artefacts sitting exactly on a boundary, and moving them both churns the
    transcript and drags unrelated labels into a two-label repair: the
    2026-07-08 dry run pulled 0.0s stubs off SPEAKER_02 and SPEAKER_12 while
    splitting SPEAKER_05. A SHORT segment that has words is a real turn and is
    kept.
    """
    runs = join_spans(spans, span_join_gap)
    moves: list[Move] = []
    for index, segment in enumerate(segments):
        if segment.speaker_label == to_label:
            continue
        if not segment.words and not (segment.text or "").strip():
            continue
        # Every run that overlaps this segment, not just the first: one named
        # segment can hold several stretches of the other voice. On 2026-07-14
        # a single 69s segment holds reporter narration, then the candidate,
        # then narration, then the candidate again.
        touching = [
            (max(segment.start_time, a), min(segment.end_time, b))
            for a, b in runs
            if min(segment.end_time, b) - max(segment.start_time, a) > 0
        ]
        for position, (start, end) in enumerate(touching):
            first, last = position == 0, position == len(touching) - 1
            # Edge absorption is for raw-vs-named boundary mismatch, so it
            # applies only at the segment's real edges. Extending an INNER run
            # would swallow the other speaker either side of it.
            if first and start - segment.start_time < min_piece_seconds:
                start = segment.start_time
            if last and segment.end_time - end < min_piece_seconds:
                end = segment.end_time
            if segment.words:
                if first and not any(w.start < start for w in segment.words):
                    start = segment.start_time
                if last and not any(w.start >= end for w in segment.words):
                    end = segment.end_time
                if not any(start <= w.start < end for w in segment.words):
                    continue  # the run lands in a silent gap
            whole = (len(touching) == 1
                     and start <= segment.start_time and end >= segment.end_time)
            moved_words = ([w.word for w in segment.words] if whole else
                           [w.word for w in segment.words if start <= w.start < end])
            moves.append(Move(
                index=index,
                from_label=segment.speaker_label,
                to_label=to_label,
                start=start,
                end=end,
                whole=whole,
                seconds=end - start,
                text=(" ".join(moved_words) if segment.words
                      else (segment.text or "")).strip(),
            ))
    return RelabelPlan(to_label=to_label, moves=moves)


def _cut_into_pieces(segment, moves: list[Move]) -> list:
    """Cut one segment at every move boundary and label the pieces.

    A cut point with no words on one side is skipped rather than attempted:
    such a piece is not a piece, and split_segment refuses it by design. Each
    surviving piece is assigned by where its FIRST WORD falls, which is the
    same evidence the cut itself used.
    """
    points = sorted({
        point for move in moves for point in (move.start, move.end)
        if segment.start_time < point < segment.end_time
    })
    pieces: list = []
    current = segment
    for point in points:
        if not any(w.start < point for w in current.words):
            continue
        if not any(w.start >= point for w in current.words):
            continue
        head, current = split_segment(current, point)
        pieces.append(head)
    pieces.append(current)

    out: list = []
    for piece in pieces:
        key = (piece.words[0].start if piece.words
               else (piece.start_time + piece.end_time) / 2)
        owner = next((m for m in moves if m.start <= key < m.end), None)
        out.append(replace(piece, speaker_label=owner.to_label) if owner else piece)
    return out


def apply_plan(segments, plan: RelabelPlan) -> list:
    """Return a NEW segment list with the plan's moves applied.

    The input list and its Segment objects are left untouched: a dry run has to
    be able to plan and apply on the loaded meeting without disturbing it.
    merge_adjacent_segments' habit of mutating what it is handed has already
    made valid summaries look stale once, so this module does not repeat it.

    Segment ids are renumbered contiguously, because a cut adds a row and the
    summary's stored section boundaries index into this list — the caller must
    reindex them afterwards (backfill_segment_merge.remerge_meeting does both).
    """
    by_index: dict[int, list[Move]] = {}
    for move in plan.moves:
        by_index.setdefault(move.index, []).append(move)
    out: list = []
    for index, segment in enumerate(segments):
        moves = sorted(by_index.get(index, ()), key=lambda m: m.start)
        if not moves:
            out.append(replace(segment, words=list(segment.words)))
            continue
        if len(moves) == 1 and moves[0].whole:
            out.append(replace(
                segment, speaker_label=moves[0].to_label, words=list(segment.words)
            ))
            continue
        out.extend(_cut_into_pieces(segment, moves))
    for position, segment in enumerate(out):
        segment.segment_id = position
    return out


def spans_for_raw_label(raw_segments, named_segments, raw_label: str) -> list[tuple[float, float]]:
    """Time spans that raw ``raw_label`` occupies inside the named transcript.

    This is the bridge to ``mismerge_scan.py``: a finding names a raw label
    whose turns ended up under someone else's name, and this returns exactly
    those turns' spans. Deriving them here — rather than retyping timestamps —
    keeps the detector and the repair talking about the same audio.

    Only turns the named transcript actually carries are returned, via
    ``mismerge.provenance_groups``, so a raw turn publish dropped is not
    proposed for relabelling.
    """
    from .mismerge import provenance_groups

    spans: list[tuple[float, float]] = []
    for by_raw in provenance_groups(raw_segments, named_segments).values():
        for index in by_raw.get(raw_label, []):
            segment = raw_segments[index]
            spans.append((segment.start_time, segment.end_time))
    return join_spans(spans)


def next_free_label(segments, speakers, *, reserved=None) -> str:
    """The lowest unused ``SPEAKER_NN``, counting labels with no segments left.

    A speakers entry with no segments is still a claimed label — reusing its
    number would attach the split-out person to a stranger's mapping, which is
    the identity collision this repo already guards against elsewhere.

    ``reserved`` must carry the labels ``transcript_raw.json`` uses. Provenance
    depends on a raw and a named ``SPEAKER_NN`` meaning the same person, so
    minting named SPEAKER_00 for a newly split-out voice while raw SPEAKER_00
    was somebody else would make ``mismerge_scan`` attribute this voice to
    them — quietly breaking the detector that found the bug.
    """
    used = {s.speaker_label for s in segments} | set(speakers or {}) | set(reserved or ())
    number = 0
    while f"SPEAKER_{number:02d}" in used:
        number += 1
    return f"SPEAKER_{number:02d}"
