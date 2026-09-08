"""Detect the INVERSE speaker-identity error: one label holding two people.

``review.duplicate_named_speakers`` and ``review.ambiguous_speaker_surnames``
are both name-based: they catch a rename that puts one name onto two diarized
labels. The mirror-image mistake — a review-time MERGE that folded two
different people into one label — leaves ONE label carrying ONE name, so both
detectors return ``{}`` and a clean scan is false reassurance. PR #162's
merge-time cosine guard prevents new instances; this module finds the ones that
already happened.

The signal is provenance, not names. ``transcript_raw.json`` keeps the original
diarized labels, so a raw label whose turns now sit inside a DIFFERENT named
label is a merge, exactly. That is a cheap, exhaustive gate: measured over the
corpus it selects 36 merges in 26 of 172 meetings, which is ~10 minutes of
embedding instead of the ~4 hours a blind re-embed of all 128 hours would cost.
Whether a merge was RIGHT is then decided acoustically, per turn, because
``embeddings.json`` stores one centroid per label — the intra-label spread that
would betray two voices is averaged away before it is ever persisted, and a
merge already folded both voices into that one vector.

``bimodal_split`` attempts what provenance cannot see: a label diarization
itself conflated, where raw and named agree because the two people were one
label before review began. It is a separate pass producing LEADS, not findings,
and it has a MEASURED BLIND SPOT — see its docstring. Do not treat a clean
bimodality scan as evidence a label holds one person.

Nothing here mutates or repairs anything. A voice mismatch still splits into
case (a) a split cluster and case (b) two real people, and per this project's
rule you cannot tell which without reading the transcript.

Pure: numpy only. Audio decoding and the voice model live in the caller
(``mismerge_scan.py``), so every decision rule here is unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def provenance_groups(raw_segments, named_segments) -> dict[str, dict[str, list[int]]]:
    """Map every raw turn onto the named label that now carries it.

    Returns ``{named_label: {raw_label: [raw turn indices]}}``. A named label
    with two or more raw groups absorbed a merge.

    Grouping is done at RAW turn granularity on purpose. A merge relabels the
    source's turns to the target, after which the adjacent-same-speaker merge
    can stitch a source turn and a target turn into ONE named turn — so a named
    turn can straddle both voices and cannot be attributed to either. Raw turns
    are the pre-merge diarized boundaries, so each one is a clean single-voice
    audio slice.

    A raw turn is assigned by greatest time overlap; one that no named turn
    covers is dropped (publish drops empty-text segments, and a reviewer can
    delete a junk cluster outright).
    """
    groups: dict[str, dict[str, list[int]]] = {}
    for index, raw in enumerate(raw_segments):
        best_label, best_overlap = None, 0.0
        for named in named_segments:
            overlap = _overlap(
                raw.start_time, raw.end_time, named.start_time, named.end_time
            )
            if overlap > best_overlap:
                best_label, best_overlap = named.speaker_label, overlap
        if best_label is None:
            continue
        groups.setdefault(best_label, {}).setdefault(raw.speaker_label, []).append(index)
    return groups


#: A raw label counts as absorbed only when this much of its speech, in seconds
#: AND as a share of its own total, landed inside another label. Both floors
#: exist to reject a boundary brush: the overlap region between two adjacent
#: turns is diarized independently in raw and named (boundary-snap moves words
#: across it), so a fraction of a second of spill is normal and means nothing.
#: Measured over the corpus, the real merges move a label's whole speech —
#: the smallest absorbed side is 3.7s at share 1.0.
#:
#: The share floor does drop one real shape: a PARTIAL relabel, where only part
#: of a raw label's speech moved. In 2026-07-09-debate-mi-governor-gop-primary
#: 71s of raw SPEAKER_04 (share 0.09) sits inside named SPEAKER_03. That
#: direction is filtered, but the same event still surfaces from the other
#: side — raw SPEAKER_03's own 22s (share 1.0) is the absorbed group there — so
#: the finding is not lost, only reported from the smaller party's point of
#: view. Lower `min_absorbed_share` to see partial relabels directly, at the
#: cost of every boundary brush in the corpus.
MIN_ABSORBED_SECONDS = 2.0
MIN_ABSORBED_SHARE = 0.5


@dataclass
class MergeCandidate:
    """One raw label's speech now living inside another label's, from a merge.

    ``label`` is the surviving named label. ``host_raw`` is the raw group
    holding the most speech inside it and ``absorbed_raw`` the group under
    test — note the survivor's own number can be the ABSORBED side, because a
    reviewer may merge a long run into a short label, which is exactly how a
    two-segment label's name came to relabel 28 of someone else's segments.
    Indices point into the raw segment list, whose boundaries are single-voice.
    """

    label: str
    host_raw: str
    absorbed_raw: str
    host_indices: list[int]
    absorbed_indices: list[int]
    host_seconds: float
    absorbed_seconds: float


def merge_candidates(
    raw_segments,
    named_segments,
    *,
    min_absorbed_seconds: float = MIN_ABSORBED_SECONDS,
    min_absorbed_share: float = MIN_ABSORBED_SHARE,
) -> list[MergeCandidate]:
    """Every (host, absorbed) pair a merge created, one per absorbed label.

    This is the cheap gate: it needs no audio and no model, and it is exact
    rather than heuristic — a raw label can only end up inside a different
    named label because something relabelled its segments. Pairs are returned
    for acoustic judgement, NOT as findings: most merges in the corpus are
    correct (a split cluster stitched back together).
    """
    groups = provenance_groups(raw_segments, named_segments)
    total_seconds: dict[str, float] = {}
    for segment in raw_segments:
        total_seconds[segment.speaker_label] = total_seconds.get(
            segment.speaker_label, 0.0
        ) + (segment.end_time - segment.start_time)

    def seconds_of(indices: list[int]) -> float:
        return sum(
            raw_segments[i].end_time - raw_segments[i].start_time for i in indices
        )

    candidates: list[MergeCandidate] = []
    for label, by_raw in sorted(groups.items()):
        if len(by_raw) < 2:
            continue
        measured = {raw: seconds_of(indices) for raw, indices in by_raw.items()}
        host_raw = max(sorted(measured), key=lambda raw: measured[raw])
        for absorbed_raw in sorted(by_raw):
            if absorbed_raw == host_raw:
                continue
            absorbed_seconds = measured[absorbed_raw]
            share = absorbed_seconds / (total_seconds.get(absorbed_raw) or absorbed_seconds)
            if absorbed_seconds < min_absorbed_seconds or share < min_absorbed_share:
                continue
            candidates.append(MergeCandidate(
                label=label,
                host_raw=host_raw,
                absorbed_raw=absorbed_raw,
                host_indices=list(by_raw[host_raw]),
                absorbed_indices=list(by_raw[absorbed_raw]),
                host_seconds=measured[host_raw],
                absorbed_seconds=absorbed_seconds,
            ))
    return candidates


#: A slice shorter than this is a turn-boundary fragment, and its embedding is
#: dominated by whatever leaked in from the neighbouring speaker — the exact
#: failure the Tanner Branham triage found (5 fragments stolen from two
#: different named speakers). reverify.MIN_EMBED_DUR uses 0.3s for a turn known
#: to be short; here there is no reason to accept one, so the floor is higher.
MIN_TURN_SECONDS = 0.5

#: Enough audio per side to place a centroid. Below it the answer is
#: "unmeasurable", never "the voices differ".
MIN_SIDE_SECONDS = 3.0

#: Caps on the embedding work per side. A centroid stops moving long before a
#: 4000s host side is fully embedded, and the corpus's own host sides run to
#: 500 turns, so the caps are what keeps the scan minutes rather than hours.
MAX_TURNS_PER_SIDE = 40
MAX_SECONDS_PER_SIDE = 180.0

#: Longest slice embedded from any one turn. A voice vector saturates in a few
#: seconds, so embedding a 200s monologue whole spends the whole budget on ONE
#: turn and leaves a centroid resting on two or three of them. Truncating buys
#: more turns for the same audio, which is what makes the centroid represent
#: the LABEL rather than its longest monologue.
MAX_SLICE_SECONDS = 20.0


def slice_bounds(segment, *, max_slice_seconds: float = MAX_SLICE_SECONDS) -> tuple[float, float]:
    """The (start, end) actually embedded for a turn — truncated from its start."""
    return (
        segment.start_time,
        min(segment.end_time, segment.start_time + max_slice_seconds),
    )


def select_turns(
    segments,
    indices,
    *,
    min_turn_seconds: float = MIN_TURN_SECONDS,
    max_turns: int = MAX_TURNS_PER_SIDE,
    max_seconds: float = MAX_SECONDS_PER_SIDE,
    max_slice_seconds: float = MAX_SLICE_SECONDS,
) -> list[int]:
    """Choose which of a group's turns to embed: longest first, then in time order.

    Longest-first because a longer turn carries more voice and less boundary
    contamination. The budget is spent in TRUNCATED slice seconds
    (``max_slice_seconds``), so a label of long monologues still contributes
    many turns to its centroid instead of two. The result is re-sorted by time
    so a report reads in the order a human would scrub through the recording.
    """
    usable = [
        i for i in indices
        if segments[i].end_time - segments[i].start_time >= min_turn_seconds
    ]
    usable.sort(
        key=lambda i: (-(segments[i].end_time - segments[i].start_time), i)
    )
    chosen: list[int] = []
    seconds = 0.0
    for i in usable:
        if len(chosen) >= max_turns or seconds >= max_seconds:
            break
        chosen.append(i)
        start, end = slice_bounds(segments[i], max_slice_seconds=max_slice_seconds)
        seconds += end - start
    return sorted(chosen)


def _unit_mean(vectors) -> "np.ndarray | None":
    """Mean of the usable unit vectors in a group, or None if none are usable.

    Non-finite and zero-norm rows are dropped per-vector rather than poisoning
    the group: 18 of 1035 corpus labels carry a missing, NaN or zero-norm
    centroid, and per-turn vectors hit the same pyannote NaNs that
    global_identity.decode_turn_vectors filters.
    """
    rows = []
    for vector in vectors:
        array = np.asarray(vector, dtype=float).ravel()
        if array.size == 0 or not np.all(np.isfinite(array)):
            continue
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            continue
        rows.append(array / norm)
    if not rows:
        return None
    mean = np.mean(rows, axis=0)
    norm = float(np.linalg.norm(mean))
    return None if norm == 0.0 else mean / norm


def group_similarity(host_vectors, absorbed_vectors) -> "float | None":
    """Cosine similarity between two turn groups' voice centroids.

    ``None`` means unmeasurable — an empty side, or every vector on a side
    unusable. Per review.voice_similarity's rule, a caller must never read
    that as evidence the voices differ.
    """
    host, absorbed = _unit_mean(host_vectors), _unit_mean(absorbed_vectors)
    if host is None or absorbed is None:
        return None
    if host.shape != absorbed.shape:
        return None
    return float(np.dot(host, absorbed))


@dataclass
class Assessment:
    """The acoustic verdict on one merge candidate. No mutation, no remedy.

    ``verdict`` uses review.merge_voice_verdict's vocabulary and thresholds
    deliberately: the 0.42/0.60 bands were calibrated once, on the 25-case
    duplicate-name triage, and a second scale with a second meaning would make
    the two detectors disagree about the same pair of voices.

    A ``mismatch`` is not a diagnosis. It says the two voices differ, which
    still splits into case (a) a split cluster and case (b) two real people,
    and per this project's rule you cannot tell which without reading the
    transcript.
    """

    label: str
    host_raw: str
    absorbed_raw: str
    similarity: "float | None"
    verdict: str
    host_turns_embedded: int
    absorbed_turns_embedded: int
    host_seconds_embedded: float
    absorbed_seconds_embedded: float
    host_seconds: float
    absorbed_seconds: float
    reason: "str | None" = None


def _embeddable_seconds(segments, indices, max_slice_seconds: float) -> float:
    total = 0.0
    for i in indices:
        start, end = slice_bounds(segments[i], max_slice_seconds=max_slice_seconds)
        total += end - start
    return total


def assess_candidate(
    candidate: MergeCandidate,
    raw_segments,
    embed_fn,
    *,
    min_turn_seconds: float = MIN_TURN_SECONDS,
    min_side_seconds: float = MIN_SIDE_SECONDS,
    max_turns: int = MAX_TURNS_PER_SIDE,
    max_seconds: float = MAX_SECONDS_PER_SIDE,
    max_slice_seconds: float = MAX_SLICE_SECONDS,
) -> Assessment:
    """Re-embed both sides of a merge from the audio and band their similarity.

    ``embed_fn(start, end)`` returns that slice's voice vector, or None when it
    cannot be embedded — the same contract as reverify's. Per-turn embedding is
    unavoidable here: embeddings.json holds ONE centroid per label, and a
    merge already averaged both voices into it, so the spread that would betray
    two people is destroyed before it is persisted.

    A side with too little embeddable audio short-circuits to ``unknown``
    WITHOUT calling embed_fn, which is what keeps a corpus scan cheap.
    """
    from .review import merge_voice_verdict

    base = dict(
        label=candidate.label,
        host_raw=candidate.host_raw,
        absorbed_raw=candidate.absorbed_raw,
        host_seconds=candidate.host_seconds,
        absorbed_seconds=candidate.absorbed_seconds,
    )

    sides = {}
    for name, indices in (
        ("host", candidate.host_indices),
        ("absorbed", candidate.absorbed_indices),
    ):
        sides[name] = select_turns(
            raw_segments, indices,
            min_turn_seconds=min_turn_seconds,
            max_turns=max_turns,
            max_seconds=max_seconds,
            max_slice_seconds=max_slice_seconds,
        )

    for name in ("host", "absorbed"):
        seconds = _embeddable_seconds(raw_segments, sides[name], max_slice_seconds)
        if seconds < min_side_seconds:
            return Assessment(
                similarity=None, verdict="unknown",
                host_turns_embedded=0, absorbed_turns_embedded=0,
                host_seconds_embedded=0.0, absorbed_seconds_embedded=0.0,
                reason=(
                    f"{name} side has {seconds:.1f}s of embeddable speech "
                    f"(need {min_side_seconds:.1f}s)"
                ),
                **base,
            )

    vectors: dict[str, list] = {}
    embedded_seconds: dict[str, float] = {}
    for name in ("host", "absorbed"):
        rows, seconds = [], 0.0
        for i in sides[name]:
            start, end = slice_bounds(
                raw_segments[i], max_slice_seconds=max_slice_seconds
            )
            vector = embed_fn(start, end)
            if vector is None:
                continue
            rows.append(vector)
            seconds += end - start
        vectors[name] = rows
        embedded_seconds[name] = seconds

    similarity = group_similarity(vectors["host"], vectors["absorbed"])
    return Assessment(
        similarity=similarity,
        verdict=merge_voice_verdict(similarity),
        host_turns_embedded=len(vectors["host"]),
        absorbed_turns_embedded=len(vectors["absorbed"]),
        host_seconds_embedded=embedded_seconds["host"],
        absorbed_seconds_embedded=embedded_seconds["absorbed"],
        reason=None if similarity is not None else "no usable voice vector on one side",
        **base,
    )


def straddling_named_turns(candidate: MergeCandidate, raw_segments, named_segments) -> int:
    """How many named turns cover raw turns from BOTH sides of the candidate.

    Zero means the merge moved whole turns, so the remedy is a re-split of the
    label. Non-zero means a named turn itself spans two voices — the shape found
    in 2026-07-09-debate-mi-governor-gop-primary, where one 99s named segment
    holds an audience question AND the candidate's answer — and no label-level
    operation can fix that: the segment has to be cut first.
    """
    host = {i: True for i in candidate.host_indices}
    absorbed = {i: True for i in candidate.absorbed_indices}
    count = 0
    for named in named_segments:
        if named.speaker_label != candidate.label:
            continue
        touches_host = touches_absorbed = False
        for index, raw in enumerate(raw_segments):
            if _overlap(raw.start_time, raw.end_time,
                        named.start_time, named.end_time) <= 0:
                continue
            if index in host:
                touches_host = True
            elif index in absorbed:
                touches_absorbed = True
        if touches_host and touches_absorbed:
            count += 1
    return count


def load_meeting_segments(meeting_dir) -> tuple[list, list]:
    """Read (raw, named) segments for one meeting directory.

    transcript_raw.json is the trustworthy record of the ORIGINAL diarized
    labels: gui.review_api._persist_after_review rewrites diarization.json with
    the POST-review segments, so it carries whatever the reviewer did — the
    merge included — and cannot witness against it.
    """
    from .models import Segment

    meeting_dir = Path(meeting_dir)
    raw = json.loads((meeting_dir / "transcript_raw.json").read_text())
    named = json.loads((meeting_dir / "transcript_named.json").read_text())
    raw_list = raw if isinstance(raw, list) else raw.get("segments", [])
    named_list = named if isinstance(named, list) else named.get("segments", [])
    return (
        [Segment.from_dict(s) for s in raw_list],
        [Segment.from_dict(s) for s in named_list],
    )


def scan_corpus_candidates(
    meetings_dir,
    **kwargs,
) -> tuple[list[tuple[str, MergeCandidate]], list[tuple[str, str]]]:
    """Cheap-gate every meeting under `meetings_dir`. No audio, no model.

    Returns ``(candidates, unchecked)``. ``unchecked`` names the meetings
    provenance cannot see and why — a reviewed meeting with no
    transcript_raw.json is a coverage hole, and reporting zero findings without
    naming it would repeat exactly the false reassurance this detector exists
    to end.
    """
    candidates: list[tuple[str, MergeCandidate]] = []
    unchecked: list[tuple[str, str]] = []
    for directory in sorted(Path(meetings_dir).iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "transcript_named.json").exists():
            continue  # never reviewed; nothing to be wrong about
        if not (directory / "transcript_raw.json").exists():
            unchecked.append((directory.name, "no transcript_raw.json"))
            continue
        try:
            raw, named = load_meeting_segments(directory)
        except (json.JSONDecodeError, KeyError, OSError) as error:
            unchecked.append((directory.name, f"unreadable transcript: {error}"))
            continue
        for candidate in merge_candidates(raw, named, **kwargs):
            candidates.append((directory.name, candidate))
    return candidates, unchecked


#: Report order. 'unknown' outranks 'match' because an unmeasurable side is an
#: open question, while a match is an answered one.
_VERDICT_ORDER = {"mismatch": 0, "uncertain": 1, "unknown": 2, "match": 3}


def misattributed_floor_seconds(assessment: Assessment) -> float:
    """Least speech that must be under the wrong name if the merge was wrong.

    The surviving label carries ONE name, so whichever side that name does not
    belong to is misattributed. Provenance cannot say which side owns the name,
    so the minority side's speech is the floor — the real figure is that or
    worse, never better. It is what makes a 30s absorbed side worth as much
    attention as a 900s one: in the confirmed 2026-04-20 case a two-segment
    label's name relabelled 28 of someone else's segments.
    """
    return min(assessment.host_seconds, assessment.absorbed_seconds)


def rank_assessments(assessments: list[Assessment]) -> list[Assessment]:
    """Most-severe first: verdict band, then how much speech is provably wrong."""
    return sorted(
        assessments,
        key=lambda a: (
            _VERDICT_ORDER.get(a.verdict, 4),
            -misattributed_floor_seconds(a),
            a.label,
            a.absorbed_raw,
        ),
    )


def label_turn_indices(segments) -> dict[str, list[int]]:
    """{speaker_label: [turn indices]} in time order — the provenance-free gate."""
    grouped: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(segment.speaker_label, []).append(index)
    return grouped


#: A side of a 2-means split needs this many turns to be a person rather than
#: an artefact. One outlier turn is a boundary fragment or a cough; the Tanner
#: Branham cluster proved single stolen fragments are common.
MIN_SPLIT_SIDE_TURNS = 2


@dataclass
class BimodalSplit:
    """The best two-way split of one label's turn vectors.

    ``side_a`` / ``side_b`` index into the vector list as PASSED, so unusable
    rows are excluded from both without shifting the numbering a caller uses to
    name turns. ``similarity`` is the cosine between the two sides' centroids —
    band it with review.merge_voice_verdict, the same scale as everything else.
    """

    side_a: list[int]
    side_b: list[int]
    similarity: float


def bimodal_split(
    vectors,
    *,
    min_side_turns: int = MIN_SPLIT_SIDE_TURNS,
    iterations: int = 8,
) -> "BimodalSplit | None":
    """Split a label's turn vectors in two and measure how far apart they are.

    This is the detector for a label NO merge touched: diarization itself can
    conflate two people (global_identity records one seam join at similarity
    0.604 that merged two councilmembers), and provenance cannot see that
    because raw and named agree. A split always exists — one speaker's turns
    scatter — so the split is not the finding; its SEPARATION is.

    Seeded from the least-similar pair of turns, which is the split a
    two-person label actually has, then refined by 2-means. Returns None when
    there are too few usable vectors to give both sides ``min_side_turns``.

    🔴 MEASURED BLIND SPOT: a balanced 2-way split cannot find a MINORITY
    speaker inside a label one voice dominates. On the corpus's one confirmed
    multi-person label — 2026-04-03-lwv-brown-county-candidate-forum-auditor
    SPEAKER_09, which carries three self-introductions (the moderator plus two
    auditor candidates) across 359 turns / 2304s — this scores +0.676, deep in
    the MATCH band and rank 105 of 555. The moderator's own variation is a
    larger axis than moderator-vs-candidate, so the split lands on the wrong
    one. Two alternatives were measured and also failed: time-stratified
    sampling (+0.655, so it is not a longest-first sampling artefact) and
    leave-one-out per-turn outlier scoring (the two candidate turns ranked 7th
    and 10th lowest, below four genuine moderator turns, because short turns
    embed noisily and duration confounds the score). Detecting a minority
    intruder in a dominated label is UNSOLVED here; do not re-derive these
    three. What DID work corpus-wide is the provenance gate above.

    What it does deliver: on the 555 labels it could test it independently
    re-found five of the provenance scan's findings (2026-05-12 SPEAKER_01,
    2026-05-19 SPEAKER_16 and SPEAKER_00, 2026-05-15 SPEAKER_10, 2026-07-14
    SPEAKER_05), which is a real cross-check by a different method. But 30 of
    its 53 mismatch-band hits have a side of only two turns, i.e. a boundary
    fragment pair rather than a person, so the band is not a finding rate.
    """
    usable: list[tuple[int, np.ndarray]] = []
    for index, vector in enumerate(vectors):
        unit = _unit_mean([vector])
        if unit is not None:
            usable.append((index, unit))
    if len(usable) < 2 * min_side_turns:
        return None

    matrix = np.asarray([unit for _, unit in usable])
    gram = matrix @ matrix.T
    np.fill_diagonal(gram, np.inf)
    seed_a, seed_b = np.unravel_index(np.argmin(gram), gram.shape)
    centre_a, centre_b = matrix[seed_a], matrix[seed_b]

    assignment = np.zeros(len(usable), dtype=int)
    for _ in range(iterations):
        updated = (matrix @ centre_b > matrix @ centre_a).astype(int)
        if np.array_equal(updated, assignment) and _ > 0:
            break
        assignment = updated
        for group, centre_name in ((0, "a"), (1, "b")):
            rows = matrix[assignment == group]
            if not len(rows):
                continue
            centre = _unit_mean(list(rows))
            if centre is None:
                continue
            if centre_name == "a":
                centre_a = centre
            else:
                centre_b = centre

    side_a = [usable[i][0] for i in range(len(usable)) if assignment[i] == 0]
    side_b = [usable[i][0] for i in range(len(usable)) if assignment[i] == 1]
    if len(side_a) < min_side_turns or len(side_b) < min_side_turns:
        return None
    similarity = float(np.dot(centre_a, centre_b))
    return BimodalSplit(side_a=side_a, side_b=side_b, similarity=similarity)


def label_source_turns(raw_segments, named_segments) -> dict[str, list[int]]:
    """{named label: [raw turn indices it now carries]}, in time order.

    The provenance-free scan's input. Raw boundaries rather than named ones
    because a named turn can straddle two voices, and one blended vector hides
    exactly the bimodality being looked for.
    """
    return {
        label: sorted(index for indices in by_raw.values() for index in indices)
        for label, by_raw in provenance_groups(raw_segments, named_segments).items()
    }
