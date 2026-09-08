"""Re-cluster an existing diarization's turns over per-turn voice embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

Pure: no env loading, no Modal, no torch. The Modal fetch and the CLI live in
`scripts/recluster_forum_turns.py`.

HAZARD — `SPEAKER_UNCLUSTERED` (aliased below as `UNCLUSTERED_LABEL`) is a
known-multi-person bucket BY CONSTRUCTION: it holds turns with no embedding
(`cluster_turns`) and labels whose total speech falls below `fold_slivers`'
floor, both folded from many different speakers into one shared label. It
must never be handed a person's name in review — doing so would misattribute
every voice pooled under it, the same defect this repair exists to fix. It is
deliberately absent from `embeddings.json` (no centroid is ever computed for
it) so `src/identify.py` can never auto-name it from a voice-profile match.
Review UI changes to make this hazard visible to a human reviewer are
out of scope for this module.
"""

from __future__ import annotations

import copy


def turn_label(index: int) -> str:
    """Unique per-turn label. Zero-padded so lexical order is turn order."""
    return f"TURN_{index:04d}"


def as_unique_label_segments(segments: list[dict]) -> list[dict]:
    """Copy `segments`, relabelling each with its own unique speaker label.

    `bench/modal_app.py:1305 pipeline_extract_embeddings` averages wespeaker
    embeddings per `speaker_label` over arbitrary supplied segments, so one
    label per turn makes each returned "centroid" that turn's own embedding.
    That is why Experiment B needs no new Modal code.

    Deep-copies rather than editing in place: `merge_adjacent_segments`
    renumbering its own inputs has already cost this repo a false staleness
    diagnosis, so a probe must never mutate what it probes.
    """
    unique = []
    for index, segment in enumerate(segments):
        clone = copy.deepcopy(segment)
        clone["speaker_label"] = turn_label(index)
        unique.append(clone)
    return unique


UNCLUSTERED_LABEL = "SPEAKER_UNCLUSTERED"


def cluster_turns(
    vectors: dict[int, list[float]],
    n_turns: int,
    threshold: float,
    *,
    unclustered_label: str = UNCLUSTERED_LABEL,
) -> list[str]:
    """One speaker label per turn index, by agglomerative clustering.

    Average linkage over cosine distance. MEASURED elsewhere in this repo
    (`src/config.py:329`): "complete" scores a candidate by its worst turn pair
    and a real person's worst pair is often anti-correlated (same-person median
    -0.125), so it merges almost nothing; "centroid" pools each cluster into one
    mean and conflated two real people at the most conservative threshold
    tested. Average is the only workable choice.

    `threshold` is a cosine SIMILARITY floor: clusters join while their mean
    pairwise similarity is at or above it. Turns with no embedding land in one
    shared bucket, not in a neighbour's cluster and not in singletons of their
    own — see the module docstring.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    labels = [unclustered_label] * n_turns
    indices = sorted(vectors)
    if not indices:
        return labels

    matrix = np.asarray([vectors[i] for i in indices], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    if len(indices) == 1:
        labels[indices[0]] = "SPEAKER_00"
        return labels

    distances = pdist(matrix, metric="cosine")
    tree = linkage(distances, method="average")
    assignments = fcluster(tree, t=1.0 - threshold, criterion="distance")

    # Number labels by first appearance so output is stable across runs.
    order: dict[int, str] = {}
    for position, cluster in enumerate(assignments):
        if cluster not in order:
            order[cluster] = f"SPEAKER_{len(order):02d}"
        labels[indices[position]] = order[cluster]
    return labels


def fold_slivers(
    labels: list[str],
    segments: list[dict],
    floor_seconds: float,
    *,
    unclustered_label: str = UNCLUSTERED_LABEL,
) -> list[str]:
    """Move labels holding less than `floor_seconds` of speech into the bucket.

    Agglomerative clustering over per-turn embeddings leaves a long tail: on the real
    meeting at threshold 0.50, 104 of 114 labels hold 1.5s each while the top 9 hold
    94% of the speech. Those slivers carry too little voice evidence to attribute —
    the same reason unembeddable turns go to the bucket — and 104 phantom speakers
    would wreck a label-level review.

    `unclustered_label` mirrors `cluster_turns`' keyword of the same name and
    MUST be passed the same value when the caller used a non-default one there
    — otherwise unembeddable turns and folded slivers land in two different
    buckets, and the gate (which excludes only one bucket by name) fails on
    the other as if it were a real conflated identity.
    """
    if floor_seconds <= 0:
        return list(labels)
    totals: dict[str, float] = {}
    for segment, label in zip(segments, labels):
        totals[label] = totals.get(label, 0.0) + (
            segment["end_time"] - segment["start_time"]
        )
    return [
        unclustered_label if totals[label] < floor_seconds else label
        for label in labels
    ]


def relabel_segments(segments: list[dict], labels: list[str]) -> list[dict]:
    """Copy `segments` with new speaker labels, spans untouched."""
    out = []
    for segment, label in zip(segments, labels):
        clone = copy.deepcopy(segment)
        clone["speaker_label"] = label
        out.append(clone)
    return out


def calibrate(
    segments: list[dict],
    vectors: dict[int, list[float]],
    tune_reference: list[tuple[float, float, str]],
    thresholds: list[float],
    *,
    sliver_floor: float = 20.0,
) -> tuple[float, list[dict]]:
    """Pick a threshold against the TUNING half of the reference.

    The caller supplies the already-halved reference — `reference_half(windows,
    "tune")` — so the tune/holdout split lives in exactly one place and cannot
    drift between the calibrator and the scorer. Calibrating and reporting on
    the same 32 anchors would prove nothing.

    `sliver_floor` folds labels holding less than that many seconds into the
    shared bucket before scoring, the same way the shipped output will be
    folded — otherwise the grid would report a threshold's conflation as if
    the 100+ phantom slivers it produces were never going to be merged away,
    and the bucket itself (which holds slivers from many people BY
    CONSTRUCTION) would be scored as a conflated identity.

    That exclusion is not unconditional: `gate_verdict` (which supplies
    `conflated`, below) also bounds the bucket's SIZE against `GATE_MIN_FRACTION`
    of scored reference speech, so raising `sliver_floor` past the point where
    the bucket has absorbed too much speech makes `conflated` rise again rather
    than fall to 0 forever — without that bound, `conflated` would fall
    monotonically as `sliver_floor` rises and the tie-break below would always
    pick the highest floor offered, which is exactly the failure mode this
    bound exists to close.

    Ties break toward the HIGHER threshold: conflation misattributes quotes
    silently, fragmentation surfaces as an extra unnamed speaker the reviewer
    clears in seconds.
    """
    from .forum_gate import GATE_MIN_FRACTION, gate_verdict
    from .identity_score import identity_report

    people = sorted({p for _, _, p in tune_reference})

    grid: list[dict] = []
    for threshold in thresholds:
        labels = cluster_turns(vectors, len(segments), threshold)
        labels = fold_slivers(labels, segments, sliver_floor)
        hypothesis = [
            (s["start_time"], s["end_time"], l)
            for s, l in zip(segments, labels)
        ]
        report = identity_report(
            hypothesis, tune_reference, min_fraction=GATE_MIN_FRACTION
        )
        _, reasons = gate_verdict(
            report, max_minority=GATE_MIN_FRACTION,
            unattributed_label=UNCLUSTERED_LABEL,
        )
        grid.append({
            "threshold": threshold,
            "labels": len(set(labels)),
            "conflated": len(reasons),
            "fragmented": len(report.fragmentation),
            "people": len(people),
        })

    clean = [row for row in grid if row["conflated"] == 0]
    if clean:
        best = min(clean, key=lambda r: (r["fragmented"], -r["threshold"]))
    else:
        best = min(grid, key=lambda r: (r["conflated"], -r["threshold"]))
    return best["threshold"], grid
