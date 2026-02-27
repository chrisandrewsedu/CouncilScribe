"""Auto-merge fragmented speakers after diarization.

Pyannote frequently splits one person into multiple SPEAKER labels.
This module detects and merges speakers whose voice embeddings are
highly similar, using union-find to group connected components.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from . import config
from .models import Segment


class _UnionFind:
    """Simple union-find for grouping speaker labels."""

    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        """Return root -> [members] mapping."""
        result: dict[str, list[str]] = {}
        for item in self.parent:
            root = self.find(item)
            if root not in result:
                result[root] = []
            result[root].append(item)
        return result


def merge_similar_speakers(
    segments: list[Segment],
    speaker_embeddings: dict[str, np.ndarray],
    threshold: float | None = None,
) -> tuple[list[Segment], dict[str, np.ndarray], list[str]]:
    """Merge diarized speakers whose embeddings are above the similarity threshold.

    Uses union-find to group speakers into connected components based on
    pairwise cosine similarity. Within each group, all speakers are relabeled
    to the one with the most total speech time.

    Args:
        segments: Diarized segments (text may be empty at this stage).
        speaker_embeddings: speaker_label -> centroid embedding.
        threshold: Cosine similarity threshold for merging. Defaults to config value.

    Returns:
        Tuple of (merged_segments, merged_embeddings, merge_log).
        - merged_segments: Segments with relabeled speakers.
        - merged_embeddings: Recomputed centroids for merged groups.
        - merge_log: Human-readable list of merge actions.
    """
    if threshold is None:
        threshold = config.SPEAKER_MERGE_THRESHOLD

    labels = sorted(speaker_embeddings.keys())
    if len(labels) < 2:
        return segments, dict(speaker_embeddings), []

    # Compute speech time per label for choosing group representative
    speech_time: dict[str, float] = {}
    for seg in segments:
        lbl = seg.speaker_label
        speech_time[lbl] = speech_time.get(lbl, 0.0) + (seg.end_time - seg.start_time)

    # Build union-find from pairwise similarities
    uf = _UnionFind(labels)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = 1.0 - cosine(speaker_embeddings[labels[i]], speaker_embeddings[labels[j]])
            if sim >= threshold:
                uf.union(labels[i], labels[j])

    # Determine group representatives (most speech time) and build remap
    groups = uf.groups()
    remap: dict[str, str] = {}
    merge_log: list[str] = []

    for root, members in groups.items():
        if len(members) == 1:
            continue  # no merge needed

        # Pick representative: the member with the most speech time
        representative = max(members, key=lambda l: speech_time.get(l, 0.0))
        merged_others = sorted(m for m in members if m != representative)
        for m in merged_others:
            remap[m] = representative
        merge_log.append(
            f"{' + '.join(merged_others)} merged into {representative}"
        )

    if not remap:
        return segments, dict(speaker_embeddings), []

    # Relabel segments
    for seg in segments:
        if seg.speaker_label in remap:
            seg.speaker_label = remap[seg.speaker_label]

    # Recompute merged centroids by averaging all original embeddings in the group
    merged_embeddings: dict[str, np.ndarray] = {}
    for root, members in groups.items():
        representative = max(members, key=lambda l: speech_time.get(l, 0.0))
        embs = [speaker_embeddings[m] for m in members if m in speaker_embeddings]
        if embs:
            merged_embeddings[representative] = np.mean(embs, axis=0)

    # Add any labels that weren't in any multi-member group
    for label in labels:
        if label not in merged_embeddings and label not in remap:
            merged_embeddings[label] = speaker_embeddings[label]

    return segments, merged_embeddings, merge_log
