"""Tests for re-clustering an existing turn set over per-turn embeddings."""
from bench.forum_recluster import as_unique_label_segments, turn_label

SEGMENTS = [
    {"segment_id": 0, "start_time": 0.0, "end_time": 4.0, "speaker_label": "SPEAKER_09"},
    {"segment_id": 1, "start_time": 4.0, "end_time": 9.0, "speaker_label": "SPEAKER_09"},
    {"segment_id": 2, "start_time": 9.0, "end_time": 9.1, "speaker_label": "SPEAKER_03"},
]


def test_turn_labels_are_zero_padded_and_ordered():
    assert turn_label(0) == "TURN_0000"
    assert turn_label(42) == "TURN_0042"
    assert turn_label(478) == "TURN_0478"


def test_every_turn_gets_its_own_label():
    """pipeline_extract_embeddings averages per speaker_label. One label per
    turn therefore makes each returned 'centroid' that turn's own embedding,
    which is why this needs no new Modal code."""
    unique = as_unique_label_segments(SEGMENTS)
    assert [s["speaker_label"] for s in unique] == ["TURN_0000", "TURN_0001", "TURN_0002"]
    assert len({s["speaker_label"] for s in unique}) == len(SEGMENTS)


def test_spans_are_preserved_exactly():
    unique = as_unique_label_segments(SEGMENTS)
    assert [(s["start_time"], s["end_time"]) for s in unique] == [
        (0.0, 4.0), (4.0, 9.0), (9.0, 9.1)
    ]


def test_the_original_segments_are_not_mutated():
    """merge_adjacent_segments renumbering its inputs in place has burned this
    repo before; a probe must never edit the thing it is probing."""
    as_unique_label_segments(SEGMENTS)
    assert [s["speaker_label"] for s in SEGMENTS] == [
        "SPEAKER_09", "SPEAKER_09", "SPEAKER_03"
    ]


import numpy as np
import pytest

from bench.forum_recluster import calibrate, cluster_turns


def _vec(*values):
    v = np.array(values, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


# Two tight voices, far apart on the unit sphere.
A1, A2, A3 = _vec(1, 0, 0.01), _vec(1, 0.02, 0), _vec(1, 0, 0.03)
B1, B2 = _vec(0, 1, 0.01), _vec(0.02, 1, 0)


def test_two_voices_become_two_labels():
    labels = cluster_turns({0: A1, 1: A2, 2: B1, 3: B2}, n_turns=4, threshold=0.5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_every_turn_gets_a_label_even_with_no_embedding():
    """32 turns fall under the worker's 0.3s floor. They must still occupy
    their audio — a turn that vanishes here vanishes from the transcript."""
    labels = cluster_turns({0: A1, 1: A2}, n_turns=4, threshold=0.5)
    assert len(labels) == 4
    assert labels[2] == labels[3] == "SPEAKER_UNCLUSTERED"


def test_unembeddable_turns_share_one_bucket_not_singletons():
    """Assigning them by adjacency would guess at exactly the
    question-to-answer boundaries that matter most, and 32 singleton labels
    would wreck a label-level review."""
    labels = cluster_turns({1: A1}, n_turns=5, threshold=0.5)
    bucket = [l for i, l in enumerate(labels) if i != 1]
    assert set(bucket) == {"SPEAKER_UNCLUSTERED"}


def test_a_high_threshold_splits_and_a_low_one_merges():
    vectors = {0: A1, 1: A2, 2: A3, 3: B1, 4: B2}
    merged = cluster_turns(vectors, n_turns=5, threshold=0.0)
    split = cluster_turns(vectors, n_turns=5, threshold=0.999)
    assert len(set(merged)) < len(set(split))


def test_calibrate_returns_a_grid_over_the_tuning_half():
    """Tuning and reporting on the same anchors proves nothing, so calibration
    is handed the tuning half only — and that half still contains moderator
    turns, because the split is by window, not by turn."""
    segments = [
        {"segment_id": i, "start_time": float(i * 10), "end_time": float(i * 10 + 9),
         "speaker_label": "SPEAKER_09", "text": t}
        for i, t in enumerate([
            "Ms. Bond, same question.", "My first answer runs on.",
            "Miss Cobian, same question.", "Her first answer runs on.",
            "Ms. Bond, what is your view?", "My second answer.",
            "Miss Cobian, what is your view?", "Her second answer.",
        ])
    ]
    vectors = {0: A1, 1: A1, 2: B1, 3: B1, 4: A2, 5: A2, 6: B2, 7: B2}
    from bench.forum_anchor_reference import (
        LWV_AUDITOR_SPEAKERS,
        anchor_reference_windows,
    )
    from bench.forum_gate import reference_half

    windows = anchor_reference_windows(segments, LWV_AUDITOR_SPEAKERS)
    tune = reference_half(windows, "tune")
    assert "MODERATOR" in {p for _, _, p in tune}

    best, grid = calibrate(segments, vectors, tune, [0.2, 0.5, 0.8])
    assert isinstance(best, float)
    assert len(grid) == 3
    assert {"threshold", "labels", "conflated", "fragmented"} <= set(grid[0])
