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
