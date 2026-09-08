"""check_consistency.py's disk-side reader.

The orphan check needs each transcript's speakers dict, and the segment count
already opens that file — these run to 5 MB, so both come out of one read.
"""
from __future__ import annotations

import json

from check_consistency import _read_named


def _write(tmp_path, data):
    path = tmp_path / "transcript_named.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_read_named_returns_the_text_bearing_segment_count(tmp_path):
    path = _write(tmp_path, {"segments": [{"text": "hello"}, {"text": "there"}]})
    segs, _ = _read_named(path)
    assert segs == 2


def test_read_named_does_not_count_blank_segments(tmp_path):
    # Publish drops empty-text segments, so the DB count must be compared
    # against the same subset.
    path = _write(tmp_path, {"segments": [{"text": "hello"}, {"text": "   "}, {}]})
    segs, _ = _read_named(path)
    assert segs == 1


def test_read_named_returns_speakers_shaped_for_the_orphan_audit(tmp_path):
    path = _write(tmp_path, {
        "segments": [],
        "speakers": {"S0": {"speaker_label": "S0", "speaker_name": "A One"}},
    })
    _, meeting_data = _read_named(path)
    from src.speaker_orphans import keep_labels
    assert keep_labels(meeting_data) == {"S0"}


def test_read_named_survives_a_transcript_with_no_speakers_key(tmp_path):
    path = _write(tmp_path, {"segments": [{"text": "hi"}]})
    segs, meeting_data = _read_named(path)
    assert segs == 1
    assert meeting_data == {"speakers": {}, "empty_labels": []}


def test_empty_label_is_reported_as_its_own_issue(tmp_path):
    """A present-but-empty label must surface here, not hide behind the orphan check.

    Goes through _read_named on a real file rather than a hand-built dict. An
    earlier version of this test passed its own segments-bearing `meeting_data`
    — a shape the production caller never produces — and so passed while the
    wiring reported EVERY label as empty on almost every meeting.
    """
    import json

    from check_consistency import _read_named, empty_label_details

    affected = tmp_path / "a.json"
    affected.write_text(json.dumps({
        "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"},
                     "SPEAKER_01": {"speaker_label": "SPEAKER_01"}},
        "segments": [
            {"speaker_label": "SPEAKER_00", "text": "hi"},
            {"speaker_label": "SPEAKER_01", "text": ""},
        ],
    }))
    healthy = tmp_path / "b.json"
    healthy.write_text(json.dumps({
        "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"}},
        "segments": [{"speaker_label": "SPEAKER_00", "text": "hi"}],
    }))

    _, data_a = _read_named(affected)
    _, data_b = _read_named(healthy)
    details = empty_label_details({"a": {"meeting_data": data_a},
                                   "b": {"meeting_data": data_b}})

    assert "b" not in details
    assert "SPEAKER_01" in details["a"]


def test_read_named_reports_empty_labels_from_the_real_file(tmp_path):
    """_read_named must carry the empty-label verdict, not the segments.

    It deliberately returns only {"speakers": ...} — these files run to 5 MB and
    holding every segment for 172 meetings is the cost it exists to avoid. So a
    caller that needs segment text CANNOT get it from meeting_data, and a check
    written against a segments-bearing dict silently reports every label as
    empty. That is exactly what shipped: the flood was invisible to a unit test
    that passed its own well-formed input.
    """
    import json

    from check_consistency import _read_named

    named = tmp_path / "transcript_named.json"
    named.write_text(json.dumps({
        "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"},
                     "SPEAKER_01": {"speaker_label": "SPEAKER_01"}},
        "segments": [
            {"speaker_label": "SPEAKER_00", "text": "real words"},
            {"speaker_label": "SPEAKER_01", "text": ""},
        ],
    }))

    count, meeting_data = _read_named(named)

    assert count == 1
    assert meeting_data["empty_labels"] == ["SPEAKER_01"]


def test_empty_label_details_reads_the_precomputed_verdict(tmp_path):
    import json

    from check_consistency import _read_named, empty_label_details

    healthy = tmp_path / "healthy.json"
    healthy.write_text(json.dumps({
        "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"}},
        "segments": [{"speaker_label": "SPEAKER_00", "text": "words"}],
    }))
    _, data = _read_named(healthy)

    assert empty_label_details({"a": {"meeting_data": data}}) == {}
