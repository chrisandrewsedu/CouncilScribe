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
    assert meeting_data == {"speakers": {}}


def test_empty_label_is_reported_as_its_own_issue():
    """A present-but-empty label must surface here, not hide behind the orphan check.

    The orphan check passes it (the label is in speakers, so it is not stale),
    yet publish writes a speaker row for it that serves nothing. Prod had three
    such rows while the orphan audit reported zero problems.
    """
    from check_consistency import empty_label_details

    disk = {
        "a": {"meeting_data": {
            "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"},
                         "SPEAKER_01": {"speaker_label": "SPEAKER_01"}},
            "segments": [
                {"speaker_label": "SPEAKER_00", "text": "hi"},
                {"speaker_label": "SPEAKER_01", "text": ""},
            ],
        }},
        "b": {"meeting_data": {
            "speakers": {"SPEAKER_00": {"speaker_label": "SPEAKER_00"}},
            "segments": [{"speaker_label": "SPEAKER_00", "text": "hi"}],
        }},
    }

    details = empty_label_details(disk)

    assert "b" not in details
    assert "SPEAKER_01" in details["a"]
