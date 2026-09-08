"""Tests for the forum diarization gate."""
import json

from bench.identity_score import identity_report
from bench.forum_gate import gate_verdict, load_turns, reference_half

REFERENCE = [
    (0.0, 30.0, "BOND"),
    (30.0, 60.0, "KOBIAN"),
    (60.0, 90.0, "MODERATOR"),
]


def test_load_turns_reads_segment_dicts(tmp_path):
    path = tmp_path / "turns.json"
    path.write_text(json.dumps([
        {"segment_id": 0, "start_time": 1.0, "end_time": 2.0, "speaker_label": "SPEAKER_00"},
        {"segment_id": 1, "start_time": 2.0, "end_time": 3.5, "speaker_label": "SPEAKER_01"},
    ]))
    assert load_turns(path) == [(1.0, 2.0, "SPEAKER_00"), (2.0, 3.5, "SPEAKER_01")]


def test_one_label_per_person_passes_the_gate():
    hypothesis = [(0.0, 30.0, "S0"), (30.0, 60.0, "S1"), (60.0, 90.0, "S2")]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    passed, reasons = gate_verdict(report, max_minority=0.05)
    assert passed
    assert reasons == []


def test_a_label_holding_two_people_fails_the_gate():
    """The incumbent's shape: one label swallows two of the three people."""
    hypothesis = [(0.0, 60.0, "S0"), (60.0, 90.0, "S1")]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    passed, reasons = gate_verdict(report, max_minority=0.05)
    assert not passed
    assert any("S0" in reason for reason in reasons)


def test_reference_halves_split_windows_and_keep_the_moderator():
    """Slicing the flat turn list by parity would strip every moderator turn.
    Halving by WINDOW keeps each half a whole, self-contained reference."""
    windows = [
        [(0.0, 5.0, "MODERATOR"), (5.0, 30.0, "BOND")],
        [(30.0, 35.0, "MODERATOR"), (35.0, 60.0, "KOBIAN")],
        [(60.0, 65.0, "MODERATOR"), (65.0, 90.0, "BOND")],
    ]
    assert len(reference_half(windows, "all")) == 6
    assert reference_half(windows, "tune") == windows[1]
    assert reference_half(windows, "holdout") == windows[0] + windows[2]
    for half in ("tune", "holdout"):
        assert "MODERATOR" in {p for _, _, p in reference_half(windows, half)}


def test_fragmentation_alone_does_not_fail_the_gate():
    """An extra unnamed speaker costs the reviewer seconds; a silent merge
    misattributes quotes. The gate is asymmetric on purpose."""
    hypothesis = [(0.0, 15.0, "S0"), (15.0, 30.0, "S3"),
                  (30.0, 60.0, "S1"), (60.0, 90.0, "S2")]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    passed, reasons = gate_verdict(report, max_minority=0.05)
    assert passed
    assert [f.person for f in report.fragmentation] == ["BOND"]
