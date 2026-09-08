"""Tests for the forum diarization gate."""
import json

from bench.identity_score import identity_report
from bench.forum_gate import (
    GATE_MAX_UNATTRIBUTED_SHARE,
    gate_verdict,
    load_turns,
    reference_half,
)

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
    passed, reasons = gate_verdict(report)
    assert passed
    assert reasons == []


def test_a_label_holding_two_people_fails_the_gate():
    """The incumbent's shape: one label swallows two of the three people."""
    hypothesis = [(0.0, 60.0, "S0"), (60.0, 90.0, "S1")]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    passed, reasons = gate_verdict(report)
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
    passed, reasons = gate_verdict(report)
    assert passed
    assert [f.person for f in report.fragmentation] == ["BOND"]


from bench.forum_recluster import UNCLUSTERED_LABEL


def test_the_unattributed_bucket_does_not_count_as_conflation_within_bound():
    """The bucket is where turns with too little voice evidence are parked. It holds
    slivers from many people BY CONSTRUCTION, so scoring it as a speaker identity
    would guarantee a failure and punish the design for being honest.

    The bound is self-scaling: GATE_MAX_UNATTRIBUTED_SHARE(3 people) = 1/4 =
    25%. Here the bucket holds only the back half of MODERATOR's last window
    (15s of the 90s scored reference, 16.7%) — under that bound — so this
    tests the EXCLUSION-from-conflation behaviour under the real production
    bound, not an inflated one. See
    test_an_oversized_unattributed_bucket_fails_the_gate for the same shape
    of bucket judged against a bound it exceeds."""
    hypothesis = [(0.0, 30.0, "S0"), (30.0, 60.0, "S1"),
                  (60.0, 75.0, "S2"), (75.0, 90.0, UNCLUSTERED_LABEL)]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    bound = GATE_MAX_UNATTRIBUTED_SHARE(report.reference_people)
    assert bound == 0.25
    passed, reasons = gate_verdict(report, unattributed_label=UNCLUSTERED_LABEL,
                                   max_unattributed_share=bound)
    assert passed, reasons


def test_a_real_label_still_counts_as_conflation():
    hypothesis = [(0.0, 60.0, "S0"), (60.0, 90.0, "S1")]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    bound = GATE_MAX_UNATTRIBUTED_SHARE(report.reference_people)
    passed, _ = gate_verdict(report, unattributed_label=UNCLUSTERED_LABEL,
                             max_unattributed_share=bound)
    assert not passed


def test_an_oversized_unattributed_bucket_fails_the_gate():
    """Same shape as the old passing case, but 30s of 90s scored reference
    (33%) is still past the self-scaling bound for this 3-person reference
    (GATE_MAX_UNATTRIBUTED_SHARE(3) = 25%), so the bucket itself must fail
    the gate even though it holds no single conflated label."""
    hypothesis = [(0.0, 30.0, "S0"), (30.0, 60.0, "S1"),
                  (60.0, 75.0, UNCLUSTERED_LABEL), (75.0, 90.0, UNCLUSTERED_LABEL)]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    bound = GATE_MAX_UNATTRIBUTED_SHARE(report.reference_people)
    passed, reasons = gate_verdict(report, unattributed_label=UNCLUSTERED_LABEL,
                                   max_unattributed_share=bound)
    assert not passed
    assert any(UNCLUSTERED_LABEL in reason for reason in reasons)


def test_folding_everything_into_the_bucket_fails_the_gate():
    """The gameable case this bound exists to close: fold every turn into the
    unattributed bucket — one label over the entire reference, a WORSE form
    of the defect this repair exists to fix — and the old exclusion-with-no-
    bound logic returned PASS with zero reasons. Share is 100%, over the
    self-scaling bound for any reference_people >= 0, so this must always
    fail."""
    hypothesis = [(0.0, 90.0, UNCLUSTERED_LABEL)]
    report = identity_report(hypothesis, REFERENCE, min_fraction=0.05)
    bound = GATE_MAX_UNATTRIBUTED_SHARE(report.reference_people)
    passed, reasons = gate_verdict(report, unattributed_label=UNCLUSTERED_LABEL,
                                   max_unattributed_share=bound)
    assert not passed
    assert reasons
