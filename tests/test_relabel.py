"""Tests for per-segment relabelling — the capability merge_speakers cannot provide.

`review.merge_speakers` is all-or-nothing to ONE target, so it cannot undo a
wrong merge or hand a single stolen span back to its real owner. Both are needed
to repair what `mismerge_scan.py` finds: a label holding two people, sometimes
with one named SEGMENT spanning both voices (2026-07-09 publishes a 99s segment
holding an audience question AND the candidate's whole answer).

Cutting a segment is only faithful when the segment carries word timings, so a
text-only segment is refused rather than guessed at.
"""

from __future__ import annotations

import pytest

from src.models import Segment, Word


def _words(spec):
    """[(word, start, end), ...] -> [Word]"""
    return [Word(word=w, start=s, end=e) for w, s, e in spec]


def _seg(seg_id, start, end, label, words=None, text=None, name=None):
    word_list = _words(words or [])
    return Segment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        speaker_label=label,
        text=text if text is not None else " ".join(w.word for w in word_list),
        words=word_list,
        speaker_name=name,
    )


# --- cutting one segment ----------------------------------------------------

def test_split_segment_divides_words_at_the_cut():
    from src.relabel import split_segment

    segment = _seg(4, 10.0, 20.0, "SPEAKER_03", words=[
        ("what's", 10.0, 10.5), ("your", 10.6, 10.9), ("question?", 11.0, 11.8),
        ("Well,", 15.0, 15.4), ("I", 15.5, 15.6), ("grew", 15.7, 16.0),
    ])

    before, after = split_segment(segment, 14.0)

    assert before.text == "what's your question?"
    assert after.text == "Well, I grew"
    assert (before.start_time, before.end_time) == (10.0, 14.0)
    assert (after.start_time, after.end_time) == (14.0, 20.0)


def test_split_segment_keeps_every_word():
    from src.relabel import split_segment

    segment = _seg(0, 0.0, 10.0, "SPEAKER_00", words=[
        ("a", 0.0, 1.0), ("b", 2.0, 3.0), ("c", 6.0, 7.0), ("d", 8.0, 9.0),
    ])

    before, after = split_segment(segment, 5.0)

    assert [w.word for w in before.words] + [w.word for w in after.words] == ["a", "b", "c", "d"]


def test_split_segment_carries_the_speaker_over_to_both_halves():
    from src.relabel import split_segment

    segment = _seg(7, 0.0, 10.0, "SPEAKER_03", words=[("x", 0.0, 1.0), ("y", 6.0, 7.0)],
                   name="Unidentified Speaker")

    before, after = split_segment(segment, 5.0)

    for half in (before, after):
        assert half.speaker_label == "SPEAKER_03"
        assert half.speaker_name == "Unidentified Speaker"


def test_split_segment_refuses_a_cut_outside_the_segment():
    from src.relabel import split_segment

    segment = _seg(0, 10.0, 20.0, "SPEAKER_00", words=[("x", 10.0, 11.0)])

    for at in (10.0, 20.0, 5.0, 25.0):
        with pytest.raises(ValueError, match="outside"):
            split_segment(segment, at)


def test_split_segment_refuses_a_segment_with_no_word_timings():
    from src.relabel import split_segment

    # Without word times there is no honest place to divide the text, and
    # guessing would fabricate who said what — the exact failure being repaired.
    segment = _seg(0, 10.0, 20.0, "SPEAKER_00", words=[], text="some words here")

    with pytest.raises(ValueError, match="word timings"):
        split_segment(segment, 15.0)


def test_split_segment_refuses_a_cut_that_would_empty_a_half():
    from src.relabel import split_segment

    # Every word on one side means the cut separates nothing; the caller's span
    # is wrong and should be reported, not silently turned into a no-op.
    segment = _seg(0, 10.0, 20.0, "SPEAKER_00", words=[("x", 18.0, 19.0)])

    with pytest.raises(ValueError, match="no words"):
        split_segment(segment, 15.0)


# --- planning a relabel from time spans -------------------------------------

def _q_and_a():
    """2026-07-09's shape: one segment holding a question AND the answer."""
    return [
        _seg(0, 3002.1, 3006.6, "SPEAKER_05", words=[("Jamie,", 3002.1, 3002.5)]),
        _seg(1, 3006.8, 3106.0, "SPEAKER_03", words=[
            ("it's", 3006.8, 3007.0), ("for", 3007.1, 3007.3),
            ("Perry", 3007.4, 3007.8), ("Johnson.", 3007.9, 3008.4),
            ("Well,", 3029.6, 3030.0), ("I", 3030.1, 3030.2),
            ("grew", 3030.3, 3030.7), ("up", 3030.8, 3031.0),
        ]),
    ]


def test_plan_relabel_cuts_a_segment_that_holds_two_voices():
    from src.relabel import plan_relabel

    plan = plan_relabel(_q_and_a(), [(3029.6, 3106.0)], "SPEAKER_04")

    assert len(plan.moves) == 1
    move = plan.moves[0]
    assert move.index == 1
    assert move.from_label == "SPEAKER_03"
    assert move.to_label == "SPEAKER_04"
    assert move.whole is False
    assert (move.start, move.end) == (3029.6, 3106.0)


def test_plan_relabel_moves_a_wholly_covered_segment_without_cutting():
    from src.relabel import plan_relabel

    segments = [
        _seg(0, 0.0, 10.0, "SPEAKER_05", words=[("mine", 0.0, 1.0)]),
        _seg(1, 20.0, 30.0, "SPEAKER_05", words=[("theirs", 20.0, 21.0)]),
    ]

    plan = plan_relabel(segments, [(20.0, 30.0)], "SPEAKER_09")

    assert [(m.index, m.whole) for m in plan.moves] == [(1, True)]


def test_plan_relabel_ignores_segments_no_span_touches():
    from src.relabel import plan_relabel

    segments = [_seg(0, 0.0, 10.0, "SPEAKER_05", words=[("mine", 0.0, 1.0)])]

    assert plan_relabel(segments, [(100.0, 200.0)], "SPEAKER_09").moves == []


def test_plan_relabel_skips_a_segment_already_on_the_target():
    from src.relabel import plan_relabel

    # Re-running a repair must be a no-op, not a second round of cuts.
    segments = [_seg(0, 0.0, 10.0, "SPEAKER_09", words=[("theirs", 0.0, 1.0)])]

    assert plan_relabel(segments, [(0.0, 10.0)], "SPEAKER_09").moves == []


def test_plan_relabel_absorbs_a_sliver_instead_of_cutting_it_off():
    from src.relabel import plan_relabel

    # Spans come from RAW turn boundaries, which sit near but not exactly on the
    # named ones (boundary-snap moves words across them). A 0.1s residual is
    # that mismatch, not a piece of another person.
    segments = [_seg(0, 20.0, 30.0, "SPEAKER_05", words=[("theirs", 20.0, 21.0)])]

    plan = plan_relabel(segments, [(20.1, 30.0)], "SPEAKER_09", min_piece_seconds=0.25)

    assert [(m.index, m.whole) for m in plan.moves] == [(0, True)]


def test_plan_relabel_handles_a_span_inside_a_segment():
    from src.relabel import plan_relabel

    segments = [_seg(0, 0.0, 30.0, "SPEAKER_05", words=[
        ("mine", 0.0, 1.0), ("theirs", 12.0, 13.0), ("mine-again", 25.0, 26.0),
    ])]

    plan = plan_relabel(segments, [(10.0, 20.0)], "SPEAKER_09")

    move = plan.moves[0]
    assert move.whole is False
    assert (move.start, move.end) == (10.0, 20.0)


def test_plan_relabel_merges_adjacent_spans_before_planning():
    from src.relabel import plan_relabel

    # mismerge_scan reports one span per RAW turn, so a presenter's run arrives
    # as dozens of near-contiguous spans. Planning a cut per span would shred
    # the segment; the run is one piece.
    segments = [_seg(0, 0.0, 30.0, "SPEAKER_05", words=[
        ("a", 0.0, 1.0), ("b", 11.0, 12.0), ("c", 13.0, 14.0), ("d", 25.0, 26.0),
    ])]

    plan = plan_relabel(segments, [(10.0, 12.5), (12.6, 20.0)], "SPEAKER_09")

    assert len(plan.moves) == 1
    assert (plan.moves[0].start, plan.moves[0].end) == (10.0, 20.0)


# --- applying it ------------------------------------------------------------

def test_apply_plan_relabels_a_wholly_moved_segment():
    from src.relabel import apply_plan, plan_relabel

    segments = [
        _seg(0, 0.0, 10.0, "SPEAKER_05", words=[("mine", 0.0, 1.0)]),
        _seg(1, 20.0, 30.0, "SPEAKER_05", words=[("theirs", 20.0, 21.0)]),
    ]

    result = apply_plan(segments, plan_relabel(segments, [(20.0, 30.0)], "SPEAKER_09"))

    assert [s.speaker_label for s in result] == ["SPEAKER_05", "SPEAKER_09"]


def test_apply_plan_cuts_and_moves_only_the_covered_half():
    from src.relabel import apply_plan, plan_relabel

    segments = _q_and_a()
    result = apply_plan(segments, plan_relabel(segments, [(3029.6, 3106.0)], "SPEAKER_04"))

    assert len(result) == 3
    question, answer = result[1], result[2]
    assert question.speaker_label == "SPEAKER_03"
    assert question.text == "it's for Perry Johnson."
    assert answer.speaker_label == "SPEAKER_04"
    assert answer.text == "Well, I grew up"


def test_apply_plan_renumbers_segment_ids_contiguously():
    from src.relabel import apply_plan, plan_relabel

    segments = _q_and_a()
    result = apply_plan(segments, plan_relabel(segments, [(3029.6, 3106.0)], "SPEAKER_04"))

    assert [s.segment_id for s in result] == [0, 1, 2]


def test_apply_plan_leaves_the_input_untouched():
    from src.relabel import apply_plan, plan_relabel

    # merge_adjacent_segments mutates what it is handed and that has already
    # made valid summaries look stale once. A dry run must be able to plan and
    # apply on a copy without disturbing the loaded meeting.
    segments = _q_and_a()
    plan = plan_relabel(segments, [(3029.6, 3106.0)], "SPEAKER_04")

    apply_plan(segments, plan)

    assert len(segments) == 2
    assert segments[1].speaker_label == "SPEAKER_03"
    assert segments[1].end_time == 3106.0


# --- deriving spans from provenance -----------------------------------------

def test_spans_for_raw_label_returns_that_labels_own_time():
    from src.relabel import spans_for_raw_label

    # The repair input is "whatever raw SPEAKER_03 said", which is exactly what
    # mismerge_scan flags. Deriving the spans here keeps the two tools honest
    # about meaning the same turns.
    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_05", words=[("mine", 0.0, 1.0)]),
        _seg(1, 10.0, 20.0, "SPEAKER_03", words=[("theirs", 10.0, 11.0)]),
        _seg(2, 20.0, 30.0, "SPEAKER_05", words=[("mine", 20.0, 21.0)]),
    ]
    named = [_seg(0, 0.0, 30.0, "SPEAKER_05", words=[
        ("mine", 0.0, 1.0), ("theirs", 10.0, 11.0), ("mine", 20.0, 21.0),
    ])]

    assert spans_for_raw_label(raw, named, "SPEAKER_03") == [(10.0, 20.0)]


def test_spans_for_raw_label_is_empty_for_an_unknown_label():
    from src.relabel import spans_for_raw_label

    raw = [_seg(0, 0.0, 10.0, "SPEAKER_05", words=[("mine", 0.0, 1.0)])]

    assert spans_for_raw_label(raw, raw, "SPEAKER_99") == []


# --- minting a label for the person being split out -------------------------

def test_next_free_label_skips_every_label_in_use():
    from src.relabel import next_free_label

    segments = [
        _seg(0, 0.0, 1.0, "SPEAKER_00"),
        _seg(1, 1.0, 2.0, "SPEAKER_02"),
    ]

    assert next_free_label(segments, {"SPEAKER_01": object()}) == "SPEAKER_03"


def test_next_free_label_counts_labels_with_no_segments_left():
    from src.relabel import next_free_label

    # A speakers entry with no segments is still a claimed label; reusing its
    # number would attach the split-out person to a stranger's mapping.
    assert next_free_label([], {"SPEAKER_00": object(), "SPEAKER_01": object()}) == "SPEAKER_02"


def test_plan_relabel_previews_only_the_text_that_moves():
    from src.relabel import plan_relabel

    # A diff that shows the whole segment's text hides what is actually being
    # moved — on 2026-07-09 that meant showing the audience question while the
    # move was the candidate's answer.
    segments = [_seg(0, 0.0, 30.0, "SPEAKER_03", words=[
        ("who's", 0.0, 0.5), ("asking?", 0.6, 1.0),
        ("Well,", 15.0, 15.4), ("I", 15.5, 15.6), ("grew", 15.7, 16.0),
    ])]

    move = plan_relabel(segments, [(14.0, 30.0)], "SPEAKER_04").moves[0]

    assert move.text == "Well, I grew"


def test_plan_relabel_previews_the_whole_text_when_the_whole_segment_moves():
    from src.relabel import plan_relabel

    segments = [_seg(0, 0.0, 10.0, "SPEAKER_03", words=[
        ("all", 0.0, 0.5), ("of", 0.6, 0.8), ("it", 0.9, 1.0),
    ])]

    assert plan_relabel(segments, [(0.0, 10.0)], "SPEAKER_04").moves[0].text == "all of it"


def test_plan_relabel_skips_an_empty_boundary_segment():
    from src.relabel import plan_relabel

    # Publish-era artefacts: zero-length, wordless segments sitting exactly on a
    # boundary. Moving them churns the transcript and, worse, drags a THIRD
    # label into a two-label repair — the 2026-07-08 dry run pulled 0.0s stubs
    # off SPEAKER_02 and SPEAKER_12 while splitting SPEAKER_05.
    segments = [
        _seg(0, 100.0, 100.0, "SPEAKER_02", words=[], text=""),
        _seg(1, 100.0, 100.1, "SPEAKER_12", words=[], text=""),
        _seg(2, 100.1, 130.0, "SPEAKER_05", words=[("real", 100.1, 101.0)]),
    ]

    plan = plan_relabel(segments, [(100.0, 130.0)], "SPEAKER_09")

    assert [m.index for m in plan.moves] == [2]


def test_plan_relabel_keeps_a_short_segment_that_has_words():
    from src.relabel import plan_relabel

    # A brief but real turn ("Yep.") is not an artefact.
    segments = [_seg(0, 100.0, 100.4, "SPEAKER_05", words=[("Yep.", 100.0, 100.3)])]

    assert len(plan_relabel(segments, [(100.0, 101.0)], "SPEAKER_09").moves) == 1


def test_next_free_label_avoids_a_number_the_raw_transcript_uses():
    from src.relabel import next_free_label

    # Provenance depends on raw and named label NUMBERS meaning the same person.
    # Minting named SPEAKER_00 for a new person while raw SPEAKER_00 was someone
    # else would make mismerge_scan attribute this split-out voice to them.
    segments = [_seg(0, 0.0, 1.0, "SPEAKER_01")]

    assert next_free_label(segments, {}, reserved={"SPEAKER_00"}) == "SPEAKER_02"


def test_plan_relabel_treats_a_wordless_head_as_a_whole_move():
    from src.relabel import plan_relabel

    # Real case from the 2026-07-08 dry run: the span starts inside the segment
    # but every word starts after it, so there is nothing to leave behind. A
    # "cut" there has no words on one side and split_segment rightly refuses —
    # the plan must call it a whole move instead of proposing an impossible cut.
    segments = [_seg(0, 5990.0, 6010.0, "SPEAKER_05", words=[
        ("Are", 5998.0, 5998.3), ("you", 5998.4, 5998.6),
    ])]

    plan = plan_relabel(segments, [(5997.5, 6010.0)], "SPEAKER_14")

    assert [(m.whole, m.start) for m in plan.moves] == [(True, 5990.0)]


def test_plan_relabel_treats_a_wordless_tail_as_a_whole_move():
    from src.relabel import plan_relabel

    segments = [_seg(0, 100.0, 130.0, "SPEAKER_05", words=[
        ("hello", 100.0, 100.5), ("there", 100.6, 101.0),
    ])]

    plan = plan_relabel(segments, [(100.0, 110.0)], "SPEAKER_14")

    assert [(m.whole, m.end) for m in plan.moves] == [(True, 130.0)]


def test_plan_relabel_skips_a_span_that_covers_no_words():
    from src.relabel import plan_relabel

    # The span lands in a silent gap between words. Cutting there would leave an
    # empty middle segment; there is nothing to move.
    segments = [_seg(0, 0.0, 30.0, "SPEAKER_05", words=[
        ("start", 0.0, 1.0), ("end", 28.0, 29.0),
    ])]

    assert plan_relabel(segments, [(10.0, 20.0)], "SPEAKER_14").moves == []


# --- several spans inside one segment ---------------------------------------

def _two_runs_one_segment():
    """2026-07-14's real shape: ONE 69s named segment holding reporter
    narration, then the candidate, then narration, then the candidate again."""
    return [_seg(0, 96.3, 165.2, "SPEAKER_05", words=[
        ("Republican", 96.3, 96.9), ("challenger", 97.0, 97.6),
        ("This", 103.2, 103.5), ("is", 103.6, 103.7), ("my", 103.8, 103.9),
        ("theme", 104.0, 104.4),
        ("Michigan's", 111.4, 112.0), ("income", 112.1, 112.6),
        ("tax", 112.7, 113.0),
        ("I", 126.8, 127.0), ("suggested", 127.1, 127.7),
        ("the", 127.8, 127.9), ("audit", 128.0, 128.5),
        ("Congressman", 144.7, 145.4), ("James", 145.5, 146.0),
    ])]


def test_plan_relabel_reports_every_run_inside_one_segment():
    from src.relabel import plan_relabel

    plan = plan_relabel(_two_runs_one_segment(),
                        [(103.154, 111.389), (126.779, 144.582)], "SPEAKER_03")

    assert len(plan.moves) == 2
    assert [m.index for m in plan.moves] == [0, 0]
    assert [round(m.start, 1) for m in plan.moves] == [103.2, 126.8]


def test_plan_relabel_does_not_extend_an_inner_run_to_the_segment_edge():
    from src.relabel import plan_relabel

    # Edge absorption exists for raw-vs-named boundary mismatch. Applying it to
    # a run in the MIDDLE would swallow the other speaker either side of it.
    plan = plan_relabel(_two_runs_one_segment(),
                        [(103.154, 111.389), (126.779, 144.582)], "SPEAKER_03")

    assert all(m.whole is False for m in plan.moves)
    assert plan.moves[0].start > 96.3
    assert plan.moves[1].end < 165.2


def test_apply_plan_cuts_one_segment_into_alternating_owners():
    from src.relabel import apply_plan, plan_relabel

    segments = _two_runs_one_segment()
    plan = plan_relabel(segments, [(103.154, 111.389), (126.779, 144.582)], "SPEAKER_03")

    result = apply_plan(segments, plan)

    assert [s.speaker_label for s in result] == [
        "SPEAKER_05", "SPEAKER_03", "SPEAKER_05", "SPEAKER_03", "SPEAKER_05",
    ]
    assert result[1].text == "This is my theme"
    assert result[3].text == "I suggested the audit"


def test_apply_plan_keeps_every_word_across_several_cuts():
    from src.relabel import apply_plan, plan_relabel

    segments = _two_runs_one_segment()
    original = [w.word for w in segments[0].words]
    plan = plan_relabel(segments, [(103.154, 111.389), (126.779, 144.582)], "SPEAKER_03")

    result = apply_plan(segments, plan)

    assert [w.word for s in result for w in s.words] == original


# --- the write path: the embedded summary must not be left stale ------------

def _payload_with_summary():
    """A transcript_named.json shaped payload whose summary indexes its segments.

    Shaped so a cut in the middle of segment 1 genuinely shifts the section's
    indices: the two SPEAKER_00 pieces either side of the moved span cannot
    re-merge across it, but the tail piece CAN merge with segment 2, so the
    section that pointed at index 2 ends up at index 3.
    """
    segments = [
        _seg(0, 0.0, 10.0, "SPEAKER_09", words=[("intro", 0.0, 1.0)], name="Z"),
        _seg(1, 10.0, 30.0, "SPEAKER_00", words=[
            ("mine", 10.0, 11.0), ("theirs", 15.0, 16.0), ("again", 25.0, 26.0),
        ], name="A"),
        _seg(2, 30.0, 40.0, "SPEAKER_00", words=[("last", 30.0, 31.0)], name="A"),
    ]
    return {
        "meeting_id": "m1",
        "city": None,
        "date": "2026-01-01",
        "segments": [s.to_dict() for s in segments],
        "speakers": {
            "SPEAKER_09": {"speaker_label": "SPEAKER_09", "speaker_name": "Z"},
            "SPEAKER_00": {"speaker_label": "SPEAKER_00", "speaker_name": "A"},
        },
        "summary": {
            "executive_summary": "x",
            "sections": [{
                "section_type": "discussion", "title": "T", "content": "c",
                "start_time": 30.0, "end_time": 40.0,
                "start_segment": 2, "end_segment": 2,
            }],
        },
    }


def test_relabel_payload_reindexes_the_embedded_summary_after_a_cut():
    from src.relabel import relabel_payload

    # Cutting segment 1 adds rows, so the section that pointed at index 2 now
    # points at the wrong segment. Leaving it is exactly the staleness trap
    # that has made valid summaries look broken before — and editing the raw
    # dict instead of going through Meeting would skip this silently.
    result = relabel_payload(_payload_with_summary(), [(14.0, 20.0)], "SPEAKER_01")

    section = result["payload"]["summary"]["sections"][0]
    assert (section["start_segment"], section["end_segment"]) == (3, 3)
    assert result["sections_reindexed"] == 1


def test_relabel_payload_moves_the_span_and_names_the_new_label():
    from src.relabel import relabel_payload

    result = relabel_payload(_payload_with_summary(), [(14.0, 20.0)], "SPEAKER_01",
                             name="B")

    labels = [s["speaker_label"] for s in result["payload"]["segments"]]
    assert labels == ["SPEAKER_09", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert result["payload"]["speakers"]["SPEAKER_01"]["speaker_name"] == "B"
    moved = result["payload"]["segments"][2]
    assert moved["speaker_name"] == "B"
    assert moved["text"] == "theirs"


def test_relabel_payload_flags_an_unnamed_new_label_for_review():
    from src.relabel import relabel_payload

    result = relabel_payload(_payload_with_summary(), [(14.0, 20.0)], "SPEAKER_01")

    assert result["payload"]["speakers"]["SPEAKER_01"]["needs_review"] is True
    assert result["payload"]["speakers"]["SPEAKER_01"].get("speaker_name") is None


def test_relabel_payload_drops_a_label_left_with_no_segments():
    from src.relabel import relabel_payload

    payload = _payload_with_summary()
    result = relabel_payload(payload, [(0.0, 40.0)], "SPEAKER_01", name="B")

    assert "SPEAKER_00" not in result["payload"]["speakers"]
    assert result["emptied"] == ["SPEAKER_00", "SPEAKER_09"]


def test_relabel_payload_preserves_keys_it_does_not_touch():
    from src.relabel import relabel_payload

    payload = _payload_with_summary()
    payload["event_kind"] = "debate"
    payload["race_id"] = "r1"

    result = relabel_payload(payload, [(14.0, 20.0)], "SPEAKER_01")

    assert result["payload"]["event_kind"] == "debate"
    assert result["payload"]["race_id"] == "r1"
