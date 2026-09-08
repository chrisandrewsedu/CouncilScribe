"""Tests for building a diarization reference from a forum moderator's handoffs.

The moderator of a candidate forum names who speaks next ("Ms. Bond, same
question"). That makes a per-turn reference derivable from TEXT alone, with no
voice model involved — which is the only way to score a clustering when the
label under test is the one that swallowed three people.
"""
import re

from bench.forum_anchor_reference import (
    LWV_AUDITOR_FORUM_END,
    LWV_AUDITOR_SPEAKERS,
    anchor_reference_turns,
    anchor_reference_windows,
    find_anchors,
)

SPEAKERS = {
    "BOND": re.compile(r"\b(ms\.?|miss|mrs\.?|candidate|andy|vasquez)\s+(vasquez\s+)?bond\b", re.I),
    "KOBIAN": re.compile(r"\b(kobian|cobian|kobe|koby|teresa)\b", re.I),
}


def seg(i, start, end, text):
    return {"segment_id": i, "start_time": start, "end_time": end,
            "speaker_label": "SPEAKER_09", "text": text}


def test_a_named_handoff_gives_the_following_turns_to_that_person():
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question. What would you do?"),
        seg(1, 5.0, 25.0, "I would start by reviewing the ledgers."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 30.0, 50.0, "I would begin with the payroll system."),
    ]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert turns == [
        (0.0, 5.0, "MODERATOR"),
        (5.0, 25.0, "BOND"),
        (25.0, 30.0, "MODERATOR"),
        (30.0, 50.0, "KOBIAN"),
    ]


def test_a_handoff_split_across_two_turns_still_anchors():
    """DEFECT 1. Diarized turns cut mid-sentence, so the cue and the name land in
    different turns: '...we will now move to closing remarks.' / 'Kobian.'
    Measured on the real meeting: this shape occurs at turns 217/218 and
    329/330, and losing it lets two windows over-run by ~135s each."""
    segments = [
        seg(0, 0.0, 5.0, "We will now move to closing remarks."),
        seg(1, 5.0, 6.0, "Kobian."),
        seg(2, 6.0, 40.0, "Something you don't know about me is..."),
    ]
    anchors = find_anchors(segments, SPEAKERS)
    assert anchors == [(1, "KOBIAN")]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert turns[-1] == (6.0, 40.0, "KOBIAN")


def test_a_lookahead_anchor_is_not_anchored_a_second_time_by_the_outer_loop():
    """DEFECT (round-1 review). The cue segment names nobody, so the lookahead
    branch anchors its successor. But if that successor ALSO independently
    matches HANDOFF (as well as naming a person), the outer loop's own visit to
    that same index must not anchor it again — otherwise the reference carries
    one span twice. Measured on the real meeting at segments 10/11 ('...opening
    statement.' / 'ask candidate Andy Bond to start.') and 106/107 ('...' /
    'Let's start with Miss Bond.')."""
    segments = [
        seg(0, 0.0, 5.0, "We will now move to her opening statement."),
        seg(1, 5.0, 6.0, "Let's start with Miss Bond."),
        seg(2, 6.0, 40.0, "Thank you for having me."),
    ]
    anchors = find_anchors(segments, SPEAKERS)
    assert anchors == [(1, "BOND")]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert turns == [
        (5.0, 6.0, "MODERATOR"),
        (6.0, 40.0, "BOND"),
    ]


def test_a_surname_that_is_also_a_common_noun_does_not_anchor():
    """DEFECT 2. 'Bond' is also a common noun in this meeting ('the county's
    debt bond rating'). Measured: turn 112 is a genuine handoff to Kobian whose
    text also says 'bond rating', so a bare \\bbond\\b match saw BOTH names,
    discarded the anchor, and gave Kobian's 138s answer to Bond."""
    segments = [
        seg(0, 0.0, 8.0, "Miss Cobian, same question. With the county's debt "
                         "bond rating recently decreased, what would you do?"),
        seg(1, 8.0, 60.0, "I have sat in on a lot of those meetings."),
    ]
    anchors = find_anchors(segments, SPEAKERS)
    assert anchors == [(0, "KOBIAN")]


def test_a_handoff_naming_both_candidates_is_discarded():
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond and Ms. Cobian, what is your view?"),
        seg(1, 5.0, 20.0, "Well, I think..."),
    ]
    assert find_anchors(segments, SPEAKERS) == []
    assert anchor_reference_turns(segments, SPEAKERS) == []


def test_turns_after_end_time_are_excluded():
    """The meet-and-greet has no handoffs, so it must not inherit the last
    anchor — that is what made a naive build give one candidate 2007s."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 25.0, "My answer."),
        seg(2, 100.0, 200.0, "Hello again, I'm the sheriff."),
    ]
    turns = anchor_reference_turns(segments, SPEAKERS, end_time=50.0)
    assert turns == [(0.0, 5.0, "MODERATOR"), (5.0, 25.0, "BOND")]


def test_text_before_the_first_anchor_is_uncovered():
    segments = [
        seg(0, 0.0, 60.0, "Welcome everyone to the candidate forum."),
        seg(1, 60.0, 65.0, "Ms. Bond, same question."),
        seg(2, 65.0, 90.0, "My answer."),
    ]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert (0.0, 60.0, "MODERATOR") not in turns
    assert turns[0] == (60.0, 65.0, "MODERATOR")


def test_windows_are_returned_one_per_anchor():
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 25.0, "My answer."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 30.0, 50.0, "Her answer."),
    ]
    windows = anchor_reference_windows(segments, SPEAKERS)
    assert len(windows) == 2
    assert windows[0] == [(0.0, 5.0, "MODERATOR"), (5.0, 25.0, "BOND")]
    assert windows[1] == [(25.0, 30.0, "MODERATOR"), (30.0, 50.0, "KOBIAN")]
    assert anchor_reference_turns(segments, SPEAKERS) == windows[0] + windows[1]


def test_every_window_carries_its_own_moderator_turn():
    """A tune/holdout split must slice WINDOWS, not the flat turn list. The flat
    list alternates moderator, person, moderator, person, so taking every other
    turn would hand calibration a reference containing no moderator at all — and
    the moderator is the label that swallowed both candidates."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 25.0, "My answer."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 30.0, 50.0, "Her answer."),
    ]
    for window in anchor_reference_windows(segments, SPEAKERS):
        assert "MODERATOR" in {person for _, _, person in window}


def test_the_shipped_lwv_constants_are_wired_up():
    assert LWV_AUDITOR_FORUM_END == 2650.0
    assert set(LWV_AUDITOR_SPEAKERS) == {"BOND", "KOBIAN"}
    assert LWV_AUDITOR_SPEAKERS["BOND"].search("Ms. Bond, same question")
    assert not LWV_AUDITOR_SPEAKERS["BOND"].search("the debt bond rating fell")
