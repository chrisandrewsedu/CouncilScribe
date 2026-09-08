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
    """UPDATED for Task 5b: a trailing third handoff is added purely to BOUND
    the KOBIAN window — an unbounded (final) window is now dropped outright
    (test_the_unbounded_final_window_is_dropped), so without it this window
    would vanish from the result."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question. What would you do?"),
        seg(1, 5.0, 25.0, "I would start by reviewing the ledgers."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 30.0, 50.0, "I would begin with the payroll system."),
        seg(4, 50.0, 55.0, "Ms. Bond, same question."),
        seg(5, 55.0, 60.0, "Closing thought."),
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
    329/330, and losing it lets two windows over-run by ~135s each.

    UPDATED for Task 5b: a trailing handoff is added purely to BOUND the
    KOBIAN window — an unbounded (final) window is now dropped outright."""
    segments = [
        seg(0, 0.0, 5.0, "We will now move to closing remarks."),
        seg(1, 5.0, 6.0, "Kobian."),
        seg(2, 6.0, 40.0, "Something you don't know about me is..."),
        seg(3, 40.0, 45.0, "Ms. Bond, same question."),
        seg(4, 45.0, 50.0, "Her turn."),
    ]
    anchors = find_anchors(segments, SPEAKERS)
    assert anchors == [(1, "KOBIAN"), (3, "BOND")]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert turns[-1] == (6.0, 40.0, "KOBIAN")


def test_a_lookahead_anchor_is_not_anchored_a_second_time_by_the_outer_loop():
    """DEFECT (round-1 review). The cue segment names nobody, so the lookahead
    branch anchors its successor. But if that successor ALSO independently
    matches HANDOFF (as well as naming a person), the outer loop's own visit to
    that same index must not anchor it again — otherwise the reference carries
    one span twice. Measured on the real meeting at segments 10/11 ('...opening
    statement.' / 'ask candidate Andy Bond to start.') and 106/107 ('...' /
    'Let's start with Miss Bond.').

    UPDATED for Task 5b: a trailing handoff is added purely to BOUND the BOND
    window — an unbounded (final) window is now dropped outright."""
    segments = [
        seg(0, 0.0, 5.0, "We will now move to her opening statement."),
        seg(1, 5.0, 6.0, "Let's start with Miss Bond."),
        seg(2, 6.0, 40.0, "Thank you for having me."),
        seg(3, 40.0, 45.0, "Miss Cobian, same question."),
        seg(4, 45.0, 50.0, "Her turn."),
    ]
    anchors = find_anchors(segments, SPEAKERS)
    assert anchors == [(1, "BOND"), (3, "KOBIAN")]
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
    anchor — that is what made a naive build give one candidate 2007s.

    UPDATED for Task 5b: a second, in-range handoff is added purely to BOUND
    the BOND window (an unbounded/final window is now dropped outright); the
    sheriff's segment past end_time is excluded from becoming a third anchor
    by `find_anchors`'s own end_time check, exercising the same guarantee."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 25.0, "My answer."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 100.0, 200.0, "Hello again, I'm the sheriff."),
    ]
    turns = anchor_reference_turns(segments, SPEAKERS, end_time=50.0)
    assert turns == [(0.0, 5.0, "MODERATOR"), (5.0, 25.0, "BOND")]


def test_text_before_the_first_anchor_is_uncovered():
    """UPDATED for Task 5b: a trailing handoff is added purely to BOUND the
    BOND window — an unbounded (final) window is now dropped outright."""
    segments = [
        seg(0, 0.0, 60.0, "Welcome everyone to the candidate forum."),
        seg(1, 60.0, 65.0, "Ms. Bond, same question."),
        seg(2, 65.0, 90.0, "My answer."),
        seg(3, 90.0, 95.0, "Miss Cobian, same question."),
        seg(4, 95.0, 120.0, "Her answer."),
    ]
    turns = anchor_reference_turns(segments, SPEAKERS)
    assert (0.0, 60.0, "MODERATOR") not in turns
    assert turns[0] == (60.0, 65.0, "MODERATOR")


def test_windows_are_returned_one_per_anchor():
    """UPDATED for Task 5b: a third, trailing handoff is added purely to BOUND
    the second (KOBIAN) window — an unbounded (final) window is now dropped
    outright (see test_the_unbounded_final_window_is_dropped)."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 25.0, "My answer."),
        seg(2, 25.0, 30.0, "Miss Cobian, same question."),
        seg(3, 30.0, 50.0, "Her answer."),
        seg(4, 50.0, 55.0, "Ms. Bond, same question."),
        seg(5, 55.0, 80.0, "Her second answer."),
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


from bench.forum_anchor_reference import is_moderator_speech


def test_a_read_aloud_question_is_moderator_speech():
    """The moderator names the candidate, THEN reads the question out. Turn 142 of
    the real meeting is 'How would you make county financial information more
    understandable and accessible to residents?' — moderator, not candidate."""
    assert is_moderator_speech(seg(0, 0.0, 5.0,
        "How would you make county financial information more accessible to residents?"))


def test_a_procedural_aside_is_moderator_speech():
    assert is_moderator_speech(seg(0, 0.0, 5.0,
        "Seeing no further questions from the audience, we will now move to closing remarks."))


def test_an_answer_is_not_moderator_speech():
    assert not is_moderator_speech(seg(0, 0.0, 5.0,
        "So I think my experience with the Treasurer's office will help me transition."))


def test_the_question_preamble_after_an_anchor_is_not_given_to_the_candidate():
    """Anchor names the candidate; the next turn is still the moderator reading the
    question. Only the answer belongs to the candidate."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, this next question is for you."),
        seg(1, 5.0, 11.0, "How will you ensure accurate financial reporting?"),
        seg(2, 11.0, 60.0, "I would start by reconciling the ledgers every month."),
        seg(3, 60.0, 65.0, "Miss Cobian, same question."),
        seg(4, 65.0, 110.0, "I would begin with the payroll system."),
    ]
    turns = anchor_reference_windows(segments, SPEAKERS)[0]
    assert (5.0, 11.0, "BOND") not in turns
    assert (11.0, 60.0, "BOND") in turns


def test_the_window_ends_when_the_moderator_retakes_the_floor():
    """NOTE: seg(2)'s text was corrected from the brief's verbatim "And we will
    move on to our next question." to end in a question mark — the literal
    text didn't actually match MODERATOR_SPEECH (no HANDOFF phrase, no trailing
    "?"), which made this test fail even with the implementation exactly as
    specified. See task-5b-report.md."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 40.0, "My answer runs for a while."),
        seg(2, 40.0, 50.0, "Shall we move on to our next question?"),
        seg(3, 50.0, 80.0, "More speech that we cannot attribute."),
        seg(4, 80.0, 85.0, "Miss Cobian, same question."),
        seg(5, 85.0, 120.0, "Her answer."),
    ]
    turns = anchor_reference_windows(segments, SPEAKERS)[0]
    assert (5.0, 40.0, "BOND") in turns
    assert not any(start >= 40.0 for start, _, person in turns if person == "BOND")


def test_the_unbounded_final_window_is_dropped():
    """The last anchor has no next anchor to bound it, so its window would run to
    end_time and absorb whatever follows. On the real meeting that swallowed 141.9s
    of the moderator's closing script and called it BOND."""
    segments = [
        seg(0, 0.0, 5.0, "Ms. Bond, same question."),
        seg(1, 5.0, 30.0, "My answer."),
        seg(2, 30.0, 35.0, "Miss Cobian, same question."),
        seg(3, 35.0, 60.0, "Her answer."),
        seg(4, 60.0, 200.0, "Thank you all for coming, please verify your voter registration."),
    ]
    windows = anchor_reference_windows(segments, SPEAKERS)
    assert len(windows) == 1
    assert not any(person == "KOBIAN" for _, _, person in windows[0])
