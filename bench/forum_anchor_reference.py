"""Build a diarization reference from a forum moderator's named handoffs.

`bench.identity_score.named_reference_turns` needs a human-reviewed
`transcript_named.json`. That is unavailable exactly when it is most needed: if
one diarized label swallowed three people, the reviewed transcript inherits the
error and cannot referee its own repair.

A moderated candidate forum supplies a second, independent source of truth. The
moderator names who speaks next before nearly every answer, so the window from
each named handoff to the next belongs to that person. That reference comes from
TEXT alone — no voice model, no embeddings — so it can referee a clustering
without circularity.

Measured on 2026-04-03-lwv-brown-county-candidate-forum-auditor: 34 anchors
covering 2068s of 2239s of forum speech (92%).

Pure: no torch, no Modal, no I/O beyond what the caller passes in.
"""

from __future__ import annotations

import re

from .identity_score import Turns

#: Phrases with which a moderator yields the floor. Deliberately broad: a false
#: positive that names nobody is discarded for free, while a missed handoff lets
#: an answer window over-run into the next person's answer, which is the
#: expensive error.
HANDOFF = re.compile(
    r"(same question|next question is for you|what about you|we'?ll start with|"
    r"let'?s start with|go first|you have your opening|opening statement|"
    r"closing remarks|to start|do you have any|are there any other|"
    r"what do you believe|what is your|what are the|how will you|"
    r"would you change|what'?s one area|how would you approach|can you expa)",
    re.I,
)

#: MEASURED. "Bond" is also a common noun in this meeting — the county's debt
#: bond rating is discussed at length — so the surname requires an honorific or
#: first name. Without that guard, turn 112 ("Miss Cobian, same question. With
#: the county's debt bond rating...") matches BOTH candidates, is discarded as
#: ambiguous, and hands Kobian's 138s answer to Bond. "Kobian" has no such
#: collision, so it stays loose and absorbs the ASR's variants (Cobian, Kobe).
LWV_AUDITOR_SPEAKERS: dict[str, re.Pattern] = {
    "BOND": re.compile(
        r"\b(ms\.?|miss|mrs\.?|candidate|andy|vasquez)\s+(vasquez\s+)?bond\b", re.I
    ),
    "KOBIAN": re.compile(r"\b(kobian|cobian|kobe|koby|teresa)\b", re.I),
}

#: The forum Q&A ends here; the meet-and-greet that follows has no handoffs and
#: therefore no reference. Stocksdale opens it at t=2696.5.
LWV_AUDITOR_FORUM_END = 2650.0

#: A turn is the moderator's, wherever it falls, if it carries a handoff cue or is
#: interrogative. MEASURED: the moderator names the candidate and THEN reads the
#: question out, so the turn after an anchor is usually still the moderator; and the
#: moderator makes procedural asides mid-window ("we will move on to our next
#: question"). Attributing question-asking to the person answering is wrong by ROLE,
#: independent of any score.
MODERATOR_SPEECH = re.compile(HANDOFF.pattern + r"|\?\s*$", re.I)


def is_moderator_speech(segment: dict) -> bool:
    """True if this turn reads as the moderator asking or managing the floor."""
    return bool(MODERATOR_SPEECH.search((segment.get("text") or "").strip()))


def _named(text: str, speakers: dict[str, re.Pattern]) -> str | None:
    """The one person this text names, or None if it names none or several."""
    hits = [person for person, pattern in speakers.items() if pattern.search(text)]
    return hits[0] if len(hits) == 1 else None


def find_anchors(
    segments: list[dict],
    speakers: dict[str, re.Pattern],
    *,
    handoff: re.Pattern = HANDOFF,
    end_time: float | None = None,
) -> list[tuple[int, str]]:
    """Locate (segment index, person) for each handoff that names one person.

    A cue and the name it carries can land in different segments, because
    diarized turns cut mid-sentence ("...move to closing remarks." / "Kobian.").
    When the cue segment names nobody, its successor is consulted and, if that
    one names a person, becomes the anchor — the floor is yielded when the name
    is spoken, not when the cue began.

    A segment claimed via that lookahead is not eligible to anchor again on its
    own turn through the outer loop's direct-match branch: if segment N names
    nobody but also matches HANDOFF, and segment N+1 both matches HANDOFF and
    names a person, the lookahead from N already anchors N+1. Without tracking
    that, the outer loop's own visit to N+1 anchors it a second time, and the
    reference carries the same span twice.
    """
    anchors: list[tuple[int, str]] = []
    anchored_indices: set[int] = set()
    for index, segment in enumerate(segments):
        if end_time is not None and segment["start_time"] > end_time:
            break
        if index in anchored_indices:
            continue
        text = segment.get("text") or ""
        if not handoff.search(text):
            continue
        person = _named(text, speakers)
        if person is not None:
            anchors.append((index, person))
            anchored_indices.add(index)
            continue
        if index + 1 < len(segments):
            person = _named(segments[index + 1].get("text") or "", speakers)
            if person is not None:
                anchors.append((index + 1, person))
                anchored_indices.add(index + 1)
    return anchors


def anchor_reference_windows(
    segments: list[dict],
    speakers: dict[str, re.Pattern],
    *,
    handoff: re.Pattern = HANDOFF,
    moderator: str = "MODERATOR",
    end_time: float | None = None,
) -> list[Turns]:
    """One reference window per anchor, each opening with the moderator's turn.

    Each anchor segment is the moderator's; every segment from there to the next
    anchor belongs to the person the anchor named. Segments before the first
    anchor are left UNCOVERED rather than guessed at — the opening remarks are
    the moderator's, but saying so from position alone would put an assumption
    into the thing that is supposed to referee assumptions.

    Windows, not the flat turn list, are the unit any tune/holdout split must
    slice. The flat list alternates moderator, person, moderator, person, so
    taking every other TURN yields a reference with no moderator in it — and the
    moderator is the label this repair exists to break apart. Dropping whole
    anchors instead would be worse: a window runs until the NEXT anchor, so
    removing anchors silently doubles the surviving windows' extent.
    """
    anchors = find_anchors(
        segments, speakers, handoff=handoff, end_time=end_time
    )
    windows: list[Turns] = []
    for position, (index, person) in enumerate(anchors):
        # The FINAL window has no next anchor to bound it, so it would run to
        # end_time and absorb everything after the last handoff. On the real
        # meeting that swallowed 141.9s of the moderator's closing script. An
        # unbounded window is unattributable by construction — drop it.
        if position + 1 >= len(anchors):
            break
        stop = anchors[position + 1][0]
        segment = segments[index]
        window: Turns = [(segment["start_time"], segment["end_time"], moderator)]
        started = False
        for following in range(index + 1, stop):
            segment = segments[following]
            if end_time is not None and segment["start_time"] > end_time:
                break
            if is_moderator_speech(segment):
                if started:
                    break        # the moderator retook the floor; stop attributing
                continue         # still the moderator's question preamble
            started = True
            window.append((segment["start_time"], segment["end_time"], person))
        if len(window) > 1:
            windows.append(window)
    return windows


def anchor_reference_turns(
    segments: list[dict],
    speakers: dict[str, re.Pattern],
    *,
    handoff: re.Pattern = HANDOFF,
    moderator: str = "MODERATOR",
    end_time: float | None = None,
) -> Turns:
    """Every anchor window's turns, concatenated in meeting order."""
    windows = anchor_reference_windows(
        segments, speakers, handoff=handoff, moderator=moderator, end_time=end_time
    )
    return [turn for window in windows for turn in window]
