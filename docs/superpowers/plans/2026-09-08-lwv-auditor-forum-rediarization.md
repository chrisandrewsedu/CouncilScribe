# LWV Auditor Forum Re-diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the conflated `SPEAKER_09` label in `2026-04-03-lwv-brown-county-candidate-forum-auditor` into its three real people, and prove the split against a text-only reference before any human sees it.

**Architecture:** Build a measurement first. The moderator's named handoffs give a per-turn reference over the forum, independent of any voice model; `bench/identity_score.identity_report` already turns a reference plus a hypothesis into fragmentation and conflation counts. With that gate in place, score the incumbent, then two candidate clusterings — OSS pyannote 3.1 on Modal (Experiment A), and a re-clustering of Precision-2's existing turns over per-turn wespeaker embeddings (Experiment B) — and land whichever passes.

**Tech Stack:** Python 3, pytest, numpy, scipy (agglomerative clustering), Modal (L4 GPU), pyannote wespeaker embeddings. No DB writes, no LLM calls until the pipeline re-run in Task 6.

**Spec:** `docs/superpowers/specs/2026-09-08-lwv-auditor-forum-rediarization-design.md`

## Global Constraints

- Python is `~/Documents/GitHub/on-the-record/.venv/bin/python`. Never system `python3` (3.14, missing project deps).
- A script that needs `.env.local` loads it with `from gui.env import load_env_local` then `load_env_local()` — the house pattern (`scripts/sweep_chunk_thresholds.py:36`). Never hand-roll dotenv loading, and never tell the operator to `set -a; . ./.env.local`.
- **Neither script in this plan needs env, so neither calls that loader.** Scoring touches no service; Modal authenticates from `~/.modal.toml` and the worker receives `HF_TOKEN` from a Modal secret. Adding `load_env_local()` at module scope would make these importable modules `os.environ.setdefault` every key in `.env.local`, `DATABASE_URL` included — the leak `tests/conftest.py:26` documents. Do not add it.
- Meeting id is `2026-04-03-lwv-brown-county-candidate-forum-auditor`, living at `~/CouncilScribe/meetings/<id>/`. Quote it in shell commands.
- `transcript_raw.json` is the trustworthy record of original diarized labels. `diarization.json` currently holds post-review merged segments, rewritten by `gui.review_api._persist_after_review`.
- The meeting is NOT live. Nothing publishes. Work stops at the quality gate.
- Conflation is strictly worse than fragmentation. Every threshold tie breaks toward fewer merges.
- No production default changes. One meeting is not a calibration.
- Run tests with `.venv/bin/python -m pytest`. `pytest.ini` sets `testpaths = tests`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `bench/forum_anchor_reference.py` (create) | Pure. Turn a forum's raw segments into reference `Turns` using the moderator's named handoffs. No I/O, no torch. Sibling of `bench/identity_score.named_reference_turns`. |
| `tests/test_forum_anchor_reference.py` (create) | Unit tests for anchor detection, the two known defects, and window termination. |
| `bench/forum_gate.py` (create) | Pure. Turn loading, the tune/holdout split, and the pass/fail verdict. Lives in `bench/` because that is where this repo puts testable measurement logic and `bench/` is an importable package. |
| `scripts/score_forum_diarization.py` (create) | Thin CLI over `bench.forum_gate`. Used unchanged by Tasks 2, 3, 5 and 6. |
| `bench/forum_recluster.py` (create) | Pure. Per-turn relabelling, agglomerative clustering, threshold calibration. No Modal, no torch. |
| `scripts/recluster_forum_turns.py` (create) | Thin CLI over `bench.forum_recluster`, plus the one Modal call that fetches per-turn embeddings. |
| `run_local.py:109-116` (modify) | `_diarization_model_name` gains the re-clustered Precision-2 provenance string. |
| `~/CouncilScribe/meetings/<id>/` (modify, Task 6 only) | Backed up first, then `diarization.json` + `embeddings.json` replaced. |

**Why the pure halves live in `bench/` and not in `scripts/`:** `scripts/` has no
`__init__.py`, nothing in `tests/` imports from it, and every script there calls
`load_env_local()` at module scope. Importing one from a test would fire that
loader and `os.environ.setdefault` every key in `.env.local`, `DATABASE_URL`
included — the precise leak `tests/conftest.py:26` documents, which is why it
captures `LIVE_DB_URL` before test modules import. `bench/` is a real package,
`tests/test_identity_score.py` already imports from it, and nothing there loads
env. Neither new script needs `.env.local` at all: scoring touches no service,
and Modal authenticates from `~/.modal.toml` while `HF_TOKEN` is injected inside
the worker by a Modal secret.

`bench/identity_score.py` is **read and reused, not modified**. Note for reviewers: `identity_report` calls `map_labels_to_reference` twice (`bench/identity_score.py:227` and `:232`) where once would do. It is a pure function over a few hundred turns, so this is cosmetic. Leave it alone; it is not this plan's business.

---

### Task 1: Handoff-anchor reference builder

The gate rests entirely on this file, so it is built test-first and its two known defects are pinned by tests before the fix exists.

**Files:**
- Create: `bench/forum_anchor_reference.py`
- Test: `tests/test_forum_anchor_reference.py`

**Interfaces:**
- Consumes: `Turns = list[tuple[float, float, str]]` from `bench.identity_score`.
- Produces:
  - `find_anchors(segments: list[dict], speakers: dict[str, re.Pattern], *, handoff: re.Pattern = HANDOFF, end_time: float | None = None) -> list[tuple[int, str]]`
  - `anchor_reference_windows(segments, speakers, *, handoff=HANDOFF, moderator="MODERATOR", end_time=None) -> list[Turns]` — one `Turns` list per anchor window, in order. **This is the unit a tune/holdout split must use.** Splitting the flat turn list by index parity would drop every moderator turn, because the flat list alternates moderator, person, moderator, person.
  - `anchor_reference_turns(segments, speakers, *, handoff=HANDOFF, moderator="MODERATOR", end_time=None) -> Turns` — the windows concatenated.
  - `HANDOFF: re.Pattern`
  - `LWV_AUDITOR_SPEAKERS: dict[str, re.Pattern]`
  - `LWV_AUDITOR_FORUM_END: float = 2650.0`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forum_anchor_reference.py`:

```python
"""Tests for building a diarization reference from a forum moderator's handoffs.

The moderator of a candidate forum names who speaks next ("Ms. Bond, same
question"). That makes a per-turn reference derivable from TEXT alone, with no
voice model involved — which is the only way to score a clustering when the
label under test is the one that swallowed three people.
"""
import re

import pytest

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_forum_anchor_reference.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'bench.forum_anchor_reference'`

- [ ] **Step 3: Write the implementation**

Create `bench/forum_anchor_reference.py`:

```python
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

Measured on 2026-04-03-lwv-brown-county-candidate-forum-auditor: 32 anchors
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
    """
    anchors: list[tuple[int, str]] = []
    for index, segment in enumerate(segments):
        if end_time is not None and segment["start_time"] > end_time:
            break
        text = segment.get("text") or ""
        if not handoff.search(text):
            continue
        person = _named(text, speakers)
        if person is not None:
            anchors.append((index, person))
            continue
        if index + 1 < len(segments):
            person = _named(segments[index + 1].get("text") or "", speakers)
            if person is not None:
                anchors.append((index + 1, person))
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
        segment = segments[index]
        window: Turns = [(segment["start_time"], segment["end_time"], moderator)]
        stop = anchors[position + 1][0] if position + 1 < len(anchors) else len(segments)
        for following in range(index + 1, stop):
            segment = segments[following]
            if end_time is not None and segment["start_time"] > end_time:
                break
            window.append((segment["start_time"], segment["end_time"], person))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_forum_anchor_reference.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Verify against the real meeting**

Run:

```bash
.venv/bin/python -c "
import json, collections
from bench.forum_anchor_reference import (anchor_reference_turns,
    LWV_AUDITOR_SPEAKERS, LWV_AUDITOR_FORUM_END)
M='/Users/chrisandrews/CouncilScribe/meetings/2026-04-03-lwv-brown-county-candidate-forum-auditor'
raw=json.load(open(f'{M}/transcript_raw.json'))
t=anchor_reference_turns(raw, LWV_AUDITOR_SPEAKERS, end_time=LWV_AUDITOR_FORUM_END)
by=collections.Counter()
for a,b,p in t: by[p]+=b-a
forum=sum(s['end_time']-s['start_time'] for s in raw if s['start_time']<=LWV_AUDITOR_FORUM_END)
print('turns', len(t), 'coverage %.0f%%' % (100*sum(by.values())/forum))
for p,d in by.most_common(): print(f'  {p:10s} {d:7.1f}s')
print('BOND/KOBIAN ratio %.2f' % (by['BOND']/by['KOBIAN']))
"
```

Expected, and these are the numbers to hold the implementation to:

```
anchors 32 turns 344 coverage 92%
  BOND       1063.5s
  KOBIAN      804.1s
  MODERATOR   199.9s
BOND/KOBIAN ratio 1.32
```

The ratio is 1.32 and not 1.00 for a known reason: Bond's opening statement ran ~90s against Kobian's ~43s (Kobian stopped early — "That's really all I got"), and Bond's answers are consistently longer. A ratio near 2.0 means the common-noun or split-anchor guard has regressed.

- [ ] **Step 6: Commit**

```bash
git add bench/forum_anchor_reference.py tests/test_forum_anchor_reference.py
git commit -m "bench: derive a diarization reference from a forum's handoffs

A reviewed transcript cannot referee its own repair when one label
swallowed three people. A moderated forum supplies an independent
source: the moderator names who speaks next, so the window from each
named handoff to the next belongs to that person — from text alone.

Two guards are load-bearing and both are measured on the LWV auditor
forum. A cue and its name can land in different diarized turns
('...closing remarks.' / 'Kobian.'), which happens twice and costs
~135s per miss. And 'Bond' is also a common noun in a meeting that
discusses the county's debt bond rating, which made a genuine handoff
to Kobian look ambiguous and gave her answer to Bond.

With both: 32 anchors, 92% of forum speech, Bond/Kobian 1.32 (was 2.0)."
```

---

### Task 2: Score the incumbent, establishing the baseline

No repair is credible without the number it improved on. This task produces the CLI that Tasks 3 and 5 reuse unchanged.

**Files:**
- Create: `bench/forum_gate.py`
- Create: `scripts/score_forum_diarization.py`
- Test: `tests/test_forum_gate.py`

**Interfaces:**
- Consumes: `anchor_reference_turns`, `LWV_AUDITOR_SPEAKERS`, `LWV_AUDITOR_FORUM_END` from Task 1; `identity_report` from `bench.identity_score`.
- Produces:
  - `load_turns(path: Path) -> Turns` — reads a JSON list of segment dicts (`start_time`, `end_time`, `speaker_label`) into `Turns`.
  - `gate_verdict(report, max_minority: float) -> tuple[bool, list[str]]` — `(passed, reasons)`. Fails on any conflation; fragmentation never fails.
  - `reference_half(windows: list[Turns], half: str) -> Turns` — `half` is `"all"`, `"tune"` (odd windows) or `"holdout"` (even windows). Slices WINDOWS, never the flat turn list.
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_forum_gate.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forum_gate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.score_forum_diarization'`

- [ ] **Step 3: Write the implementation**

Create `bench/forum_gate.py`:

```python
"""Pass/fail a candidate forum diarization against a handoff-derived reference.

Pure: no env loading, no Modal, no I/O beyond a path the caller hands in. The
CLI wrapper is `scripts/score_forum_diarization.py`.

The gate is asymmetric on purpose. Any conflation fails it; fragmentation never
does. An extra unnamed speaker costs a reviewer seconds at label level, while a
silent merge misattributes quotes to a candidate in a live race.
"""

from __future__ import annotations

import json
from pathlib import Path

from .identity_score import Turns

#: This repair's gate.
GATE_MIN_FRACTION = 0.05
#: `identity_score`'s own default, reported alongside so this meeting's numbers
#: stay comparable with every other diarization measurement in the repo.
COMPARABLE_MIN_FRACTION = 0.02


def load_turns(path: Path) -> Turns:
    """Read a JSON list of segment dicts into scoring turns."""
    segments = json.loads(Path(path).read_text())
    return [
        (float(s["start_time"]), float(s["end_time"]), str(s["speaker_label"]))
        for s in segments
    ]


def reference_half(windows: list[Turns], half: str) -> Turns:
    """All windows, the odd ones (tune) or the even ones (holdout).

    Halving by WINDOW, never by turn: the flat reference alternates moderator,
    person, moderator, person, so a parity slice of turns would hand one half a
    reference with no moderator in it — and the moderator is the label this
    repair exists to break apart.
    """
    if half == "all":
        chosen = windows
    elif half == "tune":
        chosen = windows[1::2]
    elif half == "holdout":
        chosen = windows[0::2]
    else:
        raise ValueError(f"half must be all/tune/holdout, got {half!r}")
    return [turn for window in chosen for turn in window]


def gate_verdict(report, max_minority: float) -> tuple[bool, list[str]]:
    """Pass unless some label holds two reference people above the floor.

    `max_minority` is the floor the caller already passed to `identity_report`;
    it is accepted here so the verdict line can state the bar it applied.
    """
    reasons = [
        f"label {c.label} holds {len(c.people)} people: "
        + ", ".join(f"{p} {c.seconds[p]:.1f}s" for p in c.people)
        for c in report.conflation
    ]
    return (not reasons), reasons
```

Create `scripts/score_forum_diarization.py`:

```python
#!/usr/bin/env python
"""Score a candidate forum diarization against the moderator's own handoffs.

The label under repair swallowed three people, so the reviewed transcript
inherits the error and cannot referee its own fix. `bench.forum_anchor_reference`
builds an independent reference from the moderator's named handoffs, and
`bench.identity_score.identity_report` turns that into fragmentation and
conflation counts.

Reports at two floors, always both: `COMPARABLE_MIN_FRACTION` for continuity
with the repo's other diarization measurements, and `GATE_MIN_FRACTION` for this
repair's verdict.

Loads no env: it touches no database and no service. Usage:
  .venv/bin/python scripts/score_forum_diarization.py \
      ~/CouncilScribe/meetings/<id>/transcript_raw.json --label incumbent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.forum_anchor_reference import (  # noqa: E402
    LWV_AUDITOR_FORUM_END,
    LWV_AUDITOR_SPEAKERS,
    anchor_reference_windows,
)
from bench.forum_gate import (  # noqa: E402
    COMPARABLE_MIN_FRACTION,
    GATE_MIN_FRACTION,
    gate_verdict,
    load_turns,
    reference_half,
)
from bench.identity_score import identity_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turns_json", type=Path,
                        help="JSON list of segments with start_time/end_time/speaker_label")
    parser.add_argument("--raw-json", type=Path, default=None,
                        help="Segments the reference is built from. Defaults to "
                             "turns_json; pass the ORIGINAL transcript_raw.json "
                             "when scoring a turn set that carries no text.")
    parser.add_argument("--label", default="candidate", help="Name for this run in the output")
    parser.add_argument("--forum-end", type=float, default=LWV_AUDITOR_FORUM_END)
    parser.add_argument("--half", choices=("all", "tune", "holdout"), default="all",
                        help="Which anchor windows to score against. Use 'holdout' "
                             "to score a clustering whose threshold was tuned on "
                             "'tune', so the reported number is not the tuned one.")
    args = parser.parse_args(argv)

    reference_source = json.loads((args.raw_json or args.turns_json).read_text())
    windows = anchor_reference_windows(
        reference_source, LWV_AUDITOR_SPEAKERS, end_time=args.forum_end
    )
    reference = reference_half(windows, args.half)
    if not reference:
        print("! reference is empty — no handoffs matched. Refusing to score.")
        return 2

    hypothesis = load_turns(args.turns_json)
    covered = sum(end - start for start, end, _ in reference)
    print(f"== {args.label} ==")
    print(f"reference half: {args.half} ({len(windows)} anchor windows total)")
    print(f"reference: {len(reference)} turns, {covered:.0f}s, "
          f"{len({p for _, _, p in reference})} people")
    print(f"hypothesis: {len(hypothesis)} turns, "
          f"{len({l for _, _, l in hypothesis})} labels")

    for floor in (COMPARABLE_MIN_FRACTION, GATE_MIN_FRACTION):
        report = identity_report(hypothesis, reference, min_fraction=floor)
        passed, reasons = gate_verdict(report, max_minority=floor)
        tag = "GATE" if floor == GATE_MIN_FRACTION else "comparable"
        print(f"\n-- min_fraction {floor:.2f} ({tag}) --")
        print(f"  conflation:    {report.conflation_summary}")
        print(f"  fragmentation: {report.fragmentation_summary}")
        if report.unmapped_labels:
            print(f"  unmapped labels (reference gap, not an error): "
                  f"{', '.join(report.unmapped_labels)}")
        if floor == GATE_MIN_FRACTION:
            print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
            for reason in reasons:
                print(f"    - {reason}")
            return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forum_gate.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Score the incumbent and record the baseline**

Run:

```bash
.venv/bin/python scripts/score_forum_diarization.py \
  ~/CouncilScribe/meetings/2026-04-03-lwv-brown-county-candidate-forum-auditor/transcript_raw.json \
  --label "incumbent (pyannote.ai Precision-2)"
```

Expected: exit code 1, VERDICT FAIL, with `SPEAKER_09` named as holding BOND, KOBIAN and MODERATOR. Paste the full output into the commit message — it is the number every later result is measured against.

If it does NOT fail, stop and escalate. The incumbent is known-bad from three self-introductions inside one label; a pass means the reference or the scorer is wrong, not that the meeting is fine.

- [ ] **Step 6: Commit**

```bash
git add bench/forum_gate.py scripts/score_forum_diarization.py tests/test_forum_gate.py
git commit -m "bench: gate forum diarization on the moderator's handoffs

Reports at two floors — 0.02 to stay comparable with every other
diarization measurement in this repo, 0.05 as this repair's gate — and
fails only on conflation. Fragmentation costs a reviewer seconds at
label level; a silent merge misattributes quotes in a live race.

Baseline recorded for the LWV auditor forum: <paste output>"
```

---

### Task 3: Experiment A — OSS pyannote 3.1 on Modal

**Files:**
- Modify: none
- Artifacts: `/tmp/.../experiment-a-turns.json` (scratchpad, not committed)

**Interfaces:**
- Consumes: `scripts/score_forum_diarization.py` from Task 2 (unchanged).
- Produces: a scored verdict recorded in `docs/superpowers/plans/2026-09-08-lwv-auditor-forum-rediarization.md` results section.

- [ ] **Step 1: Upload the audio and run OSS diarization on Modal**

`chunk_minutes=0` is the single-pass path, which is what the kind gate would choose anyway: `forum` is absent from `config.DIARIZE_CHUNK_EVENT_KINDS`, and 61 minutes is one window regardless. `--num-speakers` is NOT used — the forum has three voices but the meet-and-greet count is unknown, and forcing a wrong K trades one conflation for another.

Run:

```bash
.venv/bin/python -c "
import json
from pathlib import Path
from gui.env import load_env_local; load_env_local()
from src.modal_compute import run_diarization, upload_audio
MID='2026-04-03-lwv-brown-county-candidate-forum-auditor'
wav=Path.home()/'CouncilScribe/meetings'/MID/'audio.wav'
upload_audio(wav, MID)
segments, centroids = run_diarization(wav, MID, use_merge=False, diarizer='oss', chunk_minutes=0)
out=Path('/tmp/experiment-a-turns.json'); out.write_text(json.dumps(segments))
print('labels', len({s[\"speaker_label\"] for s in segments}), 'turns', len(segments))
print('wrote', out)
"
```

- [ ] **Step 2: Score it**

The OSS turn set carries no text, so the reference must be built from the original raw transcript:

```bash
.venv/bin/python scripts/score_forum_diarization.py /tmp/experiment-a-turns.json \
  --raw-json ~/CouncilScribe/meetings/2026-04-03-lwv-brown-county-candidate-forum-auditor/transcript_raw.json \
  --label "experiment A (OSS pyannote 3.1, single-pass)"
```

No `--half` here, deliberately: Experiment A has no tuned threshold, so no part
of the reference was spent on it and it is entitled to be judged on all of it.
Experiment B is the one that must be scored on the holdout.

- [ ] **Step 3: Record the result in the plan's Results section**

Append the label count, conflation summary and verdict under "## Results" at the bottom of this plan file. Record it whether it passed or failed — a measured failure of the documented dense-forum mechanism is worth as much as a pass, and the spec commits to publishing both.

- [ ] **Step 4: Commit the result**

```bash
git add docs/superpowers/plans/2026-09-08-lwv-auditor-forum-rediarization.md
git commit -m "results: experiment A (OSS pyannote 3.1) on the LWV auditor forum

<PASS or FAIL, with label count and conflation summary>"
```

- [ ] **Step 5: Branch**

If Experiment A PASSED the gate, skip Tasks 4 and 5 and go to Task 6 path A.
If it FAILED, continue to Task 4. This is the expected branch: `src/config.py:248` predicts pyannote's own clustering merges speakers on exactly this dense, fast-turn, many-voice shape.

---

### Task 4: Experiment B, part 1 — per-turn embeddings from Modal

**Files:**
- Create: `bench/forum_recluster.py` (the pure relabelling half; clustering lands in Task 5)
- Create: `scripts/recluster_forum_turns.py` (the Modal call only; the CLI lands in Task 5)
- Test: `tests/test_forum_recluster.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces:
  - `turn_label(index: int) -> str` — `"TURN_0000"` style, zero-padded to 4.
  - `as_unique_label_segments(segments: list[dict]) -> list[dict]` — one segment per turn, each carrying its own unique `speaker_label`.
  - `fetch_turn_embeddings(meeting_id: str, segments: list[dict]) -> dict[int, list[float]]` — index → vector, missing keys for turns the worker could not embed.

The trick that avoids new Modal code: `bench/modal_app.py:1305 pipeline_extract_embeddings(meeting_id, segments_json)` accepts arbitrary segments and returns one wespeaker centroid per `speaker_label`. Give every turn a unique label and its "centroid" is that turn's embedding.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forum_recluster.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forum_recluster.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.recluster_forum_turns'`

- [ ] **Step 3: Write the implementation**

Create `bench/forum_recluster.py`:

```python
"""Re-cluster an existing diarization's turns over per-turn voice embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

Pure: no env loading, no Modal, no torch. The Modal fetch and the CLI live in
`scripts/recluster_forum_turns.py`.
"""

from __future__ import annotations

import copy


def turn_label(index: int) -> str:
    """Unique per-turn label. Zero-padded so lexical order is turn order."""
    return f"TURN_{index:04d}"


def as_unique_label_segments(segments: list[dict]) -> list[dict]:
    """Copy `segments`, relabelling each with its own unique speaker label.

    `bench/modal_app.py:1305 pipeline_extract_embeddings` averages wespeaker
    embeddings per `speaker_label` over arbitrary supplied segments, so one
    label per turn makes each returned "centroid" that turn's own embedding.
    That is why Experiment B needs no new Modal code.

    Deep-copies rather than editing in place: `merge_adjacent_segments`
    renumbering its own inputs has already cost this repo a false staleness
    diagnosis, so a probe must never mutate what it probes.
    """
    unique = []
    for index, segment in enumerate(segments):
        clone = copy.deepcopy(segment)
        clone["speaker_label"] = turn_label(index)
        unique.append(clone)
    return unique
```

Create `scripts/recluster_forum_turns.py`:

```python
#!/usr/bin/env python
"""CLI for re-clustering an existing diarization over per-turn embeddings.

pyannote.ai Precision-2 segmented this meeting correctly — every
question-to-answer boundary in transcript_raw.json is clean — and then assigned
three people to one label. The boundaries are worth keeping; only the clustering
needs redoing.

The clustering itself is pure and lives in `bench.forum_recluster`; this file
holds the one Modal call and the command-line surface.

Loads no env: Modal authenticates from ~/.modal.toml, and the worker gets
HF_TOKEN from a Modal secret. No DB writes, no LLM. Usage:
  .venv/bin/python scripts/recluster_forum_turns.py <meeting-id> --out turns.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.forum_recluster import (  # noqa: E402
    as_unique_label_segments,
    turn_label,
)


def fetch_turn_embeddings(
    meeting_id: str, segments: list[dict]
) -> dict[int, list[float]]:
    """One wespeaker vector per turn, from an L4 on Modal.

    Turns shorter than the worker's 0.3s floor, and any whose embedding is
    non-finite, come back absent rather than zeroed — the caller decides what
    to do with a turn that has no voice evidence.
    """
    from src.modal_compute import _modal_app

    app = _modal_app()
    payload = json.dumps(as_unique_label_segments(segments))
    with app.app.run():
        raw = app.pipeline_extract_embeddings.remote(meeting_id, payload)
    by_label = json.loads(raw)
    return {
        index: by_label[turn_label(index)]
        for index in range(len(segments))
        if turn_label(index) in by_label
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forum_recluster.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Fetch the real embeddings and cache them**

Audio was already uploaded in Task 3 Step 1. If Task 3 was skipped, call `upload_audio` first.

Run:

```bash
.venv/bin/python -c "
import json
from pathlib import Path
from scripts.recluster_forum_turns import fetch_turn_embeddings
MID='2026-04-03-lwv-brown-county-candidate-forum-auditor'
raw=json.loads((Path.home()/'CouncilScribe/meetings'/MID/'transcript_raw.json').read_text())
vecs=fetch_turn_embeddings(MID, raw)
Path('/tmp/turn-embeddings.json').write_text(json.dumps(vecs))
print(f'{len(vecs)} of {len(raw)} turns embedded; {len(raw)-len(vecs)} unembeddable')
print('dim', len(next(iter(vecs.values()))))
"
```

Expected: about 447 of 479 embedded, about 32 unembeddable (those are the turns under the worker's 0.3s floor, carrying 2.6s of speech in total), dim 256.

If far more than ~32 come back unembeddable, stop: that means the worker rejected turns for a reason other than length, and clustering on the remainder would silently drop real speech.

- [ ] **Step 6: Commit**

```bash
git add bench/forum_recluster.py scripts/recluster_forum_turns.py tests/test_forum_recluster.py
git commit -m "scripts: fetch per-turn voice embeddings for an existing diarization

Precision-2 segmented this meeting correctly and clustered it wrongly,
so the boundaries are worth keeping. Giving every turn a unique
speaker_label makes pipeline_extract_embeddings' per-label averaging
return one vector per turn, so this needs no new Modal code."
```

---

### Task 5: Experiment B, part 2 — cluster, calibrate, score

**Files:**
- Modify: `bench/forum_recluster.py` (add `cluster_turns`, `relabel_segments`, `calibrate`)
- Modify: `scripts/recluster_forum_turns.py` (add the CLI `main`)
- Modify: `tests/test_forum_recluster.py` (add clustering tests)

**Interfaces:**
- Consumes: `fetch_turn_embeddings`, `turn_label` from Task 4; `anchor_reference_turns` from Task 1; `identity_report`, `gate_verdict` from Task 2.
- Produces:
  - `cluster_turns(vectors: dict[int, list[float]], n_turns: int, threshold: float, *, unclustered_label: str = "SPEAKER_UNCLUSTERED") -> list[str]` — one label per turn index, length `n_turns`.
  - `relabel_segments(segments: list[dict], labels: list[str]) -> list[dict]` — copy of `segments` with new labels, spans untouched.
  - `calibrate(segments, vectors, tune_reference: Turns, thresholds: list[float]) -> tuple[float, list[dict]]` — the caller supplies the ALREADY-HALVED reference (`reference_half(windows, "tune")`), so the split lives in one place.
  - `main(argv)` writing a candidate turn set to `--out`.

Average linkage is not a free choice: `src/config.py:329` records that `complete` merges almost nothing, because a real person's worst turn pair is often anti-correlated (same-person median -0.125), and that `centroid` conflated two real people at the most conservative threshold tested.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forum_recluster.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_forum_recluster.py -v`
Expected: FAIL, `ImportError: cannot import name 'cluster_turns'`

- [ ] **Step 3: Write the implementation**

Append to `bench/forum_recluster.py`:

```python
UNCLUSTERED_LABEL = "SPEAKER_UNCLUSTERED"


def cluster_turns(
    vectors: dict[int, list[float]],
    n_turns: int,
    threshold: float,
    *,
    unclustered_label: str = UNCLUSTERED_LABEL,
) -> list[str]:
    """One speaker label per turn index, by agglomerative clustering.

    Average linkage over cosine distance. MEASURED elsewhere in this repo
    (`src/config.py:329`): "complete" scores a candidate by its worst turn pair
    and a real person's worst pair is often anti-correlated (same-person median
    -0.125), so it merges almost nothing; "centroid" pools each cluster into one
    mean and conflated two real people at the most conservative threshold
    tested. Average is the only workable choice.

    `threshold` is a cosine SIMILARITY floor: clusters join while their mean
    pairwise similarity is at or above it. Turns with no embedding land in one
    shared bucket, not in a neighbour's cluster and not in singletons of their
    own — see the module docstring.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    labels = [unclustered_label] * n_turns
    indices = sorted(vectors)
    if not indices:
        return labels

    matrix = np.asarray([vectors[i] for i in indices], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    if len(indices) == 1:
        labels[indices[0]] = "SPEAKER_00"
        return labels

    distances = pdist(matrix, metric="cosine")
    tree = linkage(distances, method="average")
    assignments = fcluster(tree, t=1.0 - threshold, criterion="distance")

    # Number labels by first appearance so output is stable across runs.
    order: dict[int, str] = {}
    for position, cluster in enumerate(assignments):
        if cluster not in order:
            order[cluster] = f"SPEAKER_{len(order):02d}"
        labels[indices[position]] = order[cluster]
    return labels


def relabel_segments(segments: list[dict], labels: list[str]) -> list[dict]:
    """Copy `segments` with new speaker labels, spans untouched."""
    out = []
    for segment, label in zip(segments, labels):
        clone = copy.deepcopy(segment)
        clone["speaker_label"] = label
        out.append(clone)
    return out


def calibrate(
    segments: list[dict],
    vectors: dict[int, list[float]],
    tune_reference: list[tuple[float, float, str]],
    thresholds: list[float],
) -> tuple[float, list[dict]]:
    """Pick a threshold against the TUNING half of the reference.

    The caller supplies the already-halved reference — `reference_half(windows,
    "tune")` — so the tune/holdout split lives in exactly one place and cannot
    drift between the calibrator and the scorer. Calibrating and reporting on
    the same 32 anchors would prove nothing.

    Ties break toward the HIGHER threshold: conflation misattributes quotes
    silently, fragmentation surfaces as an extra unnamed speaker the reviewer
    clears in seconds.
    """
    from .forum_gate import GATE_MIN_FRACTION
    from .identity_score import identity_report

    people = sorted({p for _, _, p in tune_reference})

    grid: list[dict] = []
    for threshold in thresholds:
        labels = cluster_turns(vectors, len(segments), threshold)
        hypothesis = [
            (s["start_time"], s["end_time"], l)
            for s, l in zip(segments, labels)
        ]
        report = identity_report(
            hypothesis, tune_reference, min_fraction=GATE_MIN_FRACTION
        )
        grid.append({
            "threshold": threshold,
            "labels": len(set(labels)),
            "conflated": len(report.conflation),
            "fragmented": len(report.fragmentation),
            "people": len(people),
        })

    clean = [row for row in grid if row["conflated"] == 0]
    if clean:
        best = min(clean, key=lambda r: (r["fragmented"], -r["threshold"]))
    else:
        best = min(grid, key=lambda r: (r["conflated"], -r["threshold"]))
    return best["threshold"], grid
```

Append to `scripts/recluster_forum_turns.py`:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse

    from bench.forum_anchor_reference import (
        LWV_AUDITOR_FORUM_END,
        LWV_AUDITOR_SPEAKERS,
        anchor_reference_windows,
    )
    from bench.forum_gate import reference_half
    from bench.forum_recluster import calibrate, cluster_turns, relabel_segments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Cached turn-embeddings JSON. Fetched from Modal if absent.")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    args = parser.parse_args(argv)

    from src import config

    meeting_dir = config.MEETINGS_DIR / args.meeting_id
    segments = json.loads((meeting_dir / "transcript_raw.json").read_text())

    if args.embeddings and args.embeddings.exists():
        vectors = {int(k): v for k, v in json.loads(args.embeddings.read_text()).items()}
    else:
        vectors = fetch_turn_embeddings(args.meeting_id, segments)
        if args.embeddings:
            args.embeddings.write_text(json.dumps(vectors))
    print(f"{len(vectors)} of {len(segments)} turns embedded")

    windows = anchor_reference_windows(
        segments, LWV_AUDITOR_SPEAKERS, end_time=LWV_AUDITOR_FORUM_END
    )
    tune = reference_half(windows, "tune")
    print(f"{len(windows)} anchor windows; calibrating on the {len(tune)}-turn "
          f"tuning half, scoring later on the holdout half")

    best, grid = calibrate(segments, vectors, tune, args.thresholds)
    print("\nthreshold  labels  conflated  fragmented   (tuning half only)")
    for row in grid:
        mark = " <-- chosen" if row["threshold"] == best else ""
        print(f"  {row['threshold']:.2f}      {row['labels']:3d}       "
              f"{row['conflated']:2d}         {row['fragmented']:2d}{mark}")

    labels = cluster_turns(vectors, len(segments), best)
    args.out.write_text(json.dumps(relabel_segments(segments, labels)))
    print(f"\nwrote {args.out} at threshold {best:.2f} "
          f"({len(set(labels))} labels)")
    print("Now score it on the held-out half:")
    print(f"  .venv/bin/python scripts/score_forum_diarization.py {args.out} \\")
    print(f"      --raw-json {meeting_dir / 'transcript_raw.json'} \\")
    print("      --half holdout --label 'experiment B'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_forum_recluster.py -v`
Expected: PASS, 9 tests (4 from Task 4, 5 added here).

- [ ] **Step 5: Run the calibration on the real meeting**

Run:

```bash
.venv/bin/python scripts/recluster_forum_turns.py \
  2026-04-03-lwv-brown-county-candidate-forum-auditor \
  --embeddings /tmp/turn-embeddings.json \
  --out /tmp/experiment-b-turns.json
```

Read the grid before moving on. A healthy grid has a plateau: a run of thresholds with 0 conflated and a slowly rising label count. If EVERY threshold conflates, the per-turn signal is too weak for this audio and Experiment B has failed — record that and escalate rather than lowering the gate.

- [ ] **Step 6: Score on the held-out half**

Run:

```bash
.venv/bin/python scripts/score_forum_diarization.py /tmp/experiment-b-turns.json \
  --raw-json ~/CouncilScribe/meetings/2026-04-03-lwv-brown-county-candidate-forum-auditor/transcript_raw.json \
  --half holdout \
  --label "experiment B (Precision-2 boundaries, re-clustered)"
```

`--half holdout` is not optional. The threshold was chosen against the tuning
half, so scoring against `all` would report a number the threshold was fitted to.

Expected: VERDICT PASS, with MODERATOR, BOND and KOBIAN on three distinct labels.

- [ ] **Step 7: Record results and commit**

Append the threshold grid and the held-out verdict to this plan's Results section.

```bash
git add bench/forum_recluster.py scripts/recluster_forum_turns.py tests/test_forum_recluster.py \
        docs/superpowers/plans/2026-09-08-lwv-auditor-forum-rediarization.md
git commit -m "scripts: cluster forum turns over per-turn embeddings, gated

Average linkage over cosine distance, threshold calibrated on odd
anchor windows and scored on the even ones so the reported number is
not the tuned one. Turns with no embedding share one bucket label
rather than joining a neighbour or becoming 32 singletons.

<threshold grid + held-out verdict>"
```

---

### Task 6: Land the winner in the meeting

Destructive from Step 2 onward. Backups first, and there is no undo on a merge.

**Files:**
- Modify: `run_local.py:109-116`
- Test: `tests/test_run_local_chunk_gate.py` (add one test — it already covers `run_local` helpers)
- Modify: `~/CouncilScribe/meetings/<id>/{diarization,embeddings}.json`

**Interfaces:**
- Consumes: `/tmp/experiment-{a,b}-turns.json` from Tasks 3 and 5.
- Produces: a re-run meeting whose `transcript_named.json` carries correct labels.

- [ ] **Step 1: Add the provenance value, test-first**

`processing_metadata.diarization_model` is set from the `--diarizer` flag (`run_local.py:1176`). On path B that would record a lie: segmentation is still Precision-2 and only the clustering changed. Add a value that says so.

Append to `tests/test_run_local_chunk_gate.py`:

```python
def test_reclustered_precision_2_has_its_own_provenance_string():
    """Path B keeps Precision-2's boundaries and replaces only its clustering.
    Recording it as plain 'pyannote/ai-precision-2' would claim the shipped
    labels came from a model that never produced them."""
    import run_local

    assert run_local._diarization_model_name("api") == "pyannote/ai-precision-2"
    assert run_local._diarization_model_name("api-recluster") == (
        "pyannote/ai-precision-2+recluster"
    )
```

Run: `.venv/bin/python -m pytest tests/test_run_local_chunk_gate.py -k provenance -v`
Expected: FAIL.

Then edit `run_local.py:109-116` to:

```python
def _diarization_model_name(diarizer: str) -> str:
    if diarizer == "api":
        return "pyannote/ai-precision-2"
    if diarizer == "api-recluster":
        # Precision-2's segmentation kept, its clustering replaced by a
        # per-turn re-clustering. Distinct from "api" because the labels this
        # meeting ships did NOT come out of Precision-2.
        return "pyannote/ai-precision-2+recluster"
    if diarizer == "vibevoice":
        from src.vibevoice import VIBEVOICE_MODEL_ID, VIBEVOICE_MODEL_REVISION

        return f"{VIBEVOICE_MODEL_ID}@{VIBEVOICE_MODEL_REVISION}"
    return config.DIARIZATION_MODEL
```

Run: `.venv/bin/python -m pytest tests/test_run_local_chunk_gate.py -v`
Expected: PASS.

Commit:

```bash
git add run_local.py tests/test_run_local_chunk_gate.py
git commit -m "run_local: name re-clustered Precision-2 as its own provenance

Keeping Precision-2's boundaries while replacing its clustering
produces labels Precision-2 never emitted. Recording those as plain
'pyannote/ai-precision-2' would make the corpus claim a model produced
output it did not."
```

- [ ] **Step 2: Back up the meeting**

```bash
MID=2026-04-03-lwv-brown-county-candidate-forum-auditor
D=~/CouncilScribe/meetings/$MID
B=$D/backups/transcript-repair-$(date +%Y%m%d-%H%M%S)
mkdir -p "$B"
cp "$D"/transcript_named.json "$D"/transcript_raw.json "$D"/diarization.json \
   "$D"/embeddings.json "$D"/summary.json "$B"/
ls -la "$B"
```

Expected: five files copied. Do not proceed if any is missing.

- [ ] **Step 3a: If Experiment A won**

```bash
.venv/bin/python run_local.py \
  --resume 2026-04-03-lwv-brown-county-candidate-forum-auditor \
  --redo diarize --diarizer oss --compute modal --no-publish
```

Skip to Step 4.

- [ ] **Step 3b: If Experiment B won**

Stage 2 reloads from `diarization.json` when diarization is already complete (`run_local.py:1071`), so writing the winning labels there and re-running from `transcribe` re-aligns `captions.vtt` onto them. `transcript_raw.json` is regenerated by the transcribe stage (`run_local.py:1334`), so raw-versus-named provenance describes the diarization that actually exists.

```bash
.venv/bin/python -c "
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
MID='2026-04-03-lwv-brown-county-candidate-forum-auditor'
D=Path.home()/'CouncilScribe/meetings'/MID
turns=json.loads(Path('/tmp/experiment-b-turns.json').read_text())
vecs={int(k):v for k,v in json.loads(Path('/tmp/turn-embeddings.json').read_text()).items()}

segments=[{'segment_id':i,'start_time':t['start_time'],'end_time':t['end_time'],
           'speaker_label':t['speaker_label']} for i,t in enumerate(turns)]
(D/'diarization.json').write_text(json.dumps(segments, indent=1))

# Per-label centroids from the same per-turn vectors the clustering used.
groups=defaultdict(list)
for i,t in enumerate(turns):
    if i in vecs: groups[t['speaker_label']].append(vecs[i])
cent={k: np.mean(v,axis=0).tolist() for k,v in groups.items()}
(D/'embeddings.json').write_text(json.dumps(cent))
print('wrote', len(segments), 'segments,', len(cent), 'centroids')
print('labels:', sorted({s[\"speaker_label\"] for s in segments}))
"
```

Then re-run from transcribe:

```bash
.venv/bin/python run_local.py \
  --resume 2026-04-03-lwv-brown-county-candidate-forum-auditor \
  --redo transcribe --diarizer api-recluster --no-publish
```

- [ ] **Step 4: Confirm the repair landed**

```bash
.venv/bin/python -c "
import json, collections
from pathlib import Path
MID='2026-04-03-lwv-brown-county-candidate-forum-auditor'
d=json.loads((Path.home()/'CouncilScribe/meetings'/MID/'transcript_named.json').read_text())
print('diarization_model:', d['processing_metadata']['diarization_model'])
c=collections.Counter(); dur=collections.Counter(); names={}
for s in d['segments']:
    k=s['speaker_label']; c[k]+=1; dur[k]+=s['end_time']-s['start_time']
    names.setdefault(k, s.get('speaker_name'))
total=sum(dur.values())
for k,_ in c.most_common():
    print(f'  {k:22s} segs={c[k]:4d} dur={dur[k]:7.1f}s ({100*dur[k]/total:4.1f}%) {names[k]}')
"
```

Expected: no label holds anything near 75% of speech, and the two candidates appear as distinct named speakers. If one label still dominates, stop — the write did not take effect.

- [ ] **Step 5: Re-score the shipped result**

```bash
MID=2026-04-03-lwv-brown-county-candidate-forum-auditor
for HALF in holdout all; do
  .venv/bin/python scripts/score_forum_diarization.py \
    ~/CouncilScribe/meetings/$MID/transcript_raw.json \
    --half $HALF --label "shipped ($HALF)"
done
```

Both are recorded. `holdout` is the honest number, on evidence no threshold ever
saw. `all` is the fuller picture now that the threshold is frozen and nothing
further will be tuned.

Expected: VERDICT PASS on both. This scores what actually landed, after the transcribe stage's own boundary snapping and segment merging, which is not necessarily byte-identical to the candidate turn set that was scored earlier.

If this FAILS while the candidate passed, the merge stage re-introduced conflation. Do not paper over it — record it and escalate.

---

### Task 7: Hand over

**Files:**
- Modify: `docs/superpowers/plans/2026-09-08-lwv-auditor-forum-rediarization.md` (Results section)

- [ ] **Step 1: Report the label set and the unverified labels**

```bash
.venv/bin/python -c "
import json
from pathlib import Path
MID='2026-04-03-lwv-brown-county-candidate-forum-auditor'
D=Path.home()/'CouncilScribe/meetings'/MID
q=json.loads((D/'quality.json').read_text())
print('verdict:', q['verdict'], '|', q['reason'])
print('effective_coverage:', q['effective_coverage'])
FORUM_END=2650.0
d=json.loads((D/'transcript_named.json').read_text())
spans={}
for s in d['segments']:
    k=s['speaker_label']
    lo,hi,_=spans.get(k,(1e9,-1,None))
    spans[k]=(min(lo,s['start_time']), max(hi,s['end_time']), s.get('speaker_name'))
print()
print('labels touching the meet-and-greet (NO reference — unverified):')
for k,(lo,hi,name) in sorted(spans.items(), key=lambda kv: kv[1][0]):
    if hi > FORUM_END:
        print(f'  {k:22s} {lo:7.1f}-{hi:7.1f}  {name}')
"
```

- [ ] **Step 2: Write the Results section**

Fill in the "## Results" section of this plan with: the incumbent baseline, Experiment A's verdict, Experiment B's threshold grid and held-out verdict, the shipped label set, and the explicit list of labels that touch the meet-and-greet and are therefore unverified.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no new failures against the pre-existing baseline. Record the pass/fail counts.

- [ ] **Step 4: Commit and hand over**

```bash
git add docs/superpowers/plans/2026-09-08-lwv-auditor-forum-rediarization.md
git commit -m "results: LWV auditor forum re-diarization

<baseline -> shipped, with the unverified meet-and-greet labels named>"
```

Hand over with, explicitly: the meeting is not live and nothing published; the label set is ready for a label-level name-and-link pass in the GUI; and the labels spanning the meet-and-greet were never covered by the reference and are unverified, not verified-good.

---

## Results

_Filled in by Tasks 3, 5, 6 and 7._

| Run | Labels | Conflation (gate, 0.05) | Verdict |
| --- | --- | --- | --- |
| incumbent (Precision-2) | | | |
| experiment A (OSS pyannote 3.1) | 7 labels, 1199 turns | 48.0% (BOND holds 8.6s of 17.9s under label SPEAKER_00) | **FAIL** — `SPEAKER_04` holds BOND 965.8s + KOBIAN 749.4s + MODERATOR 189.7s |
| experiment B (re-clustered) | | | |
| shipped | | | |

### Experiment A (OSS pyannote 3.1, single-pass) — FAIL

Ran on Modal (L4 GPU, `chunk_minutes=0`, `use_merge=False`, no `--num-speakers`),
scored against the full handoff-derived reference (no `--half`: this run has no
tuned threshold, so it is entitled to be judged on all of it, unlike Experiment
B which must be scored on the holdout). The OSS diarization produced 7 labels
over 1199 turns (versus the incumbent's 10 labels over 479 turns — roughly 2.5x
more turns from the same audio). It failed the gate the same way the incumbent
did: one label, `SPEAKER_04`, swallows all three reference people (BOND 965.8s +
KOBIAN 749.4s + MODERATOR 189.7s = 1904.9s of the reference's 2068s), plus a
small secondary conflation where `SPEAKER_00` holds both BOND (8.6s) and KOBIAN
(9.3s). This is the predicted outcome: `src/config.py:248` documents that
pyannote's own clustering merges speakers when many voices each hold little
speech and turns are short, which is exactly this meeting's shape. A different
pyannote pipeline (OSS 3.1 vs. Precision-2) reproduced the same conflation
mechanism rather than avoiding it. Continuing to Task 4 (Experiment B:
re-cluster Precision-2's existing turn boundaries over per-turn embeddings).
