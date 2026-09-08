# Speaker Identity Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the GUI review page's speaker card so the four identity outcomes — roster politician, local person, unidentified, not-a-speaker — are all visible, all one interaction away, and the current one is obvious.

**Architecture:** A labelled "Who is this?" radio chooser replaces the flat `.actions` row. All four chips always render; the server marks the current one `checked` and un-hides its panel, so the initial paint needs no JavaScript. One new `src/review.clear_speaker_status` makes the two "marked" states reversible. Every write that lands a real identity runs **clear status → rename → assign**, an order forced by two existing functions that each destroy the other's work if run second.

**Tech Stack:** Python 3.12 + FastAPI + Jinja2 templates, vanilla JS (no framework, no build step), pytest with `fastapi.testclient.TestClient`.

## Global Constraints

- **Interpreter:** always `/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python`. This worktree has no `.venv` of its own; system `python3` is 3.14 and lacks the project deps.
- **Do not change** the bodies of `src/review.py` `link_speaker`, `assign_local_person`, `rename_speaker`, `mark_unidentified`, `mark_non_speaker`, or `clear_local_person`. The one-identity-per-speaker invariant (ev-accounts migration 623) is enforced there and stays as-is.
- **Do not touch** `src/publish.py`, `src/quality.py`, `src/enroll.py`, or `run_local.py`. The terminal review flow must stay byte-identical.
- **No new database column and no new prod query.** In particular the roster panel must not add a per-speaker politician lookup to page render — it uses the card's own `speaker_name`.
- **The `name` form field is optional** on the `link` and `local-person` routes. With it absent, both routes behave exactly as they do today.
- **Identity precedence** is exactly `src/review.py:70 identity_label`: `non_speaker` → `unidentified` → `politician_id` → `politician_slug` → `local_slug` → nothing.
- **Full-suite gate:** every task ends with its own tests green; the final task runs the whole suite. **Measured baseline on this worktree before any of this work: `2360 passed, 3 skipped` in ~11s.** The 3 skips need `DATABASE_URL` exported and are expected.
- **Markup contract:** the render tests match on exact attribute strings, so keep each asserted attribute pair on one source line. A Jinja line break inside a tag inserts a newline plus indentation into the output and breaks a substring match.

---

### Task 1: `clear_speaker_status` — the way back

Nothing in `src/` or `gui/` ever clears `speaker_status`, so `mark_unidentified` and `mark_non_speaker` are one-way doors. This adds the only new function in `src/`.

**Files:**
- Modify: `src/review.py` (add after `clear_local_person`, which ends at line 423)
- Test: `tests/test_review_local_people.py`

**Interfaces:**
- Consumes: `src.models.SpeakerMapping`; the existing `mark_unidentified` / `mark_non_speaker` field conventions.
- Produces: `clear_speaker_status(mappings, segments, label) -> SpeakerMapping | None`. Mutates `mappings` and `segments` in place. Returns the mapping on success, `None` on no-op (unknown label, or status already clear).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_review_local_people.py`:

```python
from src.review import clear_speaker_status, mark_non_speaker, mark_unidentified


class _Seg:
    """Minimal segment stand-in: clear_speaker_status only reads speaker_label
    and writes speaker_name."""

    def __init__(self, label, name=None):
        self.speaker_label = label
        self.speaker_name = name


def test_clear_speaker_status_clears_an_unidentified_mark_and_its_handle():
    """mark_unidentified writes a synthetic unidentified-<meeting>-<label> handle
    into local_slug. That handle is a voice-profile key, not a site-local person,
    so clearing the status must drop it too — otherwise the picker would show a
    private handle as a real local person."""
    mappings, segments = {}, [_Seg("S0")]
    mark_unidentified(mappings, segments, "S0", "2026-02-04-council")
    assert mappings["S0"].local_slug == "unidentified-2026-02-04-council-s0"

    m = clear_speaker_status(mappings, segments, "S0")
    assert m is mappings["S0"]
    assert m.speaker_status is None
    assert m.local_slug is None
    assert m.local_role is None


def test_clear_speaker_status_clears_a_non_speaker_mark():
    mappings, segments = {}, [_Seg("S0")]
    mark_non_speaker(mappings, segments, "S0")
    m = clear_speaker_status(mappings, segments, "S0")
    assert m.speaker_status is None
    assert m.local_slug is None


def test_clear_speaker_status_clears_the_placeholder_name_on_mapping_and_segments():
    """'Unidentified Speaker' / 'Non-speaker' label the STATUS, not a person, so
    they must not outlive the status they described."""
    mappings, segments = {}, [_Seg("S0"), _Seg("S1"), _Seg("S0")]
    mark_non_speaker(mappings, segments, "S0", "Pledge of Allegiance")
    assert [s.speaker_name for s in segments] == ["Pledge of Allegiance", None,
                                                  "Pledge of Allegiance"]

    clear_speaker_status(mappings, segments, "S0")
    assert mappings["S0"].speaker_name is None
    assert [s.speaker_name for s in segments] == [None, None, None]


def test_clear_speaker_status_resets_confidence_and_method():
    """mark_* asserted human certainty about the MARK. Once the mark is gone that
    certainty is gone with it, so the speaker returns to Needs attention."""
    mappings, segments = {}, [_Seg("S0")]
    mark_unidentified(mappings, segments, "S0", "2026-02-04-council")
    assert (mappings["S0"].confidence, mappings["S0"].id_method) == (1.0, "human_review")

    m = clear_speaker_status(mappings, segments, "S0")
    assert m.confidence == 0.0
    assert m.id_method is None


def test_clear_speaker_status_on_an_unknown_label_is_a_noop():
    assert clear_speaker_status({}, [], "S9") is None


def test_clear_speaker_status_on_an_already_clear_mapping_is_a_noop():
    """A no-op is not success: the GUI route maps None to 404 so an Undo button
    on a speaker that was never marked cannot report that it did something."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Susan Brackney",
                                     local_slug="susan-brackney",
                                     local_role="public_comment", speaker_status=None)}
    assert clear_speaker_status(mappings, [], "S0") is None
    # A genuine local person is left completely untouched.
    assert mappings["S0"].local_slug == "susan-brackney"
    assert mappings["S0"].local_role == "public_comment"
    assert mappings["S0"].speaker_name == "Susan Brackney"


def test_clear_speaker_status_leaves_a_politician_link_alone():
    """Clearing a stale mark is not an unlink. Only the mark and its own
    placeholder fields are dropped."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Non-speaker",
                                     politician_id="uuid-mk",
                                     politician_slug="marcy-kaptur",
                                     speaker_status="non_speaker")}
    m = clear_speaker_status(mappings, [], "S0")
    assert m.politician_id == "uuid-mk"
    assert m.politician_slug == "marcy-kaptur"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_review_local_people.py -k clear_speaker_status -v
```

Expected: collection error — `ImportError: cannot import name 'clear_speaker_status' from 'src.review'`.

- [ ] **Step 3: Write the implementation**

In `src/review.py`, immediately after `clear_local_person` (which ends at line 423, before `def mark_unidentified`):

```python
def clear_speaker_status(mappings, segments, label):
    """Drop 'unidentified' / 'non_speaker' so a label can hold a real identity again.

    Returns None (no mutation) when the label is unknown or its status is already
    clear. A no-op is not success: the GUI route maps None to 404, so an Undo
    button on a speaker that was never marked cannot report that it acted.

    mark_unidentified and mark_non_speaker are otherwise one-way doors — nothing
    else in src/ or gui/ ever clears speaker_status — which left a mis-clicked
    "Not a speaker" unrecoverable and permanently hid the local-person path.

    Three groups of fields go with the mark and must not outlive it:

    - `local_slug` after an 'unidentified' mark is the synthetic
      unidentified-<meeting>-<label> handle from make_unidentified_slug, whose
      only job is keeping two distinct unknowns out of one voice-profile
      enrollment key. It is not a site-local person and must not be presented as
      one. A 'non_speaker' mark clears local_slug outright, so clearing it again
      is a harmless no-op — hence no branch on the status value.
    - `speaker_name` is 'Unidentified Speaker', 'Non-speaker', or a reviewer's
      display_label FOR THE MARK. It names the status, not a person.
    - confidence 1.0 / id_method 'human_review' asserted human certainty about
      the mark. With the mark gone the speaker has no identity, so it returns to
      needs-review rather than staying falsely confirmed.

    A politician link is deliberately left alone: clearing a stale mark is not an
    unlink.
    """
    mapping = mappings.get(label)
    if mapping is None or getattr(mapping, "speaker_status", None) is None:
        return None

    mapping.speaker_status = None
    mapping.local_slug = None
    mapping.local_role = None
    mapping.speaker_name = None
    mapping.confidence = 0.0
    mapping.id_method = None
    mapping.needs_review = True

    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = None

    return mapping
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_review_local_people.py -v
```

Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add src/review.py tests/test_review_local_people.py
git commit -m "feat(review): clear_speaker_status, the way back from a mark

mark_unidentified and mark_non_speaker were one-way doors: nothing in src/
or gui/ ever cleared speaker_status, so a mis-clicked 'Not a speaker' was
unrecoverable and the local-person path stayed hidden for good.

Drops the mark plus the fields that belong to it — the synthetic
unidentified-<meeting>-<label> handle (a voice-profile key, not a person),
the placeholder name, and the human-certainty confidence/id_method — and
leaves any politician link alone, because clearing a stale mark is not an
unlink."
```

---

### Task 2: `SpeakerCard.identity_kind` — one name for the current state

The card renders its state three different ways today (`🔗 linked: <uuid>`, `local: slug · role`, `statusbadge` chips). Every later task needs one authoritative answer instead.

**Files:**
- Modify: `gui/models.py` (add to `SpeakerCard`, after the `has_local_person` property)
- Test: `tests/test_gui_review.py`

**Interfaces:**
- Consumes: existing `SpeakerCard` fields `speaker_status`, `politician_id`, `politician_slug`, `local_slug`.
- Produces: `SpeakerCard.identity_kind -> str`, one of exactly `"roster"`, `"local"`, `"unidentified"`, `"non_speaker"`, `"none"`. Also `SpeakerCard.identity_pill -> str`, the reader-facing wording: `"roster"`, `"local"`, `"unidentified"`, `"not a speaker"`, `"no identity"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_review.py`:

```python
def _card_ident(**kw):
    from gui.models import SpeakerCard
    base = dict(label="S", name="Brian Sterling", confidence=1.0,
                method="human_review", minutes=2.0, seg_count=3)
    base.update(kw)
    return SpeakerCard(**base)


def test_identity_kind_covers_all_five_states():
    assert _card_ident().identity_kind == "none"
    assert _card_ident(politician_id="uuid-1").identity_kind == "roster"
    assert _card_ident(politician_slug="xavier-becerra").identity_kind == "roster"
    assert _card_ident(local_slug="brian-sterling").identity_kind == "local"
    assert _card_ident(speaker_status="unidentified",
                       local_slug="unidentified-m-s0").identity_kind == "unidentified"
    assert _card_ident(speaker_status="non_speaker").identity_kind == "non_speaker"


def test_identity_kind_matches_review_identity_label_precedence():
    """The picker must never disagree with what publish will store, so the
    precedence here is exactly src/review.py identity_label's: status beats
    links, and politician_* beats local_slug."""
    # A stray link under a mark: the mark wins.
    assert _card_ident(speaker_status="non_speaker",
                       politician_id="uuid-1").identity_kind == "non_speaker"
    assert _card_ident(speaker_status="unidentified",
                       politician_id="uuid-1").identity_kind == "unidentified"
    # Both a roster link and a local slug: roster wins.
    assert _card_ident(politician_id="uuid-1",
                       local_slug="brian-sterling").identity_kind == "roster"


def test_identity_pill_wording():
    assert _card_ident().identity_pill == "no identity"
    assert _card_ident(politician_id="uuid-1").identity_pill == "roster"
    assert _card_ident(local_slug="s").identity_pill == "local"
    assert _card_ident(speaker_status="unidentified").identity_pill == "unidentified"
    assert _card_ident(speaker_status="non_speaker").identity_pill == "not a speaker"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -k "identity_kind or identity_pill" -v
```

Expected: FAIL with `AttributeError: 'SpeakerCard' object has no attribute 'identity_kind'`.

- [ ] **Step 3: Write the implementation**

In `gui/models.py`, inside `SpeakerCard`, immediately after the `has_local_person` property:

```python
    # Reader-facing wording for each identity_kind. Separate from the kind token
    # so the template never has to spell a status out and the two can't drift.
    _IDENTITY_PILLS = {
        "roster": "roster",
        "local": "local",
        "unidentified": "unidentified",
        "non_speaker": "not a speaker",
        "none": "no identity",
    }

    @property
    def identity_kind(self) -> str:
        """'roster' | 'local' | 'unidentified' | 'non_speaker' | 'none'.

        Which of the four identity outcomes is currently in force, as one token
        the picker can switch on. Precedence is exactly src/review.identity_label's
        — status beats links, politician_* beats local_slug — so the picker can
        never disagree with what publish will store. Derived, never stored.
        """
        if self.speaker_status == "non_speaker":
            return "non_speaker"
        if self.speaker_status == "unidentified":
            return "unidentified"
        if self.politician_id or self.politician_slug:
            return "roster"
        if self.local_slug:
            return "local"
        return "none"

    @property
    def identity_pill(self) -> str:
        """Short label for the identity pill in the card head."""
        return self._IDENTITY_PILLS[self.identity_kind]
```

Note: `_IDENTITY_PILLS` is a plain class attribute on a `@dataclass`. It has no
type annotation, so `dataclass` does not treat it as a field — that is why the
annotation is deliberately omitted.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -v
```

Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add gui/models.py tests/test_gui_review.py
git commit -m "feat(gui): SpeakerCard.identity_kind, one answer for the current state

The card rendered its identity three different ways in three places — a
linked line, an inline local badge, and status chips — so nothing could ask
'which of the four is in force?' in one place.

Precedence copies src/review.identity_label exactly, so the picker can never
disagree with what publish will store."
```

---

### Task 3: The three writers — clear status, then rename, then assign

`apply_link` and `apply_make_local_person` gain an optional name and learn to clear a mark. A new `apply_clear_speaker_status` backs the Undo buttons.

The order is forced by two existing functions, and getting it wrong silently destroys data:
- `clear_speaker_status` blanks the name, so it must run **before** the rename.
- `rename_speaker` drops any prior identity when the name changes (`src/review.py:203`, "a human-assigned name is authoritative"), so it must run **before** the assignment.

**Files:**
- Modify: `gui/review_api.py` — `apply_link` (currently at line 228), `apply_make_local_person` (line 262), plus a new `apply_clear_speaker_status`
- Test: `tests/test_gui_review.py`

**Interfaces:**
- Consumes: `src.review.clear_speaker_status` from Task 1; existing `review_api._load_meeting_ctx`, `persist_review`.
- Produces:
  - `apply_link(meeting_id, label, politician_slug, politician_id, name="") -> bool`
  - `apply_make_local_person(meeting_id, label, slug, role_raw, name="") -> bool` (still raises `ValueError` on a bad or colliding slug)
  - `apply_clear_speaker_status(meeting_id, label) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_review.py`:

```python
from gui.review_api import (
    apply_clear_speaker_status,
    apply_make_local_person,
    apply_mark_unidentified,
)


def _card_for(meeting_id, label):
    page = load_review_page(meeting_id)
    return [c for c in (page.confirmed + page.needs_attention) if c.label == label][0]


def test_apply_link_clears_a_mark(tagged_meeting_dir, tmp_meetings_dir):
    """Reaching a roster identity from a marked state must take one click. Before
    this, speaker_status survived the link and kept the card showing a stale
    'unidentified' badge with the local-person path hidden for good."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_unidentified("2026-02-04-council", "SPEAKER_01")
    assert _card_for("2026-02-04-council", "SPEAKER_01").identity_kind == "unidentified"

    assert apply_link("2026-02-04-council", "SPEAKER_01", "", "uuid-becerra") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.identity_kind == "roster"
    assert card.speaker_status is None
    assert card.local_slug is None       # the synthetic handle is gone too


def test_apply_make_local_person_clears_a_mark(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge")
    assert _card_for("2026-02-04-council", "SPEAKER_01").identity_kind == "non_speaker"

    assert apply_make_local_person("2026-02-04-council", "SPEAKER_01",
                                   "brian-sterling", "official") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.identity_kind == "local"
    assert card.local_slug == "brian-sterling"


def test_apply_link_with_a_name_stores_both(tagged_meeting_dir, tmp_meetings_dir):
    """The ordering regression test. rename_speaker drops any prior identity when
    the name changes, so renaming AFTER the link would wipe the link that was
    just made. Both fields must survive."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_link("2026-02-04-council", "SPEAKER_01", "", "uuid-becerra",
                      name="Xavier Becerra") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.name == "Xavier Becerra"
    assert card.politician_id == "uuid-becerra"
    assert card.identity_kind == "roster"


def test_apply_link_with_a_name_over_a_mark_stores_both(tagged_meeting_dir, tmp_meetings_dir):
    """All three steps at once: clear the mark, take the name, keep the link.
    Clearing after the rename would blank the name; renaming after the link
    would drop the link."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Music")

    assert apply_link("2026-02-04-council", "SPEAKER_01", "", "uuid-becerra",
                      name="Xavier Becerra") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.name == "Xavier Becerra"
    assert card.politician_id == "uuid-becerra"
    assert card.speaker_status is None


def test_apply_make_local_person_with_a_name_stores_name_slug_and_role(
        tagged_meeting_dir, tmp_meetings_dir):
    """publish writes `speaker_name or slug` as a local person's PUBLIC name, so a
    nameless local person reaches readers as the raw slug. The name has to land
    in the same action."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_make_local_person("2026-02-04-council", "SPEAKER_01",
                                   "brian-sterling", "official",
                                   name="Brian Sterling") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.name == "Brian Sterling"
    assert card.local_slug == "brian-sterling"
    assert card.local_role == "official"


def test_apply_writers_unchanged_without_a_name(tagged_meeting_dir, tmp_meetings_dir):
    """Existing callers pass no name; their behaviour must not shift."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    # SPEAKER_00 is named "Mayor Johnson" on disk; linking without a name keeps it.
    assert apply_link("2026-02-04-council", "SPEAKER_00", "mayor-johnson", "") is True
    assert _card_for("2026-02-04-council", "SPEAKER_00").name == "Mayor Johnson"


def test_apply_clear_speaker_status_and_its_guards(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge")

    assert apply_clear_speaker_status("2026-02-04-council", "SPEAKER_01") is True
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.identity_kind == "none"
    assert card.name is None

    # A second clear is a no-op, and a no-op is not success.
    assert apply_clear_speaker_status("2026-02-04-council", "SPEAKER_01") is False
    assert apply_clear_speaker_status("2026-02-04-council", "SPEAKER_99") is False
    assert apply_clear_speaker_status("ghost", "SPEAKER_01") is False
    assert apply_clear_speaker_status("../x", "SPEAKER_01") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -k "clears_a_mark or with_a_name or clear_speaker_status or unchanged_without" -v
```

Expected: collection error — `ImportError: cannot import name 'apply_clear_speaker_status' from 'gui.review_api'`.

- [ ] **Step 3: Write the implementation**

First, add a shared helper to `gui/review_api.py`, immediately above `apply_link`:

```python
def _reset_and_rename(meeting, label: str, name: str, roster) -> None:
    """Prepare a label to take a real identity: drop any unidentified/non-speaker
    mark, then apply an optional reviewer-supplied name.

    The order of these two, and of the caller's assignment after them, is forced
    and each wrong order loses data silently:

    - clear_speaker_status blanks the placeholder name, so it must run BEFORE the
      rename, or it erases the name just supplied.
    - rename_speaker drops any prior identity when the name changes (it treats
      the old link as belonging to the old name), so it must run BEFORE the
      caller's link/local-person assignment, or it erases the identity just set.

    Hence: clear status -> rename -> assign. Both steps here no-op when there is
    nothing to do, so a plain link with no name behaves exactly as before.
    """
    from src import review

    review.clear_speaker_status(meeting.speakers, meeting.segments, label)
    if (name or "").strip():
        review.rename_speaker(meeting.speakers, meeting.segments, label,
                              name.strip(), roster=roster)
```

Then replace `apply_link` (currently lines 228-244) with:

```python
def apply_link(meeting_id: str, label: str, politician_slug: str, politician_id: str,
               name: str = "") -> bool:
    """Link a speaker to an essentials politician/candidate and persist. Accepts a
    slug OR an id (candidates have an id but no slug). False on unsafe/unknown
    meeting or label, or when BOTH slug and id are empty.

    `name` is optional. The picker sends the display name of the person the
    reviewer just clicked, so the transcript's speaker_name cannot disagree with
    the linked person; callers that omit it keep the previous behaviour exactly.
    """
    slug = (politician_slug or "").strip()
    pid = (politician_id or "").strip()
    if not slug and not pid:
        return False
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review
    _reset_and_rename(meeting, label, name, roster)
    review.link_speaker(meeting.speakers, label, slug or None, pid or None)
    persist_review(meeting, meeting_dir)
    return True
```

Note the third element of `ctx` was previously discarded as `_roster`; it is now
used, so the name changes to `roster`.

Then replace `apply_make_local_person` (currently lines 262-282) with:

```python
def apply_make_local_person(meeting_id: str, label: str, slug: str, role_raw: str,
                            name: str = "") -> bool:
    """Make a speaker a site-local person and persist.

    `role_raw` is whatever the reviewer typed or picked; it goes through
    resolve_local_role, which guarantees a storable shape, so a role can never be
    invalid here. Returns False on an unsafe/unknown meeting or label. Raises
    ValueError on a slug that is malformed or already held by another label —
    a distinct failure the route reports as 400 rather than 404.

    `name` is optional but the picker always sends it, because publish writes
    `speaker_name or slug` as a local person's PUBLIC name: a nameless local
    person reaches readers as the raw slug.
    """
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review
    from src.event_kinds import resolve_local_role

    role = resolve_local_role(role_raw, meeting.event_kind)
    _reset_and_rename(meeting, label, name, roster)
    review.assign_local_person(meeting.speakers, label, slug, role)   # may raise ValueError
    persist_review(meeting, meeting_dir)
    return True
```

Then add `apply_clear_speaker_status` immediately after `apply_clear_local_person`:

```python
def apply_clear_speaker_status(meeting_id: str, label: str) -> bool:
    """Undo an unidentified / non-speaker mark and persist. False on an
    unsafe/unknown meeting or label, and also when review.clear_speaker_status
    itself no-ops (the speaker was never marked) — a no-op is not success, so an
    Undo on an unmarked speaker reports 404 rather than a silent success."""
    ctx = _load_meeting_ctx(meeting_id)
    if ctx is None:
        return False
    meeting, meeting_dir, _roster = ctx
    known = {s.speaker_label for s in meeting.segments} | set(meeting.speakers)
    if label not in known:
        return False
    from src import review

    if review.clear_speaker_status(meeting.speakers, meeting.segments, label) is None:
        return False
    persist_review(meeting, meeting_dir)
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -v
```

Expected: PASS, including the pre-existing `test_apply_link_and_unlink`, `test_apply_link_guards`, `test_apply_link_by_id_only` and `test_apply_link_requires_slug_or_id`.

- [ ] **Step 5: Commit**

```bash
git add gui/review_api.py tests/test_gui_review.py
git commit -m "feat(gui): identity writers clear the mark and carry the name

Reaching a roster or local identity from a marked speaker took no clicks at
all before — it was impossible, because speaker_status survived every write.
Both writers now clear the mark first.

They also take an optional name. publish writes \`speaker_name or slug\` as a
local person's public name, so a nameless local person reaches readers as
the raw slug; on the roster side the name stops the transcript disagreeing
with the person linked.

The order is forced and both wrong orders lose data silently:
clear_speaker_status blanks the name, so it precedes the rename;
rename_speaker drops a prior identity, so it precedes the assignment."
```

---

### Task 4: The `clear-status` route and the two `name` fields

**Files:**
- Modify: `gui/app.py` — `link_speaker_route` (line 392), `make_local_person_route` (line 426), plus a new route after `clear_local_person_route` (line 439)
- Test: `tests/test_gui_review.py`

**Interfaces:**
- Consumes: `apply_link`, `apply_make_local_person`, `apply_clear_speaker_status` from Task 3, and the `_card_for(meeting_id, label)` test helper Task 3 appended to `tests/test_gui_review.py`. `TestClient` and `create_app` are already imported near line 190 of that file.
- Produces: `POST /meetings/{meeting_id}/speakers/{label}/clear-status` → 303 on success, 404 on unknown/unsafe/no-op. `name` accepted as an optional form field on `.../link` and `.../local-person`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_review.py`:

```python
def test_clear_status_route(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/not-speaker",
                data={"display_label": "Pledge"}, follow_redirects=False)

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/clear-status",
                    follow_redirects=False)
    assert r.status_code == 303
    # Second clear is a no-op -> 404, not a silent success.
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/clear-status",
                       follow_redirects=False).status_code == 404
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_99/clear-status",
                       follow_redirects=False).status_code == 404
    assert client.post("/meetings/ghost/speakers/SPEAKER_01/clear-status",
                       follow_redirects=False).status_code == 404


def test_link_route_accepts_a_name(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/link",
                    data={"politician_slug": "", "politician_id": "uuid-becerra",
                          "name": "Xavier Becerra"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _card_for("2026-02-04-council", "SPEAKER_01").name == "Xavier Becerra"


def test_local_person_route_accepts_a_name(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/local-person",
                    data={"slug": "brian-sterling", "role": "official",
                          "name": "Brian Sterling"},
                    follow_redirects=False)
    assert r.status_code == 303
    card = _card_for("2026-02-04-council", "SPEAKER_01")
    assert card.name == "Brian Sterling"
    assert card.local_slug == "brian-sterling"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -k "clear_status_route or route_accepts_a_name" -v
```

Expected: FAIL — `test_clear_status_route` gets 405 (no such route); the two name tests get 303 but the name assertion fails, because the route drops the unknown form field.

- [ ] **Step 3: Write the implementation**

In `gui/app.py`, change `link_speaker_route` to accept and forward `name`:

```python
    @app.post("/meetings/{meeting_id}/speakers/{label}/link")
    def link_speaker_route(meeting_id: str, label: str,
                           politician_slug: str = Form(""), politician_id: str = Form(""),
                           name: str = Form("")):
        redirect = RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
        if not politician_slug.strip() and not politician_id.strip():
            return redirect  # nothing to link
        if not review_api.apply_link(meeting_id, label, politician_slug, politician_id,
                                     name=name):
            raise HTTPException(status_code=404)
        return redirect
```

Change `make_local_person_route` likewise:

```python
    @app.post("/meetings/{meeting_id}/speakers/{label}/local-person")
    def make_local_person_route(meeting_id: str, label: str,
                               slug: str = Form(""), role: str = Form(""),
                               name: str = Form("")):
        try:
            ok = review_api.apply_make_local_person(meeting_id, label, slug, role,
                                                    name=name)
        except ValueError as exc:
            # Malformed or colliding slug. Reported, not silently ignored: the
            # form prefills a valid default, so this is a deliberate bad value.
            raise HTTPException(status_code=400, detail=str(exc))
        if not ok:
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
```

Add the new route immediately after `clear_local_person_route`:

```python
    @app.post("/meetings/{meeting_id}/speakers/{label}/clear-status")
    def clear_speaker_status_route(meeting_id: str, label: str):
        if not review_api.apply_clear_speaker_status(meeting_id, label):
            raise HTTPException(status_code=404)
        return RedirectResponse(url=f"/meetings/{meeting_id}/review", status_code=303)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gui/app.py tests/test_gui_review.py
git commit -m "feat(gui): clear-status route and optional name on the identity writes

POST .../speakers/{label}/clear-status backs the Undo control on a marked
speaker; a no-op clear is 404, so Undo on an unmarked speaker cannot report
success. link and local-person now forward an optional name."
```

---

### Task 5: The "Who is this?" chooser

This is the visible fix. The card's single flat `.actions` row becomes three blocks: the identity chooser, then a separate row for rename/merge/enroll, which are not identity outcomes.

**Files:**
- Modify: `gui/templates/panels/_macros.html` (rewrite the `card` macro, currently the whole 114-line file)
- Modify: `gui/static/style.css` (append the chooser styles after line 77)
- Modify: `gui/static/workspace.js` (append one delegated listener near the existing `.link-search` input listener)
- Test: `tests/test_gui_review.py`

**Interfaces:**
- Consumes: `SpeakerCard.identity_kind` and `.identity_pill` from Task 2; the `clear-status` route and `name` fields from Task 4; existing card fields `default_slug`, `local_slug`, `local_role`, `politician_id`, `politician_slug`, `name`, `duplicate_labels`, `merge_hints`, `merge_mismatches`, `is_enrollable`, `is_enrolled`, `thin_sample`, `profile_strength`, `profile_hint`, `accept_name`, `clip_seeks`, `sample_text`, `hints`.
- Produces: markup contracts the render tests assert on — `class="ident"` wrapper, one `<input type="radio" name="ident-{{ label }}">` per outcome with `value` in `roster|local|unidentified|non_speaker`, and one `<div class="ident-panel" data-ident="<value>">` per outcome, all but the current one carrying `hidden`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_review.py`:

```python
def _linked_body(tagged_meeting_dir):
    """Review page HTML for a meeting whose SPEAKER_00 is roster-linked."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import json as _json
    data = _json.loads((mdir / "transcript_named.json").read_text())
    data["speakers"]["SPEAKER_00"]["politician_id"] = "uuid-mj"
    (mdir / "transcript_named.json").write_text(_json.dumps(data))
    return TestClient(create_app()).get("/meetings/2026-02-04-council/review").text


def test_a_linked_speaker_still_offers_the_local_person_panel(
        tagged_meeting_dir, tmp_meetings_dir):
    """THE REGRESSION THAT PROMPTED THIS WORK. The old macro gated the
    make-local-person form on `{% elif not c.is_linked %}`, so a roster-linked
    speaker rendered no such form and the reviewer had to click Unlink first.
    Most speakers needing hand-identification are NOT politicians, so the local
    path is the common case, not a fallback."""
    body = _linked_body(tagged_meeting_dir)
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_00/local-person"' in body


def test_every_card_offers_all_four_outcomes(tagged_meeting_dir, tmp_meetings_dir):
    body = _linked_body(tagged_meeting_dir)
    for label in ("SPEAKER_00", "SPEAKER_01"):
        for kind in ("roster", "local", "unidentified", "non_speaker"):
            assert f'name="ident-{label}" value="{kind}"' in body, f"{label}/{kind}"


def _card_html(body, label):
    """The HTML of one speaker's card, sliced out of the review page so the
    per-card assertions below cannot be satisfied by a sibling card."""
    chunks = body.split('<div class="card ')
    hits = [ch for ch in chunks if f'name="ident-{label}"' in ch]
    assert len(hits) == 1, f"expected exactly one card for {label}, got {len(hits)}"
    return hits[0]


def test_the_current_outcome_is_the_checked_one(tagged_meeting_dir, tmp_meetings_dir):
    """SPEAKER_00 is roster-linked, SPEAKER_01 has no identity at all. Exactly
    one chip is checked on the linked card, and none on the other — 'no identity'
    is a real state, and pre-checking a chip there would assert a choice nobody
    made."""
    body = _linked_body(tagged_meeting_dir)

    linked = _card_html(body, "SPEAKER_00")
    assert linked.count('name="ident-SPEAKER_00"') == 4
    assert linked.count("checked") == 1
    assert 'value="roster" checked' in linked

    plain = _card_html(body, "SPEAKER_01")
    assert plain.count('name="ident-SPEAKER_01"') == 4
    assert plain.count("checked") == 0


def test_only_the_current_panel_is_revealed(tagged_meeting_dir, tmp_meetings_dir):
    """Server-rendered reveal: the initial paint needs no JavaScript, so exactly
    one of a card's four panels lacks `hidden`."""
    import re
    body = _linked_body(tagged_meeting_dir)

    linked = _card_html(body, "SPEAKER_00")
    panels = re.findall(r'<div class="ident-panel" data-ident="(\w+)"([^>]*)>', linked)
    assert len(panels) == 4, f"expected 4 panels, got {panels}"
    revealed = [kind for kind, attrs in panels if "hidden" not in attrs]
    assert revealed == ["roster"]

    # A speaker with no identity reveals nothing: the four chips are the prompt.
    plain = re.findall(r'<div class="ident-panel" data-ident="(\w+)"([^>]*)>',
                       _card_html(body, "SPEAKER_01"))
    assert [k for k, a in plain if "hidden" not in a] == []


def test_a_local_person_card_warns_what_the_roster_panel_would_drop(
        tagged_meeting_dir, tmp_meetings_dir):
    """link_speaker clears local_slug/local_role, so picking a politician
    destroys the local person. That used to happen silently."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_make_local_person("2026-02-04-council", "SPEAKER_01", "brian-sterling",
                            "official", name="Brian Sterling")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "brian-sterling" in body
    assert "drops the local person" in body


def test_a_marked_card_offers_an_undo(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_01/clear-status"' in body


def test_the_local_person_panel_asks_for_a_name_and_a_role(
        tagged_meeting_dir, tmp_meetings_dir):
    """The role field must NOT prefill. local_roles_for('council')[0] is
    'public_comment' and for a news_clip it is 'candidate' — a silent wrong
    default on a person published to readers."""
    body = _linked_body(tagged_meeting_dir)
    assert 'name="name" value="Mayor Johnson" required' in body   # prefilled from the card
    assert 'name="role" value="" required' in body                # blank, explicit


def test_a_marked_card_does_not_prefill_its_placeholders_as_a_local_person(
        tagged_meeting_dir, tmp_meetings_dir):
    """Under an 'unidentified' mark, local_slug is the synthetic
    unidentified-<meeting>-<label> handle (a voice-profile key) and speaker_name
    is 'Unidentified Speaker'. Offering either as the local-person default would
    invite the reviewer to publish a private handle or a status word as a
    person's public name."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_unidentified("2026-02-04-council", "SPEAKER_01")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    card = _card_html(body, "SPEAKER_01")

    assert 'name="slug" value="unidentified-2026-02-04-council-speaker_01"' not in card
    assert 'name="name" value="Unidentified Speaker" required' not in card
    assert 'name="name" value="" required' in card       # blank, reviewer must type one


def test_the_identity_pill_renders(tagged_meeting_dir, tmp_meetings_dir):
    body = _linked_body(tagged_meeting_dir)
    assert 'class="identpill' in body
    assert ">roster<" in body
    assert ">no identity<" in body


def test_workspace_js_reveals_the_chosen_panel(tmp_meetings_dir):
    js = Path("gui/static/workspace.js").read_text()
    assert "ident-panel" in js
    assert 'data-ident' in js
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py -k "four_outcomes or checked_one or current_panel or would_drop or offers_an_undo or asks_for_a_name or identity_pill_renders or reveals_the_chosen or still_offers_the_local" -v
```

Expected: FAIL — every assertion on the new markup, and `test_a_linked_speaker_still_offers_the_local_person_panel` in particular, which is the reported bug.

- [ ] **Step 3: Rewrite the card macro**

Replace the entire contents of `gui/templates/panels/_macros.html` with:

```jinja
{# One speaker in the review page, as three blocks:

   1. head + evidence (name, confidence, voice hints, sample, clips)
   2. "Who is this?" — the four identity outcomes, all four always present
   3. "Also" — rename, merge, enroll: real operations, but not identity outcomes

   Block 2 replaces a single flat flex row that mixed all of the above. In that
   row the make-local-person form was gated on `not c.is_linked` and on
   `speaker_status is none`, so a roster-linked or already-marked speaker
   rendered no local-person control at all and the reviewer had to clear the
   other field first. Most speakers who need hand-identification are not
   politicians — anchors, reporters, party officials, members of the public — so
   the local path is the common case, not a fallback.

   The chosen panel is revealed BY THE SERVER (every other panel carries
   `hidden`), so the initial paint needs no JavaScript; workspace.js only
   handles switching. #}
{# The <input> stays on ONE source line: the render tests match the exact string
   `name="ident-<label>" value="<kind>" checked`, and a Jinja line break inside
   the tag would inject a newline plus indentation between the attributes. #}
{% macro ident_chip(c, kind, text) %}
<label class="ident-chip{{ ' current' if c.identity_kind == kind }}">
  <input type="radio" name="ident-{{ c.label }}" value="{{ kind }}"{{ ' checked' if c.identity_kind == kind }}>
  <span>{{ text }}</span>
</label>
{% endmacro %}

{# What choosing `kind` would destroy. link_speaker clears local_slug/local_role
   and assign_local_person clears politician_*, both to hold migration 623's
   one-identity-per-speaker invariant; clearing a mark drops its placeholder
   name. Saying so before the click is the whole point — it used to happen
   silently. #}
{% macro ident_cost(c, kind) %}
  {% if kind != c.identity_kind %}
    {% if c.identity_kind == 'local' %}
    <p class="ident-cost">Saving this drops the local person <code>{{ c.local_slug }}</code>.</p>
    {% elif c.identity_kind == 'roster' %}
    <p class="ident-cost">Saving this drops the roster link.</p>
    {% elif c.identity_kind == 'unidentified' %}
    <p class="ident-cost">Saving this clears the unidentified mark.</p>
    {% elif c.identity_kind == 'non_speaker' %}
    <p class="ident-cost">Saving this clears the not-a-speaker mark.</p>
    {% endif %}
  {% endif %}
{% endmacro %}

{% macro card(c, meeting_id, all_cards, local_role_options=[]) %}
<div class="card {{ 'confirmed' if c.is_confirmed else 'attention' }}">
  <div class="card-head">
    <span class="label">{{ c.label }}</span>
    <span class="cname">{{ c.display_name }}</span>
    <span class="identpill ident-{{ c.identity_kind }}">{{ c.identity_pill }}</span>
    {% if c.confidence > 0 %}<span class="conf">conf {{ '%.2f'|format(c.confidence) }}</span>{% endif %}
    <span class="mins">{{ '%.1f'|format(c.minutes) }}m · {{ c.seg_count }} segs</span>
  </div>

  {% if c.duplicate_labels %}
  <div class="dup-name">
    ⚠ Same name as {{ c.duplicate_labels|join(', ') }} — two labels can't be the same person
    {% for other in c.duplicate_labels %}
    <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/merge" class="merge-inline"
          data-merge-mismatch="{{ other if other in c.merge_mismatches else '' }}">
      <input type="hidden" name="target" value="{{ other }}">
      <button type="submit">Merge into {{ other }}{% if c.merge_hints.get(other) %} ({{ c.merge_hints[other] }}){% endif %}</button>
    </form>
    {% endfor %}
  </div>
  {% endif %}

  {% for hname, hscore in c.hints %}
    <div class="hint">▸ voice match: {{ hname }} ({{ '%.2f'|format(hscore) }})</div>
  {% endfor %}
  {% if c.sample_text %}<p class="sample">“{{ c.sample_text[:200] }}”</p>{% endif %}
  {% if c.clip_seeks %}
  <div class="clips">
    {% for s in c.clip_seeks %}
    <button type="button" class="clip" data-seek="{{ '%.2f'|format(s) }}">▶ clip {{ loop.index }}</button>
    {% endfor %}
  </div>
  {% endif %}

  <div class="ident">
    <h3>Who is this?</h3>
    <div class="ident-chips">
      {{ ident_chip(c, 'roster', 'Roster politician') }}
      {{ ident_chip(c, 'local', 'Local person') }}
      {{ ident_chip(c, 'unidentified', 'Unidentified') }}
      {{ ident_chip(c, 'non_speaker', 'Not a speaker') }}
    </div>

    <div class="ident-panel" data-ident="roster" {{ 'hidden' if c.identity_kind != 'roster' }}>
      <p class="ident-help">Someone on an essentials roster — an officeholder or a candidate.</p>
      {% if c.identity_kind == 'roster' %}
      <p class="ident-now">Currently: <strong>{{ c.display_name }}</strong>
        <code class="pslug">{{ c.politician_id or c.politician_slug }}</code></p>
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/unlink">
        <button type="submit" class="unlink">Unlink</button>
      </form>
      {% endif %}
      {{ ident_cost(c, 'roster') }}
      {# The search results are rendered by workspace.js as one POST form per
         hit, each carrying that person's name so speaker_name cannot disagree
         with the person linked. #}
      <div class="link-search"
           data-search-url="/api/politicians/search"
           data-link-action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/link">
        <input type="text" placeholder="Search a name…" autocomplete="off">
        <div class="link-results"></div>
      </div>
    </div>

    <div class="ident-panel" data-ident="local" {{ 'hidden' if c.identity_kind != 'local' }}>
      <p class="ident-help">A person on this site only — not on any roster.
        The name below is the one readers see.</p>
      {% if c.identity_kind == 'local' %}
      <p class="ident-now">Currently: <strong>{{ c.display_name }}</strong>
        <code>{{ c.local_slug }}</code>{% if c.local_role %} · {{ c.local_role }}{% endif %}</p>
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/local-person/clear">
        <button type="submit" class="unlink">Clear local person</button>
      </form>
      {% endif %}
      {{ ident_cost(c, 'local') }}
      {# name is required because publish writes `speaker_name or slug` as a local
         person's PUBLIC name: without it a reader sees the raw slug. role is
         required and blank because resolve_local_role('') silently returns
         local_roles_for(kind)[0] — 'candidate' for a news_clip, which is wrong
         for an anchor and was accepted without a word. #}
      {# Neither prefill may carry a marked speaker's placeholders across. Under an
         'unidentified' mark, local_slug is the synthetic
         unidentified-<meeting>-<label> handle and speaker_name is 'Unidentified
         Speaker'; under 'non_speaker' the name is 'Non-speaker'. Offering either
         as the default would invite the reviewer to publish a voice-profile key
         or a status word as a person. So both fall back for any marked state. #}
      {% set is_marked = c.identity_kind in ('unidentified', 'non_speaker') %}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/local-person"
            class="local-person">
        <label>Name
          <input type="text" name="name" value="{{ '' if is_marked else (c.name or '') }}" required
                 autocomplete="off" placeholder="Brian Sterling"></label>
        <label>Slug
          <input type="text" name="slug" value="{{ c.local_slug if c.identity_kind == 'local' else c.default_slug }}" required
                 pattern="[a-z0-9][a-z0-9_-]{0,99}"
                 title="lowercase letters, digits, hyphen or underscore"></label>
        <label>Role
          <input type="text" name="role" value="" required
                 list="roles-{{ c.label }}" autocomplete="off" placeholder="choose or type…"></label>
        <datalist id="roles-{{ c.label }}">
          {% for r in local_role_options %}<option value="{{ r }}"></option>{% endfor %}
        </datalist>
        <button type="submit">Save as local person</button>
      </form>
    </div>

    <div class="ident-panel" data-ident="unidentified" {{ 'hidden' if c.identity_kind != 'unidentified' }}>
      <p class="ident-help">A real, distinct person whose name we don't know. Gets a
        private handle so their voice never merges with another stranger's.</p>
      {% if c.identity_kind == 'unidentified' %}
      <p class="ident-now">Marked unidentified <code>{{ c.local_slug }}</code></p>
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/clear-status">
        <button type="submit" class="unlink">Undo this mark</button>
      </form>
      {% else %}
      {{ ident_cost(c, 'unidentified') }}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/unidentified">
        <button type="submit" class="mark">Mark as unidentified</button>
      </form>
      {% endif %}
    </div>

    <div class="ident-panel" data-ident="non_speaker" {{ 'hidden' if c.identity_kind != 'non_speaker' }}>
      <p class="ident-help">Not a person at all — music, a pledge, a station ID.
        Never enrolled as a voice.</p>
      {% if c.identity_kind == 'non_speaker' %}
      <p class="ident-now">Marked not a speaker</p>
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/clear-status">
        <button type="submit" class="unlink">Undo this mark</button>
      </form>
      {% else %}
      {{ ident_cost(c, 'non_speaker') }}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/not-speaker">
        <button type="submit" class="mark">Mark as not a speaker</button>
      </form>
      {% endif %}
    </div>
  </div>

  <div class="also">
    <h3>Also</h3>
    <div class="actions">
      {% if not c.is_confirmed and c.accept_name %}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/name">
        <input type="hidden" name="name" value="{{ c.accept_name }}">
        <button type="submit" class="accept">✓ Accept {{ c.accept_name }}</button>
      </form>
      {% endif %}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/name" class="rename">
        <label>Display name
          <input type="text" name="name" value="{{ c.name or '' }}"
                 placeholder="Type a name…" autocomplete="off"></label>
        <button type="submit">Save</button>
      </form>
      {% if all_cards|length > 1 %}
      <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/merge" class="merge"
            data-merge-mismatch="{{ c.merge_mismatches|join(',') }}">
        <label>Same person as
          <select name="target">
            <option value="">Merge into…</option>
            {% for o in all_cards %}{% if o.label != c.label %}
            <option value="{{ o.label }}">{{ o.label }} — {{ o.display_name }}{% if c.merge_hints.get(o.label) %} ({{ c.merge_hints[o.label] }}){% endif %}</option>
            {% endif %}{% endfor %}
          </select></label>
        <button type="submit">Merge</button>
      </form>
      {% endif %}
      {% if c.is_enrollable %}
        {% if c.is_enrolled %}
        <span class="voice-saved">✓ voice saved</span>
        {% else %}
        <form method="post" action="/meetings/{{ meeting_id }}/speakers/{{ c.label }}/enroll">
          <button type="submit" class="enroll">Save this voice for future meetings</button>
          {% if c.thin_sample %}<span class="thin">⚠ short sample</span>{% endif %}
        </form>
        {% endif %}
        <span class="profile-strength {{ c.profile_strength }}" title="{% if c.profile_strength == 'strong' %}Already well-established — enroll only if this captures a new mic/setting.{% elif c.profile_strength == 'new' %}No profile yet — enrolling a clean sample here really helps future auto-ID.{% else %}Still building — a clean sample here strengthens future auto-ID.{% endif %}">{{ c.profile_hint }}</span>
      {% endif %}
    </div>
  </div>
</div>
{% endmacro %}
```

- [ ] **Step 4: Add the chooser styles**

Append to `gui/static/style.css`:

```css
/* --- Speaker identity chooser -------------------------------------------
   The four identity outcomes. All four chips always render; the checked one
   is the speaker's current identity and its panel is the only visible one. */
.identpill { font-size: 0.72rem; text-transform: lowercase; padding: 0.05rem 0.4rem;
             border-radius: 0.4rem; background: #eee; color: #666; }
.identpill.ident-roster { background: #e8eefc; color: #2a4a8a; }
.identpill.ident-local { background: #eaf5ec; color: #2b6b40; }
.identpill.ident-unidentified { background: #eeeeff; color: #444455; }
.identpill.ident-non_speaker { background: #f0f0f0; color: #777; }
.identpill.ident-none { background: #fdf0e3; color: #8a5a1a; }

.ident, .also { margin-top: 0.7rem; border-top: 1px solid #eceef2; padding-top: 0.5rem; }
.ident h3, .also h3 { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                      color: #8a94a6; margin: 0 0 0.4rem; font-weight: 600; }
.ident-chips { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.ident-chip { display: inline-flex; align-items: center; gap: 0.3rem; cursor: pointer;
              font-size: 0.85rem; padding: 0.2rem 0.55rem; border: 1px solid #d3d9e3;
              border-radius: 999px; background: #fafbfd; }
.ident-chip:hover { background: #f1f4fa; }
.ident-chip.current { border-color: #2a4a8a; background: #e8eefc; font-weight: 600; }

.ident-panel { margin-top: 0.5rem; padding: 0.5rem 0.6rem; border: 1px solid #e3e7ee;
               border-radius: 0.5rem; background: #fbfcfe;
               display: flex; flex-direction: column; gap: 0.4rem; align-items: flex-start; }
.ident-help { margin: 0; font-size: 0.8rem; color: #667; }
.ident-now { margin: 0; font-size: 0.85rem; }
.ident-cost { margin: 0; font-size: 0.8rem; color: #8a5a1a; background: #fdf3e6;
              border-radius: 0.3rem; padding: 0.2rem 0.4rem; }
.ident-panel form { display: flex; gap: 0.4rem; align-items: flex-end; flex-wrap: wrap; margin: 0; }
.ident-panel label, .also label { display: inline-flex; flex-direction: column; gap: 0.15rem;
                                  font-size: 0.72rem; color: #8a94a6; text-transform: uppercase;
                                  letter-spacing: 0.04em; }
.ident-panel label input, .also label input, .also label select { text-transform: none;
                                  letter-spacing: normal; font-size: 0.85rem; color: #222; }
```

- [ ] **Step 5: Add the reveal handler**

In `gui/static/workspace.js`, immediately before the `// ---- HLS attach` comment block, add:

```javascript
  // Identity chooser: reveal the panel for the chosen outcome. Delegated on
  // document, so it survives a panel re-fetch with no re-init — the server
  // already rendered the current outcome checked and its panel un-hidden, so
  // this only handles switching.
  document.addEventListener("change", (e) => {
    const radio = e.target;
    if (!(radio instanceof HTMLInputElement) || radio.type !== "radio") return;
    if (!radio.name.startsWith("ident-")) return;
    const block = radio.closest(".ident");
    if (!block) return;
    block.querySelectorAll(".ident-panel").forEach((p) => {
      p.hidden = p.getAttribute("data-ident") !== radio.value;
    });
  });
```

Then, in the existing `.link-search` results renderer, carry the picked
person's name into the link form. Change the `results.innerHTML = list.map(...)`
return value so the form gains a name field — replace:

```javascript
        return (
          '<form method="post" action="' + action + '">' +
          '<input type="hidden" name="politician_slug" value="' + esc(r.politician_slug) + '">' +
          '<input type="hidden" name="politician_id" value="' + esc(r.politician_id) + '">' +
          '<button type="submit" class="link-result">' + inner + "</button></form>"
        );
```

with:

```javascript
        return (
          '<form method="post" action="' + action + '">' +
          '<input type="hidden" name="politician_slug" value="' + esc(r.politician_slug) + '">' +
          '<input type="hidden" name="politician_id" value="' + esc(r.politician_id) + '">' +
          // Carry the name the reviewer just clicked, so the transcript's
          // speaker_name cannot disagree with the person linked.
          '<input type="hidden" name="name" value="' + esc(r.full_name || r.display) + '">' +
          '<button type="submit" class="link-result">' + inner + "</button></form>"
        );
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest tests/test_gui_review.py tests/test_gui_workspace.py -v
```

Expected: PASS. Two pre-existing render tests are known to be at risk, and both
should still pass — check rather than assume:

- `test_review_page_has_link_widget_and_unlink` wants `link-search`,
  `/api/politicians/search`, and the SPEAKER_00 `unlink` action. All three
  survive: the search widget moves inside the roster panel, and the Unlink form
  is rendered there whenever `identity_kind == 'roster'`. That test links
  SPEAKER_00 by `politician_slug`, which `identity_kind` reads as `roster`.
- `test_status_badge_renders` wants `"not-a-speaker"` or `"non-speaker"` in the
  body after marking SPEAKER_01. The new pill reads `not a speaker` with spaces,
  which matches NEITHER. Update that test to assert on the new contract:
  `assert ">not a speaker<" in body`, and note the change in the commit message.

If any other pre-existing assertion fails, it is asserting on markup this task
deliberately replaced: update the assertion to the new contract and say so in
the commit message. Do NOT delete such a test.

- [ ] **Step 7: Commit**

```bash
git add gui/templates/panels/_macros.html gui/static/style.css gui/static/workspace.js tests/test_gui_review.py
git commit -m "feat(gui): 'Who is this?' chooser for the four identity outcomes

The card's single flat row mixed four unlabelled text inputs (name, slug,
role, politician search) with merge and enroll, and it gated the
make-local-person form on \`not c.is_linked\` and \`speaker_status is none\`.
So a roster-linked or already-marked speaker rendered NO local-person control
and the reviewer had to clear the other field first — and the local path is
the common case, since most speakers needing hand-identification are anchors,
reporters, officials and members of the public, not politicians.

Now: all four outcomes always render as chips, the checked one is the current
identity, and each panel says what saving it would drop. The server renders
the reveal, so the initial paint needs no JavaScript. Rename, merge and
enroll move to their own row — they are operations, not identities.

The role input no longer prefills local_role_options[0], which was
'candidate' for a news_clip and reached readers unchallenged."
```

---

### Task 6: Full suite, then verify in the browser

**Files:**
- Modify: `docs/superpowers/specs/2026-09-08-speaker-identity-picker-design.md` (status line only)
- Test: the whole suite

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: a verified feature and a spec marked implemented.

- [ ] **Step 1: Run the full suite**

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m pytest -q
```

Expected: `2360 passed, 3 skipped` plus the tests this plan adds. The measured
baseline before this work was exactly `2360 passed, 3 skipped`; the 3 skips need
`DATABASE_URL` exported and are expected.

If anything outside `tests/test_gui_review.py` or
`tests/test_review_local_people.py` fails, stop and read it — this plan touches
no other subsystem, so such a failure is a real regression, not a stale
assertion.

- [ ] **Step 2: Serve this worktree**

The GUI already running on port 8000 serves the MAIN checkout, not this
worktree, so it will not show these changes. This worktree has no `.venv`.
Serve the worktree on a free port with the main checkout's interpreter:

```bash
/Users/chrisandrews/Documents/GitHub/on-the-record/.venv/bin/python -m uvicorn gui.asgi:app --host 127.0.0.1 --port 8010
```

Run it from the worktree root so `gui.asgi` resolves to the worktree's code.

- [ ] **Step 3: Walk the reachability table in the browser**

Open `http://127.0.0.1:8010/meetings/2026-05-12-interview?tab=review`. This is
the real KTLA interview from the bug report: Annie Rose Ramos (local person),
Xavier Becerra (roster-linked), Frank Buckley (no identity), Jessica Holmes
(local person).

Note: the page embeds a YouTube iframe that blanks screenshots. Remove it first
with `javascript_tool`: `document.querySelectorAll('iframe').forEach(f=>f.remove())`.

Confirm each of these:

1. Every one of the four cards shows all four chips, with exactly one marked current.
2. **Xavier Becerra's card reaches the local-person panel with no Unlink** — the reported bug.
3. Annie Rose Ramos's roster panel says it would drop `annie-rose-ramos`.
4. The local-person panel's Role box is empty, not `candidate`.
5. Clicking a chip reveals that panel and hides the others, with no page reload.
6. Mark a speaker not-a-speaker, then use **Undo this mark**, and confirm the card returns to `no identity` in *Needs attention*.

Do this on a scratch copy if you would rather not write to
`~/CouncilScribe/meetings/2026-05-12-interview` — that meeting is live on the
site. `cp -R` the directory under a new meeting id and drive that instead.

- [ ] **Step 4: Mark the spec implemented**

Change the spec's status line from
`Status: design approved, not yet implemented` to
`Status: implemented`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-09-08-speaker-identity-picker-design.md
git commit -m "docs(review): mark the identity-picker spec implemented"
```

---

## Self-review

**Spec coverage.** Every numbered problem in the spec maps to a task:

| Spec problem | Task |
|---|---|
| 1. linked speaker has no local-person control | 5 (test named for it) |
| 2. marked speaker loses it; `speaker_status` never cleared | 1, 3, 4, 5 |
| 3. four unlabelled inputs in one row | 5 (`<label>` on every input; blocks split) |
| 4. role prefills `candidate` | 5 (blank + required) |
| 5. politician search silently destroys a local person | 5 (`ident_cost` macro) |
| 6. state rendered three ways | 2 (`identity_kind`/`identity_pill`), 5 (one pill) |
| 7. rename/merge/enroll mixed in | 5 (the *Also* block) |
| 8. name missing from the identity choice | 3, 4, 5 |
| Reachability table | 3, 4, 5; walked in 6 |

**Name consistency.** `clear_speaker_status(mappings, segments, label)` in Task 1
is called with exactly that signature by `_reset_and_rename` and
`apply_clear_speaker_status` in Task 3. `identity_kind` and `identity_pill` from
Task 2 are used by the Task 5 template and the Task 3 tests. `apply_link(...,
name="")` and `apply_make_local_person(..., name="")` from Task 3 are called with
`name=` by the Task 4 routes. The `ident-{{ label }}` radio group name,
`.ident-panel`, and `data-ident` are produced by the Task 5 template and consumed
by the Task 5 JS and tests.

**Known follow-up, deliberately out of scope.** `src/publish.py`
`_upsert_local_people` still publishes `speaker_name or slug`, so a nameless
local person created by a non-GUI caller can still reach readers as a slug. The
GUI can no longer produce that state, which is what this plan set out to fix.
