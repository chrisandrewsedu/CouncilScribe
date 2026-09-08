# Speaker identity picker — GUI review page

Date: 2026-09-08
Status: design approved, not yet implemented

## Problem

The review page's speaker card offers four identity outcomes — roster politician,
local person, unidentified, not-a-speaker — but they are not equally reachable,
and the card does not say plainly which one is currently in force.

Measured on the live GUI at `/meetings/2026-05-12-interview?tab=review` (the KTLA
interview: Annie Rose Ramos, Xavier Becerra, Frank Buckley, Jessica Holmes):

1. **A linked speaker has no local-person control at all.** `_macros.html`
   gates the make-local-person form on `{% elif not c.is_linked %}`, so Xavier
   Becerra's card renders no such form. The reviewer must click **Unlink** first.
   This is the reported "must clear another field first".
2. **A speaker marked unidentified or not-a-speaker also loses it**, via
   `{% if c.speaker_status is none %}`. Worse, nothing anywhere in `src/` or
   `gui/` ever clears `speaker_status`, so both marks are one-way doors: the
   label can never hold a real identity again through the GUI.
3. **Four unlabelled text inputs share one wrapping flex row** — rename name,
   local slug, local role, politician search. Only placeholder text
   distinguishes them.
4. **The role field prefills `local_role_options[0]`.** For a `news_clip` that
   is `candidate` (news_clip is absent from `LOCAL_ROLE_SETS`, so it falls back
   to `DEFAULT_LOCAL_ROLES`), which is wrong for a TV anchor and is accepted
   silently.
5. **The politician search stays visible when a local person is set**, and
   picking a result silently destroys that local person, because `link_speaker`
   clears `local_slug`/`local_role`. No warning is shown.
6. **Current identity is rendered three different ways in three places**:
   `🔗 linked: <uuid>` above the action row, `local: slug · role` inside it, and
   `statusbadge` chips near the card head.
7. Rename, merge and voice-enroll are mixed into the same row as the identity
   choice, though none of them is an identity outcome.
8. **The name is missing from the choice that needs it most.** `src/publish.py`
   writes `mapping.speaker_name or slug` as a local person's public name, so a
   local person made without a name is published to readers as the raw slug —
   `frank-buckley`. The make-local-person form collects a slug and a role but
   no name; the reviewer is expected to notice the separate rename box.

This matters because the local-person path is the common case for the speakers
that actually need hand-identification — news anchors, reporters, party
officials, members of the public — yet today it is the hardest one to reach.

## Non-goals

- The one-identity-per-speaker invariant (ev-accounts migration 623) does not
  change. `link_speaker` and `assign_local_person` keep their current bodies.
- The terminal review flow (`run_local.py --review`) is not touched.
- Publish, the confidence gate, enrollment and merge semantics are not touched.
- No new database column and no new prod query. In particular the linked panel
  must not add a per-speaker politician lookup to page render.

## Design

### Card structure

Three stacked blocks replace the single flat `.actions` row.

```
SPEAKER_04   Frank Buckley   [local person]   conf 1.00 · 0.6m · 3 segs
▸ voice match: …
“sample text …”
▶ clip 1  ▶ clip 2  ▶ clip 3

── Who is this? ────────────────────────────────────────────
 (•) Local person   ( ) Roster politician   ( ) Unidentified   ( ) Not a speaker
 ┌ Local person ─────────────────────────────────────────┐
 │ A person on this site only — not on any roster.       │
 │ Name [ Frank Buckley ]                                │
 │ Slug [ frank-buckley ]   Role [ ▾ required ]          │
 │ [ Save as local person ]                              │
 └───────────────────────────────────────────────────────┘

── Also ───────────────────────────────────────────────────
 Display name  [ Frank Buckley ] [ Save ]
 Same person as [ Merge into… ▾ ] [ Merge ]
 Voice ✓ saved · Profile new
```

Every card renders all four chips, always — no chip is ever conditionally
hidden. Exactly one is `checked`, matching the speaker's current identity.
Selecting a chip reveals that chip's panel and hides the other three.

The chips are plain `<input type="radio">`, one radio group per speaker label,
and the server renders the correct chip `checked` with the correct panel
un-hidden. The initial paint therefore needs no JavaScript; only switching does.

### The four panels

| Chip | Panel when not current | Panel when current |
|---|---|---|
| Roster politician | search box + results (existing `.link-search` widget) | `Currently: Xavier Becerra` · roster id `0f74219c…` + **Unlink** |
| Local person | name, slug (prefilled from `default_local_slug`), role, **Save as local person** | `Currently: Annie Rose Ramos · annie-rose-ramos · anchor_ktla5` + **Clear local person** |
| Unidentified | one line of explanation + **Mark as unidentified** | `Marked unidentified` · handle + **Undo this mark** |
| Not a speaker | one line of explanation + **Mark as not a speaker** | `Marked not a speaker` + **Undo this mark** |

Mutual exclusivity is stated rather than left to be discovered. When a different
identity is already set, the panel the reviewer opens carries a line naming what
saving will drop, for example:

> Saving this drops the local person `annie-rose-ramos`.
> Saving this also clears the not-a-speaker mark.

Two smaller corrections belong inside this block:

- The roster panel names the **person**, taken from the card's own
  `speaker_name`, with the id shown small as provenance. The bare UUID alone
  tells the reviewer nothing. No database call is added.
- The role field starts **blank and `required`**, replacing the
  `local_role_options[0]` prefill. `resolve_local_role("")` returns
  `roles[0]`, so an empty submission would silently store `candidate` for a
  `news_clip`. Requiring an explicit role is a form-level change only; the
  server's coercion rules are unchanged.

### The name travels with the identity

An identity without a name is not publishable: `_upsert_local_people` writes
`mapping.speaker_name or slug`, so a nameless local person reaches readers as
the raw slug. Both real-identity panels therefore carry the name.

- The **local person** panel gains a `name` field, `required`, prefilled from
  the card's current name. `apply_make_local_person` renames before assigning.
- The **roster** panel sends the picked person's name with the link — the
  reviewer has just clicked that name, so this costs them nothing and stops the
  transcript's `speaker_name` from disagreeing with the linked person.
  `apply_link` renames before linking.

Ordering is load-bearing, and two separate constraints pin it down:

- `clear_speaker_status` blanks the placeholder name, so it must run **before**
  the rename. Otherwise it would erase the name just supplied.
- `rename_speaker` drops any prior identity when the name changes — it treats
  the old link as belonging to the old name — so it must run **before** the
  assignment. Otherwise it would erase the link just made.

So each write is: **clear status → rename → assign identity**. Each step is
skipped when it has nothing to do: `clear_speaker_status` no-ops when the status
is already clear, and the rename is skipped when no name was supplied.

The `name` form field stays **optional** on both routes. When it is absent the
routes behave exactly as they do today, so existing callers and tests are
unaffected.

The **Display name** box in the *Also* block stays, and writes the same
`speaker_name` the local-person panel's Name field writes. It is the general
escape hatch — the only way to name a roster-linked or marked speaker — so the
two are one field reached from two places, not two competing values. The
local-person panel labels its input `Name` and says it is the name readers will
see, because for a local person that name is the published one.

### Current state, in one place

New derived property on `SpeakerCard`:

```python
@property
def identity_kind(self) -> str:
    """'roster' | 'local' | 'unidentified' | 'non_speaker' | 'none'."""
```

It uses the same precedence as `src/review.py:70 identity_label` — status beats
links, and `politician_id`/`politician_slug` beat `local_slug` — so the picker
can never disagree with what publish will store. No new stored field.

The three scattered state renderings collapse into one identity pill in the card
head, reading `roster` · `local` · `unidentified` · `not a speaker` ·
`no identity`; the detail lives inside the chosen panel.

When `identity_kind` is `none`, no chip is checked and no panel is revealed. The
four chips are then the call to action.

### The way back

New in `src/review.py`:

```python
def clear_speaker_status(mappings, segments, label):
    """Drop 'unidentified' / 'non_speaker' so the label can hold a real identity
    again. Returns None (no mutation) when the label is unknown or its status is
    already clear — a no-op is not success."""
```

Behaviour:

- Sets `speaker_status = None`.
- Clears `local_slug` and `local_role` when the cleared status was
  `unidentified`. That slug is the synthetic `unidentified-<meeting>-<label>`
  handle written by `mark_unidentified` / `link_to_unidentified_handle`, not a
  real local person. Leaving it behind would present a private voice-profile
  handle as a site-local person.
- Clears `speaker_name` and syncs the affected segments. The stored name is a
  status placeholder (`"Unidentified Speaker"`, `"Non-speaker"`, or a reviewer's
  `display_label` for the marked state) and not a person's name, so it must not
  survive the status it labelled.
- Resets `confidence` to `0.0` and `id_method` to `None`, so the speaker returns
  to *Needs attention* — which is true: it now has no identity.

`link_speaker` and `assign_local_person` are **not** changed. Instead
`gui/review_api.py` calls `clear_speaker_status` first inside `apply_link` and
`apply_make_local_person`, so choosing a real identity from a marked state takes
one click. This confines the behaviour change to the GUI path and leaves the
terminal flow byte-identical.

A speaker whose mark is cleared without a replacement name is left nameless on
purpose — it has no identity, and *Needs attention* is where it belongs. The
GUI cannot publish that state as a local person, because the local-person panel
requires a name; a caller outside the GUI still could, which is the same
exposure `_upsert_local_people` already carries today.

One new route, `POST /meetings/{meeting_id}/speakers/{label}/clear-status`,
backs the **Undo this mark** buttons, mirroring the existing
`local-person/clear` route: `404` on an unsafe or unknown meeting or label, and
`404` when `clear_speaker_status` no-ops.

### Reachability, after the change

From any starting state, each of the four outcomes is one interaction away:

| From \ To | roster | local | unidentified | non-speaker |
|---|---|---|---|---|
| none | search + pick | name + slug + role + save | mark | mark |
| roster | (current) + Unlink | name + slug + role + save | mark | mark |
| local | search + pick | (current) + Clear | mark | mark |
| unidentified | search + pick | name + slug + role + save | (current) + Undo | mark |
| non-speaker | search + pick | name + slug + role + save | mark | (current) + Undo |

No cell requires clearing another field first. Every cell that reaches a real
identity also lands a name, so no path can publish a speaker as a bare slug.

## Files touched

| File | Change |
|---|---|
| `src/review.py` | new `clear_speaker_status` |
| `gui/models.py` | new `SpeakerCard.identity_kind` |
| `gui/review_api.py` | `apply_link` / `apply_make_local_person` take an optional name and do rename → clear status → assign; new `apply_clear_speaker_status` |
| `gui/app.py` | new `clear-status` route; optional `name` form field on `link` and `local-person` |
| `gui/templates/panels/_macros.html` | card rewritten into the three blocks |
| `gui/static/style.css` | chip and panel styles |
| `gui/static/workspace.js` | one delegated `change` listener that reveals the chosen panel |
| `tests/test_review.py` | `clear_speaker_status` unit tests |
| `tests/test_gui_review.py` | `identity_kind`, api, route and render tests |

## Tests

Written first, red before green.

`tests/test_review.py` — `clear_speaker_status`:

- clears `speaker_status` and the synthetic handle for an `unidentified` mapping
- clears `speaker_status` for a `non_speaker` mapping and leaves no local slug
- clears the placeholder name on the mapping and on every matching segment
- resets `confidence` and `id_method`
- returns `None` for an unknown label
- returns `None` for a mapping whose status is already `None`, and leaves a
  genuine local person's slug and role untouched

`tests/test_gui_review.py`:

- `identity_kind` returns each of the five values, with `speaker_status`
  winning over a stray politician link and `politician_id` winning over
  `local_slug`
- `apply_link` on a marked speaker clears the mark and stores the link
- `apply_make_local_person` on a marked speaker clears the mark and stores the
  local person
- `apply_link` with a name stores both the name and the link — the ordering
  regression test, because a rename after the link would wipe it
- `apply_make_local_person` with a name stores the name, slug and role together
- both keep their current behaviour when no name is supplied
- `apply_clear_speaker_status` returns `False` for an unknown meeting or label
  and `False` on a no-op; the route returns `404` in those cases and `303`
  on success
- render: **a linked speaker's card contains the local-person panel** — the
  regression that prompted this work
- render: a card with a local person contains the roster panel and a line naming
  what saving would drop
- render: all four chips appear on every card, with exactly one checked
- render: a marked card carries an undo control
- render: the role input is empty and `required`, and the local-person panel
  carries a `required` name input prefilled from the card's name
- `gui/static/workspace.js` carries the panel-reveal handler

## Verification

`.venv/bin/python -m pytest tests/test_review.py tests/test_gui_review.py` green,
then the full suite. Then the GUI in the browser on
`/meetings/2026-05-12-interview?tab=review`, checking each row of the
reachability table above against the four real speakers on that page, including
Xavier Becerra (linked) reaching the local-person path with no Unlink.
