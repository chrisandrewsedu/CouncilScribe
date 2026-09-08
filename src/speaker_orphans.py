"""Find meetings.speakers rows whose label no longer exists in the local transcript.

Publish upserts speakers by (meeting_id, label) and only removes vanished labels
as a side effect of a full republish (publish._delete_vanished_speakers). Nothing
counts them, so a label merged away in review leaves a live row behind until some
unrelated republish happens to sweep it up — which is how
2026-07-25-jerri-green-… carried a stale row for weeks and dropped 10 -> 9
speakers on an unrelated 2026-09-08 republish.

An inflated speaker_count is the cosmetic half. The half that costs data:
memo_reconcile.match_speaker resolves a clerk-memo member's last name by
suffix-matching display_name across the meeting's speaker rows, and returns
(None, "ambiguous …") on 2+ hits — silently skipping that member's vote record.
A stale orphan is a second hit nothing else in the codebase can see: publish's
duplicate-name gate reads the local transcript's mappings, never the DB rows.
That is the Zulich-nay failure (PR #137), reachable without any rename.

Read-only. This module decides nothing; it reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from src.models import SpeakerMapping
from src.review import (
    _SURNAME_TOKEN,
    ambiguous_speaker_surnames,
    duplicate_named_speakers,
)

# The local_slug overload: an unidentified speaker's handle is
# review.make_unidentified_slug's "unidentified-<meeting>-<label>". It is the only
# signal in a meetings.speakers row that the display_name is a placeholder rather
# than an identity, and an orphan has no transcript mapping left to ask.
_UNIDENTIFIED_SLUG_PREFIX = "unidentified-"


@dataclass
class DbSpeaker:
    """Snapshot of a meetings.speakers row, plus how many segments still point at it."""
    label: str
    display_name: Optional[str] = None
    id: Optional[str] = None
    politician_slug: Optional[str] = None
    local_slug: Optional[str] = None
    segment_count: int = 0


@dataclass
class SurnameRisk:
    """A memo-matchable last word claimed by 2+ speaker rows, at least one stale."""
    surname: str
    labels: list[str]
    orphan_labels: list[str]
    names: list[str]


@dataclass
class MeetingAudit:
    slug: str
    judgeable: bool
    reason: Optional[str] = None
    orphans: list[DbSpeaker] = field(default_factory=list)
    surname_risks: list[SurnameRisk] = field(default_factory=list)
    db_row_count: int = 0
    kept_label_count: int = 0
    stored_speaker_count: Optional[int] = None

    @property
    def at_stake(self) -> bool:
        """Whether an unjudgeable verdict actually leaves risk behind.

        A meeting with no speaker rows has nothing that could be stale — a
        'scheduled' agenda row the poller created for a meeting that has not
        happened yet is the common case. One WITH rows and no local transcript is
        a genuine blind spot: the audit can neither clear it nor condemn it.
        """
        return not self.judgeable and self.db_row_count > 0

    @property
    def orphans_serving_segments(self) -> list[DbSpeaker]:
        """Orphans that published segments still point at.

        Not an obstacle to removal: publish._replace_segments deletes every
        segment for the meeting BEFORE _delete_vanished_speakers runs, so the
        orphan always has zero references by then and the NOT EXISTS guard never
        blocks in the normal flow (it is ordering insurance, as its docstring
        says). What these prove is the reverse — no publish has happened since
        the label was merged away locally, so prod is still serving those
        segments attributed to a speaker who no longer exists in the transcript.
        The meeting is stale, not just its speaker_count.
        """
        return [o for o in self.orphans if o.segment_count]


def keep_labels(meeting_data: Mapping) -> set[str]:
    """The labels a publish of this transcript would keep.

    Mirrors publish._upsert_speakers, which iterates speakers.values() and writes
    mapping.speaker_label — the field, not the dict key. Reading the key instead
    would invent phantom orphans on any transcript where the two disagree.
    """
    speakers = (meeting_data or {}).get("speakers") or {}
    return {
        (mapping.get("speaker_label") or key)
        for key, mapping in speakers.items()
    }


def speaker_status_by_label(meeting_data: Mapping) -> dict[str, Optional[str]]:
    """{label: speaker_status} from the transcript — the authoritative source for
    whether a display_name is an identity or a placeholder."""
    speakers = (meeting_data or {}).get("speakers") or {}
    return {
        (mapping.get("speaker_label") or key): mapping.get("speaker_status")
        for key, mapping in speakers.items()
    }


def _inferred_status(row: DbSpeaker) -> Optional[str]:
    slug = row.local_slug or ""
    return "unidentified" if slug.startswith(_UNIDENTIFIED_SLUG_PREFIX) else None


def _as_mappings(
    rows: Iterable[DbSpeaker], status_by_label: Mapping[str, Optional[str]]
) -> dict[str, SpeakerMapping]:
    """Adapt DB rows to what review's collision checks read, so the surname rule
    is reused rather than re-derived. Status comes from the transcript when the
    label still has a mapping, and from the local_slug overload when it doesn't."""
    return {
        row.label: SpeakerMapping(
            speaker_label=row.label,
            speaker_name=row.display_name,
            speaker_status=status_by_label.get(row.label, _inferred_status(row)),
        )
        for row in rows
    }


def surname_risks(
    rows: Iterable[DbSpeaker],
    status_by_label: Mapping[str, Optional[str]],
    orphan_labels: Iterable[str],
) -> list[SurnameRisk]:
    """Groups of speaker rows mutually ambiguous to memo_reconcile.match_speaker,
    restricted to those containing a stale row.

    Both of review's checks are needed and neither alone suffices:
    ambiguous_speaker_surnames covers different names sharing a last word, and
    duplicate_named_speakers covers identical names — which it deliberately omits
    to avoid double-warning, but which match_speaker still sees as two hits.
    A group of two healthy rows is review's warning to resolve by hand, not
    evidence of a stale row, so it is left out here.
    """
    rows = list(rows)
    orphans = set(orphan_labels)
    mappings = _as_mappings(rows, status_by_label)

    groups: dict[str, set[str]] = {}
    for surname, labels in ambiguous_speaker_surnames(mappings).items():
        groups.setdefault(surname, set()).update(labels)
    for name, labels in duplicate_named_speakers(mappings).items():
        words = name.split()
        # Same _SURNAME_TOKEN gate review applies: a last word no clerk memo can
        # name ("(Video)", "1") is not a memo-matchable surname, just noise.
        if words and _SURNAME_TOKEN.fullmatch(words[-1]):
            groups.setdefault(words[-1], set()).update(labels)

    out = []
    for surname, labels in sorted(groups.items()):
        hit = sorted(labels & orphans)
        if not hit:
            continue
        out.append(SurnameRisk(
            surname=surname,
            labels=sorted(labels),
            orphan_labels=hit,
            names=sorted({
                mappings[l].speaker_name.strip()
                for l in labels if mappings[l].speaker_name
            }),
        ))
    return out


# The one read-only query the audit needs, LEFT JOINed so a meeting with zero
# speaker rows still appears. Column order matches rows_by_meeting.
_AUDIT_QUERY = """
SELECT m.slug,
       m.speaker_count,
       sp.id::text,
       sp.label,
       sp.display_name,
       sp.politician_slug,
       sp.local_slug,
       COALESCE(sc.n, 0)
  FROM meetings.meetings m
  LEFT JOIN meetings.speakers sp ON sp.meeting_id = m.id
  LEFT JOIN (SELECT speaker_id, COUNT(*) AS n
               FROM meetings.segments
              WHERE speaker_id IS NOT NULL
              GROUP BY speaker_id) sc ON sc.speaker_id = sp.id
 ORDER BY m.slug, sp.label
"""


def audit_query(by_slug: bool = False) -> str:
    """The audit's SELECT, optionally narrowed to a %s list of slugs.

    Kept here rather than string-surgered at each call site: a replace() that
    silently stopped matching would widen the query to every meeting, or drop
    the rows entirely, and the caller would report a false clean either way.
    """
    if not by_slug:
        return _AUDIT_QUERY
    return _AUDIT_QUERY.replace(
        " ORDER BY", " WHERE m.slug = ANY(%s)\n ORDER BY", 1
    )


def rows_by_meeting(joined) -> dict[str, tuple[Optional[int], list[DbSpeaker]]]:
    """Group audit_query() rows into {slug: (stored_speaker_count, [DbSpeaker])}.

    A meeting with no speaker rows arrives as a single row whose speaker columns
    are all NULL; it keeps its entry with an empty list rather than gaining a
    speaker labelled None.
    """
    out: dict[str, tuple[Optional[int], list[DbSpeaker]]] = {}
    for slug, count, sp_id, label, name, pol, local, segs in joined:
        stored, rows = out.setdefault(slug, (count, []))
        if label is None:
            continue
        rows.append(DbSpeaker(
            label=label, display_name=name, id=sp_id,
            politician_slug=pol, local_slug=local, segment_count=segs or 0,
        ))
    return out


def audit_meeting(
    slug: str,
    rows: Iterable[DbSpeaker],
    meeting_data: Optional[Mapping],
    stored_speaker_count: Optional[int] = None,
) -> MeetingAudit:
    """Compare one meeting's speaker rows against its local transcript.

    meeting_data None means the transcript is absent — the meeting cannot be
    judged, and reporting every row as stale would be a fabrication. An empty
    speakers dict is treated the same way, mirroring
    publish._delete_vanished_speakers: an empty keep-set is a malformed artifact,
    not an instruction to wipe the meeting.
    """
    rows = list(rows)
    base = dict(slug=slug, db_row_count=len(rows),
                stored_speaker_count=stored_speaker_count)

    if meeting_data is None:
        return MeetingAudit(judgeable=False, reason="no local transcript", **base)

    keep = keep_labels(meeting_data)
    if not keep:
        return MeetingAudit(
            judgeable=False,
            reason="local transcript has no speaker mappings",
            kept_label_count=0,
            **base,
        )

    orphans = [row for row in rows if row.label not in keep]
    return MeetingAudit(
        judgeable=True,
        orphans=orphans,
        surname_risks=surname_risks(
            rows, speaker_status_by_label(meeting_data),
            [o.label for o in orphans],
        ),
        kept_label_count=len(keep),
        **base,
    )


def orphan_details(audits: Iterable[MeetingAudit]) -> dict[str, str]:
    """{slug: one-line description of its orphan rows}, for check_consistency.py.

    The value carries no slug: it merges into that script's existing
    per-meeting drift line, so a meeting with both a segment mismatch and an
    orphan reports once and counts once.

    Unjudgeable meetings are absent: this feeds an exit code, and a missing
    local transcript is not evidence of drift in prod.

    A surname collision is called out inline, naming the surviving rows it
    collides with, because that is the half which costs data — a memo member's
    vote record silently skipped. The rest is an inflated speaker_count.
    """
    out: dict[str, str] = {}
    for audit in audits:
        if not audit.orphans:
            continue
        stale = ", ".join(
            f"{o.label} ({o.display_name!r})" for o in audit.orphans
        )
        detail = f"{len(audit.orphans)} orphan speaker row(s): {stale}"
        if audit.surname_risks:
            collisions = []
            for risk in audit.surname_risks:
                others = [l for l in risk.labels if l not in risk.orphan_labels]
                with_whom = ", ".join(others) if others else "another stale row"
                collisions.append(f"{risk.surname.title()!r} shared with {with_whom}")
            detail += (" — AMBIGUOUS to memo matching, a vote record for that "
                       f"member would be skipped: {'; '.join(collisions)}")
        out[audit.slug] = detail
    return out


def stale_publish_warnings(audit: MeetingAudit) -> list[dict]:
    """`review.enrollment_warnings`-shaped entries for the GUI review page.

    Same {kind, label, detail} shape those use, so the "Before you publish"
    block renders these with no template change — it keys off `kind` and prints
    `detail`.

    This is the only signal a GUI reviewer can get. Label surgery (merge,
    rename, mark-unidentified) rewrites the local transcript and nothing in the
    GUI consults prod, so dropping a label from a LIVE meeting looks finished
    while the old row keeps serving. `enrollment_warnings` itself cannot cover
    this: it reads the local mappings, and an orphan exists only in the DB.

    One entry per orphan label so each names its own row, matching how the
    reviewer acts on them.
    """
    by_label = {
        label: risk
        for risk in audit.surname_risks
        for label in risk.orphan_labels
    }
    warns = []
    for orphan in audit.orphans:
        detail = (
            f"the live site still has speaker {orphan.label} "
            f"({orphan.display_name!r}), which this transcript no longer has — "
            "republish to remove it"
        )
        risk = by_label.get(orphan.label)
        if risk:
            others = [l for l in risk.labels if l not in risk.orphan_labels]
            detail += (
                f"; it shares the last name {risk.surname.title()!r} with "
                f"{', '.join(others) or 'another stale row'}, so a memo vote "
                "record for that member is being skipped as ambiguous"
            )
        warns.append({
            "kind": "stale_published_speaker",
            "label": orphan.label,
            "detail": detail,
        })
    return warns
