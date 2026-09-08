"""Publish processed meetings to the meetings.* schema via direct Postgres.

Idempotent by meeting slug:
  - Meeting row: upsert keyed on slug (INSERT or UPDATE)
  - Speaker rows: upsert keyed on (meeting_id, label)
  - Segment rows: delete-then-insert so re-publishes never leave orphan rows

Word-level timestamps are deliberately not published (they stay in
transcript.json on disk).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import psycopg2
import psycopg2.extras

from .event_entities import validate_event_entities
from .event_kinds import validate_event_kind
from .models import Meeting

SEGMENT_BATCH_SIZE = 500

# Ownership partition for meetings.votes: each writer deletes/inserts only
# its own vote_type stripe, so a re-publish can never wipe memo-reconciled
# votes and a re-reconcile can never wipe federal floor votes.
FLOOR_VOTE_TYPE = "recorded"     # written by _replace_votes (federal CREC)
MEMO_VOTE_TYPE = "roll call"     # written by reconcile_memo (clerk memo)

_DIRECT_FILE_EXTENSIONS = (".mp4", ".m4v", ".mov", ".webm")
_AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".ogg", ".wav")

_YOUTUBE_PATH_PREFIXES = ("/embed/", "/shorts/", "/live/", "/v/")


def extract_youtube_id(url: str) -> Optional[str]:
    """Return the YouTube video id for any common YouTube URL shape, else None."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")

    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None

    if host in ("youtube.com", "youtube-nocookie.com", "music.youtube.com"):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
            return vid or None
        for prefix in _YOUTUBE_PATH_PREFIXES:
            if parsed.path.startswith(prefix):
                vid = parsed.path[len(prefix):].split("/")[0]
                return vid or None

    return None


def resolve_playback(audio_source: str) -> tuple[Optional[str], Optional[str]]:
    """Map an audio_source to a (playback_kind, playback_url) pair for the site.

    Kinds: 'youtube' (url is the video id), 'file' (direct media URL),
    'audio' (direct MP3/M4A/etc. podcast or radio enclosure), 'hls' (.m3u8).
    Unknown providers return (None, None); the site renders transcript-only
    with a plain source link.
    """
    source = (audio_source or "").strip()
    if not source.startswith(("http://", "https://")):
        return None, None

    video_id = extract_youtube_id(source)
    if video_id:
        return "youtube", video_id

    parsed = urlparse(source)
    path = parsed.path.lower()

    if "catstv.net" in parsed.netloc:
        try:
            from .download import _extract_blob_url_from_page

            return "file", _extract_blob_url_from_page(source)
        except Exception:
            return None, None

    if path.endswith(_DIRECT_FILE_EXTENSIONS):
        return "file", source

    if path.endswith(_AUDIO_EXTENSIONS):
        return "audio", source

    if path.endswith(".m3u8"):
        return "hls", source

    return None, None


def playback_for_meeting(meeting) -> tuple[Optional[str], Optional[str]]:
    """(playback_kind, playback_url) for a meeting.

    Prefers a resolved enclosure URL (podcast/CMS audio file) over the citation
    audio_source, which for those sources is the human-facing episode page URL
    (kept as the citation source_url, not a playable media URL).
    """
    pm = getattr(meeting, "processing_metadata", None)
    enclosure = getattr(pm, "source_audio_url", None) if pm else None
    return resolve_playback(enclosure or (meeting.audio_source or ""))


@dataclass
class PublishResult:
    meeting_id: str
    segments: int
    speakers: int
    # Stale meetings.speakers rows this publish swept away (labels the transcript
    # no longer has). Returned rather than only printed: in the GUI publish runs
    # in-process, so the print lands in the uvicorn terminal and the reviewer
    # never learns a stale row existed. Defaults to 0 for positional callers.
    removed_speakers: int = 0


def _require_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "Publishing requires DATABASE_URL (add it to .env.local). "
            "Get it from Supabase dashboard: Project Settings → Database → "
            "Connection string (URI mode, port 5432)."
        )
    return url


def _validate_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise RuntimeError(
            f"Meeting date {date_str!r} is not YYYY-MM-DD; cannot publish. "
            "Fix the date in transcript_named.json and retry."
        )


def _resolve_chamber_id(cur, body_slug: Optional[str]) -> Optional[str]:
    if body_slug is None:
        return None

    cur.execute(
        """
        SELECT id
        FROM essentials.chambers
        WHERE slug = %s
        ORDER BY id
        LIMIT 2
        """,
        (body_slug,),
    )
    rows = cur.fetchall()
    if len(rows) != 1:
        # Multi-chamber body slugs (e.g. full council) can't be pinned to one
        # seat — treat as unchambered rather than blocking publish.
        return None
    return str(rows[0][0])


def resolve_races_for_politicians(cur, politician_ids) -> list[str]:
    """All distinct essentials races the given linked politicians belong to.

    A meeting's races are the union of its linked candidates' races. Returns
    every distinct race_id (no "exactly one" gate) so multi-race forums are
    represented; [] when there are no ids or no race_candidates rows. Casts to
    uuid[] because essentials.race_candidates.politician_id is a uuid column and
    psycopg2 sends a Python list as text[].
    """
    ids = [pid for pid in (politician_ids or []) if pid]
    if not ids:
        return []
    cur.execute(
        """
        SELECT DISTINCT race_id
        FROM essentials.race_candidates
        WHERE politician_id = ANY(%s::uuid[])
        """,
        (ids,),
    )
    return [str(r[0]) for r in cur.fetchall()]


def _reconcile_event_races(cur, meeting: Meeting, meeting_uuid: str) -> list[str]:
    """Derive the meeting's races from its linked candidates and reconcile the
    meetings.event_races join table (delete this meeting's rows, insert the
    current set). Returns the race ids written.

    debate/forum require >=1 derived race: an empty set raises (aborting the
    publish transaction) — recoverable by linking candidates, then re-publishing.
    council/school_board legitimately have no races; an empty set just clears
    stale rows.
    """
    pol_ids = [m.politician_id for m in meeting.speakers.values() if m.politician_id]
    races = resolve_races_for_politicians(cur, pol_ids)

    if not races and meeting.event_kind in ("debate", "forum"):
        raise RuntimeError(
            f"{meeting.meeting_id}: {meeting.event_kind} resolved to no race — "
            "no linked candidate maps to an essentials race yet. Link candidates, "
            "then re-publish."
        )

    cur.execute("DELETE FROM meetings.event_races WHERE meeting_id = %s", (meeting_uuid,))
    for race_id in races:
        cur.execute(
            "INSERT INTO meetings.event_races (meeting_id, race_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (meeting_uuid, race_id),
        )
    return races


def _upsert_meeting(cur, meeting: Meeting, body_slug: Optional[str]) -> str:
    """Insert or update the meeting row. Returns the meetings.meetings UUID."""
    # Backstop: never let a guessed/missing classification reach the DB.
    validate_event_kind(meeting.event_kind or "")  # raises ValueError if None/empty/invalid
    if not (meeting.meeting_type or "").strip():
        raise ValueError(f"{meeting.meeting_id}: meeting_type is required to publish")
    if meeting.event_kind in ("council", "school_board") and not (meeting.city or "").strip():
        raise ValueError(
            f"{meeting.meeting_id}: city is required to publish a {meeting.event_kind} meeting"
        )
    chamber_id = _resolve_chamber_id(cur, body_slug)
    entity_error = validate_event_entities(
        meeting.event_kind,
        chamber_id,
        None,
    )
    if entity_error:
        raise RuntimeError(entity_error)

    source = (meeting.audio_source or "").strip()
    is_url = source.startswith(("http://", "https://"))
    kind, playback_url = playback_for_meeting(meeting)
    date = _validate_date(meeting.date)
    summary = meeting.summary.to_dict() if meeting.summary else None
    proc_meta = (
        meeting.processing_metadata.to_dict()
        if meeting.processing_metadata
        else None
    )

    cur.execute(
        "SELECT id FROM meetings.meetings WHERE slug = %s",
        (meeting.meeting_id,),
    )
    row = cur.fetchone()

    if row:
        meeting_uuid = row[0]
        cur.execute(
            """
            UPDATE meetings.meetings SET
              city = %s,
              date = %s,
              meeting_type = %s,
              title = %s,
              event_kind = %s,
              duration_seconds = %s,
              audio_source = %s,
              video_url = %s,
              status = %s,
              chamber_id = %s,
              source_url = %s,
              playback_kind = %s,
              clip_start_seconds = %s,
              clip_end_seconds = %s,
              thumbnail_url = %s,
              summary = %s,
              processing_metadata = %s,
              updated_at = NOW()
            WHERE id = %s
            """,
            (
                meeting.city,
                date,
                meeting.meeting_type,
                meeting.title,
                meeting.event_kind,
                meeting.duration_seconds or None,
                source or None,
                playback_url,
                "published",
                chamber_id,
                source if is_url else None,
                kind,
                meeting.clip_start_seconds,
                meeting.clip_end_seconds,
                meeting.thumbnail_url,
                psycopg2.extras.Json(summary),
                psycopg2.extras.Json(proc_meta),
                meeting_uuid,
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO meetings.meetings
              (id, city, date, meeting_type, title, event_kind, duration_seconds,
               audio_source, video_url, status,
               chamber_id, source_url, playback_kind, clip_start_seconds, clip_end_seconds, thumbnail_url, slug,
               summary, processing_metadata,
               created_at, updated_at)
            VALUES
              (gen_random_uuid(), %s, %s, %s, %s, %s, %s,
               %s, %s, %s,
               %s, %s, %s, %s, %s, %s, %s,
               %s, %s,
               NOW(), NOW())
            RETURNING id
            """,
            (
                meeting.city,
                date,
                meeting.meeting_type,
                meeting.title,
                meeting.event_kind,
                meeting.duration_seconds or None,
                source or None,
                playback_url,
                "published",
                chamber_id,
                source if is_url else None,
                kind,
                meeting.clip_start_seconds,
                meeting.clip_end_seconds,
                meeting.thumbnail_url,
                meeting.meeting_id,
                psycopg2.extras.Json(summary),
                psycopg2.extras.Json(proc_meta),
            ),
        )
        meeting_uuid = cur.fetchone()[0]

    return meeting_uuid


def _upsert_event_orgs(cur, meeting_slug: str, event_orgs: list) -> None:
    """Delete then re-insert event_orgs for this meeting. Idempotent."""
    cur.execute(
        "DELETE FROM meetings.event_orgs WHERE meeting_id = %s",
        (meeting_slug,),
    )
    for org_name in event_orgs:
        cur.execute(
            """
            INSERT INTO meetings.event_orgs (id, meeting_id, org_name, created_at)
            VALUES (gen_random_uuid(), %s, %s, NOW())
            """,
            (meeting_slug, org_name),
        )


def _published_local_slug(mapping) -> "str | None":
    """The local_slug to publish for a speaker, or None to publish no local
    person. An unidentified handle is a placeholder, not a public entity, so it
    is suppressed until promoted to a real person.

    An essentials identity also suppresses it, because migration 623's invariant is
    one identity per speaker: either politician_* or local_slug, never both. The
    federal floor path breaks that on its own — crec_identify stashes the bioguide in
    local_slug for every resolved member, then resolve_politician_id adds an
    essentials link on top and nothing clears the stash — so a resolved member would
    otherwise mint a local_people row duplicating a politician who already exists."""
    if getattr(mapping, "speaker_status", None) == "unidentified":
        return None
    if mapping.politician_id or mapping.politician_slug:
        return None
    return mapping.local_slug


def _upsert_local_people(cur, meeting: Meeting) -> None:
    """Upsert local_people rows for any speaker mapping with local_slug set.

    Must be called BEFORE _upsert_speakers so the FK from meetings.speakers.local_slug
    to meetings.local_people.slug is satisfied at write time.

    `role` is written exactly as review recorded it, and NULL when review never
    recorded one. Defaulting an unset role to a concrete value would assert a
    fact nobody established: only the terminal prompt in run_local.py sets
    local_role, so every person reviewed in the GUI has none, and moderators,
    staff and public commenters would all publish as that default.
    Requires meetings.local_people.role to be nullable — made so by ev-accounts migration
    CA_0001, applied to prod 2026-08-21.
    """
    for mapping in meeting.speakers.values():
        slug = _published_local_slug(mapping)
        if not slug:
            continue
        cur.execute(
            """
            INSERT INTO meetings.local_people
              (slug, name, role, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
            ON CONFLICT (slug) DO UPDATE SET
              name = EXCLUDED.name,
              role = EXCLUDED.role,
              updated_at = NOW()
            """,
            (
                slug,
                mapping.speaker_name or slug,
                mapping.local_role,
            ),
        )


def _upsert_speakers(
    cur, meeting: Meeting, meeting_uuid: str
) -> dict[str, str]:
    """Upsert speaker rows. Returns {speaker_label: speaker_uuid}."""
    label_to_uuid: dict[str, str] = {}

    for mapping in meeting.speakers.values():
        cur.execute(
            "SELECT id FROM meetings.speakers WHERE meeting_id = %s AND label = %s",
            (meeting_uuid, mapping.speaker_label),
        )
        row = cur.fetchone()

        if row:
            speaker_uuid = row[0]
            cur.execute(
                """
                UPDATE meetings.speakers SET
                  display_name = %s,
                  politician_slug = %s,
                  politician_id = %s,
                  confidence = %s,
                  id_method = %s,
                  local_slug = %s
                WHERE id = %s
                """,
                (
                    mapping.speaker_name,
                    mapping.politician_slug,
                    mapping.politician_id,
                    mapping.confidence,
                    mapping.id_method,
                    _published_local_slug(mapping),
                    speaker_uuid,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO meetings.speakers
                  (id, meeting_id, label, display_name,
                   politician_slug, politician_id, confidence, id_method,
                   local_slug, created_at)
                VALUES
                  (gen_random_uuid(), %s, %s, %s,
                   %s, %s, %s, %s,
                   %s, NOW())
                RETURNING id
                """,
                (
                    meeting_uuid,
                    mapping.speaker_label,
                    mapping.speaker_name,
                    mapping.politician_slug,
                    mapping.politician_id,
                    mapping.confidence,
                    mapping.id_method,
                    _published_local_slug(mapping),
                ),
            )
            speaker_uuid = cur.fetchone()[0]

        label_to_uuid[mapping.speaker_label] = speaker_uuid

    return label_to_uuid


def _replace_segments(
    cur,
    meeting: Meeting,
    meeting_uuid: str,
    label_to_uuid: dict[str, str],
) -> int:
    """Delete then batch-insert segments. Returns segment count."""
    cur.execute(
        "DELETE FROM meetings.segments WHERE meeting_id = %s",
        (meeting_uuid,),
    )

    slug_by_label = {
        label: m.politician_slug for label, m in meeting.speakers.items()
    }

    rows = []
    for seg in meeting.segments:
        if not seg.text:
            continue
        rows.append((
            meeting_uuid,
            label_to_uuid.get(seg.speaker_label),
            seg.segment_id,
            seg.start_time,
            seg.end_time,
            seg.text,
            seg.speaker_name,
            slug_by_label.get(seg.speaker_label),
            seg.confidence,
        ))

    for i in range(0, len(rows), SEGMENT_BATCH_SIZE):
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO meetings.segments
              (meeting_id, speaker_id, segment_index,
               start_time, end_time, text,
               speaker_name, politician_slug, confidence)
            VALUES %s
            """,
            rows[i : i + SEGMENT_BATCH_SIZE],
        )

    return len(rows)


def _delete_vanished_speakers(cur, meeting_uuid: str, keep_labels) -> int:
    """Delete speaker rows whose label is no longer in the transcript. Returns the count.

    Publish upserts speakers by (meeting_id, label) and never removed ones that
    disappeared, so a label merged away in review kept its row forever — two such rows
    were live in prod, one of them a linked politician with no segments.

    🔴 Must run AFTER _replace_segments. Until the old segments are deleted they still
    reference these rows, and meetings.segments.speaker_id would block the delete. The
    NOT EXISTS guard makes that ordering safe rather than merely conventional.

    A meeting with NO labels is treated as a malformed artifact, not as an instruction to
    wipe every speaker row: publish is destructive and has no undo.
    """
    keep = sorted(set(keep_labels))
    if not keep:
        return 0
    cur.execute(
        """
        DELETE FROM meetings.speakers sp
         WHERE sp.meeting_id = %s
           AND sp.label <> ALL(%s)
           AND NOT EXISTS (
                 SELECT 1 FROM meetings.segments sg WHERE sg.speaker_id = sp.id
               )
        """,
        (meeting_uuid, keep),
    )
    return getattr(cur, "rowcount", None) or 0


def _replace_votes(cur, meeting: Meeting, meeting_uuid: str) -> int:
    """Delete then insert this meeting's recorded floor votes into meetings.votes.

    Federal floor votes carry only the vote event (roll, tally, timestamp) — the
    400+ voters are not meeting speakers and their per-member positions already
    live in essentials.legislative_votes, so meetings.vote_records is deliberately
    NOT populated here. On-the-record owns the floor-vote stripe of meetings.votes
    for meetings it publishes (delete-then-insert within the stripe, mirroring
    _replace_segments). `result` is NOT NULL; we
    store the parsed outcome + tally ("Agreed to · 236–193"), falling back to the
    bare tally ("Yea X, Nay Y") when CREC has no parseable outcome line.
    Timestamps are expected to already be source-absolute (absolutize_meeting_times).
    Deletes/inserts only the FLOOR_VOTE_TYPE stripe: memo-reconciled votes
    (MEMO_VOTE_TYPE, written by reconcile_memo) are a separate ownership
    stripe and survive re-publish untouched.
    """
    cur.execute(
        """
        DELETE FROM meetings.vote_records
        WHERE vote_id IN (SELECT id FROM meetings.votes
                          WHERE meeting_id = %s AND vote_type = %s)
        """,
        (meeting_uuid, FLOOR_VOTE_TYPE),
    )
    cur.execute(
        "DELETE FROM meetings.votes WHERE meeting_id = %s AND vote_type = %s",
        (meeting_uuid, FLOOR_VOTE_TYPE),
    )
    rows = []
    for fv in meeting.floor_votes:
        result = (f"{fv.outcome} · {fv.yea}–{fv.nay}"   # "Agreed to · 236–193"
                  if fv.outcome else f"Yea {fv.yea}, Nay {fv.nay}")
        rows.append((
            meeting_uuid,
            f"Roll No. {fv.roll_number}",        # resolution
            fv.question,                          # description
            result,                               # result (NOT NULL): outcome+tally, else tally
            FLOOR_VOTE_TYPE,                      # vote_type
            fv.timestamp,                         # numeric seconds (absolutized), NULL if unmatched
        ))
    if rows:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO meetings.votes
              (meeting_id, resolution, description, result, vote_type, timestamp)
            VALUES %s
            """,
            rows,
        )
    return len(rows)


def scheduled_slug(body, date: str) -> str:
    """Slug for an agenda-published (Pass A) meeting: '{body.slug}-{date}'.

    This is the join key the video pass must reuse so its upsert flips this
    row to published instead of creating a duplicate."""
    return f"{body.slug}-{date}"


# Column order matches the INSERT in _replace_agenda_items below.
def build_agenda_item_rows(meeting_uuid: str, items) -> list[tuple]:
    rows = []
    for it in items:
        rows.append((
            meeting_uuid,
            it.position,
            it.item_number,
            it.title_raw,
            it.kind,
            it.legislation_ref,
            it.summary_plain,
            it.decision_plain,
            it.stage,
            it.public_comment,
            it.public_comment_note,
            it.source_url,
            "upcoming",
        ))
    return rows


def _replace_agenda_items(cur, meeting_uuid: str, items) -> int:
    """Delete-then-insert, like _replace_votes. Caller must have verified the
    meeting is NOT status='published' (the video pass owns items after that).
    Duplicate positions abort the publish via the UNIQUE (meeting_id, position)
    constraint — fail-loud by design."""
    cur.execute(
        "DELETE FROM meetings.agenda_items WHERE meeting_id = %s", (meeting_uuid,)
    )
    rows = build_agenda_item_rows(meeting_uuid, items)
    if rows:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO meetings.agenda_items
              (meeting_id, position, item_number, title_raw, kind,
               legislation_ref, summary_plain, decision_plain, stage,
               public_comment, public_comment_note, source_url, status)
            VALUES %s
            """,
            rows,
        )
    return len(rows)


def publish_scheduled_meeting(body, date: str, title: str, starts_at: str,
                              source_url: str, items) -> Optional[str]:
    """Publish a future meeting + its agenda items (Pass A).

    Returns the meeting slug on success, None when skipped because the row is
    already published (video pass owns it). Idempotent: re-polls re-run this
    and delete-then-insert refreshes the items (agenda revisions/addenda).
    """
    date = _validate_date(date)
    slug = scheduled_slug(body, date)
    conn = psycopg2.connect(_require_db_url())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, status FROM meetings.meetings WHERE slug = %s",
                    (slug,),
                )
                row = cur.fetchone()
                if row and row[1] == "published":
                    return None
                if row:
                    meeting_uuid = row[0]
                    cur.execute(
                        """
                        UPDATE meetings.meetings SET
                          title = %s,
                          starts_at = %s,
                          timezone = %s,
                          source_url = %s,
                          updated_at = NOW()
                        WHERE id = %s
                        """,
                        (title, starts_at, body.timezone, source_url, meeting_uuid),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO meetings.meetings
                          (id, city, state, date, meeting_type, title, event_kind,
                           status, slug, source_url, starts_at, timezone,
                           created_at, updated_at)
                        VALUES
                          (gen_random_uuid(), %s, %s, %s, %s, %s, %s,
                           'scheduled', %s, %s, %s, %s,
                           NOW(), NOW())
                        RETURNING id
                        """,
                        (body.city, body.state, date, body.meeting_type, title,
                         body.event_kind, slug, source_url, starts_at,
                         body.timezone),
                    )
                    meeting_uuid = cur.fetchone()[0]
                _replace_agenda_items(cur, meeting_uuid, items)
    finally:
        conn.close()
    return slug


# ---------------------------------------------------------------------------
# Pass B: align published agenda items to the processed video and flip them
# to 'happened' — in place (UPDATE only; item ids are public permalinks).
# ---------------------------------------------------------------------------

def build_alignment_updates(spans, segments) -> list[tuple]:
    """Rows for _update_aligned_items: (status, segment_start_seconds,
    segment_end_seconds, outcome, position).

    Status is 'happened' for EVERY item — the meeting happened; an item whose
    span abstained is happened-without-span (not reached), so its bounds stay
    None. Seconds come from the span's boundary segments.

    outcome may be None (span abstained or no legislation match) — the
    authority ladder's fill-only UPDATE (_update_aligned_items) treats None
    as "leave the existing outcome alone", never blanking a memo-set outcome.
    """
    rows = []
    for span in spans:
        if span.start_segment is None or span.end_segment is None:
            start_s = end_s = None
        else:
            start_s = segments[span.start_segment].start
            end_s = segments[span.end_segment].end
        rows.append(("happened", start_s, end_s, span.outcome, span.position))
    return rows


def _update_aligned_items(cur, meeting_uuid: str, updates: list[tuple]) -> None:
    """Per-row in-place UPDATE of meetings.agenda_items. NEVER deletes —
    item ids are public permalinks.

    ``updates`` tuples are (status, start_s, end_s, outcome, position) and do
    not carry the meeting uuid; SQL params are composed here in the statement's
    placeholder order: (status, start_s, end_s, outcome, meeting_uuid, position).

    outcome uses COALESCE(existing, new): alignment FILLS outcomes but never
    overwrites one already set (the memo reconciler is the only overwriter —
    authority ladder: align fills → memo overwrites → align never un-fills).
    """
    cur.executemany(
        """
        UPDATE meetings.agenda_items
        SET status = %s,
            segment_start_seconds = %s,
            segment_end_seconds = %s,
            outcome = COALESCE(outcome, %s),
            updated_at = now()
        WHERE meeting_id = %s AND position = %s
        """,
        [
            (status, start_s, end_s, outcome, meeting_uuid, position)
            for (status, start_s, end_s, outcome, position) in updates
        ],
    )


def _hms(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def align_and_flip(meeting_id: str) -> dict:
    """Align a published meeting's agenda items to its processed video and
    flip every item to 'happened' in place.

    ``meeting_id`` is the meeting SLUG — for an agenda-first meeting that is
    the scheduled slug ('{body}-{date}', e.g.
    bloomington-city-council-2026-07-29), which the video pass must also have
    used so both passes share one row. Loads the local transcript, absolutizes
    times into the source video's timeline (matching the published segments),
    runs the anchor→LLM→gates→oracle chain, and UPDATEs the item rows.

    Outcome authority ladder: this only fills a NULL outcome (COALESCE in
    _update_aligned_items) — an outcome already set by the memo reconciler
    is never overwritten here.
    """
    from . import config
    from .agenda_align import SegmentRef, align_items, apply_oracle
    from .agenda_parse import ParsedItem
    from .clip import absolutize_meeting_times
    from .legislation_oracle import _default_fetch

    meeting_dir = config.MEETINGS_DIR / meeting_id
    named_path = meeting_dir / "transcript_named.json"
    if not named_path.exists():
        raise RuntimeError(
            f"No transcript_named.json for {meeting_id!r} (expected {named_path})"
        )
    with open(named_path, "r", encoding="utf-8") as f:
        meeting = Meeting.from_dict(json.load(f))
    if meeting.clip_start_seconds is None:
        # Pre-clip-feature transcripts kept the window in pipeline state only
        # (same fallback as the publish loader).
        from .checkpoint import PipelineState

        meeting.clip_start_seconds = PipelineState(meeting_dir).clip_start_seconds
    meeting = absolutize_meeting_times(meeting)
    segments = [
        SegmentRef(
            i=i,
            start=seg.start_time,
            end=seg.end_time,
            speaker=seg.speaker_name or seg.speaker_label or "",
            text=seg.text or "",
        )
        for i, seg in enumerate(meeting.segments)
    ]

    conn = psycopg2.connect(_require_db_url())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM meetings.meetings WHERE slug = %s",
                    (meeting_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(
                        f"No published meeting with slug {meeting_id!r}."
                    )
                meeting_uuid = row[0]
                cur.execute(
                    """
                    SELECT position, item_number, title_raw, legislation_ref, outcome
                    FROM meetings.agenda_items
                    WHERE meeting_id = %s
                    ORDER BY position
                    """,
                    (meeting_uuid,),
                )
                item_rows = cur.fetchall()
        if not item_rows:
            raise RuntimeError(
                f"Meeting {meeting_id!r} has no agenda_items rows — nothing to "
                "align. This usually means the video pass published under a "
                "slug the agenda pass never used (wrong --meeting-id): the "
                "scheduled slug (e.g. bloomington-city-council-2026-07-29) is "
                "required so both passes share one meeting row."
            )
        items = [
            ParsedItem(
                position=position,
                item_number=item_number or "",
                section="",
                section_number=0,
                title_raw=title_raw or "",
                legislation_ref=legislation_ref,
            )
            for (position, item_number, title_raw, legislation_ref, outcome) in item_rows
        ]
        existing_outcomes = {
            position: outcome
            for (position, item_number, title_raw, legislation_ref, outcome) in item_rows
        }

        # LLM + oracle run OUTSIDE any transaction (slow network calls).
        from .llm_providers import make_llm_client

        client = make_llm_client()
        spans = align_items(client, items, segments)
        spans = apply_oracle(spans, items, fetch=_default_fetch)
        updates = build_alignment_updates(spans, segments)

        with conn:
            with conn.cursor() as cur:
                _update_aligned_items(cur, meeting_uuid, updates)
    finally:
        conn.close()

    by_position = {item.position: item for item in items}
    bound = [s for s in spans if s.start_segment is not None]
    outcomes = {s.position: s.outcome for s in spans if s.outcome}
    abstained = [
        {"position": s.position,
         "item_number": by_position[s.position].item_number,
         "reason": s.rejected_reason or "no span proposed"}
        for s in spans if s.start_segment is None
    ]

    print(f"\n=== Agenda alignment: {meeting_id} ===")
    print(f"  {len(items)} item(s) flipped to 'happened', "
          f"{len(bound)} bound to video spans, {len(outcomes)} outcome(s).")
    for span in spans:
        item = by_position[span.position]
        label = f"  [{item.item_number:>4}] pos {span.position:<3}"
        existing_outcome = existing_outcomes.get(span.position)
        preserved = (
            f" [existing outcome {existing_outcome!r} preserved]"
            if existing_outcome is not None and existing_outcome != span.outcome
            else ""
        )
        if span.start_segment is None:
            reason = span.rejected_reason or "no span proposed"
            print(f"{label} ABSTAINED — {reason}{preserved}")
        else:
            start_s = segments[span.start_segment].start
            end_s = segments[span.end_segment].end
            line = f"{label} {_hms(start_s)}–{_hms(end_s)}"
            line += f"  outcome: {span.outcome or 'none'}"
            if span.outcome is None and span.rejected_reason:
                line += f" ({span.rejected_reason})"
            line += preserved
            print(line)
    print("=" * 40)

    return {
        "meeting_id": meeting_id,
        "items": len(items),
        "bound": len(bound),
        "outcomes": outcomes,
        "abstained": abstained,
    }


def reconcile_memo(meeting_id: str, check: bool = False) -> dict:
    """Reconcile a meeting's item outcomes and votes from the clerk's
    post-meeting Memorandum (OnBoard file type 'Memorandum').

    ``meeting_id`` is the meeting SLUG. The memo is authoritative: item
    dispositions OVERWRITE agenda_items.outcome, and each substantive motion
    with a recorded roll call becomes a meetings.votes row ('roll call',
    timestamp NULL) — with per-member meetings.vote_records rows when the
    memo names the sides. Idempotent: this meeting's votes are
    delete-then-inserted (records first — FK). Votes are partitioned by
    vote_type: this function owns the MEMO_VOTE_TYPE stripe and never
    touches federal floor votes; re-publish (_replace_votes) likewise
    cannot wipe memo votes.

    ``check=True`` is read-only: the plan is recomputed and diffed against
    the DB's current memo-stripe votes/outcomes, but the write transaction
    never runs. Returns the usual dict plus ``"drift"`` (list of strings,
    empty when the DB matches the memo) and ``"checked": True``.
    """
    from . import config
    from .agenda_pipeline import download_file
    from .bodies import BLOOMINGTON_COMMON_COUNCIL as body  # single body today
    from .memo_parse import parse_memo
    from .memo_reconcile import (
        AgendaItemRow, SpeakerRow, build_reconcile_plan, diff_plan_against_db,
    )
    from .onboard import fetch_meetings_window
    from .pdf_text import extract_text

    conn = psycopg2.connect(_require_db_url())
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, date FROM meetings.meetings WHERE slug = %s",
                    (meeting_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"No meeting with slug {meeting_id!r}.")
                meeting_uuid, meeting_date = row
                cur.execute(
                    """
                    SELECT id, position, legislation_ref, outcome
                    FROM meetings.agenda_items
                    WHERE meeting_id = %s ORDER BY position
                    """,
                    (meeting_uuid,),
                )
                agenda_items = [
                    AgendaItemRow(str(i), p, ref, out)
                    for (i, p, ref, out) in cur.fetchall()
                ]
                cur.execute(
                    "SELECT id, display_name FROM meetings.speakers WHERE meeting_id = %s",
                    (meeting_uuid,),
                )
                speakers = [SpeakerRow(str(i), dn or "") for (i, dn) in cur.fetchall()]
                if check:
                    cur.execute(
                        """
                        SELECT v.resolution, v.result, count(r.id)
                        FROM meetings.votes v
                        LEFT JOIN meetings.vote_records r ON r.vote_id = v.id
                        WHERE v.meeting_id = %s AND v.vote_type = %s
                        GROUP BY v.id
                        """,
                        (meeting_uuid, MEMO_VOTE_TYPE),
                    )
                    existing_votes = [(res, result, int(n)) for (res, result, n) in cur.fetchall()]
                else:
                    existing_votes = []

        # Network (OnBoard + PDF) runs outside any transaction. A single-day
        # start==end window returns [] (verified live), so span ±1 day and
        # filter back to the exact date.
        from datetime import timedelta

        day = timedelta(days=1)
        window = fetch_meetings_window(
            (meeting_date - day).isoformat(),
            (meeting_date + day).isoformat(),
            title_prefix=body.meeting_title_prefix,
        )
        matches = [m for m in window if m.start[:10] == meeting_date.isoformat()]
        if not matches:
            raise RuntimeError(
                f"OnBoard has no {body.meeting_title_prefix!r} meeting on "
                f"{meeting_date.isoformat()} — cannot locate a memorandum."
            )
        if len(matches) > 1:
            print(f"  NOTE [{meeting_id}]: {len(matches)} OnBoard meetings match "
                  f"{meeting_date.isoformat()} — using the first")
        memo_url = matches[0].memo_url
        if memo_url is None:
            print(f"\n=== Memo reconcile: {meeting_id} ===")
            print("  No Memorandum posted yet (clerk posts within ~a week). "
                  "Re-run when it appears.")
            return {"meeting_id": meeting_id, "memo": None, "checked": check}

        pdf_path = config.DRIVE_ROOT / "agendas" / body.slug / meeting_id / "memo.pdf"
        download_file(memo_url, pdf_path)

        memo = parse_memo(extract_text(pdf_path))
        if not memo.items:
            raise RuntimeError(
                f"memo for {meeting_id!r} parsed to zero legislation items — "
                "template drift? No votes/outcomes written."
            )
        plan = build_reconcile_plan(memo, agenda_items, speakers)

        if check:
            drift = diff_plan_against_db(plan, agenda_items, existing_votes)
            print(f"\n=== Memo check: {meeting_id} ===")
            if drift:
                for line in drift:
                    print(f"  DRIFT: {line}")
            else:
                print("  no drift — DB matches the memo")
            for note in plan.notes:
                print(f"  NOTE: {note}")
            print("=" * 40)
            return {
                "meeting_id": meeting_id,
                "memo": memo_url,
                "outcome_updates": len(plan.outcome_updates),
                "votes": len(plan.votes),
                "records": sum(len(v.records) for v in plan.votes),
                "notes": plan.notes,
                "drift": drift,
                "checked": True,
            }

        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE meetings.agenda_items
                    SET outcome = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    plan.outcome_updates,
                )
                cur.execute(
                    """
                    DELETE FROM meetings.vote_records
                    WHERE vote_id IN
                      (SELECT id FROM meetings.votes
                       WHERE meeting_id = %s AND vote_type = %s)
                    """,
                    (meeting_uuid, MEMO_VOTE_TYPE),
                )
                cur.execute(
                    "DELETE FROM meetings.votes WHERE meeting_id = %s AND vote_type = %s",
                    (meeting_uuid, MEMO_VOTE_TYPE),
                )
                for vote in plan.votes:
                    cur.execute(
                        """
                        INSERT INTO meetings.votes
                          (meeting_id, resolution, description, result,
                           vote_type, timestamp, agenda_item_id)
                        VALUES (%s, %s, %s, %s, %s, NULL, %s)
                        RETURNING id
                        """,
                        (meeting_uuid, vote.resolution, vote.description,
                         vote.result, MEMO_VOTE_TYPE, vote.agenda_item_id),
                    )
                    vote_uuid = cur.fetchone()[0]
                    if vote.records:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO meetings.vote_records
                              (vote_id, speaker_id, position)
                            VALUES %s
                            """,
                            [(vote_uuid, sid, pos) for (sid, pos) in vote.records],
                        )
    finally:
        conn.close()

    record_count = sum(len(v.records) for v in plan.votes)

    print(f"\n=== Memo reconcile: {meeting_id} ===")
    print(f"  {len(memo.items)} memo item(s); {len(plan.outcome_updates)} outcome "
          f"update(s); {len(plan.votes)} vote(s); {record_count} member record(s).")
    for vote in plan.votes:
        attached = "" if vote.agenda_item_id else "  (no agenda item)"
        print(f"  [{vote.resolution}] {vote.result} — "
              f"{len(vote.records)} record(s){attached}")
    for note in plan.notes:
        print(f"  NOTE: {note}")
    print("=" * 40)

    return {
        "meeting_id": meeting_id,
        "memo": memo_url,
        "outcome_updates": len(plan.outcome_updates),
        "votes": len(plan.votes),
        "records": record_count,
        "notes": plan.notes,
        "checked": False,
    }


def memo_votes_present(meeting_id: str) -> bool:
    """True when the meeting (by slug) has memo-stripe votes rows. Cheap
    probe for the poller's self-heal: marker says reconciled but votes
    vanished -> re-reconcile."""
    conn = psycopg2.connect(_require_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM meetings.votes v
                JOIN meetings.meetings m ON m.id = v.meeting_id
                WHERE m.slug = %s AND v.vote_type = %s
                LIMIT 1
                """,
                (meeting_id, MEMO_VOTE_TYPE),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def _replace_topics(cur, meeting_uuid: str, meeting: "Meeting") -> None:
    """Delete-then-insert meeting_topics rows from meeting.section_topics.

    Denormalizes section metadata (title/type/times) so topic pages are a
    single query. status is always 'predicted' in this build.

    Guard runs BEFORE the delete: an empty section_topics almost always means
    classification wasn't loaded for this publish (e.g. a standalone
    --publish-meeting where topics.json wasn't read), not that the meeting
    genuinely has no topics. Deleting first would wipe previously-published
    tags on every plain re-publish. Only replace when we have a fresh set.
    """
    if not meeting.section_topics or not meeting.summary:
        return

    cur.execute(
        "DELETE FROM meetings.meeting_topics WHERE meeting_id = %s",
        (meeting_uuid,),
    )

    sections = meeting.summary.sections
    model = meeting.summary.model or None
    rows = []
    for st in meeting.section_topics:
        if st.section_index < 0 or st.section_index >= len(sections):
            continue
        sec = sections[st.section_index]
        for key in st.topic_keys:
            rows.append((
                meeting_uuid, st.section_index, key, "predicted",
                st.confidence, model,
                sec.title, sec.section_type, sec.start_time, sec.end_time,
            ))

    if rows:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO meetings.meeting_topics
              (meeting_id, section_index, topic_key, status, confidence, model,
               section_title, section_type, start_time, end_time)
            VALUES %s
            """,
            rows,
        )


def _trigger_deploy_hook() -> None:
    """POST to the Render deploy hook URL if RENDER_DEPLOY_HOOK_URL is set.

    Called after a successful DB publish so the static site rebuilds
    automatically. Failures are logged but never raised so a hook error
    never rolls back a completed publish.
    """
    url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "").strip()
    if not url:
        return

    import urllib.request

    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"  Deploy hook triggered (HTTP {resp.status})")
    except Exception as exc:
        print(f"  WARNING: Deploy hook failed — {exc}")
        print(f"    Trigger manually: curl -X POST '{url}'")


def publish_meeting(
    meeting: Meeting, body_slug: Optional[str] = None, trigger_deploy: bool = True
) -> PublishResult:
    """Push one meeting into the meetings.* schema. Idempotent by slug.

    ``trigger_deploy`` is retained for backward compatibility but is now
    ignored. The site reads data live from the API; deploys happen via git push,
    not per-publish rebuild hooks.
    """
    from .clip import absolutize_meeting_times
    meeting = absolutize_meeting_times(meeting)

    # Two named, non-placeholder speakers sharing a name is never a valid
    # published state (identify's dedupe guard enforces this, but a review
    # rename can re-create it — and downstream, memo reconciliation drops the
    # member's votes as "ambiguous"). Refuse before any DB work.
    from .review import duplicate_named_speakers
    dups = duplicate_named_speakers(meeting.speakers)
    if dups:
        parts = [
            f"{len(labels)} speakers named {meeting.speakers[labels[0]].speaker_name!r} ({', '.join(labels)})"
            for labels in dups.values()
        ]
        raise ValueError(
            f"Cannot publish {meeting.meeting_id}: {'; '.join(parts)}. "
            "Merge them in review before publishing."
        )

    db_url = _require_db_url()

    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                meeting_uuid = _upsert_meeting(cur, meeting, body_slug)
                _upsert_event_orgs(cur, meeting.meeting_id, meeting.event_orgs)
                _upsert_local_people(cur, meeting)
                label_to_uuid = _upsert_speakers(cur, meeting, meeting_uuid)
                _reconcile_event_races(cur, meeting, meeting_uuid)
                segment_count = _replace_segments(
                    cur, meeting, meeting_uuid, label_to_uuid
                )
                # After the segments are replaced: a vanished label now has none, so the
                # FK cannot block the delete.
                vanished = _delete_vanished_speakers(
                    cur, meeting_uuid, label_to_uuid.keys()
                )
                if vanished:
                    print(f"  Removed {vanished} speaker row(s) for labels no longer in the transcript")
                _replace_topics(cur, meeting_uuid, meeting)
                vote_count = _replace_votes(cur, meeting, meeting_uuid)
                if vote_count:
                    print(f"  Published {vote_count} floor vote(s) to meetings.votes")
                speaker_count = len(label_to_uuid)

                cur.execute(
                    """
                    UPDATE meetings.meetings
                    SET segment_count = %s, speaker_count = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (segment_count, speaker_count, meeting_uuid),
                )
    finally:
        conn.close()

    # The web app now reads data live from the API, so publishing no longer needs
    # to rebuild the static site (and the per-publish rebuild caused a deploy-hook
    # race that staled the meeting list). Code deploys happen via git push.
    # _trigger_deploy_hook() is intentionally not called here.

    return PublishResult(
        meeting_id=meeting.meeting_id,
        segments=segment_count,
        speakers=speaker_count,
        removed_speakers=vanished,
    )
