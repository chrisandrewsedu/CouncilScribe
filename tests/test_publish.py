"""Tests for the meetings.* publisher: playback resolution helpers.

The row-building logic was refactored in commit dd9b26e into cursor-bound
upserts (``_upsert_meeting`` / ``_upsert_speakers`` / ``_replace_segments``)
that need a live Postgres connection, so the old pure ``build_*_row`` helpers
no longer exist. Their unit tests were removed with them. The pure URL
helpers below survived the refactor unchanged and remain worth covering.
"""

import pytest

from src.bodies import BLOOMINGTON_COMMON_COUNCIL
from src.models import AgendaItem, Meeting, ProcessingMetadata, SpeakerMapping
from src.publish import _resolve_chamber_id, _upsert_event_orgs, _upsert_meeting
from src.publish import _upsert_local_people, _upsert_speakers
from src.publish import build_agenda_item_rows, scheduled_slug
from src.publish import extract_youtube_id, resolve_playback, playback_for_meeting


# ---------------------------------------------------------------------------
# resolve_playback / extract_youtube_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=AbC12345xyz", "AbC12345xyz"),
        ("https://youtube.com/watch?v=AbC12345xyz&t=120", "AbC12345xyz"),
        ("https://youtu.be/AbC12345xyz", "AbC12345xyz"),
        ("https://youtu.be/AbC12345xyz?si=share", "AbC12345xyz"),
        ("https://www.youtube.com/embed/AbC12345xyz", "AbC12345xyz"),
        ("https://www.youtube.com/shorts/AbC12345xyz", "AbC12345xyz"),
        ("https://www.youtube.com/live/AbC12345xyz", "AbC12345xyz"),
        ("https://m.youtube.com/watch?v=AbC12345xyz", "AbC12345xyz"),
        ("https://vimeo.com/12345", None),
        ("https://example.com/watch?v=nope", None),
        ("not a url", None),
    ],
)
def test_extract_youtube_id(url, expected):
    assert extract_youtube_id(url) == expected


def test_resolve_playback_youtube():
    assert resolve_playback("https://www.youtube.com/watch?v=AbC12345xyz") == (
        "youtube",
        "AbC12345xyz",
    )


def test_resolve_playback_catstv_blob_is_direct_file():
    url = "https://catstv.blob.core.windows.net/videoarchive/B_CC_260218.m4v"
    assert resolve_playback(url) == ("file", url)


def test_resolve_playback_direct_mp4():
    url = "https://example.gov/meetings/2026-02-10.mp4"
    assert resolve_playback(url) == ("file", url)


def test_resolve_playback_hls():
    url = "https://stream.example.gov/live/playlist.m3u8"
    assert resolve_playback(url) == ("hls", url)


def test_resolve_playback_unknown_provider():
    assert resolve_playback("https://www.facebook.com/video/123") == (None, None)


def test_resolve_playback_local_path():
    assert resolve_playback("/Users/operator/meeting.mp4") == (None, None)
    assert resolve_playback("") == (None, None)


def test_resolve_playback_audio_mp3():
    url = "https://cpa.ds.npr.org/s385/audio/2026/07/ep.mp3"
    assert resolve_playback(url) == ("audio", url)


def test_resolve_playback_audio_m4a():
    url = "https://cdn/ep.m4a"
    assert resolve_playback(url) == ("audio", url)


def test_resolve_playback_catstv_page_falls_back_on_error(monkeypatch):
    """A catstv.net page URL that can't be scraped degrades to (None, None)."""
    import src.download as download

    def boom(url):
        raise ValueError("no video found")

    monkeypatch.setattr(download, "_extract_blob_url_from_page", boom)
    assert resolve_playback("https://catstv.net/government.php?id=99") == (None, None)


def test_playback_for_meeting_prefers_resolved_enclosure():
    m = Meeting(meeting_id="x", city=None, date="2026-06-03",
                audio_source="https://www.ipm.org/show/askthemayor/2026-07-15/ep")
    m.processing_metadata = ProcessingMetadata(source_audio_url="https://cpa.ds.npr.org/s385/audio/ep.mp3")
    assert playback_for_meeting(m) == ("audio", "https://cpa.ds.npr.org/s385/audio/ep.mp3")


def test_playback_for_meeting_falls_back_to_audio_source():
    m = Meeting(meeting_id="x", city=None, date="2026-06-03",
                audio_source="https://www.youtube.com/watch?v=AbC12345xyz")
    assert playback_for_meeting(m)[0] == "youtube"


RACE_ID = "22222222-2222-4222-8222-222222222222"
MEETING_UUID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class RecordingCursor:
    def __init__(self, select_row=None, fetch_rows=None):
        self.select_row = select_row
        self.fetch_rows = list(fetch_rows or [])
        self.calls = []
        self._fetchone = None

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if "FROM essentials.chambers" in sql:
            return
        if "SELECT id FROM meetings.meetings" in sql:
            self._fetchone = self.select_row
        elif "RETURNING id" in sql:
            self._fetchone = ("new-uuid",)

    def fetchall(self):
        return self.fetch_rows

    def fetchone(self):
        return self._fetchone


@pytest.mark.parametrize("existing_row", [("existing-uuid",), None])
def test_upsert_meeting_writes_title_and_event_kind(existing_row):
    cur = RecordingCursor(existing_row)
    meeting = Meeting(
        meeting_id="ca-governor-debate",
        city=None,
        date="2026-06-02",
        meeting_type="Governor Debate",
        title="California Governor Debate",
        event_kind="debate",
        race_id=RACE_ID,
    )

    _upsert_meeting(cur, meeting, None)

    write_sql, write_params = cur.calls[1]
    assert "title" in write_sql
    assert "event_kind" in write_sql
    assert "California Governor Debate" in write_params
    assert "debate" in write_params


def test_upsert_rejects_missing_event_kind():
    cur = RecordingCursor()
    meeting = Meeting(
        meeting_id="mystery", city="Bloomington", date="2026-02-18",
        meeting_type="Regular Session", event_kind=None,
    )
    with pytest.raises(ValueError):
        _upsert_meeting(cur, meeting, None)
    assert cur.calls == []  # rejected before any DB work


def test_upsert_rejects_invalid_event_kind():
    cur = RecordingCursor()
    meeting = Meeting(
        meeting_id="mystery", city="Bloomington", date="2026-02-18",
        meeting_type="Regular Session", event_kind="townhall",
    )
    with pytest.raises(ValueError):
        _upsert_meeting(cur, meeting, None)


def test_upsert_rejects_missing_meeting_type():
    cur = RecordingCursor()
    meeting = Meeting(
        meeting_id="mystery", city="Bloomington", date="2026-02-18",
        meeting_type=None, event_kind="council",
    )
    with pytest.raises(ValueError):
        _upsert_meeting(cur, meeting, None)


def test_upsert_rejects_council_without_city():
    cur = RecordingCursor()
    meeting = Meeting(
        meeting_id="mystery", city=None, date="2026-02-18",
        meeting_type="Regular Session", event_kind="council",
    )
    with pytest.raises(ValueError):
        _upsert_meeting(cur, meeting, None)


def test_upsert_allows_cityless_forum():
    # Non-civic kinds legitimately have no city; guard must not block them.
    cur = RecordingCursor(select_row=("existing-uuid",))
    meeting = Meeting(
        meeting_id="forum-1", city=None, date="2026-02-18",
        meeting_type="Candidate Forum", event_kind="forum",
    )
    _upsert_meeting(cur, meeting, None)  # must not raise
    assert cur.calls  # proceeded to DB work


def test_resolve_chamber_id_returns_unique_match():
    cur = RecordingCursor(fetch_rows=[
        ("11111111-1111-4111-8111-111111111111",),
    ])
    assert _resolve_chamber_id(cur, "test-council") == (
        "11111111-1111-4111-8111-111111111111"
    )


def test_resolve_chamber_id_returns_none_for_missing_match():
    cur = RecordingCursor(fetch_rows=[])
    assert _resolve_chamber_id(cur, "missing") is None


def test_resolve_chamber_id_returns_none_for_duplicate_slug():
    cur = RecordingCursor(fetch_rows=[
        ("11111111-1111-4111-8111-111111111111",),
        ("22222222-2222-4222-8222-222222222222",),
    ])
    assert _resolve_chamber_id(cur, "duplicate") is None


# NOTE: there is no longer any entity-state rejection reachable through
# _upsert_meeting — chamber_id is optional for council/school_board (multi-seat
# bodies), race_id is no longer passed to the validator (races are derived into
# meetings.event_races), so the only remaining rule (chamber+race mutual
# exclusion) can't trigger here. The validator rules are unit-tested directly in
# tests/test_event_entities.py.


@pytest.mark.parametrize("existing_row", [("existing-uuid",), None])
def test_publish_writes_chamber_id_for_council(existing_row):
    cur = RecordingCursor(
        select_row=existing_row,
        fetch_rows=[("11111111-1111-4111-8111-111111111111",)],
    )
    meeting = Meeting(
        meeting_id="council-event",
        city="Bloomington",
        date="2026-02-18",
        meeting_type="Regular Session",
        event_kind="council",
    )

    _upsert_meeting(cur, meeting, "test-council")

    write_sql, write_params = cur.calls[-1]
    assert "chamber_id" in write_sql
    # publish no longer writes the meetings.meetings.race_id column
    assert "race_id" not in write_sql
    assert "11111111-1111-4111-8111-111111111111" in write_params


@pytest.mark.parametrize("existing_row", [("existing-uuid",), None])
def test_publish_does_not_write_race_id_for_debate(existing_row):
    """Publish stopped writing the meetings.meetings.race_id column; a debate's
    races now live in meetings.event_races (reconciled separately). The meeting
    row must carry neither the race_id column nor its value."""
    cur = RecordingCursor(select_row=existing_row)
    meeting = Meeting(
        meeting_id="debate-event",
        city=None,
        date="2026-06-02",
        meeting_type="Governor Debate",
        event_kind="debate",
        race_id=RACE_ID,
    )

    _upsert_meeting(cur, meeting, None)

    write_sql, write_params = cur.calls[-1]
    assert "chamber_id" in write_sql
    assert "race_id" not in write_sql
    assert RACE_ID not in write_params


def test_event_orgs_upserted():
    cur = RecordingCursor()
    _upsert_event_orgs(cur, MEETING_UUID, ["California Courier"])
    sqls = [sql for sql, _ in cur.calls]
    assert any("event_orgs" in sql for sql in sqls)
    params_list = [params for _, params in cur.calls]
    assert any("California Courier" in (params or ()) for params in params_list)


def test_event_orgs_upsert_empty_skips_insert():
    cur = RecordingCursor()
    _upsert_event_orgs(cur, MEETING_UUID, [])
    insert_calls = [sql for sql, _ in cur.calls if "INSERT" in sql and "event_orgs" in sql]
    assert len(insert_calls) == 0


# ---------------------------------------------------------------------------
# unidentified handles must not publish as local_people
# ---------------------------------------------------------------------------

def test_unidentified_speaker_publishes_no_local_person_and_null_local_slug():
    """An unidentified handle is a placeholder, not a public entity: no
    local_people row, and the speakers.local_slug column is written NULL so the
    FK to local_people.slug holds."""
    meeting = Meeting(
        meeting_id="event",
        city=None,
        date="2026-06-02",
        meeting_type="Event",
        event_kind="debate",
        race_id=RACE_ID,
        speakers={
            "S0": SpeakerMapping(
                speaker_label="S0",
                speaker_name="Unidentified Speaker",
                local_slug="unidentified-m-s0",
                speaker_status="unidentified",
            ),
        },
    )

    people_cur = RecordingCursor()
    _upsert_local_people(people_cur, meeting)
    local_people_inserts = [
        sql for sql, _ in people_cur.calls
        if "INSERT INTO meetings.local_people" in sql
    ]
    assert local_people_inserts == []   # no public placeholder entity

    speakers_cur = RecordingCursor(select_row=None)
    _upsert_speakers(speakers_cur, meeting, MEETING_UUID)
    insert_sql, insert_params = next(
        (sql, params) for sql, params in speakers_cur.calls
        if "INSERT INTO meetings.speakers" in sql
    )
    assert insert_params[-1] is None   # local_slug column written NULL


def test_publish_meeting_never_calls_deploy_hook(monkeypatch):
    """publish_meeting never fires _trigger_deploy_hook — the web app reads data
    live from the API so per-publish site rebuilds are no longer needed."""
    import src.publish as publish

    calls = {"deploy": 0}
    monkeypatch.setattr(publish, "_trigger_deploy_hook", lambda: calls.__setitem__("deploy", calls["deploy"] + 1))
    # Stub the whole DB transaction so we only exercise the deploy decision.
    monkeypatch.setattr(publish, "_require_db_url", lambda: "postgresql://x")

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): pass
        def fetchone(self): return ("muid",)
        def fetchall(self): return []
    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _Cur()
        def close(self): pass
    monkeypatch.setattr(publish.psycopg2, "connect", lambda *a, **k: _Conn())
    # Stub the per-step helpers so publish_meeting reaches the deploy decision.
    for fn in ("_upsert_meeting", "_upsert_event_orgs", "_upsert_local_people",
               "_reconcile_event_races", "_replace_topics"):
        monkeypatch.setattr(publish, fn, lambda *a, **k: "muid")
    monkeypatch.setattr(publish, "_upsert_speakers", lambda *a, **k: {})
    monkeypatch.setattr(publish, "_replace_segments", lambda *a, **k: 0)

    from src.models import Meeting
    m = Meeting(meeting_id="m1", city="X", date="2026-04-01")

    publish.publish_meeting(m, None, trigger_deploy=False)
    assert calls["deploy"] == 0
    publish.publish_meeting(m, None)  # trigger_deploy=True (default) — hook still not called
    assert calls["deploy"] == 0


def test_replace_votes_builds_rows(monkeypatch):
    from src import publish
    from src.models import Meeting, FloorVote

    captured = {}
    def fake_execute_values(cur, sql, rows):
        captured["sql"] = sql
        captured["rows"] = rows
    monkeypatch.setattr(publish.psycopg2.extras, "execute_values", fake_execute_values)

    class _Cur:
        def __init__(self):
            self.executes = []
        def execute(self, sql, params=None):
            self.executes.append((sql, params))

    m = Meeting(meeting_id="m", city=None, date="2019-07-11", floor_votes=[
        FloorVote(438, "On the Smith amendment", 236, 193, 0, 9, 102.6, 0, True),
        FloorVote(500, "On the Jones amendment", 300, 100, 0, 5, None, None, False),
    ])
    cur = _Cur()
    n = publish._replace_votes(cur, m, "uuid-1")

    assert n == 2
    votes_delete = next((s, p) for s, p in cur.executes if "DELETE FROM meetings.votes" in s)
    assert "vote_type = %s" in votes_delete[0]           # stripe-scoped, not a blanket delete
    assert publish.FLOOR_VOTE_TYPE in votes_delete[1]    # bound to the floor stripe
    assert captured["rows"][0] == (
        "uuid-1", "Roll No. 438", "On the Smith amendment", "Yea 236, Nay 193", "recorded", 102.6)
    assert captured["rows"][1][5] is None


def test_replace_votes_empty_deletes_and_inserts_nothing(monkeypatch):
    from src import publish
    from src.models import Meeting

    called = {"execute_values": False}
    monkeypatch.setattr(publish.psycopg2.extras, "execute_values",
                        lambda *a, **k: called.__setitem__("execute_values", True))

    class _Cur:
        def __init__(self): self.executes = []
        def execute(self, sql, params=None): self.executes.append((sql, params))

    m = Meeting(meeting_id="m", city=None, date="d")
    cur = _Cur()
    assert publish._replace_votes(cur, m, "uuid-1") == 0
    votes_delete = next((s, p) for s, p in cur.executes if "DELETE FROM meetings.votes" in s)
    assert "vote_type = %s" in votes_delete[0]           # stripe-scoped, not a blanket delete
    assert publish.FLOOR_VOTE_TYPE in votes_delete[1]    # bound to the floor stripe
    assert called["execute_values"] is False


# ---------------------------------------------------------------------------
# Pass A (scheduled meeting + agenda items) pure helpers
# ---------------------------------------------------------------------------

def test_scheduled_slug_is_body_plus_date():
    assert (
        scheduled_slug(BLOOMINGTON_COMMON_COUNCIL, "2026-07-29")
        == "bloomington-city-council-2026-07-29"
    )


def test_build_agenda_item_rows_orders_and_nulls():
    items = [
        AgendaItem(position=1, item_number="1", title_raw="Roll Call",
                   kind="procedural", source_url="https://x.gov/a.pdf"),
        AgendaItem(position=13, item_number="6A",
                   title_raw="Ordinance 2026-16 – Salaries", kind="ordinance",
                   legislation_ref="Ordinance 2026-16",
                   summary_plain="Raises pay 4 percent.",
                   stage="First reading", public_comment=False,
                   source_url="https://x.gov/a.pdf"),
    ]
    rows = build_agenda_item_rows("uuid-123", items)
    assert rows[0][0] == "uuid-123"          # meeting_id first
    assert rows[0][1] == 1                    # position
    assert rows[0][5] is None                 # legislation_ref null for roll call
    assert rows[1][2] == "6A"
    assert rows[1][12] == "upcoming"          # status literal
    assert len(rows[0]) == 13                 # column count locks INSERT arity


# ---------------------------------------------------------------------------
# Pass B (alignment flip) pure helper
# ---------------------------------------------------------------------------

def _seg(i, start, end):
    from src.agenda_align import SegmentRef
    return SegmentRef(i=i, start=start, end=end, speaker="S", text="t")


def test_build_alignment_updates_bound_span_maps_segment_seconds():
    from src.agenda_align import ItemSpan
    from src.publish import build_alignment_updates

    segments = [_seg(0, 0.0, 10.5), _seg(1, 10.5, 42.0), _seg(2, 42.0, 99.9)]
    spans = [
        ItemSpan(position=3, start_segment=1, end_segment=2, outcome="passed",
                 outcome_evidence_segment=2),
    ]
    rows = build_alignment_updates(spans, segments)
    assert rows == [("happened", 10.5, 99.9, "passed", 3)]


def test_build_alignment_updates_abstained_span_none_bounds():
    from src.agenda_align import ItemSpan
    from src.publish import build_alignment_updates

    segments = [_seg(0, 0.0, 10.0)]
    spans = [ItemSpan(position=1, rejected_reason="position missing from reply")]
    rows = build_alignment_updates(spans, segments)
    # Status is 'happened' even without a span: the meeting happened, the
    # item just wasn't bound (not reached / abstained).
    assert rows == [("happened", None, None, None, 1)]


def test_build_alignment_updates_arity_and_every_item_happened():
    from src.agenda_align import ItemSpan
    from src.publish import build_alignment_updates

    segments = [_seg(0, 0.0, 5.0), _seg(1, 5.0, 8.0)]
    spans = [
        ItemSpan(position=1, start_segment=0, end_segment=0),  # bound, no outcome
        ItemSpan(position=2),                                   # abstained
        ItemSpan(position=5, start_segment=1, end_segment=1, outcome="failed",
                 outcome_evidence_segment=1),
    ]
    rows = build_alignment_updates(spans, segments)
    assert len(rows) == 3
    assert all(len(r) == 5 for r in rows)          # arity locks the UPDATE
    assert all(r[0] == "happened" for r in rows)   # every item flips
    assert rows[0] == ("happened", 0.0, 5.0, None, 1)
    assert [r[4] for r in rows] == [1, 2, 5]       # position is last


def test_replace_votes_result_uses_outcome_when_present(monkeypatch):
    from src import publish
    from src.models import Meeting, FloorVote

    captured = {}
    def fake_execute_values(cur, sql, rows):
        captured["rows"] = rows
    monkeypatch.setattr(publish.psycopg2.extras, "execute_values", fake_execute_values)

    class _Cur:
        def __init__(self): self.executes = []
        def execute(self, sql, params=None): self.executes.append((sql, params))

    m = Meeting(meeting_id="m", city=None, date="2019-07-11", floor_votes=[
        FloorVote(438, "On the Smith amendment", 236, 193, 0, 9, 102.6, 0, True,
                  outcome="Agreed to", passed=True),
        FloorVote(500, "On the Jones amendment", 300, 100, 0, 5, None, None, False),  # no outcome
    ])
    publish._replace_votes(_Cur(), m, "uuid-1")
    assert captured["rows"][0][3] == "Agreed to · 236–193"   # outcome · yea–nay
    assert captured["rows"][1][3] == "Yea 300, Nay 100"                # fallback tally


# ---------------------------------------------------------------------------
# publish_meeting refuses duplicate-named speakers (invariant: two diarized
# labels can't be the same person — rename in review can re-create this)
# ---------------------------------------------------------------------------

def _speaker_meeting(speakers):
    return Meeting(meeting_id="2026-06-10-council", city="Bloomington",
                   date="2026-06-10", segments=[], speakers=speakers)


def test_publish_meeting_refuses_duplicate_named_speakers(monkeypatch):
    import src.publish as publish

    class NoDB:
        def connect(self, *a, **k):
            raise AssertionError("psycopg2.connect must not be reached")

    monkeypatch.setattr(publish, "psycopg2", NoDB())
    speakers = {
        "SPEAKER_19": SpeakerMapping("SPEAKER_19", "City Common Council - District 6 Zulich"),
        "SPEAKER_07": SpeakerMapping("SPEAKER_07", "City Common Council - District 6 Zulich"),
        "SPEAKER_00": SpeakerMapping("SPEAKER_00", "Mayor Johnson"),
    }
    with pytest.raises(ValueError) as exc:
        publish.publish_meeting(_speaker_meeting(speakers))
    msg = str(exc.value)
    assert "Cannot publish 2026-06-10-council" in msg
    assert "2 speakers named" in msg
    assert "'City Common Council - District 6 Zulich'" in msg
    assert "SPEAKER_07, SPEAKER_19" in msg
    assert "Merge them in review" in msg


def test_publish_meeting_clean_names_reach_db_stage(monkeypatch):
    """Placement pin: the duplicate check sits BEFORE any DB work, and a clean
    meeting sails past it (the sentinel at _require_db_url is what fires)."""
    import src.publish as publish

    def sentinel():
        raise RuntimeError("reached-db-stage")

    monkeypatch.setattr(publish, "_require_db_url", sentinel)
    speakers = {
        "SPEAKER_00": SpeakerMapping("SPEAKER_00", "Mayor Johnson"),
        "SPEAKER_01": SpeakerMapping("SPEAKER_01", "Kate Rosenbarger"),
    }
    with pytest.raises(RuntimeError, match="reached-db-stage"):
        publish.publish_meeting(_speaker_meeting(speakers))


def test_publish_meeting_duplicate_check_ignores_placeholder_statuses(monkeypatch):
    """Two 'Unidentified Speaker' or 'Non-speaker' rows are a valid published
    state — placeholder names are not identities and must not block publish."""
    import src.publish as publish

    def sentinel():
        raise RuntimeError("reached-db-stage")

    monkeypatch.setattr(publish, "_require_db_url", sentinel)
    speakers = {
        "S0": SpeakerMapping("S0", "Unidentified Speaker", local_slug="u-1",
                             speaker_status="unidentified"),
        "S1": SpeakerMapping("S1", "Unidentified Speaker", local_slug="u-2",
                             speaker_status="unidentified"),
        "S2": SpeakerMapping("S2", "Non-speaker", speaker_status="non_speaker"),
        "S3": SpeakerMapping("S3", "Non-speaker", speaker_status="non_speaker"),
    }
    with pytest.raises(RuntimeError, match="reached-db-stage"):
        publish.publish_meeting(_speaker_meeting(speakers))


# ---------------------------------------------------------------------------
# local_people.role must reflect what review recorded, never a guess
# ---------------------------------------------------------------------------

def _local_people_insert(meeting):
    """Run _upsert_local_people and return the params of the single INSERT."""
    cur = RecordingCursor()
    _upsert_local_people(cur, meeting)
    return next(
        params for sql, params in cur.calls
        if "INSERT INTO meetings.local_people" in sql
    )


def _meeting_with_local_person(local_role):
    return Meeting(
        meeting_id="event",
        city=None,
        date="2026-06-02",
        meeting_type="Event",
        event_kind="council",
        race_id=RACE_ID,
        speakers={
            "S0": SpeakerMapping(
                speaker_label="S0",
                speaker_name="Pearl Vinard",
                local_slug="pearl-vinard",
                local_role=local_role,
            ),
        },
    )


def test_local_person_without_role_publishes_null_not_candidate():
    """An unset local_role means review never recorded one. Publishing it as
    'candidate' asserts a fact nobody established — moderators, staff and
    public commenters all get mislabelled. Write NULL instead."""
    slug, name, role = _local_people_insert(_meeting_with_local_person(None))
    assert (slug, name) == ("pearl-vinard", "Pearl Vinard")
    assert role is None


def test_local_person_role_publishes_verbatim():
    """A role review did record is published unchanged, not remapped."""
    _, _, role = _local_people_insert(_meeting_with_local_person("moderator"))
    assert role == "moderator"


# ---------------------------------------------------------------------------
# one identity per speaker: an essentials link suppresses the local person
# ---------------------------------------------------------------------------

def _meeting_with_dual_identity(**identity):
    """A speaker carrying BOTH an essentials identity and a local_slug.

    This is what the federal floor path produces: crec_identify stashes the
    bioguide in local_slug for every resolved member, then resolve_politician_id
    adds an essentials link on top and nothing clears the stash.
    """
    return Meeting(
        meeting_id="2026-07-16-house-floor",
        city=None,
        date="2026-07-16",
        meeting_type="Floor",
        event_kind="floor",
        race_id=None,
        speakers={
            "S0": SpeakerMapping(
                speaker_label="S0",
                speaker_name="Marcy Kaptur",
                local_slug="congress-K000009",
                **identity,
            ),
        },
    )


def _published_local_slug_column(meeting):
    """The local_slug value _upsert_speakers writes, on the INSERT path."""
    cur = RecordingCursor(select_row=None)
    _upsert_speakers(cur, meeting, MEETING_UUID)
    _, params = next(
        (sql, p) for sql, p in cur.calls if "INSERT INTO meetings.speakers" in sql
    )
    return params[-1]


def test_speaker_with_essentials_id_publishes_no_local_person():
    """migration 623: either an essentials identity OR a local person, never both.
    A resolved politician is not a site-local person, so the bioguide stash in
    local_slug must not mint a duplicate local_people row for them."""
    meeting = _meeting_with_dual_identity(politician_id="1938e59f-bd7c-45fb-8dd1-2f7591a0fc3d")

    cur = RecordingCursor()
    _upsert_local_people(cur, meeting)
    assert [sql for sql, _ in cur.calls if "INSERT INTO meetings.local_people" in sql] == []

    assert _published_local_slug_column(meeting) is None


def test_speaker_with_essentials_slug_publishes_no_local_person():
    """Same for a legacy slug-only link — 623 words the invariant as politician_slug."""
    meeting = _meeting_with_dual_identity(politician_slug="marcy-kaptur")

    cur = RecordingCursor()
    _upsert_local_people(cur, meeting)
    assert [sql for sql, _ in cur.calls if "INSERT INTO meetings.local_people" in sql] == []

    assert _published_local_slug_column(meeting) is None


def test_local_only_speaker_still_publishes_a_local_person():
    """The genuine case is untouched: no essentials identity means the local person
    is the only identity there is."""
    meeting = _meeting_with_dual_identity()   # neither politician_id nor politician_slug

    cur = RecordingCursor()
    _upsert_local_people(cur, meeting)
    assert len([sql for sql, _ in cur.calls if "INSERT INTO meetings.local_people" in sql]) == 1

    assert _published_local_slug_column(meeting) == "congress-K000009"


# ---------------------------------------------------------------------------
# speaker rows for labels that vanished from the transcript
# ---------------------------------------------------------------------------

def _vanished_delete_calls(cur):
    return [(sql, params) for sql, params in cur.calls
            if "DELETE FROM meetings.speakers" in sql]


def test_delete_vanished_speakers_removes_labels_no_longer_in_the_transcript():
    """Publish upserts speakers by (meeting_id, label) and never removed ones that
    disappeared — after a merge, the merged-away label's row lingered forever. Two
    such rows existed in prod, one of them a linked politician with no segments."""
    from src.publish import _delete_vanished_speakers

    cur = RecordingCursor()
    _delete_vanished_speakers(cur, MEETING_UUID, ["SPEAKER_00", "SPEAKER_01"])

    calls = _vanished_delete_calls(cur)
    assert len(calls) == 1
    sql, params = calls[0]
    assert params[0] == MEETING_UUID          # scoped to this meeting
    assert sorted(params[1]) == ["SPEAKER_00", "SPEAKER_01"]   # keeps current labels
    # never delete a row segments still point at, whatever the call order
    assert "NOT EXISTS" in sql


def test_delete_vanished_speakers_refuses_to_wipe_when_there_are_no_labels():
    """An empty speakers dict is a malformed artifact, not an instruction to delete
    every speaker row for the meeting. Publish is destructive and has no undo."""
    from src.publish import _delete_vanished_speakers

    cur = RecordingCursor()
    deleted = _delete_vanished_speakers(cur, MEETING_UUID, [])

    assert _vanished_delete_calls(cur) == []
    assert deleted == 0


def test_publish_deletes_vanished_speakers_after_replacing_segments():
    """Ordering is load-bearing: until the old segments are gone they still reference
    these rows, and meetings.segments.speaker_id would block the delete."""
    import inspect
    from src import publish

    src = inspect.getsource(publish.publish_meeting)
    assert src.index("_replace_segments") < src.index("_delete_vanished_speakers")


def test_publish_result_reports_how_many_stale_speaker_rows_it_removed(monkeypatch):
    """The count was only ever printed. In the GUI, publish_meeting runs in-process
    and that print goes to the uvicorn terminal, never to the browser — so the one
    signal that a stale row existed was invisible to a GUI reviewer."""
    from src import publish

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): pass
        def fetchone(self): return ("muid",)
        def fetchall(self): return []
    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _Cur()
        def close(self): pass
    monkeypatch.setattr(publish, "_require_db_url", lambda: "postgresql://x")
    monkeypatch.setattr(publish.psycopg2, "connect", lambda *a, **k: _Conn())
    for fn in ("_upsert_meeting", "_upsert_event_orgs", "_upsert_local_people",
               "_reconcile_event_races", "_replace_topics"):
        monkeypatch.setattr(publish, fn, lambda *a, **k: "muid")
    monkeypatch.setattr(publish, "_upsert_speakers", lambda *a, **k: {})
    monkeypatch.setattr(publish, "_replace_segments", lambda *a, **k: 0)
    monkeypatch.setattr(publish, "_delete_vanished_speakers", lambda *a, **k: 2)

    from src.models import Meeting
    result = publish.publish_meeting(
        Meeting(meeting_id="m1", city="X", date="2026-04-01"), None,
        trigger_deploy=False,
    )
    assert result.removed_speakers == 2


def test_publish_result_removed_speakers_defaults_to_zero():
    # Existing callers build PublishResult positionally with three fields.
    from src.publish import PublishResult
    assert PublishResult("m1", 12, 3).removed_speakers == 0
