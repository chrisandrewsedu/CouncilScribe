from __future__ import annotations

import gui.publish_api as pub


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []
    def execute(self, sql, params=None):
        self.executed.append((sql, params))
    def fetchone(self):
        return self._rows.pop(0) if self._rows else None
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeConn:
    def __init__(self, rows):
        self.cursor_obj = _FakeCursor(rows)
        self.committed = False
    def cursor(self): return self.cursor_obj
    def commit(self): self.committed = True
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_db(monkeypatch, rows):
    conn = _FakeConn(list(rows))
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    monkeypatch.setattr(pub.psycopg2, "connect", lambda url: conn)
    return conn


class _FakeCursorAll(_FakeCursor):
    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows


def test_live_published_slugs_returns_set(monkeypatch):
    # One query now returns (slug, speaker_count); the slug set derives from it.
    conn = _FakeConn([])
    conn.cursor_obj = _FakeCursorAll([("2026-02-04-council", 6),
                                      ("2026-03-04-council", None)])
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    monkeypatch.setattr(pub.psycopg2, "connect", lambda url: conn)
    assert pub.live_published_slugs() == {"2026-02-04-council", "2026-03-04-council"}


def test_live_meeting_speaker_counts_returns_prods_counts(monkeypatch):
    # The count rides along so the library can show prod's speaker_count beside
    # the local one — an orphan speaker row inflates prod's.
    conn = _FakeConn([])
    conn.cursor_obj = _FakeCursorAll([("2026-02-04-council", 6),
                                      ("2026-03-04-council", None)])
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    monkeypatch.setattr(pub.psycopg2, "connect", lambda url: conn)
    assert pub.live_meeting_speaker_counts() == {"2026-02-04-council": 6,
                                                "2026-03-04-council": None}


def test_live_meeting_speaker_counts_none_without_db(monkeypatch):
    monkeypatch.setattr(pub, "_db_url", lambda: None)
    assert pub.live_meeting_speaker_counts() is None


def test_live_meeting_speaker_counts_none_on_error(monkeypatch):
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    def boom(url):
        raise RuntimeError("db down")
    monkeypatch.setattr(pub.psycopg2, "connect", boom)
    assert pub.live_meeting_speaker_counts() is None


def test_live_published_slugs_none_without_db(monkeypatch):
    monkeypatch.setattr(pub, "_db_url", lambda: None)
    assert pub.live_published_slugs() is None      # unknown, not empty -> no false "not live"


def test_live_published_slugs_none_on_error(monkeypatch):
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    def boom(url):
        raise RuntimeError("db down")
    monkeypatch.setattr(pub.psycopg2, "connect", boom)
    assert pub.live_published_slugs() is None


def test_meeting_published_id_found(monkeypatch):
    conn = _patch_db(monkeypatch, [("uuid-123",)])
    assert pub.meeting_published_id("2026-02-04-council") == "uuid-123"


def test_meeting_published_id_absent(monkeypatch):
    _patch_db(monkeypatch, [])  # no row
    assert pub.meeting_published_id("ghost") is None


def test_meeting_published_id_no_db_url(monkeypatch):
    monkeypatch.setattr(pub, "_db_url", lambda: None)
    assert pub.meeting_published_id("x") is None  # not configured -> None, no crash


def test_update_supabase_metadata_updates_when_published(monkeypatch):
    conn = _patch_db(monkeypatch, [("uuid-123",)])  # SELECT id finds a row
    ok = pub.update_supabase_metadata("2026-02-04-council", {
        "title": "Fixed Title", "city": "Bloomington", "date": "2026-02-04",
        "meeting_type": "Special Session", "event_kind": "council"})
    assert ok is True
    assert conn.committed is True
    # an UPDATE ... WHERE slug was issued
    sqls = " ".join(sql for sql, _ in conn.cursor_obj.executed).lower()
    assert "update meetings.meetings" in sqls and "where slug" in sqls


def test_update_supabase_metadata_skips_when_unpublished(monkeypatch):
    conn = _patch_db(monkeypatch, [])  # no row
    ok = pub.update_supabase_metadata("ghost", {"title": "x"})
    assert ok is False
    assert conn.committed is False


import json
import pytest


def _write_meeting(mdir):
    from src.models import Meeting, Segment, SpeakerMapping
    m = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-02-04",
                meeting_type="Regular Session", title=None, event_kind="council",
                segments=[Segment(segment_id=0, start_time=0.0, end_time=5.0,
                                  speaker_label="SPEAKER_00", speaker_name="X")],
                speakers={"SPEAKER_00": SpeakerMapping(speaker_label="SPEAKER_00", speaker_name="X")})
    (mdir / "transcript_named.json").write_text(json.dumps(m.to_dict()))


def test_apply_metadata_edit_writes_local_and_freezes_slug(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    # no DB configured -> Supabase skipped, local still written
    monkeypatch.setattr(pub, "_db_url", lambda: None)

    res = pub.apply_metadata_edit("2026-02-04-council",
                                  {"title": "Budget Hearing", "meeting_type": "Special Session"})
    assert res["local"] is True
    assert res["supabase"] is False

    data = json.loads((mdir / "transcript_named.json").read_text())
    assert data["title"] == "Budget Hearing"
    assert data["meeting_type"] == "Special Session"
    assert data["meeting_id"] == "2026-02-04-council"   # FROZEN — slug/id unchanged
    assert mdir.name == "2026-02-04-council"             # dir not renamed
    # pipeline_state display fields updated too
    from src.checkpoint import PipelineState
    assert PipelineState(mdir).meeting_type == "Special Session"


def test_apply_metadata_edit_pushes_to_supabase_when_published(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    calls = {}
    monkeypatch.setattr(pub, "update_supabase_metadata",
                        lambda mid, fields: (calls.__setitem__("args", (mid, fields)), True)[1])
    res = pub.apply_metadata_edit("2026-02-04-council", {"title": "New"})
    assert res["supabase"] is True
    assert calls["args"][0] == "2026-02-04-council"
    assert calls["args"][1]["title"] == "New"


def test_apply_metadata_edit_unknown_meeting(tmp_meetings_dir):
    assert pub.apply_metadata_edit("ghost", {"title": "x"}) is None


from fastapi.testclient import TestClient
from gui.app import create_app


def test_edit_form_prefills(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/edit").text
    assert 'value="Regular Session"' in body    # meeting_type prefilled
    assert 'value="Bloomington"' in body        # city prefilled
    assert 'action="/meetings/2026-02-04-council/edit"' in body


def test_edit_form_unknown_meeting_404(tmp_meetings_dir):
    assert TestClient(create_app()).get("/meetings/ghost/edit").status_code == 404


def test_post_edit_applies_and_redirects(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    import gui.publish_api as pub2
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    monkeypatch.setattr(pub2, "_db_url", lambda: None)  # no real DB in test
    client = TestClient(create_app())
    resp = client.post("/meetings/2026-02-04-council/edit",
                       data={"title": "Budget Hearing", "city": "Bloomington",
                             "date": "2026-02-04", "meeting_type": "Special Session",
                             "event_kind": "council"}, follow_redirects=False)
    assert resp.status_code == 303
    import json as _json
    assert _json.loads((mdir / "transcript_named.json").read_text())["title"] == "Budget Hearing"


def _publish_meeting_ctx(tagged_meeting_dir, review_status):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    from src.checkpoint import PipelineState
    st = PipelineState(mdir); st.review_status = review_status; st.save()
    return mdir


def test_apply_publish_blocked_by_gate(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="review")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    called = {"n": 0}
    import src.publish as sp
    monkeypatch.setattr(sp, "publish_meeting", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is False and res["reason"] == "gate"
    assert res["review_status"] == "review"
    assert called["n"] == 0                     # gate blocked -> never published


def test_apply_publish_force_overrides_gate(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="review")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None: PublishResult(meeting.meeting_id, 12, 3))
    res = pub.apply_publish("2026-02-04-council", force=True)
    assert res["ok"] is True and res["segments"] == 12 and res["speakers"] == 3


def test_apply_publish_passes_gate(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    seen = {}
    def fake_pub(meeting, body_slug=None):
        seen["body_slug"] = body_slug
        return PublishResult(meeting.meeting_id, 5, 2)
    monkeypatch.setattr(sp, "publish_meeting", fake_pub)
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is True
    assert "body_slug" in seen                  # body_slug forwarded from state


def test_apply_publish_attaches_thumbnail(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    order = []
    monkeypatch.setattr(pub, "attach_thumbnail",
                        lambda meeting, meeting_dir: order.append("thumb"))
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None: order.append("publish")
                        or PublishResult(meeting.meeting_id, 5, 2))
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is True
    assert order == ["thumb", "publish"]        # thumbnail extracted before publish


def test_apply_publish_thumbnail_failure_is_nonfatal(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    # attach_thumbnail is best-effort and never raises, but even if it did the
    # publish must still succeed — publish must never break over a thumbnail.
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None: PublishResult(meeting.meeting_id, 5, 2))
    # real attach_thumbnail runs (no ffmpeg output in test dir) -> no-op, no crash
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is True


def test_apply_publish_no_db(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: None)
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is False and res["reason"] == "no_db"


def test_apply_publish_error_is_caught(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    def boom(*a, **k):
        raise RuntimeError("db exploded")
    monkeypatch.setattr(sp, "publish_meeting", boom)
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["ok"] is False and res["reason"] == "error" and "db exploded" in res["error"]


def test_apply_publish_unknown_meeting(tmp_meetings_dir):
    assert pub.apply_publish("ghost", force=False)["reason"] == "unknown"


def test_publish_confirm_shows_gate(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "meeting_published_id", lambda mid: None)  # not yet published
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/publish").text
    assert "pass" in body.lower()
    assert 'action="/meetings/2026-02-04-council/publish"' in body


def test_publish_confirm_unknown_404(tmp_meetings_dir):
    assert TestClient(create_app()).get("/meetings/ghost/publish").status_code == 404


def test_post_publish_success(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "apply_publish",
                        lambda mid, force=False: {"ok": True, "meeting_id": mid, "segments": 5, "speakers": 2})
    resp = TestClient(create_app()).post("/meetings/2026-02-04-council/publish", data={})
    assert resp.status_code == 200
    assert "Published" in resp.text
    assert "5" in resp.text                       # segment count shown


def test_post_publish_gate_blocked_shown(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="review")
    monkeypatch.setattr(pub, "apply_publish",
                        lambda mid, force=False: {"ok": False, "reason": "gate", "review_status": "review"})
    resp = TestClient(create_app()).post("/meetings/2026-02-04-council/publish", data={})
    assert resp.status_code == 200
    assert "error-banner" in resp.text
    assert "gate" in resp.text.lower()


def test_apply_publish_surfaces_the_removed_stale_row_count(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None:
                        PublishResult(meeting.meeting_id, 5, 2, removed_speakers=1))
    res = pub.apply_publish("2026-02-04-council", force=False)
    assert res["removed_speakers"] == 1


def test_publish_route_reports_stale_rows_it_removed(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None:
                        PublishResult(meeting.meeting_id, 5, 2, removed_speakers=1))
    body = TestClient(create_app()).post("/meetings/2026-02-04-council/publish").text
    assert "1 stale speaker row" in body


def test_publish_route_says_nothing_about_stale_rows_when_none_were_removed(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    _publish_meeting_ctx(tagged_meeting_dir, review_status="pass")
    monkeypatch.setattr(pub, "_db_url", lambda: "postgres://fake")
    import src.publish as sp
    from src.publish import PublishResult
    monkeypatch.setattr(sp, "publish_meeting",
                        lambda meeting, body_slug=None: PublishResult(meeting.meeting_id, 5, 2))
    body = TestClient(create_app()).post("/meetings/2026-02-04-council/publish").text
    assert "stale speaker row" not in body
