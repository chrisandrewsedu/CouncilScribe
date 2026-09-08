from __future__ import annotations

from gui.models import MeetingSummary, stage_label


def test_stage_label_maps_each_stage_to_friendly_text():
    assert stage_label(0) == "Not started"
    assert stage_label(1) == "Audio ingested"
    assert stage_label(2) == "Speakers separated"
    assert stage_label(3) == "Transcribed"
    assert stage_label(4) == "Identified — ready to review"
    assert stage_label(5) == "Summarized"
    assert stage_label(6) == "Voices enrolled"
    assert stage_label(7) == "Exported"  # local export, NOT live-site publish


def test_stage_label_tolerates_unknown_stage():
    assert stage_label(99) == "Unknown (99)"


def test_meeting_summary_display_name_prefers_title():
    s = MeetingSummary(
        meeting_id="2026-02-04-regular-session",
        title="Budget Hearing",
        city="Bloomington",
        meeting_type="Regular Session",
        date="2026-02-04",
        event_kind="council",
        completed_stage=4,
    )
    assert s.display_name == "Budget Hearing"
    assert s.stage_label == "Identified — ready to review"


def test_meeting_summary_display_name_falls_back_to_city_and_type():
    s = MeetingSummary(
        meeting_id="2026-02-04-regular-session",
        title=None,
        city="Bloomington",
        meeting_type="Regular Session",
        date="2026-02-04",
        event_kind="council",
        completed_stage=2,
    )
    assert s.display_name == "Bloomington Regular Session"


def test_meeting_summary_context_line_composes_available_fields():
    s = MeetingSummary(
        meeting_id="m", title=None, city="Bloomington", meeting_type="Regular Session",
        date="2026-02-10", event_kind="council", completed_stage=4,
        body_slug="bloomington-common-council",
    )
    # city + prettified body, de-duplicated, joined with ' · '
    assert s.context_line == "Bloomington · Bloomington Common Council"

    s2 = MeetingSummary(
        meeting_id="m2", title=None, city=None, meeting_type="Interview", date="2026-05-01",
        event_kind="news_clip", completed_stage=5,
        event_orgs=["CBS"], race_label="CA Governor · 2026",
    )
    assert s2.context_line == "CBS · CA Governor · 2026"

    s3 = MeetingSummary(meeting_id="m3", title=None, city=None, meeting_type=None,
                        date=None, event_kind="floor", completed_stage=3)
    assert s3.context_line == ""   # nothing to show


def test_meeting_summary_status_key():
    def s(**kw):
        base = dict(meeting_id="m", title=None, city=None, meeting_type=None, date=None,
                    event_kind=None, completed_stage=0)
        base.update(kw)
        return MeetingSummary(**base)
    assert s(completed_stage=2).status_key == "processing"          # pre-identify
    assert s(completed_stage=4).status_key == "needs-review"        # reviewable, gate not passed
    assert s(completed_stage=5, review_status="pass").status_key == "ready"
    assert s(completed_stage=7, review_status="pass", is_live=True).status_key == "live"
    assert s(completed_stage=7, review_status="review").status_key == "needs-review"


import json

from gui.library import scan_meetings


def test_scan_meetings_reads_state_and_sorts_by_date_desc(tagged_meeting_dir, tmp_meetings_dir):
    import os, time
    # tagged_meeting_dir writes pipeline_state.json with completed_stage + body_slug.
    older = tagged_meeting_dir(
        "bloomington-common-council",
        meeting_id="2026-01-10-regular-session",
        completed_stage=4,
    )
    newer = tagged_meeting_dir(
        "bloomington-common-council",
        meeting_id="2026-03-02-special-session",
        completed_stage=2,
    )
    # Enrich one state file with the newer metadata keys the GUI displays.
    state_path = older / "pipeline_state.json"
    data = json.loads(state_path.read_text())
    data.update({"city": "Bloomington", "meeting_type": "Regular Session",
                 "date": "2026-01-10", "event_kind": "council"})
    state_path.write_text(json.dumps(data))

    # The library now sorts by processed_at (pipeline_state.json mtime), not clip
    # date, so pin explicit mtimes here to make the intended order deterministic.
    now = time.time()
    os.utime(older / "pipeline_state.json", (now - 1000, now - 1000))
    os.utime(newer / "pipeline_state.json", (now, now))

    summaries = scan_meetings(tmp_meetings_dir)

    assert [s.meeting_id for s in summaries] == [
        "2026-03-02-special-session",  # more recently processed (mtime) first
        "2026-01-10-regular-session",
    ]
    older_summary = summaries[1]
    assert older_summary.city == "Bloomington"
    assert older_summary.completed_stage == 4
    assert older_summary.stage_label == "Identified — ready to review"


def test_scan_meetings_reads_title_from_named_transcript(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    # transcript_named.json holds the Meeting dict; title lives there, not in state.
    (mdir / "transcript_named.json").write_text(json.dumps({"title": "Budget Hearing"}))

    summaries = scan_meetings(tmp_meetings_dir)

    assert summaries[0].title == "Budget Hearing"
    assert summaries[0].display_name == "Budget Hearing"


def test_scan_meetings_missing_dir_returns_empty(tmp_path):
    assert scan_meetings(tmp_path / "does-not-exist") == []


def test_scan_meetings_skips_dirs_without_state(tmp_meetings_dir):
    (tmp_meetings_dir / "stray-dir").mkdir()
    assert scan_meetings(tmp_meetings_dir) == []


def test_scan_meetings_skips_dir_with_invalid_json(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    bad = tmp_meetings_dir / "2026-05-01-broken-session"
    bad.mkdir()
    (bad / "pipeline_state.json").write_text("{ not json")

    summaries = scan_meetings(tmp_meetings_dir)

    # Bad dir skipped, no exception; only the valid meeting is returned.
    assert [s.meeting_id for s in summaries] == ["2026-02-04-regular-session"]


def test_scan_meetings_skips_dir_with_out_of_range_stage(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    # completed_stage 99 is out of range for PipelineStage and raises in _load().
    tagged_meeting_dir("x", meeting_id="2026-05-01-broken-session", completed_stage=99)

    summaries = scan_meetings(tmp_meetings_dir)

    assert [s.meeting_id for s in summaries] == ["2026-02-04-regular-session"]


from fastapi.testclient import TestClient

from gui.app import create_app


def test_library_route_renders_meetings(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    client = TestClient(create_app())

    resp = client.get("/")

    assert resp.status_code == 200
    body = resp.text
    assert "2026-02-04-regular-session" in body
    assert "Identified — ready to review" in body


def test_library_route_empty_state(tmp_meetings_dir):
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "No meetings processed yet" in resp.text


def test_library_route_survives_invalid_json_dir(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    bad = tmp_meetings_dir / "2026-05-01-broken-session"
    bad.mkdir()
    (bad / "pipeline_state.json").write_text("{ not json")
    client = TestClient(create_app())

    resp = client.get("/")

    assert resp.status_code == 200
    assert "2026-02-04-regular-session" in resp.text
    assert "2026-05-01-broken-session" not in resp.text


def test_library_route_survives_out_of_range_stage_dir(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-regular-session", completed_stage=4)
    tagged_meeting_dir("x", meeting_id="2026-05-01-broken-session", completed_stage=99)
    client = TestClient(create_app())

    resp = client.get("/")

    assert resp.status_code == 200
    assert "2026-02-04-regular-session" in resp.text
    assert "2026-05-01-broken-session" not in resp.text


def test_main_module_exposes_app_factory():
    import gui.__main__ as entry
    assert hasattr(entry, "main")
    # create_app is importable and returns a FastAPI instance
    from gui.app import create_app
    from fastapi import FastAPI
    assert isinstance(create_app(), FastAPI)


from gui.models import gate_badge


def test_gate_badge_pass_with_coverage():
    level, text = gate_badge("pass", 0.972)
    assert level == "pass"
    assert text == "97% trusted"


def test_gate_badge_pass_without_coverage():
    assert gate_badge("pass", None) == ("pass", "passed")


def test_gate_badge_review_and_failed():
    assert gate_badge("review", None) == ("review", "needs review")
    assert gate_badge("failed", 0.4) == ("failed", "failed")


def test_gate_badge_none():
    assert gate_badge(None, None) == ("none", "—")


def test_duration_label_formats_hours_and_minutes():
    from gui.models import duration_label
    assert duration_label(10325.26) == "2h 52m"
    assert duration_label(2820) == "47m"
    assert duration_label(None) == "—"
    assert duration_label(0) == "—"


def test_meeting_summary_exposes_new_display_helpers():
    s = MeetingSummary(
        meeting_id="m", title="T", city=None, meeting_type=None, date=None,
        event_kind="council", completed_stage=5,
        speaker_count=12, duration_seconds=10325.26,
        review_status="pass", trusted_coverage=0.972, has_thumbnail=True,
    )
    assert s.speakers_label == "12"
    assert s.duration_label == "2h 52m"
    assert s.gate_badge == ("pass", "97% trusted")


def test_meeting_summary_new_fields_default_to_absent():
    s = MeetingSummary(
        meeting_id="m", title=None, city=None, meeting_type=None, date=None,
        event_kind=None, completed_stage=0,
    )
    assert s.speakers_label == "—"
    assert s.duration_label == "—"
    assert s.gate_badge == ("none", "—")
    assert s.has_thumbnail is False


def test_scan_meetings_reads_named_speaker_count_and_duration(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    # transcript_named.json: identified/merged speakers + duration live here.
    (mdir / "transcript_named.json").write_text(json.dumps({
        "title": "Council",
        "duration_seconds": 10325.26,
        # Meeting.to_dict() writes speakers as a dict keyed by speaker_label.
        "speakers": {"SPEAKER_00": {}, "SPEAKER_01": {}, "SPEAKER_02": {}},
    }))
    # gate fields come from state.
    state = mdir / "pipeline_state.json"
    data = json.loads(state.read_text())
    data.update({"review_status": "pass", "trusted_coverage": 0.972})
    state.write_text(json.dumps(data))

    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.speaker_count == 3
    assert round(s.duration_seconds) == 10325
    assert s.gate_badge == ("pass", "97% trusted")


def test_scan_meetings_speaker_count_falls_back_to_diarization(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-03-01-council", completed_stage=2)
    # No transcript_named yet (pre-identification); count comes from diarization labels.
    (mdir / "diarization.json").write_text(json.dumps([
        {"speaker_label": "SPEAKER_00"}, {"speaker_label": "SPEAKER_00"},
        {"speaker_label": "SPEAKER_01"},
    ]))
    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.speaker_count == 2  # unique labels


def test_scan_meetings_thumbnail_flag(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-05-council", completed_stage=4)
    (mdir / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.has_thumbnail is True


def test_scan_meetings_enrichment_absent_is_graceful(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-06-council", completed_stage=1)
    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.speaker_count is None
    assert s.duration_seconds is None
    assert s.has_thumbnail is False
    assert s.gate_badge == ("none", "—")


def test_scan_meetings_populates_context_fields(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("bloomington-common-council",
                              meeting_id="2026-02-10-council", completed_stage=4)
    # body_slug comes from state (tagged_meeting_dir sets it to the source arg);
    # race_id from state; event_orgs from transcript_named.
    state = mdir / "pipeline_state.json"
    data = json.loads(state.read_text())
    data.update({"city": "Bloomington", "race_id": "uuid-r"})
    state.write_text(json.dumps(data))
    (mdir / "transcript_named.json").write_text(json.dumps(
        {"title": "Council", "event_orgs": ["CATS", "WFHB"]}))

    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.body_slug == "bloomington-common-council"
    assert s.race_id == "uuid-r"
    assert s.event_orgs == ["CATS", "WFHB"]
    assert "Bloomington Common Council" in s.context_line


def test_scan_meetings_context_fields_absent_are_graceful(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-11-council", completed_stage=1)
    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.event_orgs == [] and s.race_id is None


from gui.paths import is_safe_meeting_id


def test_is_safe_meeting_id_rejects_traversal():
    assert is_safe_meeting_id("2026-02-04-council") is True
    assert is_safe_meeting_id("..") is False
    assert is_safe_meeting_id(".") is False
    assert is_safe_meeting_id("a/b") is False
    assert is_safe_meeting_id("") is False
    assert is_safe_meeting_id("/abs") is False


def test_thumbnail_route_serves_existing_jpg(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    client = TestClient(create_app())
    resp = client.get("/meetings/2026-02-04-council/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8\xff\xe0jpegbytes"


def test_thumbnail_route_404_when_missing(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    client = TestClient(create_app())
    assert client.get("/meetings/2026-02-04-council/thumbnail").status_code == 404


def test_thumbnail_route_404_on_unsafe_id(tmp_meetings_dir):
    client = TestClient(create_app())
    # A dot-segment id must never resolve outside MEETINGS_DIR.
    assert client.get("/meetings/../thumbnail").status_code in (404, 400)


def test_library_route_renders_enrichment_columns(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    (mdir / "transcript_named.json").write_text(json.dumps({
        "title": "Council", "duration_seconds": 10325.26,
        "speakers": {"A": {}, "B": {}, "C": {}},
    }))
    (mdir / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff\xe0j")
    state = mdir / "pipeline_state.json"
    data = json.loads(state.read_text()); data.update({"review_status": "pass", "trusted_coverage": 0.972})
    state.write_text(json.dumps(data))

    body = TestClient(create_app()).get("/").text
    assert "97% trusted" in body            # gate badge
    assert "2h 52m" in body                 # duration
    assert ">3<" in body or "3 speakers" in body  # speaker count (see template choice below)
    assert "/meetings/2026-02-04-council/thumbnail" in body  # thumbnail img src


def test_library_has_new_meeting_link(tmp_meetings_dir):
    body = TestClient(create_app()).get("/").text
    assert 'href="/new"' in body


def test_meeting_summary_context_line_includes_guest():
    s = MeetingSummary(
        meeting_id="m", title=None, city=None, meeting_type="Interview", date="2026-05-01",
        event_kind="news_clip", completed_stage=5,
        event_orgs=["CBS"], race_label="CA Governor · 2026", guest="Xavier Becerra",
    )
    assert s.context_line == "CBS · CA Governor · 2026 · guest Xavier Becerra"


def test_scan_meetings_reads_guest(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-01-interview", completed_stage=5)
    st = mdir / "pipeline_state.json"
    data = json.loads(st.read_text()); data.update({"guest": "Xavier Becerra"}); st.write_text(json.dumps(data))
    s = scan_meetings(tmp_meetings_dir)[0]
    assert s.guest == "Xavier Becerra"
    assert "guest Xavier Becerra" in s.context_line


def test_humanize_kind_labels():
    from gui.formmeta import humanize_kind
    assert humanize_kind("news_clip") == "News Clip"
    assert humanize_kind("school_board") == "School Board"
    assert humanize_kind("council") == "Council"
    assert humanize_kind("") == ""
    assert humanize_kind(None) == ""


def test_library_humanizes_kind_display(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-01-interview", completed_stage=5)
    st = mdir / "pipeline_state.json"
    data = json.loads(st.read_text()); data.update({"event_kind": "news_clip"}); st.write_text(json.dumps(data))
    body = TestClient(create_app()).get("/").text
    assert "News Clip" in body             # humanized display (dropdown label + column)
    assert 'value="news_clip"' in body     # raw value preserved for the filter
    assert 'data-kind="news_clip"' in body  # raw value preserved for filtering


# --- live-site status (distinct from the local export stage) --------------------

def test_live_badge_states():
    def s(is_live):
        return MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                              date=None, event_kind=None, completed_stage=7, is_live=is_live)
    assert s(None).live_badge is None                 # not checked -> no badge
    assert s(True).live_badge == ("live", "Live")
    assert s(False).live_badge == ("notlive", "Not live")


def test_scan_meetings_marks_live_from_slug_set(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    tagged_meeting_dir("x", meeting_id="2026-03-04-council", completed_stage=7)
    out = {s.meeting_id: s for s in
           scan_meetings(tmp_meetings_dir, live_slugs={"2026-02-04-council"})}
    assert out["2026-02-04-council"].is_live is True
    assert out["2026-03-04-council"].is_live is False   # exported locally, not live


def test_scan_meetings_live_none_when_not_checked(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    s = scan_meetings(tmp_meetings_dir)[0]              # no live_slugs passed
    assert s.is_live is None and s.live_badge is None


def test_library_route_shows_live_badge(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    tagged_meeting_dir("x", meeting_id="2026-03-04-council", completed_stage=7)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts",
                        lambda: {"2026-02-04-council": None})
    body = TestClient(create_app()).get("/").text
    assert "Live" in body and "Not live" in body
    assert "Exported" in body            # stage 7 no longer mislabeled "Published"


def test_library_route_no_live_badge_without_db(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts", lambda: None)  # DB not configured
    body = TestClient(create_app()).get("/").text
    assert "live-badge" not in body      # no Live/Not-live badge rendered (only the "—" placeholder)


def test_library_route_renders_filter_bar_and_context(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    import gui.races as races
    monkeypatch.setattr(races, "race_labels", lambda ids: {"uuid-r": "CA Governor · 2026"})
    mdir = tagged_meeting_dir("bloomington-common-council",
                              meeting_id="2026-02-10-council", completed_stage=4)
    st = mdir / "pipeline_state.json"
    data = json.loads(st.read_text())
    data.update({"city": "Bloomington", "event_kind": "council"})
    st.write_text(json.dumps(data))
    body = TestClient(create_app()).get("/").text
    # filter bar
    assert 'id="lib-search"' in body and 'id="lib-kind"' in body and 'id="lib-status"' in body
    assert "library.js" in body
    # per-row data attributes for client-side filtering
    assert 'data-status="needs-review"' in body
    assert 'data-kind="council"' in body
    # context subline rendered
    assert "Bloomington Common Council" in body
    # row links to the bare workspace URL (stage-aware), NOT /review
    assert 'href="/meetings/2026-02-10-council"' in body


def test_library_route_attaches_race_label(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    import gui.races as races
    seen = {}

    def _fake_race_labels(ids):
        seen["ids"] = set(ids)
        return {"uuid-r": "TX Senate · 2026"}

    monkeypatch.setattr(races, "race_labels", _fake_race_labels)
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-01-interview", completed_stage=5)
    st = mdir / "pipeline_state.json"
    data = json.loads(st.read_text()); data.update({"race_id": "uuid-r", "event_kind": "news_clip"})
    st.write_text(json.dumps(data))
    body = TestClient(create_app()).get("/").text
    assert "uuid-r" in seen["ids"]           # route asked for the label
    assert "TX Senate · 2026" in body        # and rendered it


def test_library_js_filters_by_search_kind_status(tmp_meetings_dir):
    from pathlib import Path
    js = Path("gui/static/library.js").read_text()
    assert "lib-search" in js and "lib-kind" in js and "lib-status" in js
    assert "data-search" in js


def test_processed_label_relative():
    import time
    from gui.models import MeetingSummary
    def s(pa):
        return MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                              date=None, event_kind=None, completed_stage=0, processed_at=pa)
    now = time.time()
    assert s(now - 90).processed_label.endswith("m ago")     # "1m ago"
    assert s(now - 7200).processed_label.endswith("h ago")   # "2h ago"
    assert s(None).processed_label == "—"


def test_scan_meetings_sorts_by_processed_recency(tagged_meeting_dir, tmp_meetings_dir):
    import os, time
    old = tagged_meeting_dir("x", meeting_id="2026-01-01-old", completed_stage=4)
    new = tagged_meeting_dir("x", meeting_id="2026-01-02-new", completed_stage=4)
    # Make "old" the more-recently-processed one (older clip date, newer mtime).
    now = time.time()
    os.utime(new / "pipeline_state.json", (now - 1000, now - 1000))
    os.utime(old / "pipeline_state.json", (now, now))
    ids = [s.meeting_id for s in scan_meetings(tmp_meetings_dir)]
    assert ids == ["2026-01-01-old", "2026-01-02-new"]       # by mtime desc, not clip date


def test_library_renders_processed_column(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    body = TestClient(create_app()).get("/").text
    assert "<th>Processed</th>" in body


def test_library_renders_batch_header_and_pending(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    import gui.batch as batch
    monkeypatch.setattr(batch, "status",
                        lambda: {"counts": {"running": 2, "pending": 1, "max": 8},
                                 "running": [],
                                 "pending": [{"pending_id": 9, "label": "Ken Paxton",
                                              "event_kind": "news_clip", "derived_id": "d"}]})
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    body = TestClient(create_app()).get("/").text
    assert 'id="batch-running-count"' in body
    assert "Ken Paxton" in body                                  # pending chip
    assert 'action="/batch/pending/9/remove"' in body            # remove form
    assert 'action="/batch/max"' in body                         # max-concurrent control


def test_library_rows_have_meeting_id_hook(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    body = TestClient(create_app()).get("/").text
    assert 'data-meeting-id="2026-02-04-council"' in body
    assert 'class="status-cell"' in body


def test_library_js_polls_batch_status(tmp_meetings_dir):
    from pathlib import Path
    js = Path("gui/static/library.js").read_text()
    assert "/batch/status" in js and "status-cell" in js


# --- prod speaker_count beside the local one: an orphan row inflates prod's ---

def test_speakers_label_is_just_the_local_count_when_prod_agrees():
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=6, live_speaker_count=6)
    assert s.speakers_label == "6"


def test_speakers_label_shows_prods_count_when_it_is_higher():
    # The orphan symptom: prod says 7 because a merged-away label still has a
    # meetings.speakers row. Invisible until the two counts sit side by side.
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=6, live_speaker_count=7)
    assert s.speakers_label == "6 (live: 7)"


def test_speakers_label_shows_a_lower_prod_count_too():
    # Drift in either direction means prod is out of date; do not special-case
    # only the inflating one.
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=6, live_speaker_count=4)
    assert s.speakers_label == "6 (live: 4)"


def test_speakers_label_ignores_an_unknown_prod_count():
    # None = not published, or no DB. Neither is drift.
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=6, live_speaker_count=None)
    assert s.speakers_label == "6"


def test_library_route_shows_the_live_speaker_count_when_it_differs(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts",
                        lambda: {"2026-02-04-council": 99})
    body = TestClient(create_app()).get("/").text
    assert "live: 99" in body
    assert "Live" in body            # the live badge still derives from the same query


def test_library_route_without_a_db_shows_no_live_speaker_count(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=7)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts", lambda: None)
    body = TestClient(create_app()).get("/").text
    assert "live:" not in body
    assert "live-badge" not in body


def test_live_published_slugs_still_returns_a_set_of_slugs(monkeypatch):
    # Kept as the published touchpoint; it now derives from the count query.
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts", lambda: {"a": 3, "b": None})
    assert pub.live_published_slugs() == {"a", "b"}


def test_live_published_slugs_still_reports_unknown_as_none(monkeypatch):
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "live_meeting_speaker_counts", lambda: None)
    assert pub.live_published_slugs() is None


def test_speakers_label_shows_prods_count_when_there_is_no_local_one():
    # No local transcript to count, but prod has speakers. That is the
    # not-judgeable case the orphan audit reports separately, and showing prod's
    # number is informative rather than misleading — a bare "—" hides it.
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=None, live_speaker_count=7)
    assert s.speakers_label == "— (live: 7)"


def test_speakers_label_is_a_bare_dash_when_neither_count_is_known():
    from gui.models import MeetingSummary
    s = MeetingSummary(meeting_id="m", title=None, city=None, meeting_type=None,
                       date=None, event_kind=None, completed_stage=7,
                       speaker_count=None, live_speaker_count=None)
    assert s.speakers_label == "—"
# --- _speaker_count: transcript_named 'speakers' is a DICT, not a list ---------

from gui.library import _speaker_count


def _write_diarization(mdir, labels):
    (mdir / "diarization.json").write_text(
        json.dumps([{"speaker_label": l} for l in labels]), encoding="utf-8")


def test_speaker_count_prefers_named_speakers_dict(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    # Real transcript_named.json keys 'speakers' by label -> dict (src/models.py).
    named = {"speakers": {"Mayor Thomson": {}, "SPEAKER_01": {}, "Clerk": {}}}
    _write_diarization(mdir, ["SPEAKER_00", "SPEAKER_01"])  # raw count differs

    assert _speaker_count(mdir, named) == 3  # identified speakers win over raw


def test_speaker_count_falls_back_when_named_speakers_absent(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_diarization(mdir, ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])

    assert _speaker_count(mdir, {"title": "Council"}) == 2  # unique raw labels


def test_speaker_count_falls_back_when_named_speakers_empty(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_diarization(mdir, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])

    assert _speaker_count(mdir, {"speakers": {}}) == 3  # not 0


def test_speaker_count_none_when_neither_source_exists(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()

    assert _speaker_count(mdir, None) is None
    assert _speaker_count(mdir, {"speakers": {}}) is None
