from __future__ import annotations

from pathlib import Path

from gui.models import SpeakerCard, ReviewPageData, CONFIDENT_THRESHOLD


def _card(label, name, conf, method="human_review"):
    return SpeakerCard(
        label=label, name=name, confidence=conf, method=method,
        minutes=3.0, seg_count=4, sample_text="hello", hints=[], clip_seeks=[12.0],
    )


def test_confident_threshold_value():
    assert CONFIDENT_THRESHOLD == 0.85


def test_speaker_card_is_confirmed_requires_name_and_high_confidence():
    assert _card("S0", "Mayor Johnson", 0.91).is_confirmed is True
    assert _card("S1", "Mayor Johnson", 0.5).is_confirmed is False   # low conf
    assert _card("S2", None, 0.99).is_confirmed is False              # no name
    assert _card("S3", "(unidentified)", 0.99).is_confirmed is False  # placeholder name


def _card_m(method, conf=0.97):
    return SpeakerCard(
        label="S", name="Ted Brown", confidence=conf, method=method,
        minutes=3.0, seg_count=4,
    )


def test_speaker_card_is_confirmed_requires_trusted_method():
    """Confirmed must mean gate-trusted, not merely high-confidence. A speaker the
    pipeline auto-identified (auto_linked/llm) is NOT confirmed even at high
    confidence, so the review UI surfaces it for a one-click human Accept and the
    UI's 'confirmed' can't diverge from the gate's 'trusted'."""
    # Human-authoritative / trusted-tier methods -> confirmed.
    assert _card_m("human_review", conf=1.0).is_confirmed is True
    assert _card_m("voice_profile", conf=0.9).is_confirmed is True
    # High confidence but auto-identified (untrusted tier) -> NOT confirmed.
    assert _card_m("auto_linked").is_confirmed is False
    assert _card_m("llm").is_confirmed is False
    assert _card_m("name_addressing").is_confirmed is False
    assert _card_m(None).is_confirmed is False


def test_speaker_card_display_name_placeholder():
    assert _card("S0", None, 0.0).display_name == "(unidentified)"
    assert _card("S0", "Mayor Johnson", 0.9).display_name == "Mayor Johnson"


def test_review_page_data_holds_groups():
    page = ReviewPageData(
        meeting_id="m", display_name="Council", media_kind="video",
        needs_attention=[_card("S1", None, 0.0)],
        confirmed=[_card("S0", "Mayor Johnson", 0.9)],
    )
    assert page.speaker_count == 2
    assert page.needs_attention[0].label == "S1"
    assert page.confirmed[0].label == "S0"


import json

import pytest

from gui.review_api import find_meeting_media, load_review_page


def _write_meeting(mdir, *, clip_start=None):
    """Write a transcript_named.json with 2 speakers (one confident, one not)."""
    from src.models import Meeting, Segment, SpeakerMapping

    segs = [
        Segment(segment_id=0, start_time=10.0, end_time=70.0, speaker_label="SPEAKER_00",
                text="Good evening and welcome to the council meeting.", speaker_name="Mayor Johnson"),
        Segment(segment_id=1, start_time=80.0, end_time=95.0, speaker_label="SPEAKER_01",
                text="Point of order.", speaker_name=None),
    ]
    speakers = {
        "SPEAKER_00": SpeakerMapping(speaker_label="SPEAKER_00", speaker_name="Mayor Johnson",
                                     confidence=0.95, id_method="voice_profile"),
        "SPEAKER_01": SpeakerMapping(speaker_label="SPEAKER_01", speaker_name=None, confidence=0.0),
    }
    meeting = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-02-04",
                      meeting_type="Regular Session", event_kind="council",
                      segments=segs, speakers=speakers, clip_start_seconds=clip_start)
    (mdir / "transcript_named.json").write_text(json.dumps(meeting.to_dict()))


def test_find_meeting_media_prefers_video_then_audio(tmp_path):
    assert find_meeting_media(tmp_path) is None
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    assert find_meeting_media(tmp_path) == ("audio", "audio.wav")
    (tmp_path / "source.mp4").write_bytes(b"\x00\x00")
    assert find_meeting_media(tmp_path) == ("video", "source.mp4")


def test_load_review_page_missing_meeting_returns_none(tmp_meetings_dir):
    assert load_review_page("nope") is None
    assert load_review_page("../escape") is None  # unsafe id


def test_load_review_page_groups_and_orders(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    page = load_review_page("2026-02-04-council")
    assert page is not None
    assert page.display_name == "Council" or "Bloomington" in page.display_name
    # SPEAKER_00 is named @0.95 -> confirmed; SPEAKER_01 unnamed -> needs attention.
    assert [c.label for c in page.confirmed] == ["SPEAKER_00"]
    assert [c.label for c in page.needs_attention] == ["SPEAKER_01"]
    assert page.media_kind is None  # no media files written


def test_load_review_page_computes_audio_seeks_cliplocal(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    (mdir / "audio.wav").write_bytes(b"RIFF")
    page = load_review_page("2026-02-04-council")
    assert page.media_kind == "audio"
    # SPEAKER_00's longest turn starts at 10.0; audio is clip-local, 3s lead-in -> 7.0
    conf = page.confirmed[0]
    assert conf.clip_seeks and conf.clip_seeks[0] == pytest.approx(7.0)


def test_load_review_page_video_seeks_add_clip_offset(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir, clip_start=600.0)
    (mdir / "source.mp4").write_bytes(b"\x00")
    page = load_review_page("2026-02-04-council")
    assert page.media_kind == "video"
    # video is full source: seek = max(0, 10-3) + 600 = 607.0
    assert page.confirmed[0].clip_seeks[0] == pytest.approx(607.0)


def _write_meeting_with_duplicate(mdir):
    """3 speakers, two human-renamed to the same person (the Zulich collision)."""
    from src.models import Meeting, Segment, SpeakerMapping

    segs = [
        Segment(segment_id=0, start_time=10.0, end_time=70.0, speaker_label="SPEAKER_07",
                text="I vote no.", speaker_name="District 6 Zulich"),
        Segment(segment_id=1, start_time=80.0, end_time=95.0, speaker_label="SPEAKER_19",
                text="Thank you.", speaker_name="District 6 Zulich"),
        Segment(segment_id=2, start_time=100.0, end_time=130.0, speaker_label="SPEAKER_00",
                text="Motion carries.", speaker_name="Mayor Johnson"),
    ]
    speakers = {
        "SPEAKER_07": SpeakerMapping(speaker_label="SPEAKER_07", speaker_name="District 6 Zulich",
                                     confidence=1.0, id_method="human_review"),
        "SPEAKER_19": SpeakerMapping(speaker_label="SPEAKER_19", speaker_name="District 6 Zulich",
                                     confidence=1.0, id_method="human_review"),
        "SPEAKER_00": SpeakerMapping(speaker_label="SPEAKER_00", speaker_name="Mayor Johnson",
                                     confidence=0.95, id_method="voice_profile"),
    }
    meeting = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-06-10",
                      meeting_type="Regular Session", event_kind="council",
                      segments=segs, speakers=speakers)
    (mdir / "transcript_named.json").write_text(json.dumps(meeting.to_dict()))


def test_load_review_page_surfaces_duplicate_name_warning(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-06-10-council", completed_stage=4)
    _write_meeting_with_duplicate(mdir)
    page = load_review_page("2026-06-10-council")
    dup = [w for w in page.warnings if w["kind"] == "duplicate_name"]
    assert len(dup) == 1
    assert dup[0]["labels"] == ["SPEAKER_07", "SPEAKER_19"]
    by_label = {c.label: c for c in page.all_cards}
    assert by_label["SPEAKER_07"].duplicate_labels == ["SPEAKER_19"]
    assert by_label["SPEAKER_19"].duplicate_labels == ["SPEAKER_07"]
    assert by_label["SPEAKER_00"].duplicate_labels == []


def test_load_review_page_clean_meeting_has_no_warnings(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    page = load_review_page("2026-02-04-council")
    assert page.warnings == []
    assert all(c.duplicate_labels == [] for c in page.all_cards)


from fastapi.testclient import TestClient

from gui.app import create_app


def test_review_route_renders_groups(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    resp = client.get("/meetings/2026-02-04-council/review")
    assert resp.status_code == 200
    body = resp.text
    assert "Mayor Johnson" in body            # confirmed speaker
    assert "SPEAKER_01" in body               # needs-attention label
    assert "Needs attention" in body and "Confirmed" in body


def test_review_route_404_for_unknown_meeting(tmp_meetings_dir):
    client = TestClient(create_app())
    assert client.get("/meetings/ghost/review").status_code == 404


def test_media_route_serves_audio_with_range(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "audio.wav").write_bytes(b"0123456789")
    client = TestClient(create_app())

    full = client.get("/meetings/2026-02-04-council/media")
    assert full.status_code == 200
    assert full.content == b"0123456789"

    part = client.get("/meetings/2026-02-04-council/media", headers={"Range": "bytes=0-3"})
    assert part.status_code == 206
    assert part.content == b"0123"


def test_media_route_404_when_no_media(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    client = TestClient(create_app())
    assert client.get("/meetings/2026-02-04-council/media").status_code == 404


def test_media_route_404_unsafe_id(tmp_meetings_dir):
    client = TestClient(create_app())
    assert client.get("/meetings/..%2Fx/media").status_code in (404, 400)


def test_review_page_has_media_player_and_clip_buttons(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    (mdir / "audio.wav").write_bytes(b"RIFF0000")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert '/meetings/2026-02-04-council/media' in body   # media element src
    assert 'data-seek=' in body                            # at least one clip button
    assert 'workspace.js' in body                          # playback script wired


def test_library_links_to_review(tagged_meeting_dir, tmp_meetings_dir):
    tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    body = TestClient(create_app()).get("/").text
    # library now links to the bare workspace URL; the shell picks the tab by stage.
    assert 'href="/meetings/2026-02-04-council"' in body


def test_load_review_page_malformed_transcript_shape_returns_none(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    # Valid JSON but wrong shape: a list, not the Meeting object dict.
    (mdir / "transcript_named.json").write_text("[]")

    assert load_review_page("2026-02-04-council") is None
    resp = TestClient(create_app()).get("/meetings/2026-02-04-council/review")
    assert resp.status_code == 200                 # degrades to a placeholder, not 404
    assert "available yet" in resp.text            # the not-ready placeholder is shown
                                                    # (Jinja escapes "isn't" -> "isn&#39;t")


def test_load_review_page_malformed_embeddings_degrade_gracefully(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    # Valid JSON but wrong shape for embeddings (list, not a dict of label->vector).
    (mdir / "embeddings.json").write_text("[1, 2]")

    page = load_review_page("2026-02-04-council")
    assert page is not None  # bad embeddings degrade to "no hints", not a crash
    resp = TestClient(create_app()).get("/meetings/2026-02-04-council/review")
    assert resp.status_code == 200


from gui.review_api import _load_meeting_ctx, persist_review


def test_load_meeting_ctx_returns_none_for_unsafe_or_missing(tmp_meetings_dir):
    assert _load_meeting_ctx("../x") is None
    assert _load_meeting_ctx("ghost") is None


def test_persist_review_syncs_segments_and_writes_named(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    ctx = _load_meeting_ctx("2026-02-04-council")
    assert ctx is not None
    meeting, meeting_dir, _roster = ctx

    # Simulate a rename having happened on the mapping only.
    meeting.speakers["SPEAKER_01"].speaker_name = "Clerk Smith"
    meeting.speakers["SPEAKER_01"].confidence = 1.0
    meeting.speakers["SPEAKER_01"].id_method = "human_review"

    persist_review(meeting, meeting_dir)

    # transcript_named.json now carries the new name on BOTH mapping and segment.
    import json as _json
    data = _json.loads((meeting_dir / "transcript_named.json").read_text())
    assert data["speakers"]["SPEAKER_01"]["speaker_name"] == "Clerk Smith"
    seg01 = [s for s in data["segments"] if s["speaker_label"] == "SPEAKER_01"][0]
    assert seg01["speaker_name"] == "Clerk Smith"


def test_persist_review_recomputes_gate_quality_json(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    meeting, meeting_dir, _ = _load_meeting_ctx("2026-02-04-council")
    persist_review(meeting, meeting_dir)
    # Gate ran (best-effort): quality.json written and state mirrored.
    assert (meeting_dir / "quality.json").exists()
    from src.checkpoint import PipelineState
    assert PipelineState(meeting_dir).review_status in ("pass", "review", "failed")


from gui.review_api import apply_rename


def test_apply_rename_sets_name_and_confirms(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_rename("2026-02-04-council", "SPEAKER_01", "Clerk Smith") is True

    # Reload the page: SPEAKER_01 is now named + confident -> Confirmed group.
    page = load_review_page("2026-02-04-council")
    conf_labels = [c.label for c in page.confirmed]
    assert "SPEAKER_01" in conf_labels
    card = [c for c in page.confirmed if c.label == "SPEAKER_01"][0]
    assert card.name == "Clerk Smith"
    assert card.confidence == 1.0


def test_apply_rename_rejects_unknown_meeting_label_and_empty(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_rename("ghost", "SPEAKER_00", "X") is False          # no meeting
    assert apply_rename("2026-02-04-council", "SPEAKER_99", "X") is False  # unknown label
    assert apply_rename("2026-02-04-council", "SPEAKER_00", "   ") is False  # empty name
    assert apply_rename("../x", "SPEAKER_00", "X") is False           # unsafe id


def test_post_name_renames_and_redirects(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())

    resp = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/name",
                       data={"name": "Clerk Smith"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/meetings/2026-02-04-council/review"

    # Follow-up GET shows the new name.
    body = client.get("/meetings/2026-02-04-council/review").text
    assert "Clerk Smith" in body


def test_post_name_empty_is_noop_redirect(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    resp = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/name",
                       data={"name": "   "}, follow_redirects=False)
    assert resp.status_code == 303  # back to the page, no change


def test_post_name_unknown_meeting_or_label_404(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    assert client.post("/meetings/ghost/speakers/SPEAKER_00/name",
                       data={"name": "X"}, follow_redirects=False).status_code == 404
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_99/name",
                       data={"name": "X"}, follow_redirects=False).status_code == 404


def test_accept_name_prefers_current_name_then_hint():
    from gui.models import SpeakerCard
    c1 = SpeakerCard(label="S", name="Mayor Johnson", confidence=0.5, method=None,
                     minutes=1, seg_count=1)
    assert c1.accept_name == "Mayor Johnson"
    c2 = SpeakerCard(label="S", name=None, confidence=0.0, method=None,
                     minutes=1, seg_count=1, hints=[("Ada Lovelace", 0.7)])
    assert c2.accept_name == "Ada Lovelace"
    c3 = SpeakerCard(label="S", name=None, confidence=0.0, method=None, minutes=1, seg_count=1)
    assert c3.accept_name is None


def test_review_page_has_rename_form_and_accept_button(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    # A rename form posts to the name endpoint for the unnamed speaker.
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_01/name"' in body
    assert 'name="name"' in body
    # SPEAKER_00 is named at high conf -> confirmed, still editable (rename form present).
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_00/name"' in body


def test_persist_review_leaves_no_temp_file(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    meeting, meeting_dir, _ = _load_meeting_ctx("2026-02-04-council")
    persist_review(meeting, meeting_dir)
    assert (meeting_dir / "transcript_named.json").exists()
    assert not (meeting_dir / "transcript_named.json.tmp").exists()


def test_speaker_card_carries_politician_link_fields():
    from gui.models import SpeakerCard
    c = SpeakerCard(label="S", name="Tom Steyer", confidence=1.0, method="human_review",
                    minutes=2, seg_count=3, politician_slug="tom-steyer", politician_id="uuid-1")
    assert c.politician_slug == "tom-steyer"
    assert c.politician_id == "uuid-1"
    assert c.is_linked is True
    assert SpeakerCard(label="S", name=None, confidence=0, method=None,
                       minutes=0, seg_count=0).is_linked is False


def test_load_review_page_populates_link_fields(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    # link SPEAKER_00 in the on-disk meeting
    import json as _json
    data = _json.loads((mdir / "transcript_named.json").read_text())
    data["speakers"]["SPEAKER_00"]["politician_slug"] = "mayor-johnson"
    data["speakers"]["SPEAKER_00"]["politician_id"] = "uuid-mj"
    (mdir / "transcript_named.json").write_text(_json.dumps(data))

    page = load_review_page("2026-02-04-council")
    card = [c for c in page.confirmed if c.label == "SPEAKER_00"][0]
    assert card.politician_slug == "mayor-johnson"
    assert card.is_linked is True


from gui.review_api import apply_link, apply_unlink, search_politicians_safe


# These two now exercise review_api._search_politicians_http, not the direct-DB
# path: conftest's autouse _no_real_db_env strips DATABASE_URL from every test, so
# search_politicians_safe always takes the fallback branch here. The direct-DB
# branch is covered in tests/test_gui_politicians.py.
def test_search_politicians_safe_success(monkeypatch):
    import src.essentials_client as ec
    monkeypatch.setattr(ec, "search_politicians", lambda q, **kw: [
        {"politician_slug": "tom-steyer", "politician_id": "u1", "full_name": "Tom Steyer",
         "office_title": "Governor", "district_label": "", "government_name": "California",
         "is_incumbent": False},
    ])
    out = search_politicians_safe("steyer")
    assert out["error"] is None
    assert out["results"][0]["politician_slug"] == "tom-steyer"
    assert out["results"][0]["full_name"] == "Tom Steyer"


def test_search_politicians_safe_swallows_errors(monkeypatch):
    import src.essentials_client as ec
    def boom(q, **kw):
        raise ec.EssentialsClientError("nope", code="INVALID_QUERY", status=None)
    monkeypatch.setattr(ec, "search_politicians", boom)
    out = search_politicians_safe("x")
    assert out["results"] == []
    assert out["error"]  # a message, not a crash


def test_apply_link_and_unlink(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)

    assert apply_link("2026-02-04-council", "SPEAKER_01", "clerk-smith", "uuid-cs") is True
    page = load_review_page("2026-02-04-council")
    card = [c for c in (page.confirmed + page.needs_attention) if c.label == "SPEAKER_01"][0]
    assert card.politician_slug == "clerk-smith"

    assert apply_unlink("2026-02-04-council", "SPEAKER_01") is True
    page2 = load_review_page("2026-02-04-council")
    card2 = [c for c in (page2.confirmed + page2.needs_attention) if c.label == "SPEAKER_01"][0]
    assert card2.is_linked is False


def test_apply_link_guards(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_link("ghost", "SPEAKER_00", "s", "i") is False           # unknown meeting
    assert apply_link("2026-02-04-council", "SPEAKER_99", "s", "i") is False  # unknown label
    assert apply_link("../x", "SPEAKER_00", "s", "i") is False            # unsafe id


def test_apply_link_by_id_only(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    # candidate: no slug, only an essentials id
    assert apply_link("2026-02-04-council", "SPEAKER_01", "", "uuid-becerra") is True
    page = load_review_page("2026-02-04-council")
    card = [c for c in (page.confirmed + page.needs_attention) if c.label == "SPEAKER_01"][0]
    assert card.politician_id == "uuid-becerra"
    assert card.is_linked is True          # linked by id, even without a slug


def test_apply_link_requires_slug_or_id(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_link("2026-02-04-council", "SPEAKER_00", "", "") is False   # neither → no-op
    assert apply_link("2026-02-04-council", "SPEAKER_99", "s", "i") is False  # unknown label
    assert apply_link("ghost", "SPEAKER_00", "s", "i") is False              # unknown meeting


def test_search_route_returns_json(monkeypatch, tmp_meetings_dir):
    import src.essentials_client as ec
    monkeypatch.setattr(ec, "search_politicians", lambda q, **kw: [
        {"politician_slug": "tom-steyer", "politician_id": "u1", "full_name": "Tom Steyer",
         "office_title": "Governor", "district_label": "", "government_name": "CA"},
    ])
    client = TestClient(create_app())
    resp = client.get("/api/politicians/search", params={"q": "steyer"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["politician_slug"] == "tom-steyer"
    assert body["error"] is None


def test_search_route_error_is_200_empty(monkeypatch, tmp_meetings_dir):
    import src.essentials_client as ec
    monkeypatch.setattr(ec, "search_politicians",
                        lambda q, **kw: (_ for _ in ()).throw(ec.EssentialsClientError("bad", code="X", status=None)))
    client = TestClient(create_app())
    resp = client.get("/api/politicians/search", params={"q": "z"})
    assert resp.status_code == 200          # best-effort: not a 500
    assert resp.json()["results"] == []


def test_link_and_unlink_routes(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/link",
                    data={"politician_slug": "clerk-smith", "politician_id": "uuid-cs"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert "clerk-smith" in client.get("/meetings/2026-02-04-council/review").text

    r2 = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/unlink", follow_redirects=False)
    assert r2.status_code == 303

    assert client.post("/meetings/ghost/speakers/SPEAKER_00/link",
                       data={"politician_slug": "s"}, follow_redirects=False).status_code == 404


def test_review_page_has_link_widget_and_unlink(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import json as _json
    data = _json.loads((mdir / "transcript_named.json").read_text())
    data["speakers"]["SPEAKER_00"]["politician_slug"] = "mayor-johnson"
    (mdir / "transcript_named.json").write_text(_json.dumps(data))

    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    # search widget present for a speaker, wired to the search API in JS
    assert 'link-search' in body
    assert '/api/politicians/search' in body  # referenced from review.js (served inline check below)
    # unlink form for the already-linked SPEAKER_00
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_00/unlink"' in body


def test_review_js_references_search_and_link(tmp_meetings_dir):
    from pathlib import Path
    js = Path("gui/static/workspace.js").read_text()
    assert "/api/politicians/search" in js
    assert "/link" in js


import numpy as np


def _write_embeddings(mdir, dim=8, labels=("SPEAKER_00", "SPEAKER_01")):
    import json as _json
    emb = {lbl: list(np.linspace(i, i + 1, dim)) for i, lbl in enumerate(labels)}
    (mdir / "embeddings.json").write_text(_json.dumps(emb))


def test_persist_review_with_embeddings_rewrites_diar_and_emb(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)
    from gui.review_api import _load_embeddings, persist_review
    meeting, meeting_dir, _ = _load_meeting_ctx("2026-02-04-council")
    embeddings = _load_embeddings(meeting_dir)
    # drop a label from embeddings to prove the file is rewritten from the arg
    embeddings.pop("SPEAKER_01", None)

    persist_review(meeting, meeting_dir, embeddings=embeddings)

    import json as _json
    emb_on_disk = _json.loads((meeting_dir / "embeddings.json").read_text())
    assert "SPEAKER_01" not in emb_on_disk         # rewritten from the passed dict
    assert (meeting_dir / "diarization.json").exists()  # written from meeting.segments


def test_persist_review_without_embeddings_leaves_emb_untouched(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)
    from gui.review_api import persist_review
    meeting, meeting_dir, _ = _load_meeting_ctx("2026-02-04-council")
    persist_review(meeting, meeting_dir)  # no embeddings -> rename/link path
    import json as _json
    emb = _json.loads((meeting_dir / "embeddings.json").read_text())
    assert set(emb) == {"SPEAKER_00", "SPEAKER_01"}  # untouched


from gui.review_api import apply_merge


def test_apply_merge_folds_source_into_target(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)

    assert apply_merge("2026-02-04-council", "SPEAKER_01", "SPEAKER_00") is True

    import json as _json
    data = _json.loads((mdir / "transcript_named.json").read_text())
    # SPEAKER_01 is gone from speakers; its segments now belong to SPEAKER_00.
    assert "SPEAKER_01" not in data["speakers"]
    assert all(s["speaker_label"] != "SPEAKER_01" for s in data["segments"])
    emb = _json.loads((mdir / "embeddings.json").read_text())
    assert "SPEAKER_01" not in emb  # dropped from embeddings too
    # diarization.json (the riskiest cache) must be relabeled consistently too.
    diar = _json.loads((mdir / "diarization.json").read_text())
    assert all(s["speaker_label"] != "SPEAKER_01" for s in diar)


def test_apply_merge_guards(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_merge("2026-02-04-council", "SPEAKER_00", "SPEAKER_00") is False  # self-merge
    assert apply_merge("2026-02-04-council", "SPEAKER_99", "SPEAKER_00") is False  # unknown source
    assert apply_merge("2026-02-04-council", "SPEAKER_00", "SPEAKER_99") is False  # unknown target
    assert apply_merge("ghost", "SPEAKER_00", "SPEAKER_01") is False               # unknown meeting


from gui.review_api import apply_mark_non_speaker, apply_mark_unidentified


def test_apply_mark_unidentified(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_mark_unidentified("2026-02-04-council", "SPEAKER_01", "Man in blue") is True
    import json as _json
    sp = _json.loads((mdir / "transcript_named.json").read_text())["speakers"]["SPEAKER_01"]
    assert sp["speaker_status"] == "unidentified"
    assert sp["local_slug"]  # a stable handle was assigned
    assert sp["speaker_name"] == "Man in blue"


def test_apply_mark_non_speaker(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge") is True
    import json as _json
    sp = _json.loads((mdir / "transcript_named.json").read_text())["speakers"]["SPEAKER_01"]
    assert sp["speaker_status"] == "non_speaker"


def test_mark_guards(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_mark_unidentified("ghost", "SPEAKER_00", "") is False
    assert apply_mark_unidentified("2026-02-04-council", "SPEAKER_99", "") is False
    assert apply_mark_non_speaker("../x", "SPEAKER_00", "") is False


def test_speaker_card_exposes_status(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge")
    page = load_review_page("2026-02-04-council")
    card = [c for c in (page.confirmed + page.needs_attention) if c.label == "SPEAKER_01"][0]
    assert card.speaker_status == "non_speaker"


def test_merge_route(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    client = TestClient(create_app())
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/merge",
                    data={"target": "SPEAKER_00"}, follow_redirects=False)
    assert r.status_code == 303
    import json as _json
    assert "SPEAKER_01" not in _json.loads((mdir / "transcript_named.json").read_text())["speakers"]
    # self-merge / unknown target -> 404
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_00/merge",
                       data={"target": "SPEAKER_00"}, follow_redirects=False).status_code == 404


def test_unidentified_and_not_speaker_routes(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/unidentified",
                       data={"display_label": "Man in blue"}, follow_redirects=False).status_code == 303
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_00/not-speaker",
                       data={"display_label": ""}, follow_redirects=False).status_code == 303
    assert client.post("/meetings/ghost/speakers/SPEAKER_00/not-speaker",
                       data={}, follow_redirects=False).status_code == 404


def test_review_page_has_merge_and_status_controls(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    # merge form + a target option referencing the OTHER speaker
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_01/merge"' in body
    assert 'value="SPEAKER_00"' in body  # a merge target option
    # unidentified + not-speaker forms
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_01/unidentified"' in body
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_01/not-speaker"' in body


def test_status_badge_renders(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_01", "Pledge")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "not-a-speaker" in body or "non-speaker" in body  # a visible status badge


from gui.models import ENROLL_MIN_SPEECH_SECONDS


def test_enroll_min_speech_threshold():
    assert ENROLL_MIN_SPEECH_SECONDS == 30.0


def test_load_review_page_marks_enrollable_and_thin(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)  # gives SPEAKER_00 + SPEAKER_01 embeddings
    page = load_review_page("2026-02-04-council")
    cards = {c.label: c for c in (page.confirmed + page.needs_attention)}
    # SPEAKER_00: named, has embedding, 60s of speech (from _write_meeting) -> enrollable, not thin, not yet enrolled
    assert cards["SPEAKER_00"].is_enrollable is True
    assert cards["SPEAKER_00"].is_enrolled is False
    # SPEAKER_01 in _write_meeting speaks 80..95 = 15s -> thin; but unnamed -> not enrollable
    assert cards["SPEAKER_01"].is_enrollable is False
    assert cards["SPEAKER_01"].thin_sample is True


def test_load_review_page_enrollable_false_without_embedding(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)  # no embeddings.json
    page = load_review_page("2026-02-04-council")
    card = [c for c in page.confirmed if c.label == "SPEAKER_00"][0]
    assert card.is_enrollable is False  # no embedding to enroll


from gui.review_api import apply_enroll


def test_apply_enroll_writes_profile_and_is_idempotent(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)
    from src.enroll import load_profiles, resolve_mapping_enrollment

    assert apply_enroll("2026-02-04-council", "SPEAKER_00") is True
    db = load_profiles()
    # find SPEAKER_00's profile
    from src.models import SpeakerMapping
    import json as _json
    sp = _json.loads((mdir / "transcript_named.json").read_text())["speakers"]["SPEAKER_00"]
    key, _, _ = resolve_mapping_enrollment(SpeakerMapping(**{k: sp.get(k) for k in
        ("speaker_label","speaker_name","confidence","id_method","politician_slug","politician_id","local_slug","local_role","speaker_status")}))
    assert key in db.profiles
    assert "2026-02-04-council" in db.profiles[key].meetings_seen
    n_records = len(db.profiles[key].embeddings)

    # Second enroll from the SAME meeting must be a no-op (no duplicate record).
    assert apply_enroll("2026-02-04-council", "SPEAKER_00") is True
    db2 = load_profiles()
    assert len(db2.profiles[key].embeddings) == n_records  # unchanged


def test_speaker_card_profile_strength_thresholds():
    from gui.models import SpeakerCard
    def card(m, s):
        return SpeakerCard(label="X", name="A", confidence=0.9, method="voice",
                           minutes=1.0, seg_count=1, profile_meetings=m, profile_samples=s)
    assert card(0, 0).profile_strength == "new"
    assert card(0, 0).profile_hint == "New voice — no profile yet"
    assert card(1, 1).profile_strength == "building"
    assert card(1, 1).profile_hint == "Profile building — 1 sample from 1 meeting"
    assert card(2, 3).profile_strength == "building"
    assert card(3, 5).profile_strength == "strong"
    assert card(4, 6).profile_hint == "Profile strong — 6 samples from 4 meetings"


def test_load_review_page_shows_existing_profile_strength(tagged_meeting_dir, tmp_meetings_dir):
    # Enroll SPEAKER_00 (Mayor Johnson) from an EARLIER meeting to build a profile.
    prior = tagged_meeting_dir("x", meeting_id="2026-01-07-council", completed_stage=4)
    _write_meeting(prior); _write_embeddings(prior)
    assert apply_enroll("2026-01-07-council", "SPEAKER_00") is True

    # A new meeting with the same speaker, not yet enrolled here.
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    page = load_review_page("2026-02-04-council")
    card = [c for c in page.confirmed if c.label == "SPEAKER_00"][0]
    assert card.is_enrolled is False           # this meeting hasn't contributed
    assert card.profile_meetings == 1          # counts only the OTHER (prior) meeting
    assert card.profile_samples == 1
    assert card.profile_strength == "building"


def test_load_review_page_no_profile_is_new(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    page = load_review_page("2026-02-04-council")
    card = [c for c in page.confirmed if c.label == "SPEAKER_00"][0]
    assert card.profile_meetings == 0 and card.profile_strength == "new"


def test_review_page_renders_profile_hint(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "profile-strength" in body
    assert "New voice — no profile yet" in body


def test_apply_enroll_guards(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)
    assert apply_enroll("ghost", "SPEAKER_00") is False              # unknown meeting
    assert apply_enroll("2026-02-04-council", "SPEAKER_99") is False  # unknown label
    assert apply_enroll("2026-02-04-council", "SPEAKER_01") is False  # SPEAKER_01 is unnamed
    assert apply_enroll("../x", "SPEAKER_00") is False               # unsafe id


def test_apply_enroll_skips_non_speaker(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings(mdir)
    apply_mark_non_speaker("2026-02-04-council", "SPEAKER_00", "Music")
    assert apply_enroll("2026-02-04-council", "SPEAKER_00") is False  # non-speaker not enrollable


def test_enroll_route(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    client = TestClient(create_app())
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_00/enroll", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/meetings/2026-02-04-council/review"
    from src.enroll import load_profiles
    assert load_profiles().profiles  # a profile now exists

    # unknown / non-enrollable -> 404
    assert client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/enroll",
                       follow_redirects=False).status_code == 404  # unnamed
    assert client.post("/meetings/ghost/speakers/SPEAKER_00/enroll",
                       follow_redirects=False).status_code == 404


def test_review_page_shows_enroll_button(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_00/enroll"' in body
    assert "Save this voice" in body


def test_review_page_shows_saved_state_after_enroll(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir); _write_embeddings(mdir)
    apply_enroll("2026-02-04-council", "SPEAKER_00")
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "✓ voice saved" in body


def test_speaker_card_is_linked_by_id_only():
    from gui.models import SpeakerCard
    c = SpeakerCard(label="S", name="Xavier Becerra", confidence=1.0, method="human_review",
                    minutes=2, seg_count=3, politician_slug=None, politician_id="uuid-b")
    assert c.is_linked is True
    assert SpeakerCard(label="S", name=None, confidence=0, method=None,
                       minutes=0, seg_count=0).is_linked is False


def test_review_page_shows_link_for_id_only_speaker(tagged_meeting_dir, tmp_meetings_dir):
    import json as _json
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    data = _json.loads((mdir / "transcript_named.json").read_text())
    data["speakers"]["SPEAKER_00"]["politician_id"] = "uuid-b"
    data["speakers"]["SPEAKER_00"]["politician_slug"] = None
    (mdir / "transcript_named.json").write_text(_json.dumps(data))
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    # linked state shows (unlink form present) even though there's no slug
    assert 'action="/meetings/2026-02-04-council/speakers/SPEAKER_00/unlink"' in body


def test_review_page_links_to_run(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council").text
    assert 'data-tab="progress"' in body and 'href="/meetings/2026-02-04-council?tab=progress"' in body


def test_review_page_links_to_publish(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=5)
    _write_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council").text
    assert 'data-tab="publish"' in body


def test_find_meeting_media_falls_back_to_opus(tmp_path):
    from gui.review_api import find_meeting_media

    (tmp_path / "audio.opus").write_bytes(b"OPUS")
    assert find_meeting_media(tmp_path) == ("audio", "audio.opus")
    # video still wins when present
    (tmp_path / "source.mp4").write_bytes(b"\x00")
    assert find_meeting_media(tmp_path) == ("video", "source.mp4")


def test_review_route_renders_youtube_iframe(tagged_meeting_dir, tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_youtube_meeting(mdir)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert 'id="yt-player"' in body
    assert "youtube.com/embed/abc123XYZ" in body


def test_media_route_serves_opus_as_ogg(tagged_meeting_dir, tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "audio.opus").write_bytes(b"OPUSBYTES")
    client = TestClient(create_app())

    resp = client.get("/meetings/2026-02-04-council/media")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/ogg")
    assert resp.content == b"OPUSBYTES"


def _write_youtube_meeting(mdir, *, clip_start=0.0):
    from src.models import Meeting, Segment, SpeakerMapping

    segs = [
        Segment(segment_id=0, start_time=10.0, end_time=70.0, speaker_label="SPEAKER_00",
                text="Good evening.", speaker_name="Mayor Johnson"),
    ]
    speakers = {
        "SPEAKER_00": SpeakerMapping(speaker_label="SPEAKER_00", speaker_name="Mayor Johnson",
                                     confidence=0.95, id_method="voice_profile"),
    }
    meeting = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-02-04",
                      meeting_type="Regular Session", event_kind="debate",
                      segments=segs, speakers=speakers, clip_start_seconds=clip_start,
                      audio_source="https://www.youtube.com/watch?v=abc123XYZ")
    (mdir / "transcript_named.json").write_text(json.dumps(meeting.to_dict()))


def test_load_review_page_exposes_youtube_id(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import load_review_page

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_youtube_meeting(mdir)
    page = load_review_page("2026-02-04-council")
    assert page.youtube_id == "abc123XYZ"


def test_youtube_seeks_add_clip_offset_even_without_local_video(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import load_review_page

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_youtube_meeting(mdir, clip_start=600.0)
    # No local media on disk at all — streaming from YouTube.
    page = load_review_page("2026-02-04-council")
    assert page.youtube_id == "abc123XYZ"
    # seek = max(0, 10-3) + 600 = 607.0 (full-source semantics)
    assert page.confirmed[0].clip_seeks[0] == pytest.approx(607.0)


def test_cleanup_route_removes_media(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    from src import cleanup

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "transcript_named.json").write_text("{}")
    (mdir / "audio.wav").write_bytes(b"0" * 1000)
    (mdir / "source.mp4").write_bytes(b"0" * 4000)
    monkeypatch.setattr(cleanup, "compress_audio_to_opus",
                        lambda w, o, bitrate="32k": (Path(o).write_bytes(b"OPUS"), Path(o))[1])

    client = TestClient(create_app())
    resp = client.post("/meetings/2026-02-04-council/cleanup", follow_redirects=False)
    assert resp.status_code == 303
    assert not (mdir / "source.mp4").exists()
    assert (mdir / "audio.opus").exists()


def test_cleanup_route_404_for_unsafe_id(tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app

    client = TestClient(create_app())
    assert client.post("/meetings/..%2Fescape/cleanup", follow_redirects=False).status_code == 404


def test_cleanup_all_route_redirects(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    from src import cleanup

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "transcript_named.json").write_text("{}")
    (mdir / "audio.wav").write_bytes(b"0" * 1000)
    (mdir / "source.mp4").write_bytes(b"0" * 4000)
    monkeypatch.setattr(cleanup, "compress_audio_to_opus",
                        lambda w, o, bitrate="32k": (Path(o).write_bytes(b"OPUS"), Path(o))[1])

    client = TestClient(create_app())
    resp = client.post("/cleanup-all", follow_redirects=False)
    assert resp.status_code == 303
    assert not (mdir / "source.mp4").exists()


def test_delete_route_purges_on_matching_slug(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    from src import purge

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    (mdir / "transcript_named.json").write_text("{}")
    monkeypatch.setattr(purge, "_db_url", lambda: None)
    monkeypatch.setattr(purge, "_profile_contaminated", lambda slug: False)

    client = TestClient(create_app())
    resp = client.post("/meetings/2026-02-04-council/delete",
                       data={"confirm_slug": "2026-02-04-council"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    assert not mdir.exists()


def test_delete_route_noop_on_mismatched_slug(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    from src import purge

    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    monkeypatch.setattr(purge, "_db_url", lambda: None)

    client = TestClient(create_app())
    resp = client.post("/meetings/2026-02-04-council/delete",
                       data={"confirm_slug": "WRONG"}, follow_redirects=False)
    assert resp.status_code == 303
    assert mdir.exists()  # nothing deleted


def test_delete_route_404_for_unsafe_id(tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app

    client = TestClient(create_app())
    assert client.post("/meetings/..%2Fx/delete",
                       data={"confirm_slug": "x"}, follow_redirects=False).status_code == 404


def test_review_route_renders_duplicate_warning_and_merge_chip(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-06-10-council", completed_stage=4)
    _write_meeting_with_duplicate(mdir)
    client = TestClient(create_app())
    resp = client.get("/meetings/2026-06-10-council/review")
    assert resp.status_code == 200
    body = resp.text
    assert "Before you publish" in body      # warnings block is visible
    assert "Same name as" in body            # per-card duplicate chip
    # one-click merge: hidden-target forms pointing each twin at the other
    assert '<input type="hidden" name="target" value="SPEAKER_19">' in body
    assert '<input type="hidden" name="target" value="SPEAKER_07">' in body


def test_review_route_clean_meeting_renders_no_warning_block(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app())
    resp = client.get("/meetings/2026-02-04-council/review")
    assert resp.status_code == 200
    assert "Before you publish" not in resp.text
    assert "Same name as" not in resp.text


def test_review_page_exposes_local_person_fields(tagged_meeting_dir, tmp_meetings_dir):
    from src.event_kinds import local_roles_for
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    page = load_review_page("2026-02-04-council")

    # role options come from the meeting's event_kind ('council'), not a hardcoded list
    assert page.local_role_options == list(local_roles_for("council"))

    card = [c for c in page.all_cards if c.label == "SPEAKER_00"][0]
    assert card.local_slug is None
    assert card.has_local_person is False
    # prefilled from the speaker's name so the common path needs no typing
    assert card.default_slug == "mayor-johnson"


def test_apply_make_and_clear_local_person(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import apply_clear_local_person, apply_make_local_person
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)

    assert apply_make_local_person("2026-02-04-council", "SPEAKER_01",
                                   "susan-brackney", "2") is True
    page = load_review_page("2026-02-04-council")
    card = [c for c in page.all_cards if c.label == "SPEAKER_01"][0]
    assert card.local_slug == "susan-brackney"
    assert card.local_role == "staff"      # option 2 for event_kind 'council'

    assert apply_clear_local_person("2026-02-04-council", "SPEAKER_01") is True
    card2 = [c for c in load_review_page("2026-02-04-council").all_cards
             if c.label == "SPEAKER_01"][0]
    assert card2.local_slug is None


def test_apply_clear_local_person_returns_false_for_a_noop(tagged_meeting_dir, tmp_meetings_dir):
    """A label can be 'known' (it has segments) with no entry at all in
    meeting.speakers, so review.clear_local_person returns None and nothing
    changed. Reporting True here would tell the reviewer their click did
    something when it did not."""
    import json as _json
    from src.models import Meeting, Segment, SpeakerMapping
    from gui.review_api import apply_clear_local_person
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    segs = [Segment(segment_id=0, start_time=0.0, end_time=5.0,
                     speaker_label="SPEAKER_00", text="hi", speaker_name=None)]
    meeting = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-02-04",
                      meeting_type="Regular Session", event_kind="council",
                      segments=segs, speakers={})  # no mapping for SPEAKER_00 at all
    (mdir / "transcript_named.json").write_text(_json.dumps(meeting.to_dict()))

    assert apply_clear_local_person("2026-02-04-council", "SPEAKER_00") is False


def test_apply_clear_local_person_refuses_an_unidentified_speaker(tagged_meeting_dir, tmp_meetings_dir):
    """Regression guard for the Critical finding: clicking Clear on an
    unidentified speaker must not drop the synthetic unidentified-<meeting>-
    <label> handle, or two distinct unknown speakers merge onto one voice
    profile."""
    from gui.review_api import apply_clear_local_person, apply_mark_unidentified
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_mark_unidentified("2026-02-04-council", "SPEAKER_01") is True
    before = [c for c in load_review_page("2026-02-04-council").all_cards
              if c.label == "SPEAKER_01"][0]
    assert before.local_slug is not None and before.speaker_status == "unidentified"

    assert apply_clear_local_person("2026-02-04-council", "SPEAKER_01") is False

    after = [c for c in load_review_page("2026-02-04-council").all_cards
             if c.label == "SPEAKER_01"][0]
    assert after.local_slug == before.local_slug
    assert after.speaker_status == "unidentified"


def test_apply_make_local_person_accepts_a_custom_role(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import apply_make_local_person
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert apply_make_local_person("2026-02-04-council", "SPEAKER_01",
                                   "jo-doe", "City Attorney") is True
    card = [c for c in load_review_page("2026-02-04-council").all_cards
            if c.label == "SPEAKER_01"][0]
    assert card.local_role == "city_attorney"


def test_apply_make_local_person_clears_an_essentials_link(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import apply_link, apply_make_local_person
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    apply_link("2026-02-04-council", "SPEAKER_01", "clerk-smith", "uuid-cs")
    apply_make_local_person("2026-02-04-council", "SPEAKER_01", "clerk-smith-local", "staff")
    card = [c for c in load_review_page("2026-02-04-council").all_cards
            if c.label == "SPEAKER_01"][0]
    assert card.is_linked is False


def test_apply_make_local_person_guards(tagged_meeting_dir, tmp_meetings_dir):
    import pytest
    from gui.review_api import apply_make_local_person
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    # unknown / unsafe -> False
    assert apply_make_local_person("ghost", "SPEAKER_00", "a-b", "staff") is False
    assert apply_make_local_person("2026-02-04-council", "SPEAKER_99", "a-b", "staff") is False
    assert apply_make_local_person("../x", "SPEAKER_00", "a-b", "staff") is False
    # bad slug -> ValueError, a different failure the route reports differently
    with pytest.raises(ValueError):
        apply_make_local_person("2026-02-04-council", "SPEAKER_00", "Susan Brackney", "staff")
    # slug already held by another label -> ValueError
    apply_make_local_person("2026-02-04-council", "SPEAKER_00", "taken-slug", "staff")
    with pytest.raises(ValueError, match="already used"):
        apply_make_local_person("2026-02-04-council", "SPEAKER_01", "taken-slug", "staff")


def test_local_person_routes(tagged_meeting_dir, tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app(), follow_redirects=False)

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/local-person",
                    data={"slug": "susan-brackney", "role": "2"})
    assert r.status_code == 303
    card = [c for c in load_review_page("2026-02-04-council").all_cards
            if c.label == "SPEAKER_01"][0]
    assert card.local_slug == "susan-brackney"

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/local-person/clear")
    assert r.status_code == 303
    card = [c for c in load_review_page("2026-02-04-council").all_cards
            if c.label == "SPEAKER_01"][0]
    assert card.local_slug is None


def _write_meeting_local_person_states(mdir):
    """One speaker in each local-person-relevant state, for the review-page
    render tests: a plain unlinked speaker (make-local-person form), a genuine
    local person (badge + Clear), a speaker linked to a politician (neither),
    and an UNIDENTIFIED speaker (neither — the Critical's regression guard,
    since mark_unidentified also sets local_slug to a synthetic handle)."""
    from src.models import Meeting, Segment, SpeakerMapping

    segs = [
        Segment(segment_id=0, start_time=0.0, end_time=10.0, speaker_label="SPEAKER_00",
                text="Good evening.", speaker_name="Random Person"),
        Segment(segment_id=1, start_time=20.0, end_time=30.0, speaker_label="SPEAKER_01",
                text="Comment from the floor.", speaker_name="Jo Doe"),
        Segment(segment_id=2, start_time=40.0, end_time=50.0, speaker_label="SPEAKER_02",
                text="Motion carries.", speaker_name="Marcy Kaptur"),
        Segment(segment_id=3, start_time=60.0, end_time=70.0, speaker_label="SPEAKER_03",
                text="[unclear]", speaker_name="Unidentified Speaker"),
    ]
    speakers = {
        "SPEAKER_00": SpeakerMapping(speaker_label="SPEAKER_00", speaker_name="Random Person"),
        "SPEAKER_01": SpeakerMapping(speaker_label="SPEAKER_01", speaker_name="Jo Doe",
                                     local_slug="jo-doe", local_role="staff"),
        "SPEAKER_02": SpeakerMapping(speaker_label="SPEAKER_02", speaker_name="Marcy Kaptur",
                                     politician_slug="marcy-kaptur", politician_id="uuid-mk"),
        "SPEAKER_03": SpeakerMapping(speaker_label="SPEAKER_03", speaker_name="Unidentified Speaker",
                                     local_slug="unidentified-2026-02-04-council-speaker_03",
                                     speaker_status="unidentified"),
    }
    meeting = Meeting(meeting_id=mdir.name, city="Bloomington", date="2026-02-04",
                      meeting_type="Regular Session", event_kind="council",
                      segments=segs, speakers=speakers)
    (mdir / "transcript_named.json").write_text(json.dumps(meeting.to_dict()))


def test_review_page_render_local_person_branch_selection(tagged_meeting_dir, tmp_meetings_dir):
    """Render test for the _macros.html card: branch selection, verified by
    inspecting the returned HTML rather than by manual inspection. This is the
    Critical's regression guard — an unidentified speaker (SPEAKER_03) must show
    NEITHER the create form nor the badge+Clear button, since its local_slug is
    the synthetic unidentified-<meeting>-<label> handle, not a local person a
    reviewer authored."""
    from fastapi.testclient import TestClient
    from gui.app import create_app
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting_local_person_states(mdir)
    client = TestClient(create_app())
    r = client.get("/meetings/2026-02-04-council/panel/review")
    assert r.status_code == 200
    html = r.text

    def create_form_present(label):
        return f'/meetings/2026-02-04-council/speakers/{label}/local-person"' in html

    def clear_button_present(label):
        return f'/meetings/2026-02-04-council/speakers/{label}/local-person/clear"' in html

    # 1. plain unlinked speaker -> shows the create form
    assert create_form_present("SPEAKER_00") is True
    assert clear_button_present("SPEAKER_00") is False

    # 2. speaker with a local person -> shows the badge + Clear, not the create form
    assert clear_button_present("SPEAKER_01") is True
    assert create_form_present("SPEAKER_01") is False
    assert "local: jo-doe" in html

    # 3. speaker linked to a politician -> shows neither
    assert create_form_present("SPEAKER_02") is False
    assert clear_button_present("SPEAKER_02") is False

    # 4. UNIDENTIFIED speaker -> shows neither (the Critical's regression guard)
    assert create_form_present("SPEAKER_03") is False
    assert clear_button_present("SPEAKER_03") is False


def test_local_person_route_rejects_a_bad_slug_with_400(tagged_meeting_dir, tmp_meetings_dir):
    """A silently-ignored submission is the worst outcome for a reviewer, so an
    invalid slug is a visible 400 rather than the no-op redirect set_speaker_name
    uses for empty input."""
    from fastapi.testclient import TestClient
    from gui.app import create_app
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app(), follow_redirects=False)
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/local-person",
                    data={"slug": "Susan Brackney", "role": "staff"})
    assert r.status_code == 400


def test_local_person_route_unknown_label_is_404(tagged_meeting_dir, tmp_meetings_dir):
    from fastapi.testclient import TestClient
    from gui.app import create_app
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    client = TestClient(create_app(), follow_redirects=False)
    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_99/local-person",
                    data={"slug": "a-b", "role": "staff"})
    assert r.status_code == 404


# --- merge-time voice guard ---------------------------------------------------

def _write_embeddings_pair(mdir, a_vec, b_vec, a="SPEAKER_00", b="SPEAKER_01"):
    import json as _json
    (mdir / "embeddings.json").write_text(_json.dumps({a: list(a_vec), b: list(b_vec)}))


ORTHOGONAL = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])   # cos 0.0 -> mismatch


def test_merge_voice_report_bands_a_proposed_merge(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import merge_voice_report
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)
    rep = merge_voice_report("2026-02-04-council", "SPEAKER_01", "SPEAKER_00")
    assert rep["verdict"] == "mismatch"
    assert rep["blocked"] is True
    assert rep["similarity"] == pytest.approx(0.0)


def test_merge_voice_report_none_for_unknown_meeting_or_label(tagged_meeting_dir, tmp_meetings_dir):
    from gui.review_api import merge_voice_report
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    assert merge_voice_report("ghost", "SPEAKER_00", "SPEAKER_01") is None
    assert merge_voice_report("2026-02-04-council", "SPEAKER_99", "SPEAKER_00") is None
    assert merge_voice_report("2026-02-04-council", "SPEAKER_00", "SPEAKER_00") is None


def test_apply_merge_refuses_a_voice_mismatch(tagged_meeting_dir, tmp_meetings_dir):
    """The whole point: a destructive merge of two different voices must not go
    through silently, because it leaves one label holding two people and every
    name-based detector then reads clean."""
    import json as _json
    from gui.review_api import apply_merge
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)

    assert apply_merge("2026-02-04-council", "SPEAKER_01", "SPEAKER_00") is False
    # and nothing was written
    data = _json.loads((mdir / "transcript_named.json").read_text())
    assert "SPEAKER_01" in data["speakers"]
    assert "SPEAKER_01" in _json.loads((mdir / "embeddings.json").read_text())


def test_apply_merge_proceeds_on_mismatch_when_confirmed(tagged_meeting_dir, tmp_meetings_dir):
    import json as _json
    from gui.review_api import apply_merge
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)

    assert apply_merge("2026-02-04-council", "SPEAKER_01", "SPEAKER_00",
                       confirm_mismatch=True) is True
    data = _json.loads((mdir / "transcript_named.json").read_text())
    assert "SPEAKER_01" not in data["speakers"]


def test_apply_merge_never_blocked_when_similarity_is_unmeasurable(tagged_meeting_dir, tmp_meetings_dir):
    """No embeddings.json at all — 'we could not tell' must not read as a mis-merge."""
    from gui.review_api import apply_merge, merge_voice_report
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    rep = merge_voice_report("2026-02-04-council", "SPEAKER_01", "SPEAKER_00")
    assert rep["verdict"] == "unknown" and rep["blocked"] is False
    assert apply_merge("2026-02-04-council", "SPEAKER_01", "SPEAKER_00") is True


def test_review_cards_annotate_each_merge_target_with_its_voice(tagged_meeting_dir, tmp_meetings_dir):
    """The dropdown must guide the choice BEFORE the click, so each candidate
    target carries its own similarity and mismatches are named."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)
    page = load_review_page("2026-02-04-council")
    by_label = {c.label: c for c in page.all_cards}
    assert by_label["SPEAKER_00"].merge_mismatches == ["SPEAKER_01"]
    assert by_label["SPEAKER_01"].merge_mismatches == ["SPEAKER_00"]
    assert "0.00" in by_label["SPEAKER_00"].merge_hints["SPEAKER_01"]


def test_merge_route_offers_confirmation_instead_of_404_on_a_voice_mismatch(
        tagged_meeting_dir, tmp_meetings_dir):
    """A blocked merge is a real UI state, not a missing resource — the route must
    distinguish it from an unknown label (which stays a 404)."""
    import json as _json
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)
    client = TestClient(create_app())

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/merge",
                    data={"target": "SPEAKER_00"}, follow_redirects=False)
    assert r.status_code == 303                      # not 404
    data = _json.loads((mdir / "transcript_named.json").read_text())
    assert "SPEAKER_01" in data["speakers"]          # nothing merged

    r = client.post("/meetings/2026-02-04-council/speakers/SPEAKER_01/merge",
                    data={"target": "SPEAKER_00", "confirm": "1"}, follow_redirects=False)
    assert r.status_code == 303
    data = _json.loads((mdir / "transcript_named.json").read_text())
    assert "SPEAKER_01" not in data["speakers"]      # confirmed merge went through


def test_merge_route_still_404s_on_an_unknown_label(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    r = TestClient(create_app()).post(
        "/meetings/2026-02-04-council/speakers/SPEAKER_99/merge",
        data={"target": "SPEAKER_00"}, follow_redirects=False)
    assert r.status_code == 404


def test_merge_control_renders_each_target_with_its_voice_and_flags_mismatches(
        tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    _write_embeddings_pair(mdir, *ORTHOGONAL)
    body = TestClient(create_app()).get("/meetings/2026-02-04-council?tab=review").text
    assert "voice +0.00 ✗" in body            # annotated option
    assert "data-merge-mismatch" in body      # machine-readable for the confirm prompt


# --- orphan speaker rows: label surgery on a LIVE meeting leaves prod stale ---

def _prod_rows(*pairs):
    from src.speaker_orphans import DbSpeaker
    return [DbSpeaker(label=l, display_name=n) for l, n in pairs]


def test_review_warns_when_prod_holds_a_label_the_transcript_dropped(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    # The reviewer merged SPEAKER_09 away locally. Nothing else in the GUI
    # consults prod, so without this the meeting looks finished while the old
    # row keeps serving.
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "published_speaker_rows", lambda mid: _prod_rows(
        ("SPEAKER_00", "Mayor Johnson"), ("SPEAKER_09", "Councilmember Volan")))
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "Before you publish" in body
    assert "warn-stale_published_speaker" in body
    assert "SPEAKER_09" in body and "Councilmember Volan" in body
    assert "republish" in body.lower()


def test_review_shows_no_stale_warning_when_prod_matches_the_transcript(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "published_speaker_rows", lambda mid: _prod_rows(
        ("SPEAKER_00", "Mayor Johnson"), ("SPEAKER_01", None)))
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "warn-stale_published_speaker" not in body


def test_review_shows_no_stale_warning_for_an_unpublished_meeting(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    # Not live: an empty row list is "published nothing", never "everything stale".
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "published_speaker_rows", lambda mid: [])
    body = TestClient(create_app()).get("/meetings/2026-02-04-council/review").text
    assert "warn-stale_published_speaker" not in body


def test_review_page_still_renders_when_prod_cannot_be_reached(
    tagged_meeting_dir, tmp_meetings_dir, monkeypatch
):
    # None = unknown (no DB / query failed). Review must not break or invent a
    # warning it cannot substantiate.
    mdir = tagged_meeting_dir("x", meeting_id="2026-02-04-council", completed_stage=4)
    _write_meeting(mdir)
    import gui.publish_api as pub
    monkeypatch.setattr(pub, "published_speaker_rows", lambda mid: None)
    resp = TestClient(create_app()).get("/meetings/2026-02-04-council/review")
    assert resp.status_code == 200
    assert "warn-stale_published_speaker" not in resp.text


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
