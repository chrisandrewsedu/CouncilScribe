from __future__ import annotations

import json

from src.models import Meeting, Segment, SpeakerMapping, MeetingSummary, SummarySection
from backfill_segment_merge import republish_notice
from backfill_segment_merge import (
    backfill,
    merge_would_change,
    reindex_summary_sections,
    remerge_meeting,
    sections_are_stale,
)


def test_reindex_summary_sections_maps_times_to_merged_indices():
    segs = [
        Segment(segment_id=0, start_time=0.0, end_time=9.0, speaker_label="A", text="a"),
        Segment(segment_id=1, start_time=10.0, end_time=19.0, speaker_label="B", text="b"),
        Segment(segment_id=2, start_time=20.0, end_time=29.0, speaker_label="A", text="c"),
    ]
    sec = SummarySection(section_type="discussion", title="Mid", content="…",
                         start_time=10.0, end_time=19.0,
                         start_segment=99, end_segment=99)  # stale indices
    m = Meeting(meeting_id="m", city="X", date="2026-01-01", meeting_type="R",
                event_kind="council", segments=segs, speakers={},
                summary=MeetingSummary(executive_summary="", sections=[sec]))
    reindex_summary_sections(m)
    assert sec.start_segment == 1 and sec.end_segment == 1  # remapped to the middle segment


def test_reindex_summary_sections_no_summary_is_noop():
    m = Meeting(meeting_id="m", city="X", date="2026-01-01", meeting_type="R",
                event_kind="council",
                segments=[Segment(segment_id=0, start_time=0.0, end_time=1.0,
                                  speaker_label="A", text="a")], speakers={})
    reindex_summary_sections(m)  # must not raise


def _fragmented_meeting(meeting_id="2026-05-15-interview"):
    # Four consecutive same-speaker fragments + one other speaker.
    segs = [
        Segment(segment_id=0, start_time=0.0, end_time=3.0, speaker_label="S0",
                speaker_name="Bass", text="of this being"),
        Segment(segment_id=1, start_time=3.2, end_time=5.0, speaker_label="S0",
                speaker_name="Bass", text="the worst"),
        Segment(segment_id=2, start_time=5.1, end_time=7.0, speaker_label="S0",
                speaker_name="Bass", text="natural disaster"),
        Segment(segment_id=3, start_time=8.0, end_time=10.0, speaker_label="S1",
                speaker_name="Host", text="I see."),
    ]
    sec = SummarySection(section_type="discussion", title="Disaster", content="…",
                         start_time=0.0, end_time=7.0, start_segment=0, end_segment=2)
    return Meeting(meeting_id=meeting_id, city="LA", date="2026-05-15",
                   meeting_type="Interview", event_kind="news_clip",
                   segments=segs, speakers={"S0": SpeakerMapping(speaker_label="S0", speaker_name="Bass"),
                                            "S1": SpeakerMapping(speaker_label="S1", speaker_name="Host")},
                   summary=MeetingSummary(executive_summary="", sections=[sec]))


def test_remerge_meeting_collapses_and_reindexes():
    m = _fragmented_meeting()
    before, after, _ = remerge_meeting(m)
    assert before == 4 and after == 2                    # 3 Bass fragments -> 1, Host stays
    assert m.segments[0].text == "of this being the worst natural disaster"
    # section still covers the Bass block, now a single merged segment (index 0)
    assert m.summary.sections[0].start_segment == 0
    assert m.summary.sections[0].end_segment == 0


def test_backfill_rewrites_transcript_named(tagged_meeting_dir, tmp_meetings_dir, monkeypatch):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(json.dumps(_fragmented_meeting().to_dict()))
    # keep the export step from doing real work / needing deps
    import backfill_segment_merge as bf
    monkeypatch.setattr(bf, "live_published_slugs", lambda: None, raising=False)

    changed = backfill(dry_run=False)
    assert changed == 1
    data = json.loads((mdir / "transcript_named.json").read_text())
    assert len(data["segments"]) == 2                    # persisted merged


def test_backfill_dry_run_writes_nothing(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    original = json.dumps(_fragmented_meeting().to_dict())
    (mdir / "transcript_named.json").write_text(original)

    assert backfill(dry_run=True) == 1                   # reports it would change
    assert (mdir / "transcript_named.json").read_text() == original  # untouched


def test_backfill_skips_already_merged(tagged_meeting_dir, tmp_meetings_dir):
    m = _fragmented_meeting()
    remerge_meeting(m)  # already merged
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(json.dumps(m.to_dict()))
    assert backfill(dry_run=False) == 0                  # nothing to do


def _already_merged_with_stale_sections():
    """The state the three drifted meetings were actually in: segments already
    merged, so merge_adjacent_segments is a no-op, but the summary boundaries
    still index into the pre-merge numbering."""
    m = _fragmented_meeting()
    remerge_meeting(m)
    assert len(m.segments) == 2                          # merge is now a no-op
    m.summary.sections[0].start_segment = 0
    m.summary.sections[0].end_segment = 139              # pre-merge overrun
    return m


def test_backfill_persists_reindex_when_segment_count_is_unchanged(
        tagged_meeting_dir, tmp_meetings_dir):
    """The hole that missed the three meetings: `remerge_meeting` reindexed in
    memory and `if after == before: continue` threw the result away."""
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(
        json.dumps(_already_merged_with_stale_sections().to_dict()))

    assert backfill(dry_run=False) == 1
    sec = json.loads((mdir / "transcript_named.json").read_text())["summary"]["sections"][0]
    assert (sec["start_segment"], sec["end_segment"]) == (0, 0)  # section covers the Bass block


def test_backfill_dry_run_does_not_persist_reindex(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    original = json.dumps(_already_merged_with_stale_sections().to_dict())
    (mdir / "transcript_named.json").write_text(original)

    assert backfill(dry_run=True) == 1
    assert (mdir / "transcript_named.json").read_text() == original


def test_backfill_resyncs_standalone_summary_json(tagged_meeting_dir, tmp_meetings_dir):
    """summary.json is the resume-path checkpoint; it must not keep boundaries
    the authoritative embedded copy no longer has."""
    m = _already_merged_with_stale_sections()
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(json.dumps(m.to_dict()))
    stale = m.summary.to_dict()
    stale["sections"][0]["extra_key"] = "preserve me"
    (mdir / "summary.json").write_text(json.dumps(stale))

    backfill(dry_run=False)
    sec = json.loads((mdir / "summary.json").read_text())["sections"][0]
    assert (sec["start_segment"], sec["end_segment"]) == (0, 0)
    assert sec["extra_key"] == "preserve me"             # untouched apart from boundaries


def test_backfill_without_summary_json_does_not_raise(tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(
        json.dumps(_already_merged_with_stale_sections().to_dict()))
    assert not (mdir / "summary.json").exists()
    assert backfill(dry_run=False) == 1


def test_backfill_leaves_valid_boundaries_alone(tagged_meeting_dir, tmp_meetings_dir):
    """Boundaries that still index into the current segments are authoritative —
    the summariser chose them against these very segments. Re-deriving them from
    times can only lose information, so a valid summary is never rewritten even
    where the time-derived answer would differ."""
    m = _fragmented_meeting()
    remerge_meeting(m)                                   # merge is now a no-op
    sec = m.summary.sections[0]
    sec.start_segment, sec.end_segment = 0, 1            # valid ids; times say (0, 0)
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(json.dumps(m.to_dict()))

    assert backfill(dry_run=False) == 0
    kept = json.loads((mdir / "transcript_named.json").read_text())["summary"]["sections"][0]
    assert (kept["start_segment"], kept["end_segment"]) == (0, 1)


def test_merge_would_change_does_not_disturb_the_meeting():
    """merge_adjacent_segments renumbers the segment_ids of the objects it is
    given, so probing must copy the objects, not just the list. Probing in place
    renumbers the live segments and makes a valid summary look stale."""
    m = _fragmented_meeting()
    ids_before = [s.segment_id for s in m.segments]
    ends_before = [s.end_time for s in m.segments]

    assert merge_would_change(m) is True
    assert [s.segment_id for s in m.segments] == ids_before
    assert [s.end_time for s in m.segments] == ends_before
    assert len(m.segments) == 4
    assert not sections_are_stale(m)      # still valid, because nothing moved


def test_merge_would_change_is_false_once_merged():
    m = _fragmented_meeting()
    remerge_meeting(m)
    assert merge_would_change(m) is False


def test_sections_only_repairs_stale_boundaries_without_merging(
        tagged_meeting_dir, tmp_meetings_dir):
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    (mdir / "transcript_named.json").write_text(
        json.dumps(_already_merged_with_stale_sections().to_dict()))

    assert backfill(dry_run=False, sections_only=True) == 1
    data = json.loads((mdir / "transcript_named.json").read_text())
    assert len(data["segments"]) == 2                    # untouched
    sec = data["summary"]["sections"][0]
    assert (sec["start_segment"], sec["end_segment"]) == (0, 0)


def test_sections_only_defers_meetings_that_still_need_merging(
        tagged_meeting_dir, tmp_meetings_dir, capsys):
    """Reindexing a meeting whose segments are about to be renumbered would just
    have to be redone, so --sections-only leaves it for the full run."""
    m = _fragmented_meeting()                            # 4 fragments, not merged
    m.summary.sections[0].end_segment = 139              # and stale boundaries
    mdir = tagged_meeting_dir("x", meeting_id="2026-05-15-interview", completed_stage=7)
    original = json.dumps(m.to_dict())
    (mdir / "transcript_named.json").write_text(original)

    assert backfill(dry_run=False, sections_only=True) == 0
    assert (mdir / "transcript_named.json").read_text() == original
    assert "needs the segment merge" in capsys.readouterr().out


# --- republish notice --------------------------------------------------------
# The live check is best-effort: live_published_slugs() swallows any DB failure
# and returns None. That "unknown" case used to print NOTHING, and silence there
# reads as "nothing to re-publish" — which is how a backlog of 13 meetings, ALL
# of them live, stayed stale on the public site.

def test_republish_notice_names_the_live_meetings():
    out = republish_notice(["a", "b", "c"], {"b", "c", "z"})
    listed = [l.strip(" -") for l in out.splitlines() if l.startswith("    - ")]
    assert listed == ["b", "c"]        # only the changed AND live ones
    assert "z" not in listed           # live but not changed here


def test_republish_notice_says_so_when_none_are_live():
    out = republish_notice(["a", "b"], set())
    assert "none" in out.lower()
    assert "re-publish" in out.lower()


def test_republish_notice_warns_when_live_status_is_unknown():
    """The bug this closes: an unreachable DB must not render as reassurance."""
    out = republish_notice(["a", "b"], None)
    assert out.strip(), "an unknown live status must still say something"
    low = out.lower()
    assert "could not determine" in low or "unknown" in low
    assert "none of the changed meetings are live" not in low   # never claim safety
    assert ".env.local" in out    # the actual cause, and the fix


def test_republish_notice_unknown_is_visually_distinct_from_all_clear():
    assert republish_notice(["a"], None) != republish_notice(["a"], set())
