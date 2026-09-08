"""Orphan-speaker audit: DB speaker rows whose label is gone from the local transcript.

The failure these protect against is not the inflated speaker_count — it is
memo_reconcile.match_speaker resolving a memo member's last name to 2 rows
instead of 1 and silently skipping that member's vote record (the Zulich case,
PR #137). So the surname-collision half of the audit matters more than the count.
"""
from __future__ import annotations

from src.speaker_orphans import (
    DbSpeaker,
    audit_meeting,
    audit_query,
    keep_labels,
    orphan_details,
    rows_by_meeting,
    speaker_status_by_label,
    stale_publish_warnings,
)


def _named(**by_label):
    """A minimal transcript_named.json dict with the given {label: name}."""
    return {
        "speakers": {
            label: {"speaker_label": label, "speaker_name": name}
            for label, name in by_label.items()
        }
    }


# ---------------------------------------------------------------- keep_labels

def test_keep_labels_reads_the_speaker_label_field_publish_upserts_on():
    # publish._upsert_speakers keys on mapping.speaker_label, not the dict key,
    # so the audit's keep-set must too or it invents phantom orphans.
    data = {"speakers": {"ignored-key": {"speaker_label": "SPEAKER_02"}}}
    assert keep_labels(data) == {"SPEAKER_02"}


def test_keep_labels_falls_back_to_the_dict_key_when_the_field_is_absent():
    data = {"speakers": {"SPEAKER_00": {"speaker_name": "Isak Nti Asare"}}}
    assert keep_labels(data) == {"SPEAKER_00"}


def test_keep_labels_is_empty_for_a_transcript_with_no_speakers():
    assert keep_labels({"speakers": {}}) == set()


# --------------------------------------------------------- status_by_label

def test_speaker_status_by_label_carries_the_transcripts_own_status():
    data = {
        "speakers": {
            "S0": {"speaker_label": "S0", "speaker_name": "Isak Nti Asare"},
            "S1": {"speaker_label": "S1", "speaker_name": "Non-speaker",
                   "speaker_status": "non_speaker"},
        }
    }
    assert speaker_status_by_label(data) == {"S0": None, "S1": "non_speaker"}


# -------------------------------------------------------------- orphan set

def test_a_db_label_missing_from_the_transcript_is_an_orphan():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
            DbSpeaker(label="S7", display_name="Ron Smith")]
    audit = audit_meeting("m", rows, _named(S0="Isak Nti Asare"))
    assert audit.judgeable is True
    assert [o.label for o in audit.orphans] == ["S7"]


def test_a_db_label_still_in_the_transcript_is_not_an_orphan():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare")]
    audit = audit_meeting("m", rows, _named(S0="Isak Nti Asare"))
    assert audit.orphans == []


def test_a_transcript_label_with_no_db_row_is_not_an_orphan():
    # Unpublished-yet, not stale. Nothing to delete.
    audit = audit_meeting("m", [], _named(S0="Isak Nti Asare"))
    assert audit.orphans == []
    assert audit.judgeable is True


# ---------------------------------------------------- cannot-judge cases

def test_a_meeting_with_no_local_transcript_cannot_be_judged():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare")]
    audit = audit_meeting("m", rows, None)
    assert audit.judgeable is False
    assert audit.orphans == []
    assert "no local transcript" in audit.reason


def test_a_transcript_with_zero_speaker_mappings_cannot_be_judged():
    # Mirrors publish._delete_vanished_speakers: an empty keep-set is a
    # malformed artifact, never an instruction to call every row stale.
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare")]
    audit = audit_meeting("m", rows, {"speakers": {}})
    assert audit.judgeable is False
    assert audit.orphans == []
    assert "no speaker mappings" in audit.reason


# ------------------------------------------------- surname collision risk

def test_an_orphan_sharing_a_surname_with_a_live_speaker_is_flagged():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
            DbSpeaker(label="S9", display_name="Council President Asare")]
    audit = audit_meeting("m", rows, _named(S0="Isak Nti Asare"))
    assert [o.label for o in audit.orphans] == ["S9"]
    (risk,) = audit.surname_risks
    assert risk.surname == "asare"
    assert risk.labels == ["S0", "S9"]
    assert risk.orphan_labels == ["S9"]


def test_an_orphan_whose_display_name_duplicates_a_live_speaker_is_flagged():
    # publish's duplicate-name gate only reads the local transcript, so an
    # identically-named orphan row passes it and still makes memo matching
    # ambiguous. This is the case that gate cannot see.
    rows = [DbSpeaker(label="S0", display_name="Steve Volan"),
            DbSpeaker(label="S9", display_name="Steve Volan")]
    audit = audit_meeting("m", rows, _named(S0="Steve Volan"))
    (risk,) = audit.surname_risks
    assert risk.surname == "volan"
    assert risk.orphan_labels == ["S9"]


def test_an_orphan_with_a_unique_surname_is_reported_without_a_surname_risk():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
            DbSpeaker(label="S9", display_name="Ron Smith")]
    audit = audit_meeting("m", rows, _named(S0="Isak Nti Asare"))
    assert [o.label for o in audit.orphans] == ["S9"]
    assert audit.surname_risks == []


def test_a_surname_shared_by_two_live_speakers_is_not_an_orphan_risk():
    # Two sitting members can share a surname; that is review's warning to
    # resolve, not evidence of a stale row. Only groups containing an orphan.
    rows = [DbSpeaker(label="S0", display_name="Dave Rollo"),
            DbSpeaker(label="S1", display_name="Kate Rollo")]
    audit = audit_meeting("m", rows, _named(S0="Dave Rollo", S1="Kate Rollo"))
    assert audit.orphans == []
    assert audit.surname_risks == []


def test_a_placeholder_last_word_is_not_a_memo_matchable_surname():
    # No clerk memo can name "(Video)" — grouping on it is noise, and noisy
    # warnings get ignored. Same _SURNAME_TOKEN rule as review.
    rows = [DbSpeaker(label="S0", display_name="Ron Smith (Video)"),
            DbSpeaker(label="S9", display_name="Kate Jones (Video)")]
    audit = audit_meeting("m", rows, _named(S0="Ron Smith (Video)"))
    assert [o.label for o in audit.orphans] == ["S9"]
    assert audit.surname_risks == []


def test_an_unidentified_orphan_handle_is_not_a_surname_risk():
    # An orphan carries no transcript mapping, so its status is inferred from
    # the local_slug overload: 'unidentified-<meeting>-<label>'. Placeholders
    # named "Unidentified Speaker" would otherwise collide on "speaker".
    rows = [DbSpeaker(label="S0", display_name="Unidentified Speaker",
                      local_slug="unidentified-m-s0"),
            DbSpeaker(label="S9", display_name="Unidentified Speaker",
                      local_slug="unidentified-m-s9")]
    audit = audit_meeting(
        "m", rows,
        {"speakers": {"S0": {"speaker_label": "S0",
                             "speaker_name": "Unidentified Speaker",
                             "speaker_status": "unidentified"}}},
    )
    assert [o.label for o in audit.orphans] == ["S9"]
    assert audit.surname_risks == []


def test_a_row_with_no_display_name_is_never_a_surname_risk():
    rows = [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
            DbSpeaker(label="S9", display_name=None)]
    audit = audit_meeting("m", rows, _named(S0="Isak Nti Asare"))
    assert [o.label for o in audit.orphans] == ["S9"]
    assert audit.surname_risks == []


# ------------------------------------------------------------- count drift

def test_audit_reports_the_speaker_count_a_republish_would_write():
    rows = [DbSpeaker(label="S0", display_name="A One"),
            DbSpeaker(label="S9", display_name="B Two")]
    audit = audit_meeting("m", rows, _named(S0="A One"), stored_speaker_count=10)
    assert audit.db_row_count == 2
    assert audit.kept_label_count == 1
    assert audit.stored_speaker_count == 10


# ------------------------------------------- what the segments tell us

def test_an_orphan_with_live_segments_means_prod_is_serving_a_vanished_label():
    # publish._replace_segments deletes EVERY segment for the meeting before
    # _delete_vanished_speakers runs, so its NOT EXISTS guard never blocks in the
    # normal flow — a republish clears this row either way. What segments on an
    # orphan DO prove is that no publish has happened since the local merge, so
    # prod is still serving those segments attributed to a label that is gone.
    rows = [DbSpeaker(label="S0", display_name="A One"),
            DbSpeaker(label="S9", display_name="B Two", segment_count=4)]
    audit = audit_meeting("m", rows, _named(S0="A One"))
    (orphan,) = audit.orphans
    assert audit.orphans_serving_segments == [orphan]


def test_an_orphan_with_no_segments_outlived_a_later_republish():
    # The jerri-green pattern: segments were rebuilt without the label, but the
    # speaker row survived because that publish predates the vanished-row sweep.
    rows = [DbSpeaker(label="S0", display_name="A One"),
            DbSpeaker(label="S9", display_name="B Two", segment_count=0)]
    audit = audit_meeting("m", rows, _named(S0="A One"))
    assert audit.orphans_serving_segments == []


# ------------------------------------------------- grouping joined DB rows

def test_rows_by_meeting_groups_speaker_rows_under_their_slug():
    joined = [
        ("a", 2, "u1", "S0", "A One", None, None, 5),
        ("a", 2, "u2", "S1", "B Two", "b-two", None, 0),
    ]
    grouped = rows_by_meeting(joined)
    assert set(grouped) == {"a"}
    counted, rows = grouped["a"]
    assert counted == 2
    assert [(r.label, r.id, r.politician_slug, r.segment_count) for r in rows] == [
        ("S0", "u1", None, 5), ("S1", "u2", "b-two", 0),
    ]


def test_rows_by_meeting_keeps_a_meeting_with_no_speaker_rows_at_all():
    # The LEFT JOIN emits one all-NULL speaker row for such a meeting; taking it
    # literally would invent a speaker labelled None.
    grouped = rows_by_meeting([("empty", 0, None, None, None, None, None, 0)])
    assert grouped == {"empty": (0, [])}


# ------------------------------------- what an unjudgeable meeting risks

def test_an_unjudgeable_meeting_with_no_speaker_rows_has_nothing_at_stake():
    # A 'scheduled' agenda row from the Bloomington poller: no transcript yet
    # because it has not happened. Nothing to be stale.
    audit = audit_meeting("upcoming", [], None)
    assert audit.judgeable is False
    assert audit.at_stake is False


def test_an_unjudgeable_meeting_with_speaker_rows_is_at_stake():
    # Published speakers but no local transcript to check them against: the
    # audit cannot clear this meeting, and cannot condemn it either.
    audit = audit_meeting("m", [DbSpeaker(label="S0", display_name="A One")], None)
    assert audit.judgeable is False
    assert audit.at_stake is True


# ------------------------------------- rendering for check_consistency.py

def test_orphan_details_are_empty_when_prod_is_clean():
    audit = audit_meeting("m", [DbSpeaker(label="S0", display_name="A One")],
                          _named(S0="A One"))
    assert orphan_details([audit]) == {}


def test_orphan_detail_is_keyed_by_slug_and_names_the_stale_row():
    # The value merges into check_consistency's existing per-meeting MISMATCH
    # line, so it must not repeat the slug.
    audit = audit_meeting(
        "lwv-forum",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S9", display_name="B Two")],
        _named(S0="A One"),
    )
    detail = orphan_details([audit])["lwv-forum"]
    assert "lwv-forum" not in detail
    assert "S9" in detail
    assert "B Two" in detail


def test_orphan_detail_marks_a_memo_matching_collision():
    # The half that costs a vote record has to be visible in one glance.
    audit = audit_meeting(
        "council",
        [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
         DbSpeaker(label="S9", display_name="Council President Asare")],
        _named(S0="Isak Nti Asare"),
    )
    detail = orphan_details([audit])["council"]
    assert "AMBIGUOUS" in detail
    assert "Asare" in detail
    assert "S0" in detail   # names the surviving row it collides with


def test_orphan_detail_omits_the_ambiguous_marker_when_there_is_no_collision():
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S9", display_name="B Two")],
        _named(S0="A One"),
    )
    assert "AMBIGUOUS" not in orphan_details([audit])["m"]


def test_orphan_detail_lists_every_stale_row_for_the_meeting():
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S8", display_name="B Two"),
         DbSpeaker(label="S9", display_name="C Three")],
        _named(S0="A One"),
    )
    detail = orphan_details([audit])["m"]
    assert "S8" in detail and "S9" in detail


def test_an_unjudgeable_meeting_gets_no_orphan_detail():
    # This feeds an exit code: a missing local transcript is not evidence of
    # drift in prod.
    audit = audit_meeting("m", [DbSpeaker(label="S0", display_name="A One")], None)
    assert orphan_details([audit]) == {}


# ------------------------------------------------------- the query itself

def test_audit_query_is_unfiltered_by_default():
    assert "WHERE m.slug" not in audit_query()


def test_audit_query_can_be_narrowed_to_a_slug_list():
    # The narrowing was string surgery on AUDIT_QUERY in two callers; a silent
    # non-match would have returned every meeting's rows, or none.
    sql = audit_query(by_slug=True)
    assert "WHERE m.slug = ANY(%s)" in sql
    assert sql.index("WHERE m.slug") < sql.index("ORDER BY")


def test_audit_query_keeps_the_left_joins_that_report_empty_meetings():
    for sql in (audit_query(), audit_query(by_slug=True)):
        assert sql.count("LEFT JOIN") == 2


# ---------------------------------- warnings for the GUI review page

def test_stale_publish_warnings_are_empty_when_prod_matches():
    audit = audit_meeting("m", [DbSpeaker(label="S0", display_name="A One")],
                          _named(S0="A One"))
    assert stale_publish_warnings(audit) == []


def test_stale_publish_warning_has_the_enrollment_warning_shape():
    # The review template renders warnings generically by kind, so a new kind
    # needs no template change — but only if the dict shape matches.
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S9", display_name="B Two")],
        _named(S0="A One"),
    )
    (warn,) = stale_publish_warnings(audit)
    assert set(warn) >= {"kind", "label", "detail"}
    assert warn["kind"] == "stale_published_speaker"
    assert warn["label"] == "S9"
    assert "B Two" in warn["detail"]


def test_stale_publish_warning_says_a_republish_is_what_removes_it():
    # The whole point: the reviewer dropped a label from a LIVE meeting and has
    # no other signal that prod still holds it.
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S9", display_name="B Two")],
        _named(S0="A One"),
    )
    (warn,) = stale_publish_warnings(audit)
    assert "republish" in warn["detail"].lower()


def test_stale_publish_warning_names_the_vote_record_risk_when_ambiguous():
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="Isak Nti Asare"),
         DbSpeaker(label="S9", display_name="Council President Asare")],
        _named(S0="Isak Nti Asare"),
    )
    (warn,) = stale_publish_warnings(audit)
    assert "vote record" in warn["detail"]
    assert "Asare" in warn["detail"]


def test_stale_publish_warnings_are_one_per_orphan_label():
    audit = audit_meeting(
        "m",
        [DbSpeaker(label="S0", display_name="A One"),
         DbSpeaker(label="S8", display_name="B Two"),
         DbSpeaker(label="S9", display_name="C Three")],
        _named(S0="A One"),
    )
    assert [w["label"] for w in stale_publish_warnings(audit)] == ["S8", "S9"]


def test_an_unjudgeable_audit_produces_no_stale_publish_warning():
    audit = audit_meeting("m", [DbSpeaker(label="S0", display_name="A One")], None)
    assert stale_publish_warnings(audit) == []
