"""Tests for detecting the INVERSE speaker-identity error: one diarized label
holding TWO different people.

``review.duplicate_named_speakers`` / ``review.ambiguous_speaker_surnames`` are
both name-based, so they catch a rename that puts one name on two labels. A
wrong review-time MERGE leaves one label with one name, so both return ``{}``
and a clean scan is false reassurance.

The detector here works from provenance instead of names: ``transcript_raw.json``
keeps the original diarized labels, so two raw labels overlapping one named
label means a merge happened. Whether that merge joined one person's split
cluster (case a, correct) or two different people (case b, an error) is then
decided acoustically, by re-embedding each side's turns from the audio.
"""

from __future__ import annotations

import numpy as np

from src.models import Segment


def _seg(seg_id, start, end, label, text="hello there"):
    return Segment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        speaker_label=label,
        text=text,
    )


# --- provenance: which raw labels ended up inside each named label -----------

def test_provenance_groups_maps_each_raw_label_to_its_named_label():
    from src.mismerge import provenance_groups

    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 10.0, 20.0, "SPEAKER_01"),
    ]
    named = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 10.0, 20.0, "SPEAKER_01"),
    ]

    groups = provenance_groups(raw, named)

    assert groups == {
        "SPEAKER_00": {"SPEAKER_00": [0]},
        "SPEAKER_01": {"SPEAKER_01": [1]},
    }


def test_provenance_groups_reports_a_merged_label_as_two_raw_groups():
    from src.mismerge import provenance_groups

    # SPEAKER_01's turns were relabelled onto SPEAKER_00 by a review merge,
    # and adjacent-merge then stitched the whole run into one named turn.
    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 10.0, 20.0, "SPEAKER_01"),
        _seg(2, 20.0, 30.0, "SPEAKER_00"),
    ]
    named = [_seg(0, 0.0, 30.0, "SPEAKER_00")]

    groups = provenance_groups(raw, named)

    assert groups == {"SPEAKER_00": {"SPEAKER_00": [0, 2], "SPEAKER_01": [1]}}


def test_provenance_groups_assigns_a_raw_turn_by_greatest_overlap():
    from src.mismerge import provenance_groups

    # A boundary-snap shifted the named boundary, so raw turn 1 overlaps both
    # named turns — 4s of SPEAKER_01 against 1s of SPEAKER_00.
    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 9.0, 14.0, "SPEAKER_01"),
    ]
    named = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 10.0, 14.0, "SPEAKER_01"),
    ]

    groups = provenance_groups(raw, named)

    assert groups["SPEAKER_01"] == {"SPEAKER_01": [1]}
    assert groups["SPEAKER_00"] == {"SPEAKER_00": [0]}


def test_provenance_groups_drops_a_raw_turn_that_no_named_turn_covers():
    from src.mismerge import provenance_groups

    # Publish drops empty-text segments and a reviewer can delete a junk
    # cluster outright, so a raw turn with no surviving named turn is normal.
    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 50.0, 52.0, "SPEAKER_09", text=""),
    ]
    named = [_seg(0, 0.0, 10.0, "SPEAKER_00")]

    assert provenance_groups(raw, named) == {"SPEAKER_00": {"SPEAKER_00": [0]}}


# --- the cheap gate: which merges actually happened -------------------------

def test_merge_candidates_finds_the_absorbed_label_and_its_host():
    from src.mismerge import merge_candidates

    raw = [
        _seg(0, 0.0, 100.0, "SPEAKER_03"),
        _seg(1, 100.0, 130.0, "SPEAKER_07"),
        _seg(2, 130.0, 200.0, "SPEAKER_03"),
    ]
    named = [_seg(0, 0.0, 200.0, "SPEAKER_03")]

    candidates = merge_candidates(raw, named)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.label == "SPEAKER_03"
    assert candidate.host_raw == "SPEAKER_03"
    assert candidate.absorbed_raw == "SPEAKER_07"
    assert candidate.absorbed_indices == [1]
    assert candidate.host_indices == [0, 2]
    assert candidate.absorbed_seconds == 30.0
    assert candidate.host_seconds == 170.0


def test_merge_candidates_is_empty_when_no_label_was_merged():
    from src.mismerge import merge_candidates

    raw = [_seg(0, 0.0, 100.0, "SPEAKER_00"), _seg(1, 100.0, 200.0, "SPEAKER_01")]
    named = [_seg(0, 0.0, 100.0, "SPEAKER_00"), _seg(1, 100.0, 200.0, "SPEAKER_01")]

    assert merge_candidates(raw, named) == []


def test_merge_candidates_ignores_a_boundary_brush():
    from src.mismerge import merge_candidates

    # SPEAKER_01 lives in its own named label; only 0.4s of one turn spills
    # across the boundary into SPEAKER_00's. That is two windows disagreeing
    # about a boundary, not a merge.
    raw = [
        _seg(0, 0.0, 100.0, "SPEAKER_00"),
        _seg(1, 99.6, 160.0, "SPEAKER_01"),
    ]
    named = [
        _seg(0, 0.0, 100.0, "SPEAKER_00"),
        _seg(1, 100.0, 160.0, "SPEAKER_01"),
    ]

    assert merge_candidates(raw, named) == []


def test_merge_candidates_reports_every_absorbed_label_separately():
    from src.mismerge import merge_candidates

    # bloomington-city-council-2026-05-06 has this shape: two raw labels
    # folded into one survivor. Each side is judged on its own voice.
    raw = [
        _seg(0, 0.0, 100.0, "SPEAKER_26"),
        _seg(1, 100.0, 130.0, "SPEAKER_24"),
        _seg(2, 130.0, 180.0, "SPEAKER_25"),
    ]
    named = [_seg(0, 0.0, 180.0, "SPEAKER_26")]

    absorbed = sorted(c.absorbed_raw for c in merge_candidates(raw, named))
    assert absorbed == ["SPEAKER_24", "SPEAKER_25"]


def test_merge_candidates_host_is_the_group_with_the_most_speech():
    from src.mismerge import merge_candidates

    # The survivor's label need not be the dominant voice: a reviewer can
    # merge a long run INTO a short one, which is how a two-segment label's
    # name came to relabel 28 of someone else's segments.
    raw = [
        _seg(0, 0.0, 4.0, "SPEAKER_07"),
        _seg(1, 4.0, 120.0, "SPEAKER_03"),
    ]
    named = [_seg(0, 0.0, 120.0, "SPEAKER_07")]

    candidate = merge_candidates(raw, named)[0]
    assert candidate.label == "SPEAKER_07"
    assert candidate.host_raw == "SPEAKER_03"
    assert candidate.absorbed_raw == "SPEAKER_07"


# --- which turns to embed ---------------------------------------------------

def test_select_turns_prefers_the_longest_turns():
    from src.mismerge import select_turns

    segments = [
        _seg(0, 0.0, 1.0, "SPEAKER_00"),
        _seg(1, 10.0, 30.0, "SPEAKER_00"),
        _seg(2, 40.0, 45.0, "SPEAKER_00"),
    ]

    assert select_turns(segments, [0, 1, 2], max_turns=2) == [1, 2]


def test_select_turns_skips_turns_too_short_to_embed():
    from src.mismerge import select_turns

    # A sub-0.5s slice is a turn-boundary fragment; wespeaker's vector for it
    # is dominated by whatever leaked in from the neighbour.
    segments = [
        _seg(0, 0.0, 0.2, "SPEAKER_00"),
        _seg(1, 1.0, 1.4, "SPEAKER_00"),
        _seg(2, 2.0, 8.0, "SPEAKER_00"),
    ]

    assert select_turns(segments, [0, 1, 2]) == [2]


def test_select_turns_stops_once_it_has_enough_audio():
    from src.mismerge import select_turns

    # A 4000s host side does not need re-embedding in full to place a centroid.
    # The budget is spent in truncated slice seconds, so a 120s budget at a 20s
    # slice cap buys six turns rather than two 60s ones.
    segments = [_seg(i, i * 100.0, i * 100.0 + 60.0, "SPEAKER_00") for i in range(40)]

    chosen = select_turns(segments, list(range(40)), max_seconds=120.0,
                          max_slice_seconds=20.0)
    assert len(chosen) == 6


def test_select_turns_returns_indices_in_time_order():
    from src.mismerge import select_turns

    segments = [
        _seg(0, 0.0, 30.0, "SPEAKER_00"),
        _seg(1, 100.0, 140.0, "SPEAKER_00"),
        _seg(2, 200.0, 210.0, "SPEAKER_00"),
    ]

    assert select_turns(segments, [0, 1, 2], max_turns=2) == [0, 1]


# --- the acoustic test ------------------------------------------------------

def test_group_similarity_is_high_for_one_voice():
    from src.mismerge import group_similarity

    host = [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0])]
    absorbed = [np.array([0.95, 0.05, 0.0])]

    assert group_similarity(host, absorbed) > 0.99


def test_group_similarity_is_low_for_two_voices():
    from src.mismerge import group_similarity

    host = [np.array([1.0, 0.0, 0.0])]
    absorbed = [np.array([0.0, 1.0, 0.0])]

    assert group_similarity(host, absorbed) == 0.0


def test_group_similarity_is_none_when_a_side_has_no_vectors():
    from src.mismerge import group_similarity

    assert group_similarity([np.array([1.0, 0.0])], []) is None
    assert group_similarity([], []) is None


def test_group_similarity_is_none_when_a_vector_is_nan():
    from src.mismerge import group_similarity

    # 8 of 1035 corpus labels carry NaN vectors; "we could not tell" must never
    # be reported as "the voices differ".
    host = [np.array([np.nan, np.nan, np.nan])]
    absorbed = [np.array([0.0, 1.0, 0.0])]

    assert group_similarity(host, absorbed) is None


def test_group_similarity_ignores_a_single_unusable_vector():
    from src.mismerge import group_similarity

    # One NaN turn among usable ones is dropped, not fatal — the same rule
    # global_identity.decode_turn_vectors applies to turn embeddings.
    host = [np.array([np.nan, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    absorbed = [np.array([1.0, 0.0, 0.0])]

    assert group_similarity(host, absorbed) == 1.0


def test_group_similarity_ignores_a_zero_norm_vector():
    from src.mismerge import group_similarity

    host = [np.zeros(3), np.array([1.0, 0.0, 0.0])]
    absorbed = [np.array([1.0, 0.0, 0.0])]

    assert group_similarity(host, absorbed) == 1.0


# --- the verdict ------------------------------------------------------------

def _voice(vector):
    """embed_fn that returns one fixed voice for every slice."""
    return lambda start, end: np.asarray(vector, dtype=float)


def _two_voices(boundary, before, after):
    """embed_fn returning `before` for slices under `boundary`, else `after`."""
    def embed_fn(start, end):
        return np.asarray(before if start < boundary else after, dtype=float)
    return embed_fn


def _merged_pair():
    from src.mismerge import merge_candidates

    raw = [
        _seg(0, 0.0, 60.0, "SPEAKER_03"),
        _seg(1, 100.0, 130.0, "SPEAKER_07"),
        _seg(2, 200.0, 260.0, "SPEAKER_03"),
    ]
    named = [_seg(0, 0.0, 260.0, "SPEAKER_03")]
    return raw, merge_candidates(raw, named)[0]


def test_assess_candidate_reports_mismatch_for_two_different_voices():
    from src.mismerge import assess_candidate

    raw, candidate = _merged_pair()
    # host (SPEAKER_03) speaks before 100s and after 200s; the absorbed turn
    # at 100-130s carries an orthogonal voice.
    embed_fn = lambda start, end: np.array(
        [0.0, 1.0, 0.0] if 100.0 <= start < 200.0 else [1.0, 0.0, 0.0]
    )

    assessment = assess_candidate(candidate, raw, embed_fn)

    assert assessment.verdict == "mismatch"
    assert assessment.similarity == 0.0
    assert assessment.host_turns_embedded == 2
    assert assessment.absorbed_turns_embedded == 1


def test_assess_candidate_reports_match_for_one_voice():
    from src.mismerge import assess_candidate

    raw, candidate = _merged_pair()

    assessment = assess_candidate(candidate, raw, _voice([1.0, 0.0, 0.0]))

    assert assessment.verdict == "match"
    assert assessment.similarity > 0.99


def test_assess_candidate_reports_unknown_when_a_side_is_unembeddable():
    from src.mismerge import assess_candidate

    raw, candidate = _merged_pair()

    assessment = assess_candidate(candidate, raw, lambda start, end: None)

    assert assessment.verdict == "unknown"
    assert assessment.similarity is None


def test_assess_candidate_does_not_embed_a_side_with_too_little_audio():
    from src.mismerge import assess_candidate, merge_candidates

    raw = [
        _seg(0, 0.0, 60.0, "SPEAKER_03"),
        _seg(1, 100.0, 102.5, "SPEAKER_07"),
        _seg(2, 200.0, 260.0, "SPEAKER_03"),
    ]
    named = [_seg(0, 0.0, 260.0, "SPEAKER_03")]
    candidate = merge_candidates(raw, named)[0]

    calls = []

    def embed_fn(start, end):
        calls.append((start, end))
        return np.array([1.0, 0.0, 0.0])

    assessment = assess_candidate(candidate, raw, embed_fn, min_side_seconds=3.0)

    assert assessment.verdict == "unknown"
    assert assessment.reason == "absorbed side has 2.5s of embeddable speech (need 3.0s)"
    assert calls == []


def test_assess_candidate_bands_the_uncertain_middle():
    from src.mismerge import assess_candidate

    raw, candidate = _merged_pair()
    # cos = 0.5 -> between MERGE_SIM_MISMATCH and MERGE_SIM_CONFIDENT
    embed_fn = lambda start, end: np.array(
        [0.5, 0.8660254037844386] if 100.0 <= start < 200.0 else [1.0, 0.0]
    )

    assessment = assess_candidate(candidate, raw, embed_fn)

    assert assessment.verdict == "uncertain"
    assert 0.49 < assessment.similarity < 0.51


def test_assess_candidate_uses_the_calibrated_merge_bands():
    """The bands are review's, not new ones — one calibration, one meaning."""
    from src import review
    from src.mismerge import assess_candidate

    raw, candidate = _merged_pair()
    for similarity, expected in (
        (review.MERGE_SIM_MISMATCH, "mismatch"),
        (review.MERGE_SIM_CONFIDENT, "match"),
    ):
        angle = np.arccos(similarity)
        embed_fn = lambda start, end, angle=angle: np.array(
            [np.cos(angle), np.sin(angle)] if 100.0 <= start < 200.0 else [1.0, 0.0]
        )
        assert assess_candidate(candidate, raw, embed_fn).verdict == expected


# --- what the merge actually did, for triage --------------------------------

def test_straddling_named_turns_counts_turns_covering_both_voices():
    from src.mismerge import merge_candidates, straddling_named_turns

    # One named turn spans the questioner AND the answer — the shape found in
    # 2026-07-09-debate-mi-governor-gop-primary, where a 99s named segment
    # swallowed Perry Johnson's answer under an unidentified questioner's label.
    raw = [
        _seg(0, 0.0, 20.0, "SPEAKER_03"),
        _seg(1, 20.0, 100.0, "SPEAKER_04"),
    ]
    named = [_seg(0, 0.0, 100.0, "SPEAKER_03")]
    candidate = merge_candidates(raw, named)[0]

    assert straddling_named_turns(candidate, raw, named) == 1


def test_straddling_named_turns_is_zero_for_a_label_level_merge():
    from src.mismerge import merge_candidates, straddling_named_turns

    # A review merge relabels whole turns, so each named turn still holds one
    # voice; the fix is a re-split, not a re-segmentation.
    raw = [
        _seg(0, 0.0, 60.0, "SPEAKER_03"),
        _seg(1, 100.0, 130.0, "SPEAKER_07"),
    ]
    named = [
        _seg(0, 0.0, 60.0, "SPEAKER_03"),
        _seg(1, 100.0, 130.0, "SPEAKER_03"),
    ]
    candidate = merge_candidates(raw, named)[0]

    assert straddling_named_turns(candidate, raw, named) == 0


# --- reading the corpus -----------------------------------------------------

def _write_meeting(root, meeting_id, raw, named, *, raw_form="list"):
    import json

    directory = root / meeting_id
    directory.mkdir(parents=True)
    payload = [s.to_dict() for s in raw]
    (directory / "transcript_raw.json").write_text(json.dumps(
        payload if raw_form == "list" else {"segments": payload}
    ))
    (directory / "transcript_named.json").write_text(json.dumps(
        {"meeting_id": meeting_id, "segments": [s.to_dict() for s in named]}
    ))
    return directory


def test_load_meeting_segments_reads_a_bare_list_raw_transcript(tmp_path):
    from src.mismerge import load_meeting_segments

    raw = [_seg(0, 0.0, 10.0, "SPEAKER_00")]
    directory = _write_meeting(tmp_path, "m1", raw, raw)

    loaded_raw, loaded_named = load_meeting_segments(directory)

    assert [s.speaker_label for s in loaded_raw] == ["SPEAKER_00"]
    assert [s.speaker_label for s in loaded_named] == ["SPEAKER_00"]


def test_load_meeting_segments_reads_a_wrapped_raw_transcript(tmp_path):
    from src.mismerge import load_meeting_segments

    # transcript_raw.json is a bare list corpus-wide today, but the named file
    # is a dict and the two writers have drifted before.
    raw = [_seg(0, 0.0, 10.0, "SPEAKER_00")]
    directory = _write_meeting(tmp_path, "m1", raw, raw, raw_form="dict")

    loaded_raw, _ = load_meeting_segments(directory)
    assert len(loaded_raw) == 1


def test_scan_corpus_reports_a_meeting_it_cannot_check(tmp_path):
    from src.mismerge import scan_corpus_candidates

    # 6 of 178 corpus meetings have no transcript_raw.json. Provenance cannot
    # see them, and silently skipping would make the scan's "0 findings" a lie.
    raw = [
        _seg(0, 0.0, 60.0, "SPEAKER_03"),
        _seg(1, 100.0, 130.0, "SPEAKER_07"),
    ]
    named = [_seg(0, 0.0, 130.0, "SPEAKER_03")]
    _write_meeting(tmp_path, "checkable", raw, named)
    (tmp_path / "no-raw").mkdir()
    (tmp_path / "no-raw" / "transcript_named.json").write_text('{"segments": []}')

    candidates, unchecked = scan_corpus_candidates(tmp_path)

    assert [meeting for meeting, _ in candidates] == ["checkable"]
    assert unchecked == [("no-raw", "no transcript_raw.json")]


def test_scan_corpus_skips_a_directory_with_no_transcript_at_all(tmp_path):
    from src.mismerge import scan_corpus_candidates

    (tmp_path / "not-a-meeting").mkdir()

    assert scan_corpus_candidates(tmp_path) == ([], [])


# --- ranking the findings ---------------------------------------------------

def _assessment(verdict, similarity, host_seconds, absorbed_seconds):
    from src.mismerge import Assessment

    return Assessment(
        label="SPEAKER_00", host_raw="SPEAKER_00", absorbed_raw="SPEAKER_01",
        similarity=similarity, verdict=verdict,
        host_turns_embedded=1, absorbed_turns_embedded=1,
        host_seconds_embedded=host_seconds, absorbed_seconds_embedded=absorbed_seconds,
        host_seconds=host_seconds, absorbed_seconds=absorbed_seconds,
    )


def test_misattributed_floor_is_the_smaller_side():
    from src.mismerge import misattributed_floor_seconds

    # Whichever side the surviving name does NOT belong to is wrong, so the
    # minority side's speech is the least that must be misattributed.
    assert misattributed_floor_seconds(_assessment("mismatch", 0.1, 900.0, 30.0)) == 30.0


def test_rank_assessments_puts_mismatches_before_uncertain_and_unknown():
    from src.mismerge import rank_assessments

    ranked = rank_assessments([
        _assessment("match", 0.9, 100.0, 50.0),
        _assessment("unknown", None, 100.0, 50.0),
        _assessment("uncertain", 0.5, 100.0, 50.0),
        _assessment("mismatch", 0.2, 100.0, 50.0),
    ])

    assert [a.verdict for a in ranked] == [
        "mismatch", "uncertain", "unknown", "match",
    ]


def test_rank_assessments_orders_a_band_by_how_much_speech_is_wrong():
    from src.mismerge import rank_assessments

    ranked = rank_assessments([
        _assessment("mismatch", 0.2, 900.0, 20.0),
        _assessment("mismatch", 0.2, 900.0, 260.0),
    ])

    assert [a.absorbed_seconds for a in ranked] == [260.0, 20.0]


# --- provenance-free bimodality: a label no merge touched -------------------

def test_bimodal_split_separates_two_voices():
    from src.mismerge import bimodal_split

    vectors = [
        np.array([1.0, 0.0, 0.0]), np.array([0.98, 0.02, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.02, 0.98, 0.0]),
    ]

    split = bimodal_split(vectors)

    assert sorted([sorted(split.side_a), sorted(split.side_b)]) == [[0, 1], [2, 3]]
    assert split.similarity < 0.1


def test_bimodal_split_reports_one_voice_as_similar():
    from src.mismerge import bimodal_split

    # A single speaker's turn vectors scatter, so a 2-means split ALWAYS
    # exists; what distinguishes two people is that the split is far apart.
    rng = np.random.default_rng(0)
    centre = np.array([1.0, 0.0, 0.0])
    vectors = [centre + 0.05 * rng.standard_normal(3) for _ in range(12)]

    assert bimodal_split(vectors).similarity > 0.9


def test_bimodal_split_needs_enough_turns_on_both_sides():
    from src.mismerge import bimodal_split

    # One outlier turn is a boundary fragment or a cough, not a second person.
    vectors = [np.array([1.0, 0.0]), np.array([0.99, 0.01]), np.array([0.0, 1.0])]

    assert bimodal_split(vectors, min_side_turns=2) is None


def test_bimodal_split_is_none_without_enough_usable_vectors():
    from src.mismerge import bimodal_split

    assert bimodal_split([np.array([1.0, 0.0])]) is None
    assert bimodal_split([]) is None


def test_bimodal_split_drops_unusable_vectors_before_splitting():
    from src.mismerge import bimodal_split

    vectors = [
        np.array([1.0, 0.0]), np.array([0.99, 0.01]),
        np.array([np.nan, np.nan]), np.zeros(2),
        np.array([0.0, 1.0]), np.array([0.01, 0.99]),
    ]

    split = bimodal_split(vectors)

    # Indices are into the ORIGINAL list, so a caller can name the turns.
    assert sorted(split.side_a + split.side_b) == [0, 1, 4, 5]


# --- cheap gates for the provenance-free scan ------------------------------

def test_label_turn_indices_groups_a_labels_turns():
    from src.mismerge import label_turn_indices

    segments = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 10.0, 20.0, "SPEAKER_01"),
        _seg(2, 20.0, 30.0, "SPEAKER_00"),
    ]

    assert label_turn_indices(segments) == {
        "SPEAKER_00": [0, 2],
        "SPEAKER_01": [1],
    }


def test_label_source_turns_uses_raw_boundaries_for_a_named_label():
    from src.mismerge import label_source_turns

    # The provenance-free scan must embed RAW turns, not named ones: a named
    # turn can straddle two voices (the 99s swallow), and a blended vector
    # hides the very bimodality the scan is looking for.
    raw = [
        _seg(0, 0.0, 20.0, "SPEAKER_03"),
        _seg(1, 20.0, 100.0, "SPEAKER_04"),
        _seg(2, 200.0, 260.0, "SPEAKER_05"),
    ]
    named = [
        _seg(0, 0.0, 100.0, "SPEAKER_03"),
        _seg(1, 200.0, 260.0, "SPEAKER_05"),
    ]

    assert label_source_turns(raw, named) == {
        "SPEAKER_03": [0, 1],
        "SPEAKER_05": [2],
    }


# --- slice truncation: more turns per second of audio -----------------------

def test_slice_bounds_truncates_a_long_turn():
    from src.mismerge import slice_bounds

    # wespeaker needs seconds, not minutes. Truncating a 200s monologue to 20s
    # costs nothing in vector quality and buys 9 more turns in the same budget.
    segment = _seg(0, 100.0, 300.0, "SPEAKER_00")

    assert slice_bounds(segment, max_slice_seconds=20.0) == (100.0, 120.0)


def test_slice_bounds_leaves_a_short_turn_alone():
    from src.mismerge import slice_bounds

    segment = _seg(0, 10.0, 15.0, "SPEAKER_00")

    assert slice_bounds(segment, max_slice_seconds=20.0) == (10.0, 15.0)


def test_select_turns_budgets_on_the_truncated_length():
    from src.mismerge import select_turns

    # Four 100s turns against a 180s budget: untruncated that is 2 turns, but
    # a 20s cap per slice buys all four and a centroid over four turns.
    segments = [_seg(i, i * 200.0, i * 200.0 + 100.0, "SPEAKER_00") for i in range(4)]

    chosen = select_turns(segments, [0, 1, 2, 3], max_seconds=180.0,
                          max_slice_seconds=20.0)

    assert chosen == [0, 1, 2, 3]


def test_assess_candidate_embeds_truncated_slices():
    from src.mismerge import assess_candidate, merge_candidates

    raw = [
        _seg(0, 0.0, 300.0, "SPEAKER_03"),
        _seg(1, 400.0, 430.0, "SPEAKER_07"),
    ]
    named = [_seg(0, 0.0, 430.0, "SPEAKER_03")]
    candidate = merge_candidates(raw, named)[0]

    calls = []

    def embed_fn(start, end):
        calls.append((start, end))
        return np.array([1.0, 0.0])

    assess_candidate(candidate, raw, embed_fn, max_slice_seconds=20.0)

    assert calls == [(0.0, 20.0), (400.0, 420.0)]


def test_provenance_groups_drops_a_zero_length_raw_turn():
    from src.mismerge import provenance_groups

    # Zero-length turns are common in the corpus (empty-text artefacts of word
    # assignment). They overlap nothing, carry no audio, and must not be
    # credited to a label as if they were speech.
    raw = [
        _seg(0, 0.0, 10.0, "SPEAKER_00"),
        _seg(1, 5.0, 5.0, "SPEAKER_07", text=""),
    ]
    named = [_seg(0, 0.0, 10.0, "SPEAKER_00")]

    assert provenance_groups(raw, named) == {"SPEAKER_00": {"SPEAKER_00": [0]}}
