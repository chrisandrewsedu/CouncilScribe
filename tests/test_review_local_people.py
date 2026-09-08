import pytest

from src.models import SpeakerMapping
from src.review import (
    LOCAL_SLUG_PATTERN,
    LOCAL_SLUG_RE,
    assign_local_person,
    clear_local_person,
    default_local_slug,
    identity_label,
    link_speaker,
)


def test_default_local_slug_kebab_cases_the_name():
    assert default_local_slug("Susan Brackney", "SPEAKER_04") == "susan-brackney"


def test_default_local_slug_falls_back_to_the_label():
    assert default_local_slug(None, "SPEAKER_04") == "speaker-04"
    assert default_local_slug("   ", "SPEAKER_04") == "speaker-04"


def test_default_local_slug_output_is_always_valid():
    for name, label in [("Susan Brackney", "S0"), ("!!!", "S0"), ("!!!", "!!!"),
                        ("O'Brien-Smith, Jr.", "S1"), ("x" * 300, "S2")]:
        slug = default_local_slug(name, label)
        assert LOCAL_SLUG_RE.match(slug), f"({name!r}, {label!r}) produced {slug!r}"


def test_default_local_slug_is_bounded_at_one_hundred():
    assert len(default_local_slug("x" * 300, "S2")) == 100


def test_assign_local_person_sets_slug_and_role():
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Susan Brackney")}
    m = assign_local_person(mappings, "S0", "susan-brackney", "public_comment")
    assert (m.local_slug, m.local_role) == ("susan-brackney", "public_comment")


def test_assign_local_person_clears_any_essentials_identity():
    """One identity per speaker (migration 623). A local person is not a roster
    politician, so making someone local drops the essentials link rather than
    leaving publish to suppress the contradiction."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Marcy Kaptur",
                                     politician_id="uuid-mk", politician_slug="marcy-kaptur")}
    m = assign_local_person(mappings, "S0", "marcy-kaptur", "official")
    assert m.politician_id is None
    assert m.politician_slug is None


def test_assign_local_person_creates_a_mapping_for_an_unmapped_label():
    mappings = {}
    m = assign_local_person(mappings, "S7", "jane-doe", "staff")
    assert mappings["S7"] is m
    assert m.speaker_label == "S7"


def test_assign_local_person_rejects_an_invalid_slug():
    mappings = {"S0": SpeakerMapping(speaker_label="S0")}
    for bad in ["Susan Brackney", "-leading", "_leading", "", "x" * 101, "UPPER"]:
        with pytest.raises(ValueError):
            assign_local_person(mappings, "S0", bad, "staff")


def test_assign_local_person_refuses_a_slug_held_by_another_label():
    """Two diarized labels cannot be the same person."""
    mappings = {
        "S0": SpeakerMapping(speaker_label="S0", local_slug="susan-brackney"),
        "S1": SpeakerMapping(speaker_label="S1"),
    }
    with pytest.raises(ValueError, match="already used"):
        assign_local_person(mappings, "S1", "susan-brackney", "public_comment")


def test_assign_local_person_allows_reassigning_the_same_label():
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="susan-brackney",
                                     local_role="public_comment")}
    m = assign_local_person(mappings, "S0", "susan-brackney", "staff")
    assert m.local_role == "staff"


def test_clear_local_person_unsets_both_fields():
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="susan-brackney",
                                     local_role="public_comment")}
    m = clear_local_person(mappings, "S0")
    assert (m.local_slug, m.local_role) == (None, None)


def test_clear_local_person_on_unknown_label_is_a_noop():
    assert clear_local_person({}, "S9") is None


def test_clear_local_person_still_works_for_a_real_local_person():
    """The normal path (speaker_status is None) must keep working: only the
    unidentified case below is refused."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="susan-brackney",
                                     local_role="public_comment", speaker_status=None)}
    m = clear_local_person(mappings, "S0")
    assert (m.local_slug, m.local_role) == (None, None)


def test_clear_local_person_refuses_an_unidentified_handle():
    """mark_unidentified also writes local_slug — to a synthetic
    unidentified-<meeting>-<label> handle whose whole purpose is keeping two
    distinct unknown speakers from sharing one voice-profile enrollment key.
    Clearing it would collapse them onto 'unidentified_speaker'. Refuse instead
    of mutating."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0",
                                     speaker_name="Unidentified Speaker",
                                     local_slug="unidentified-2026-council-s0",
                                     speaker_status="unidentified")}
    result = clear_local_person(mappings, "S0")
    assert result is None
    m = mappings["S0"]
    assert m.local_slug == "unidentified-2026-council-s0"
    assert m.speaker_status == "unidentified"


def test_local_slug_re_fullmatch_rejects_a_trailing_newline():
    """Python's $ (used by .match) also matches just before a trailing newline,
    but Postgres's SQL ~ does not — so a raw slug like 'staff\\n' would pass
    LOCAL_SLUG_RE.match and then fail the DB CHECK at publish time.
    assign_local_person's own .strip() happens to remove a real trailing
    newline before the value ever reaches the regex, so this is exercised
    directly against the pattern object (the thing the call sites share),
    documenting why they use .fullmatch rather than .match."""
    assert LOCAL_SLUG_RE.match("staff\n")
    assert not LOCAL_SLUG_RE.fullmatch("staff\n")


def test_local_slug_pattern_is_pinned_to_its_ev_accounts_sql_twin():
    """LOCAL_SLUG_PATTERN is the hand-maintained twin of ev-accounts's
    SLUG_REGEX. A silent edit here keeps this suite green while publish starts
    aborting whole meetings on a CHECK violation, so pin the literal."""
    assert LOCAL_SLUG_PATTERN == r"^[a-z0-9][a-z0-9_-]{0,99}$", (
        "LOCAL_SLUG_PATTERN drifted from its twin: ev-accounts's SLUG_REGEX"
    )


def test_identity_label_prefers_politician_id_over_a_local_slug():
    """politician_slug is NULL for ~99.4% of essentials politicians, so a federal
    speaker carrying politician_id plus the crec bioguide stash must not read as
    a local person. Mirrors src/enroll.py:215."""
    m = SpeakerMapping(speaker_label="S0", speaker_name="Marcy Kaptur",
                       politician_id="uuid-mk", local_slug="congress-K000009")
    assert identity_label(m) == "essentials:uuid-mk"


def test_identity_label_still_prefers_a_slug_when_present():
    m = SpeakerMapping(speaker_label="S0", politician_slug="marcy-kaptur")
    assert identity_label(m) == "essentials:marcy-kaptur"


def test_identity_label_reports_a_genuine_local_person():
    m = SpeakerMapping(speaker_label="S0", local_slug="susan-brackney")
    assert identity_label(m) == "local:susan-brackney"


def test_link_speaker_clears_a_local_person():
    """One identity per speaker: an essentials link supersedes a local person, the
    mirror of assign_local_person clearing the essentials fields."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="jo-doe",
                                     local_role="staff")}
    m = link_speaker(mappings, "S0", None, "uuid-jd")
    assert m.politician_id == "uuid-jd"
    assert m.local_slug is None
    assert m.local_role is None


def test_link_speaker_by_slug_also_clears_a_local_person():
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="jo-doe",
                                     local_role="staff")}
    m = link_speaker(mappings, "S0", "jo-doe-politician", None)
    assert m.local_slug is None
    assert m.local_role is None


def test_unlinking_leaves_a_local_person_alone():
    """link_speaker(None, None) is the UNLINK path. Clearing the politician link must
    not destroy an unrelated local-person identity."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", local_slug="jo-doe",
                                     local_role="staff",
                                     politician_id="uuid-jd")}
    m = link_speaker(mappings, "S0", None, None)
    assert m.politician_id is None
    assert m.politician_slug is None
    assert m.local_slug == "jo-doe"
    assert m.local_role == "staff"


from src.review import clear_speaker_status, mark_non_speaker, mark_unidentified


class _Seg:
    """Minimal segment stand-in: clear_speaker_status only reads speaker_label
    and writes speaker_name."""

    def __init__(self, label, name=None):
        self.speaker_label = label
        self.speaker_name = name


def test_clear_speaker_status_clears_an_unidentified_mark_and_its_handle():
    """mark_unidentified writes a synthetic unidentified-<meeting>-<label> handle
    into local_slug. That handle is a voice-profile key, not a site-local person,
    so clearing the status must drop it too — otherwise the picker would show a
    private handle as a real local person."""
    mappings, segments = {}, [_Seg("S0")]
    mark_unidentified(mappings, segments, "S0", "2026-02-04-council")
    assert mappings["S0"].local_slug == "unidentified-2026-02-04-council-s0"

    m = clear_speaker_status(mappings, segments, "S0")
    assert m is mappings["S0"]
    assert m.speaker_status is None
    assert m.local_slug is None
    assert m.local_role is None


def test_clear_speaker_status_clears_a_non_speaker_mark():
    mappings, segments = {}, [_Seg("S0")]
    mark_non_speaker(mappings, segments, "S0")
    m = clear_speaker_status(mappings, segments, "S0")
    assert m.speaker_status is None
    assert m.local_slug is None


def test_clear_speaker_status_clears_the_placeholder_name_on_mapping_and_segments():
    """'Unidentified Speaker' / 'Non-speaker' label the STATUS, not a person, so
    they must not outlive the status they described."""
    mappings, segments = {}, [_Seg("S0"), _Seg("S1"), _Seg("S0")]
    mark_non_speaker(mappings, segments, "S0", "Pledge of Allegiance")
    assert [s.speaker_name for s in segments] == ["Pledge of Allegiance", None,
                                                  "Pledge of Allegiance"]

    clear_speaker_status(mappings, segments, "S0")
    assert mappings["S0"].speaker_name is None
    assert [s.speaker_name for s in segments] == [None, None, None]


def test_clear_speaker_status_resets_confidence_and_method():
    """mark_* asserted human certainty about the MARK. Once the mark is gone that
    certainty is gone with it, so the speaker returns to Needs attention."""
    mappings, segments = {}, [_Seg("S0")]
    mark_unidentified(mappings, segments, "S0", "2026-02-04-council")
    assert (mappings["S0"].confidence, mappings["S0"].id_method) == (1.0, "human_review")

    m = clear_speaker_status(mappings, segments, "S0")
    assert m.confidence == 0.0
    assert m.id_method is None


def test_clear_speaker_status_on_an_unknown_label_is_a_noop():
    assert clear_speaker_status({}, [], "S9") is None


def test_clear_speaker_status_on_an_already_clear_mapping_is_a_noop():
    """A no-op is not success: the GUI route maps None to 404 so an Undo button
    on a speaker that was never marked cannot report that it did something."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Susan Brackney",
                                     local_slug="susan-brackney",
                                     local_role="public_comment", speaker_status=None)}
    assert clear_speaker_status(mappings, [], "S0") is None
    # A genuine local person is left completely untouched.
    assert mappings["S0"].local_slug == "susan-brackney"
    assert mappings["S0"].local_role == "public_comment"
    assert mappings["S0"].speaker_name == "Susan Brackney"


def test_clear_speaker_status_leaves_a_politician_link_alone():
    """Clearing a stale mark is not an unlink. Only the mark and its own
    placeholder fields are dropped."""
    mappings = {"S0": SpeakerMapping(speaker_label="S0", speaker_name="Non-speaker",
                                     politician_id="uuid-mk",
                                     politician_slug="marcy-kaptur",
                                     speaker_status="non_speaker")}
    m = clear_speaker_status(mappings, [], "S0")
    assert m.politician_id == "uuid-mk"
    assert m.politician_slug == "marcy-kaptur"
