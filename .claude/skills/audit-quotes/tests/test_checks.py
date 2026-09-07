from scripts.checks import (
    check_note_quality, check_deid_present, check_trailing_ellipsis,
    check_partisan_tell_in_blind, check_invalid_source,
    check_unquotable_source, check_scorecard_source, check_stance_label,
    topic_live_count, topic_min_candidates, STANCE_LABEL_MAX_WORDS,
)

def row(**kw):
    base = dict(id="q1", candidate="A", topic_key="housing", race_id="r1",
                readrank_selected=True, quote_text="We must build more homes.",
                deidentified_text="We must build more homes.", editor_note="Verbatim, no edits.",
                source_name="www.youtube.com", source_url="https://youtu.be/x?t=1s")
    base.update(kw); return base

def test_note_missing_is_high_guided():
    f = check_note_quality(row(editor_note=None))
    assert f and f.check_id == "note-missing" and f.severity == "high" and f.fix_class == "guided"

def test_note_with_section_ref_flagged():
    f = check_note_quality(row(editor_note="Matches stance (§4.3); tier-1 debate."))
    assert f and f.check_id == "note-section-ref"

def test_note_too_long_flagged():
    long = "One sentence here. Two here. Three here. Four here."
    f = check_note_quality(row(editor_note=long))
    assert f and f.check_id == "note-too-long"

def test_note_of_three_sentences_allowed():
    """Two sentences stay the house preference, but an edited quote often needs a third to say
    what was cut and why — a note that earns it is not a defect."""
    three = "Trimmed from one longer sentence. The ellipsis drops a drafting aside. Verbatim otherwise."
    assert check_note_quality(row(editor_note=three)) is None

def test_good_note_passes():
    assert check_note_quality(row(editor_note="Clear housing supply position. Verbatim, no edits.")) is None

def test_deid_null_flagged():
    f = check_deid_present(row(deidentified_text=None))
    assert f and f.check_id == "deid-missing" and f.fix_class == "guided"

def test_trailing_ellipsis_flagged():
    f = check_trailing_ellipsis(row(quote_text="We must act …"))
    assert f and f.check_id == "trailing-ellipsis"

def test_partisan_tell_in_blind_flagged():
    f = check_partisan_tell_in_blind(row(deidentified_text="These Democrat policies failed."))
    assert f and f.check_id == "partisan-tell" and f.fix_class == "guided"

def test_partisan_tell_genuine_side_tells_flagged():
    for blind in ["I am a Reagan Republican.", "The Endangerment Finding is a Democrat tool.",
                  "Best not represented by a MAGA acolyte.", "GOP can't beat the ACA.",
                  "These policies betray my party."]:
        f = check_partisan_tell_in_blind(row(deidentified_text=blind))
        assert f and f.check_id == "partisan-tell", f"should flag: {blind!r}"

def test_partisan_tell_small_d_democratic_not_flagged():
    # small-d "democratic" (democracy, not the party) must not false-match
    for blind in ["Stand firmly with democratic allies by strengthening NATO.",
                  "the right to participate in the democratic process",
                  "a threat to democratic governance"]:
        assert check_partisan_tell_in_blind(row(deidentified_text=blind)) is None, blind

def test_partisan_tell_symmetric_bipartisan_framing_not_flagged():
    # naming BOTH parties reveals no side
    for blind in ["These are not Democratic or Republican priorities.",
                  "Voters — Democratic, Republican and independent — must reject this."]:
        assert check_partisan_tell_in_blind(row(deidentified_text=blind)) is None, blind

# `source-tier-4` (campaign-site URL => low tier) was deleted: provenance is directness of answer,
# not medium, and no URL pattern can see directness. It now lives in the judgment pass as
# `source-not-an-answer` (CHECKS.md §3). Behavioural coverage of its absence — and of the three
# bad-source checks surviving it — is in tests/test_audit_checks.py at the repo root, which is the
# suite `pytest tests/` actually collects.

def test_invalid_source_ontheissues_flagged():
    f = check_invalid_source(row(source_url="https://www.ontheissues.org/John_James.htm", source_name="www.ontheissues.org"))
    assert f is not None and f.check_id == "invalid-source" and f.severity == "high"

def test_invalid_source_wikipedia_flagged():
    f = check_invalid_source(row(source_url="https://en.wikipedia.org/wiki/Jocelyn_Benson", source_name="en.wikipedia.org"))
    assert f is not None and f.check_id == "invalid-source"

def test_invalid_source_youtube_not_flagged():
    assert check_invalid_source(row(source_url="https://youtu.be/x?t=1s")) is None

def test_invalid_source_ballotpedia_not_flagged():
    # ballotpedia footnotes originals and publishes Candidate Connection answers itself —
    # it is neither an aggregator to re-attribute from nor a quiz site to delete
    assert check_invalid_source(row(source_url="https://ballotpedia.org/Antony_Barran",
                                    source_name="ballotpedia.org")) is None
    assert check_unquotable_source(row(source_url="https://ballotpedia.org/Antony_Barran",
                                       source_name="ballotpedia.org")) is None

def test_unquotable_source_isidewith_flagged():
    f = check_unquotable_source(row(source_url="https://www.isidewith.com/candidates/james-sceniak",
                                    source_name="www.isidewith.com"))
    assert f is not None and f.check_id == "unquotable-source"
    assert f.severity == "high" and f.fix_class == "decision-required"

def test_unquotable_source_is_delete_not_reattribute():
    # the two classes must not collapse: a quiz site is NOT flagged as a re-attributable aggregator
    r = row(source_url="https://www.isidewith.com/candidates/james-sceniak")
    assert check_invalid_source(r) is None
    assert "delete" in check_unquotable_source(r).suggested_fix.lower()

def test_unquotable_source_youtube_not_flagged():
    assert check_unquotable_source(row(source_url="https://youtu.be/x?t=1s")) is None

def test_run_mechanical_flags_isidewith_row():
    from scripts.checks import run_mechanical
    rows = [row(id="a", candidate="A", source_url="https://www.isidewith.com/candidates/x"),
            row(id="b", candidate="B")]
    assert "unquotable-source" in {f.check_id for f in run_mechanical(rows)}

def test_topic_same_candidate_two_live_flagged_legacy():
    g = {"race_id": "r1", "topic_key": "housing",
         "quotes": [row(id="a", readrank_selected=True), row(id="b", readrank_selected=True)]}
    f = topic_live_count(g)
    assert f and f.check_id == "multiple-live" and f.severity == "high"

def test_topic_one_candidate_not_rankable():
    g = {"race_id": "r1", "topic_key": "housing", "quotes": [row(readrank_selected=True)]}
    f = topic_min_candidates(g)
    assert f and f.check_id == "not-rankable"

def test_topic_two_candidates_one_each_is_clean():
    g = {"race_id": "r1", "topic_key": "housing",
         "quotes": [row(id="a", candidate="A", readrank_selected=True),
                    row(id="b", candidate="B", readrank_selected=True)]}
    assert topic_live_count(g) is None

def test_topic_same_candidate_two_live_flagged():
    g = {"race_id": "r1", "topic_key": "housing",
         "quotes": [row(id="a", candidate="A", readrank_selected=True),
                    row(id="b", candidate="A", readrank_selected=True)]}
    f = topic_live_count(g)
    assert f and f.check_id == "multiple-live" and f.severity == "high"

def test_run_mechanical_aggregates_quote_and_topic():
    from scripts.checks import run_mechanical
    rows = [row(id="a", candidate="A", editor_note=None), row(id="b", candidate="B", editor_note=None)]
    fs = run_mechanical(rows)
    ids = {f.check_id for f in fs}
    assert "note-missing" in ids  # quote-level
    # two distinct candidates, one live each -> NOT multiple-live
    assert "multiple-live" not in ids


# --- scorecard-source ---
# A legislative scorecard publishes an advocacy group's rating and vote record for a member,
# never the member's own words. All 30 scorecard-sourced live quotes came back
# source-unverified in the 2026-08-02 full sweep — 30 of 30.

def test_scorecard_source_lcv_moc_page_flagged():
    f = check_scorecard_source(row(source_url="https://www.lcv.org/congressional-scorecard/moc/mike-carey"))
    assert f is not None and f.check_id == "scorecard-source"
    assert f.severity == "high" and f.fix_class == "decision-required"

def test_scorecard_source_lcv_members_index_flagged():
    f = check_scorecard_source(row(
        source_url="https://www.lcv.org/scorecard/members-of-congress/?congress=118&state=AL&chamber=H"))
    assert f is not None and f.check_id == "scorecard-source"

def test_scorecard_source_generalizes_to_other_advocacy_groups():
    for u in ("https://aflcio.org/scorecard/legislators/jane-doe",
              "https://heritageaction.com/scorecard/members/jane-doe",
              "https://example.org/environmental-scorecards/jane-doe"):
        assert check_scorecard_source(row(source_url=u)) is not None, u

def test_scorecard_source_does_not_match_a_headline_slug():
    # Path-anchored on purpose: an article about a scorecard is still an article.
    assert check_scorecard_source(row(
        source_url="https://www.politico.com/news/2026/03/01/scorecard-shows-house-split-00123")) is None

def test_scorecard_source_does_not_flag_rollcall_the_news_outlet():
    # CQ Roll Call is journalism, not a vote tally — an earlier draft of this pattern caught it.
    assert check_scorecard_source(row(source_url="https://rollcall.com/members/suhas-subramanyam/")) is None

def test_scorecard_source_does_not_flag_bill_or_vote_pages():
    # Deliberately out of scope: a congress.gov bill page verified a real quote in the sweep,
    # so "structurally cannot carry a quote" is false for that class.
    assert check_scorecard_source(row(
        source_url="https://www.congress.gov/bill/119th-congress/house-bill/3069/cosponsors")) is None
    assert check_scorecard_source(row(source_url="https://clerk.house.gov/Votes/202523")) is None

def test_scorecard_source_youtube_not_flagged():
    assert check_scorecard_source(row(source_url="https://youtu.be/x?t=1s")) is None


# --- stance-label ---
# A quote of <=4 words names a topic instead of taking a position on it. Calibrated against the
# corpus: all 101 live quotes at that length read as slogans or platform bullets, while the 5-6
# word band already holds real mechanism-bearing sentences.

def test_stance_label_bare_noun_phrase_flagged():
    for t in ("Universal Healthcare", "Medicare For All", "Anti-ICE", "Abolish ICE",
              "Tax the rich", "Protect the unborn"):
        f = check_stance_label(row(quote_text=t))
        assert f is not None and f.check_id == "stance-label", t
        assert f.severity == "medium" and f.fix_class == "decision-required"

def test_stance_label_counts_words_not_characters():
    f = check_stance_label(row(quote_text="Close the border."))
    assert f is not None and "3 word" in f.what

def test_stance_label_leaves_a_terse_real_sentence_alone():
    # 5+ words is where genuine, mechanism-bearing quotes start; the check must stop below them.
    for t in ("Lower Medicare eligibility to Age 55.",
              "Rein in private insurers managing Medicaid.",
              "We need all forms of energy.",
              "public funds belong in public schools"):
        assert check_stance_label(row(quote_text=t)) is None, t

def test_stance_label_boundary_is_exactly_the_constant():
    four = " ".join(["word"] * STANCE_LABEL_MAX_WORDS)
    five = " ".join(["word"] * (STANCE_LABEL_MAX_WORDS + 1))
    assert check_stance_label(row(quote_text=four)) is not None
    assert check_stance_label(row(quote_text=five)) is None

def test_stance_label_ignores_punctuation_and_hyphenates_as_one_word():
    # "all-of-the-above" is one word, not four — punctuation must not inflate the count.
    assert check_stance_label(row(quote_text="An all-of-the-above approach")) is not None
    # ...and stray quotes/ellipses must not inflate it either
    assert check_stance_label(row(quote_text='"Protecting the unborn"')) is not None

def test_stance_label_empty_quote_not_flagged():
    # An empty quote_text is a different defect; don't double-report it here.
    assert check_stance_label(row(quote_text="")) is None
    assert check_stance_label(row(quote_text=None)) is None
