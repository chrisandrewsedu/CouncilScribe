import importlib.util
import sys
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parents[1] / ".claude/skills/audit-quotes"
# checks.py does `from scripts.models import Finding`, so the skill root has to be importable.
# The repo root also holds a `scripts/` dir, but it has no __init__.py, so it is only a namespace
# portion — the skill's regular package wins the lookup regardless of ordering.
if str(_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILL_ROOT))

_SPEC_PATH = _SKILL_ROOT / "scripts/db.py"
_spec = importlib.util.spec_from_file_location("audit_db2", _SPEC_PATH)
audit_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_db)

from scripts import checks as audit_checks  # noqa: E402


def test_scope_sql_selects_question_columns():
    """Per-set checks need the question a quote answers, not just its topic."""
    sql = audit_db.build_scope_sql(race=None)
    assert "q.question_id" in sql
    assert "rq.question_text" in sql
    assert "rq.origin" in sql


def test_scope_sql_left_joins_questions_so_unattached_quotes_survive():
    """question_id is nullable; an unattached quote must still be audited."""
    sql = audit_db.build_scope_sql(race=None)
    assert "LEFT JOIN essentials.readrank_questions rq" in sql


# --- the mechanical pass no longer grades sources by medium -------------------------------
#
# `source-tier-4` matched a campaign-site URL pattern and called it a low-tier source. Provenance
# is *directness of answer*, not medium (comparability model §6-7; casebook "Directness of answer,
# not medium"), and no URL pattern can see directness — a Vote411 questionnaire is written and
# self-published yet is the most directly comparable source there is. The judgment pass owns this
# now, as `source-not-an-answer`.
#
# These drive `run_mechanical` and assert on the findings it returns, so they fail if the check is
# merely renamed, left wired into QUOTE_CHECKS, or reintroduced under another id.


def _row(**kw):
    base = dict(id="q1", candidate="A", topic_key="housing", race_id="r1",
                readrank_selected=True, quote_text="We must build more homes near transit.",
                deidentified_text="We must build more homes near transit.",
                editor_note="Verbatim, no edits. Matches the candidate's pro-supply stance.",
                source_name="www.youtube.com", source_url="https://youtu.be/x?t=1s")
    base.update(kw)
    return base


def _pair(**kw):
    """Two candidates on one topic, so `not-rankable` doesn't fire and mask the assertion."""
    return [_row(id="q1", candidate="A", **kw), _row(id="q2", candidate="B", **kw)]


def _ids(findings):
    return {f.check_id for f in findings}


def test_campaign_site_url_produces_no_mechanical_finding():
    """The exact URL the deleted check flagged. An otherwise-clean pair must come back clean."""
    findings = audit_checks.run_mechanical(
        _pair(source_url="https://www.xavierbecerra2026.com/housing",
              source_name="www.xavierbecerra2026.com"))
    assert findings == [], [f.check_id for f in findings]


def test_no_mechanical_check_grades_sources_by_medium():
    """Other campaign/written URL shapes the old regex caught — none of them are defects now."""
    for url in ["https://www.xavierbecerra2026.com/housing",
                "https://votesmith.com/issues",
                "https://electjones.org/platform",
                "https://smith2026.org/immigration",
                "https://jonesforsenate.com/priorities"]:
        findings = audit_checks.run_mechanical(_pair(source_url=url))
        assert findings == [], f"{url} -> {[f.check_id for f in findings]}"


def test_source_tier_four_is_not_emitted_by_any_check():
    """Belt-and-braces on the id itself, across every source shape the pass still inspects."""
    rows = []
    for i, url in enumerate(["https://www.xavierbecerra2026.com/housing",
                             "https://www.ontheissues.org/John_James.htm",
                             "https://www.isidewith.com/candidates/jane-doe",
                             "https://scorecard.lcv.org/scorecard/member",
                             "https://youtu.be/x?t=1s"]):
        rows.append(_row(id=f"q{i}", candidate=f"C{i}", source_url=url))
    assert "source-tier-4" not in _ids(audit_checks.run_mechanical(rows))


def test_check_source_tier_is_gone_from_the_module():
    assert not hasattr(audit_checks, "check_source_tier")


# --- the three bad-source checks are orthogonal to directness and must survive -------------


def test_aggregator_source_still_flagged():
    findings = audit_checks.run_mechanical(
        _pair(source_url="https://www.ontheissues.org/John_James.htm",
              source_name="www.ontheissues.org"))
    assert "invalid-source" in _ids(findings)
    assert all(f.severity == "high" for f in findings if f.check_id == "invalid-source")


def test_quiz_site_source_still_flagged():
    findings = audit_checks.run_mechanical(
        _pair(source_url="https://www.isidewith.com/candidates/jane-doe",
              source_name="www.isidewith.com"))
    assert "unquotable-source" in _ids(findings)


def test_scorecard_source_still_flagged():
    findings = audit_checks.run_mechanical(
        _pair(source_url="https://scorecard.lcv.org/scorecard/member/jane-doe",
              source_name="scorecard.lcv.org"))
    assert "scorecard-source" in _ids(findings)


def test_a_campaign_site_hosting_a_bad_source_pattern_still_flags_the_bad_source():
    """Directness and source validity are independent axes; dropping one must not drop the other."""
    findings = audit_checks.run_mechanical(
        _pair(source_url="https://smith2026.com/scorecard/2025"))
    assert _ids(findings) == {"scorecard-source"}
