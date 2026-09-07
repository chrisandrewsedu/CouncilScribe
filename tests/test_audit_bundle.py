import importlib.util
import re
import sys
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parents[1] / ".claude/skills/audit-quotes"
# audit.py does `from scripts.db import ...`, so the skill root has to be importable. The repo
# root also holds a `scripts/` dir, but it has no __init__.py, so it is only a namespace portion —
# the skill's regular package wins the lookup regardless of ordering.
if str(_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILL_ROOT))

_SPEC_PATH = _SKILL_ROOT / "scripts/audit.py"
_spec = importlib.util.spec_from_file_location("audit_cli", _SPEC_PATH)
audit_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_cli)

_DB_PATH = _SKILL_ROOT / "scripts/db.py"
_db_spec = importlib.util.spec_from_file_location("audit_db3", _DB_PATH)
audit_db = importlib.util.module_from_spec(_db_spec)
_db_spec.loader.exec_module(audit_db)

ROWS = [
    {"id": "q1", "race_id": "r1", "topic_key": "housing", "candidate": "A", "politician_id": "pA",
     "question_id": "Q1", "question_text": "How fast should permits move?", "question_origin": "moderator"},
    {"id": "q2", "race_id": "r1", "topic_key": "housing", "candidate": "B", "politician_id": "pB",
     "question_id": "Q1", "question_text": "How fast should permits move?", "question_origin": "moderator"},
    {"id": "q3", "race_id": "r1", "topic_key": "housing", "candidate": "A", "politician_id": "pA",
     "question_id": "Q2", "question_text": "Should the state preempt local zoning?", "question_origin": "emergent"},
    {"id": "q4", "race_id": "r1", "topic_key": "housing", "candidate": "B", "politician_id": "pB",
     "question_id": None, "question_text": None, "question_origin": None},
]


def test_two_questions_under_one_topic_stay_separate():
    bundle = audit_cli.build_bundle("r1", ROWS, stances={})
    assert set(bundle["questions"]) == {"Q1", "Q2"}
    assert [q["id"] for q in bundle["questions"]["Q1"]["quotes"]] == ["q1", "q2"]
    assert [q["id"] for q in bundle["questions"]["Q2"]["quotes"]] == ["q3"]


def test_topic_map_still_holds_every_quote():
    bundle = audit_cli.build_bundle("r1", ROWS, stances={})
    assert [q["id"] for q in bundle["topics"]["housing"]["quotes"]] == ["q1", "q2", "q3", "q4"]


def test_unattached_quotes_are_collected_not_dropped():
    """A quote with no question_id can't be judged per-set, but must stay visible."""
    bundle = audit_cli.build_bundle("r1", ROWS, stances={})
    assert bundle["unattached_quote_ids"] == ["q4"]


def test_question_metadata_is_carried():
    bundle = audit_cli.build_bundle("r1", ROWS, stances={})
    assert bundle["questions"]["Q1"]["question_text"] == "How fast should permits move?"
    assert bundle["questions"]["Q1"]["origin"] == "moderator"


def _scope_sql_output_names(sql: str) -> set[str]:
    """The column names a row from fetch_rows will actually carry."""
    select_list = sql.split("FROM essentials.quotes")[0]
    names = set()
    for item in re.split(r",(?![^(]*\))", select_list.replace("SELECT", "", 1)):
        item = item.strip()
        if not item:
            continue
        alias = re.search(r"\bAS\s+(\w+)\s*$", item, re.IGNORECASE)
        names.add(alias.group(1) if alias else item.split(".")[-1].strip())
    return names


def test_bundle_only_reads_columns_the_scope_sql_actually_returns():
    """Contract between the two modules, which the pure-SQL assertions can't cover.

    build_bundle reads question_id / question_text / question_origin off each row. Those names
    are aliases invented in db.py; rename one there (or misspell one here) and the bundle would
    silently produce an empty `questions` map against the real database while every fixture-based
    test above still passed. This is the test that fails on that drift.
    """
    names = _scope_sql_output_names(audit_db.build_scope_sql(race=None))
    for key in ("id", "topic_key", "politician_id", "race_id",
                "question_id", "question_text", "question_origin"):
        assert key in names, f"{key} is read by build_bundle but not selected by build_scope_sql"
