import importlib.util
from pathlib import Path

# Load the script module by path (it lives under .claude/skills, not on sys.path).
_SPEC_PATH = Path(__file__).resolve().parents[1] / ".claude/skills/audit-quotes/scripts/db.py"
_spec = importlib.util.spec_from_file_location("audit_db", _SPEC_PATH)
audit_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_db)

LA_GENERAL = "9e888818-c50b-4c61-a106-a0839ff2479d"


def test_race_expr_prefers_the_requested_race():
    """With --race, that race is authoritative — not the lowest-sorting one.

    Bass and Raman sit on both LA Mayor rosters; the June primary id sorts first, so the
    old `ORDER BY rc.race_id LIMIT 1` mislabelled every quote in the November race.
    """
    expr = audit_db.race_id_expr(race=LA_GENERAL)
    assert "%(race)s" in expr
    assert "ORDER BY rc.race_id" not in expr


def test_race_expr_falls_back_to_lowest_when_unscoped():
    """An unscoped sweep has no race to prefer, so the deterministic fallback stays."""
    expr = audit_db.race_id_expr(race=None)
    assert "ORDER BY rc.race_id" in expr


def test_scope_sql_embeds_the_race_expression():
    sql = audit_db.build_scope_sql(race=LA_GENERAL)
    assert audit_db.race_id_expr(race=LA_GENERAL) in sql
    assert "AS race_id" in sql


class _FakeCursor:
    """Captures what fetch_rows actually hands to psycopg2."""

    def __init__(self):
        self.sql = None
        self.params = None

    def execute(self, sql, params):
        self.sql, self.params = sql, params

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self, **kwargs):
        return self._cur


def test_fetch_rows_forwards_race_into_the_generated_sql():
    """Guards the wiring, not just the string builders.

    The three assertions above all pass even if fetch_rows stops passing `race` to
    build_scope_sql -- it would silently fall back to the lowest-sorting race id, which is the
    original bug. This is the test that actually fails on that regression.
    """
    cur = _FakeCursor()
    audit_db.fetch_rows(_FakeConn(cur), race=LA_GENERAL)
    assert audit_db.race_id_expr(race=LA_GENERAL) in cur.sql
    assert "ORDER BY rc.race_id" not in cur.sql
    assert cur.params["race"] == LA_GENERAL


def test_fetch_rows_unscoped_uses_the_fallback_expression():
    cur = _FakeCursor()
    audit_db.fetch_rows(_FakeConn(cur), race=None)
    assert "ORDER BY rc.race_id" in cur.sql
    assert cur.params["race"] is None
