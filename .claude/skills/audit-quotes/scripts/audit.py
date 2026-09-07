"""audit-quotes CLI. Default: sweep all live quotes across all races.
Modes:
  (default)         resolve scope, run mechanical checks, write context bundles + mechanical report
Flags: --candidate NAME  --topic KEY  --ids id1,id2  --include-drafts  --out DIR  --scope-label LABEL
       --verify-written / --verify-sources (opt into fetching written sources)
"""
import argparse, json, pathlib, datetime
from scripts.db import connect, fetch_rows, fetch_stance
from scripts.checks import run_mechanical
from scripts.verify_source import run_source_checks, make_page_fetcher
from scripts.report import render


def build_bundle(race_id, rows, stances):
    """Group one race's rows for the judgment agent.

    `topics` drives the per-quote checks and the existing report. `questions` drives the per-set
    checks: several questions can share one topic, so judging a whole topic as a set would compare
    answers to different questions -- the incommensurability the rubric exists to catch.
    Quotes with no question_id can't be judged per-set; they are listed rather than dropped.
    """
    bundle = {"race_id": race_id, "topics": {}, "questions": {}, "unattached_quote_ids": []}
    for r in rows:
        enriched = {**r, "stance": stances.get((r["race_id"], r.get("politician_id"), r["topic_key"]))}
        topic = bundle["topics"].setdefault(
            r["topic_key"], {"topic_key": r["topic_key"], "quotes": []})
        topic["quotes"].append(enriched)

        qid = r.get("question_id")
        if not qid:
            bundle["unattached_quote_ids"].append(r["id"])
            continue
        question = bundle["questions"].setdefault(qid, {
            "question_id": qid,
            "question_text": r.get("question_text"),
            "origin": r.get("question_origin"),
            "topic_key": r["topic_key"],
            "quotes": [],
        })
        question["quotes"].append(enriched)
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate"); ap.add_argument("--topic"); ap.add_argument("--ids")
    ap.add_argument("--race", help="Scope to one race_id (uuid). Find race_ids in a default run's report.")
    ap.add_argument("--include-drafts", action="store_true")
    ap.add_argument("--verify-written", "--verify-sources", dest="verify_written",
                    action="store_true",
                    help="Also verify non-video sources by fetching the cited page (network I/O; "
                         "pages cached under .runs/.source-cache). Off by default: the audit is "
                         "otherwise DB-only.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scope-label", default="all races")
    a = ap.parse_args()
    ids = [s.strip() for s in a.ids.split(",") if s.strip()] if a.ids else None

    conn = connect()
    rows = fetch_rows(conn, ids=ids, candidate=a.candidate, topic=a.topic, race=a.race, include_drafts=a.include_drafts)
    if not rows:
        print("No quotes matched scope."); return

    races = {r["race_id"] for r in rows}
    topics = {(r["race_id"], r["topic_key"]) for r in rows}
    print(f"SCOPE: {len(rows)} quotes | {len(races)} races | {len(topics)} race-topic groups | "
          f"drafts={'yes' if a.include_drafts else 'no'}")

    # Default output dir resolves relative to this skill (cwd-independent), so it always lands
    # under audit-quotes/.runs/ (which .gitignore covers) no matter where the CLI is invoked.
    skill_root = pathlib.Path(__file__).resolve().parents[1]  # .../.claude/skills/audit-quotes

    # The page cache is deliberately shared across runs (not per-run): a sweep and its follow-up
    # re-runs should hit each cited URL once, not once per invocation.
    fetch_page = None
    if a.verify_written:
        written = sum(1 for r in rows
                      if "youtube.com" not in (r.get("source_url") or "")
                      and "youtu.be" not in (r.get("source_url") or ""))
        print(f"WRITTEN-SOURCE VERIFICATION: on — up to {written} pages may be fetched "
              f"(cached in .runs/.source-cache; repeated URLs and cache hits cost nothing, "
              f"and requests to one host are spaced 1s apart)")
        fetch_page = make_page_fetcher(skill_root / ".runs" / ".source-cache")

    findings = run_mechanical(rows)
    findings += run_source_checks(conn, rows, fetch_page=fetch_page)
    print(f"MECHANICAL+SOURCE FINDINGS: {len(findings)}")

    run_dir = pathlib.Path(a.out) if a.out else skill_root / ".runs" / str(datetime.date.today())
    (run_dir / "context").mkdir(parents=True, exist_ok=True)
    by_race = {}
    for r in rows:
        by_race.setdefault(r["race_id"], []).append(r)
    stance_cache = {}
    for race, rrows in by_race.items():
        for r in rrows:
            key = (r["race_id"], r["politician_id"], r["topic_key"])
            if key not in stance_cache:
                stance_cache[key] = fetch_stance(
                    conn, r["politician_id"], r["topic_key"], race_id=r["race_id"]
                )
        bundle = build_bundle(race, rrows, stance_cache)
        safe = str(race).replace("/", "_")
        (run_dir / "context" / f"{safe}.json").write_text(json.dumps(bundle, indent=2, default=str))

    (run_dir / "mechanical_findings.json").write_text(
        json.dumps([f.to_dict() for f in findings], indent=2, default=str))
    report_md = render(findings, scope_label=a.scope_label + " (mechanical only)")
    (run_dir / "mechanical_report.md").write_text(report_md)
    print(f"WROTE: {run_dir}/context/*.json, mechanical_findings.json, mechanical_report.md")
    print("NEXT: run the judgment pass (see SKILL.md), then merge findings and render the full report.")

if __name__ == "__main__":
    main()
