#!/usr/bin/env python
"""Build a side-by-side quote-review page for Read & Rank showcase races.

The unit is the QUESTION, not the topic: two candidates are comparable when they answered the
same question, not when they touched the same subject. Topic is shown only as a chip, because it
is the Compass-coupling backbone rather than the organising axis.

Single-voice questions are included on purpose — seeing who is missing is the point.

    .venv/bin/python scripts/build_quote_review.py [-o out.html]
"""
import argparse
import datetime
import html
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / ".claude/skills/_shared"))
from ev_env import ev_accounts_database_url  # noqa: E402
import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

RACES = [
    ("bc936a36-287c-4ffd-abd8-5e4fd798bae5", "CA Governor", "November 3, 2026 · general"),
    ("9e888818-c50b-4c61-a106-a0839ff2479d", "Los Angeles Mayor", "November 3, 2026 · general runoff"),
]

# First-person markers that identify an office-holder on a blind card.
SELF_ID = re.compile(
    r"\b(on my watch|i declared|as governor|as mayor|when i was|my administration|"
    r"i established|under my|in my district|i signed|my number one)\b", re.I)

SQL = """
SELECT q.id::text            AS quote_id,
       q.question_id::text   AS question_id,
       rq.question_text,
       rq.origin             AS question_origin,
       q.topic_key,
       ct.short_title        AS topic_title,
       p.full_name           AS candidate,
       q.quote_text,
       q.deidentified_text,
       q.editor_note,
       q.source_name,
       q.source_url,
       q.readrank_selected
  FROM essentials.quotes q
  JOIN essentials.race_candidates rc ON rc.politician_id = q.politician_id
  JOIN essentials.politicians p      ON p.id = q.politician_id
  LEFT JOIN essentials.readrank_questions rq ON rq.id = q.question_id
  LEFT JOIN inform.compass_topics ct         ON ct.topic_key = lower(q.topic_key)
 WHERE rc.race_id = %s
   AND COALESCE(rc.candidate_status, 'active') <> 'withdrawn'
 ORDER BY rq.question_text NULLS LAST, p.full_name
"""


def flags_for(row, other_surname):
    """Problems a reviewer should see without reading every word."""
    out = []
    blind = (row["deidentified_text"] or "").strip()
    canon = (row["quote_text"] or "").strip()
    if not (row["editor_note"] or "").strip():
        out.append(("no-note", "No editor note — cannot ship"))
    if not row["source_url"]:
        out.append(("no-source", "No source URL — cannot ship"))
    if not blind:
        out.append(("no-blind", "No blind text"))
    if other_surname and other_surname.lower() in blind.lower():
        out.append(("names-opponent", f"Blind card names {other_surname}"))
    if blind and blind == canon and SELF_ID.search(canon):
        out.append(("self-id", "Blind text identical to canonical, but contains a self-identifying phrase"))
    return out


def esc(s):
    return html.escape(s or "")


OTR = "https://on-the-record.onrender.com/meetings/"
_YT = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")
_T = re.compile(r"[?&]t=(\d+)")
_DATED_SLUG = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")


def meeting_index(cur):
    """video_url -> the ingested meeting, so a bare YouTube link can become a readable one."""
    cur.execute("""SELECT video_url, id::text AS id, date,
                          COALESCE(NULLIF(btrim(title), ''), slug) AS label
                     FROM meetings.meetings WHERE video_url IS NOT NULL""")
    return {r["video_url"]: dict(r) for r in cur.fetchall()}


def pretty_label(label, date):
    """'2026-05-15-governor-debate-(cbs-and-sf-examiner)' -> 'Governor debate (cbs and sf examiner)'."""
    if not label:
        return "meeting"
    m = _DATED_SLUG.match(label)
    body = m.group(2) if m else label
    if m or "-" in body:
        body = body.replace("-", " ").strip()
        body = body[:1].upper() + body[1:]
    return body


def source_link(row, meetings):
    """Prefer our own transcript over a bare youtube.com, and always give the link a name."""
    url = row["source_url"]
    if not url:
        return '<span class="nosrc">no source</span>'
    yt = _YT.search(url)
    if yt and yt.group(1) in meetings:
        m = meetings[yt.group(1)]
        t = _T.search(url)
        otr = f'{OTR}{m["id"]}' + (f'?t={t.group(1)}' if t else "")
        name = esc(pretty_label(m["label"], m["date"]))
        when = f' <span class="dim">{m["date"]}</span>' if m.get("date") else ""
        return (f'<a href="{esc(otr)}" target="_blank" rel="noopener">{name}</a>{when}'
                f' <a class="alt" href="{esc(url)}" target="_blank" rel="noopener">YouTube ↗</a>')
    if yt:
        return (f'<a href="{esc(url)}" target="_blank" rel="noopener">YouTube (not ingested)</a>')
    if OTR.rstrip("/") in url or "on-the-record" in url or "ontherecord" in url:
        label = esc(row["source_name"] or "On the Record transcript")
        return f'<a href="{esc(url)}" target="_blank" rel="noopener">{label}</a>'
    host = re.sub(r"^www\.", "", (re.search(r"://([^/]+)", url) or [None, ""])[1])
    label = esc(row["source_name"] or host or "source")
    return (f'<a href="{esc(url)}" target="_blank" rel="noopener">{label}</a>'
            + (f' <span class="dim">{esc(host)}</span>' if row["source_name"] and host else ""))


def diff_blind(canon, blind):
    """Word-level diff of canonical -> blind, so de-identification edits are visible at a glance.

    Additions and substitutions are highlighted; removals are struck through, because on a blind
    card what was taken OUT is usually the thing that mattered.
    """
    import difflib
    a, b = (canon or "").split(), (blind or "").split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            out.append(esc(" ".join(b[j1:j2])))
        elif op == "insert":
            out.append(f'<mark>{esc(" ".join(b[j1:j2]))}</mark>')
        elif op == "delete":
            out.append(f'<del>{esc(" ".join(a[i1:i2]))}</del>')
        else:
            out.append(f'<del>{esc(" ".join(a[i1:i2]))}</del> <mark>{esc(" ".join(b[j1:j2]))}</mark>')
    return " ".join(x for x in out if x)


def build(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    meetings = meeting_index(cur)
    races = []
    for race_id, name, subtitle in RACES:
        cur.execute(SQL, (race_id,))
        rows = [dict(r) for r in cur.fetchall()]
        cands = sorted({r["candidate"] for r in rows})
        surname = {c: c.split()[-1] for c in cands}
        by_q = {}
        for r in rows:
            key = r["question_id"] or "__unattached__"
            q = by_q.setdefault(key, {
                "question_text": r["question_text"] or "(not attached to a question)",
                "origin": r["question_origin"],
                "topic": r["topic_title"] or r["topic_key"],
                "topic_key": r["topic_key"],
                "by_cand": {c: [] for c in cands},
            })
            other = next((surname[c] for c in cands if c != r["candidate"]), None)
            r["flags"] = flags_for(r, other)
            q["by_cand"][r["candidate"]].append(r)
        for q in by_q.values():
            answered = [c for c in cands if q["by_cand"][c]]
            q["answered"] = len(answered)
            q["state"] = "both" if len(answered) >= 2 else ("one" if answered else "none")
            q["live"] = sum(1 for c in cands for x in q["by_cand"][c] if x["readrank_selected"])
            q["issues"] = sum(len(x["flags"]) for c in cands for x in q["by_cand"][c])
        order = sorted(by_q.values(),
                       key=lambda q: (q["state"] != "both", q["topic"] or "", q["question_text"]))
        races.append({"name": name, "subtitle": subtitle, "cands": cands, "questions": order})
    return races, meetings


def render(races, meetings):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"""<title>Read &amp; Rank — quote review</title>
<style>
:root {{
  --bg:#fbfbfa; --fg:#1b1b19; --mut:#6b6a66; --line:#e3e1dc; --card:#fff;
  --accent:#2d5f5d; --warn:#8a4b2a; --warnbg:#fdf3ec; --live:#2d5f5d; --livebg:#e8f0ef;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161715; --fg:#eceae4; --mut:#9b9890; --line:#2e302c; --card:#1e201d;
    --accent:#7fb5b1; --warn:#d99a6c; --warnbg:#2c2119; --live:#7fb5b1; --livebg:#1d2a29; }}
}}
:root[data-theme="dark"] {{
  --bg:#161715; --fg:#eceae4; --mut:#9b9890; --line:#2e302c; --card:#1e201d;
  --accent:#7fb5b1; --warn:#d99a6c; --warnbg:#2c2119; --live:#7fb5b1; --livebg:#1d2a29;
}}
:root[data-theme="light"] {{
  --bg:#fbfbfa; --fg:#1b1b19; --mut:#6b6a66; --line:#e3e1dc; --card:#fff;
  --accent:#2d5f5d; --warn:#8a4b2a; --warnbg:#fdf3ec; --live:#2d5f5d; --livebg:#e8f0ef;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-serif,Georgia,"Iowan Old Style",serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:32px 20px 96px; }}
h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--mut); font-size:13px; margin-bottom:26px; }}
.controls {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0;
  border-bottom:1px solid var(--line); margin-bottom:26px; display:flex; gap:8px; flex-wrap:wrap;
  align-items:center; }}
button {{ font:inherit; font-size:13px; padding:5px 12px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); border-radius:999px; cursor:pointer; }}
button[aria-pressed="true"] {{ background:var(--accent); color:var(--bg); border-color:var(--accent); }}
h2 {{ font-size:20px; margin:38px 0 2px; }}
h2 .rs {{ font-size:12px; color:var(--mut); font-weight:400; font-family:ui-sans-serif,system-ui; }}
.q {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:18px 18px 6px; margin:16px 0; }}
.qhead {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; margin-bottom:14px; }}
.qtext {{ font-size:17px; font-weight:600; flex:1 1 340px; min-width:0; }}
.chip {{ font:11px/1.5 ui-sans-serif,system-ui; text-transform:uppercase; letter-spacing:.05em;
  color:var(--mut); border:1px solid var(--line); padding:1px 8px; border-radius:999px;
  white-space:nowrap; }}
.chip.live {{ color:var(--live); background:var(--livebg); border-color:transparent; }}
.chip.gap {{ color:var(--warn); background:var(--warnbg); border-color:transparent; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:760px) {{ .cols {{ grid-template-columns:1fr; }} }}
.cell {{ border-top:1px solid var(--line); padding-top:12px; min-width:0; }}
.cand {{ font:12px/1.5 ui-sans-serif,system-ui; font-weight:600; letter-spacing:.03em;
  text-transform:uppercase; color:var(--mut); margin-bottom:8px; }}
.quote {{ margin:0 0 10px; }}
.quote + .quote {{ border-top:1px dashed var(--line); padding-top:12px; }}
blockquote {{ margin:0 0 8px; padding-left:12px; border-left:2px solid var(--accent); }}
.meta {{ font:12px/1.5 ui-sans-serif,system-ui; color:var(--mut); }}
.meta a {{ color:var(--accent); }}
.note {{ font:12.5px/1.5 ui-sans-serif,system-ui; color:var(--mut); margin-top:6px;
  padding-left:12px; border-left:2px solid var(--line); }}
.blind {{ font:12.5px/1.5 ui-sans-serif,system-ui; margin-top:6px; padding:8px 10px;
  background:var(--bg); border-radius:6px; }}
.blind b {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); }}
.blind mark {{ background:#ffe08a; color:#3a2c00; font-weight:700; padding:0 2px; border-radius:2px; }}
.blind del {{ color:var(--mut); text-decoration:line-through; text-decoration-thickness:1px; opacity:.75; }}
@media (prefers-color-scheme: dark) {{ .blind mark {{ background:#7a6320; color:#ffeec2; }} }}
:root[data-theme="dark"] .blind mark {{ background:#7a6320; color:#ffeec2; }}
:root[data-theme="light"] .blind mark {{ background:#ffe08a; color:#3a2c00; }}
.meta a.alt {{ font-size:11px; color:var(--mut); margin-left:6px; }}
.dim {{ color:var(--mut); }}
.nosrc {{ color:var(--warn); }}
.flag {{ display:inline-block; font:11px/1.5 ui-sans-serif,system-ui; color:var(--warn);
  background:var(--warnbg); padding:1px 7px; border-radius:4px; margin:3px 4px 0 0; }}
.empty {{ color:var(--mut); font-style:italic; font-size:13px; padding:10px 0 16px; }}
.hidden {{ display:none; }}
</style>
<div class="wrap">
<h1>Read &amp; Rank — quote review</h1>
<div class="sub">Organised by <strong>question</strong>, not topic: two candidates are comparable
when they answered the same question, not when they touched the same subject. Topic is shown as a
chip because it is the Compass-coupling backbone, not the organising axis.
Single-voice questions are included deliberately — the gap is the point.
Generated {now}.</div>
<div class="controls">
  <button data-f="all" aria-pressed="true">All questions</button>
  <button data-f="both" aria-pressed="false">Both answered</button>
  <button data-f="one" aria-pressed="false">One voice only</button>
  <button data-f="issues" aria-pressed="false">Has issues</button>
  <button data-f="live" aria-pressed="false">Has a live quote</button>
</div>"""]

    for race in races:
        nb = sum(1 for q in race["questions"] if q["state"] == "both")
        parts.append(f'<h2>{esc(race["name"])} '
                     f'<span class="rs">{esc(race["subtitle"])} · {len(race["questions"])} questions · '
                     f'{nb} with both candidates</span></h2>')
        for q in race["questions"]:
            cls = f'q s-{q["state"]}' + (" has-issues" if q["issues"] else "") + (" has-live" if q["live"] else "")
            chips = [f'<span class="chip">{esc(q["topic"])}</span>']
            if q["origin"]:
                chips.append(f'<span class="chip">{esc(q["origin"])}</span>')
            if q["live"]:
                chips.append(f'<span class="chip live">{q["live"]} live</span>')
            if q["state"] != "both":
                chips.append('<span class="chip gap">one voice</span>')
            if q["issues"]:
                chips.append(f'<span class="chip gap">{q["issues"]} issue{"s" if q["issues"]>1 else ""}</span>')
            parts.append(f'<div class="{cls}"><div class="qhead">'
                         f'<div class="qtext">{esc(q["question_text"])}</div>{"".join(chips)}</div>'
                         f'<div class="cols">')
            for cand in race["cands"]:
                parts.append(f'<div class="cell"><div class="cand">{esc(cand)}</div>')
                items = q["by_cand"][cand]
                if not items:
                    parts.append('<div class="empty">No answer to this question.</div>')
                for it in items:
                    live = '<span class="chip live">live</span>' if it["readrank_selected"] else '<span class="chip">draft</span>'
                    src = source_link(it, meetings)
                    parts.append(f'<div class="quote"><blockquote>{esc(it["quote_text"])}</blockquote>'
                                 f'<div class="meta">{live} &nbsp;·&nbsp; {src}</div>')
                    canon = (it["quote_text"] or "").strip()
                    blind = (it["deidentified_text"] or "").strip()
                    if blind and blind != canon:
                        parts.append('<div class="blind"><b>Blind card — changes from canonical</b><br>'
                                     f'{diff_blind(canon, blind)}</div>')
                    if (it["editor_note"] or "").strip():
                        parts.append(f'<div class="note">{esc(it["editor_note"])}</div>')
                    for _, label in it["flags"]:
                        parts.append(f'<span class="flag">{esc(label)}</span>')
                    parts.append('</div>')
                parts.append('</div>')
            parts.append('</div></div>')

    parts.append("""</div>
<script>
const btns=[...document.querySelectorAll('.controls button')];
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.setAttribute('aria-pressed', x===b));
  const f=b.dataset.f;
  document.querySelectorAll('.q').forEach(q=>{
    const show = f==='all' ? true
      : f==='both'   ? q.classList.contains('s-both')
      : f==='one'    ? q.classList.contains('s-one')
      : f==='issues' ? q.classList.contains('has-issues')
      : q.classList.contains('has-live');
    q.classList.toggle('hidden', !show);
  });
});
</script>""")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="quote-review.html")
    a = ap.parse_args()
    conn = psycopg2.connect(ev_accounts_database_url(__file__))
    races, meetings = build(conn)
    pathlib.Path(a.out).write_text(render(races, meetings), encoding="utf-8")
    tot = sum(len(r["questions"]) for r in races)
    iss = sum(q["issues"] for r in races for q in r["questions"])
    print(f"wrote {a.out} — {tot} questions across {len(races)} races, {iss} flagged issues")


if __name__ == "__main__":
    main()
