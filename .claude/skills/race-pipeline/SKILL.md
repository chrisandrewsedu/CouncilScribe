---
name: race-pipeline
description: Run a Read & Rank content pipeline session - pull the highest-priority races
  from essentials.readrank_race_pipeline, advance each one lifecycle step (create race,
  build roster, source quotes, publish, audit), update the queue. Use when the user wants
  to work the 2026 race queue, source candidate quotes at scale, or check pipeline status.
---

# Race Pipeline Session

Design/spec: `read-rank/docs/superpowers/specs/2026-07-31-readrank-2026-content-pipeline-design.md`.
The queue is `essentials.readrank_race_pipeline` (ev-accounts DB). Priority is always
`election_date` asc, then `priority_tier` asc. The queue is the only cross-session memory.

## Session loop

1. **Claim work** (N = 10–20):

```sql
update essentials.readrank_race_pipeline p
set claimed_by = :session_id, claimed_at = now()
where p.id in (
  select id from essentials.readrank_race_pipeline
  where status not in ('audited','skipped','blocked')
    and (claimed_at is null or claimed_at < now() - interval '2 hours')
  order by election_date, priority_tier
  limit :N
)
returning p.id, p.race_label, p.status, p.race_id;
```

2. **Fan out one agent per race** for its pending transition (transitions below).
   Research on haiku-class, quote extraction on sonnet-class; publish/audit stay on the
   session's main (strong) model.
3. **Advance statuses**, write `notes`, release claims (`claimed_by = null, claimed_at = null`).
4. **Refresh + snapshot**:

```sql
select essentials.refresh_readrank_pipeline_counters();
select office_category, status, count(*),
       count(*) filter (where rankable_topics > 0) as rankable
from essentials.readrank_race_pipeline group by 1,2 order by 1,2;
```

Report the snapshot and list races newly awaiting the human live-selection step.

## Transitions

### needs_race → needs_roster
Create `essentials.elections` (reuse an existing election row for the same state+date+kind
if present!) and `essentials.races` rows. Mirror existing conventions: `position_name` like
existing rows ("Governor of Ohio"), `office_id` nullable (link only if an obvious
`essentials.offices` row exists), party primaries = separate race rows sharing one election,
`primary_party` set. Then set the pipeline row's `race_id`.

### needs_roster → needs_quotes
The pipeline row's `notes` holds the researched candidate JSON. Verify against the SOS list,
then insert `essentials.race_candidates` (full_name, first/last, is_incumbent,
candidate_status 'active', website_url, source like 'manual:pipeline-2026'). Ensure each
major candidate has an `essentials.politicians` row (quotes attach to politician_id) —
create minimal rows if missing. Roster is done when every ballot-listed candidate is present.

### needs_quotes → quotes_staged
Per candidate, work DOWN the source hierarchy (QUOTE-CURATION-PRINCIPLES §5, ranked by
**directness of answer** — the levels are named, not numbered, because
`discovered_sources.source_tier_guess` is a *different* 1–4 scale):

- **`answered-this-question`** — debate/forum answers, AND **candidate questionnaires**
  (Vote411, LWV, outlet questionnaires). A questionnaire is written and self-published, yet
  every candidate answers an identical prompt and every ballot-qualified candidate is
  invited — the most directly comparable source there is. Prefer these.
- **`adjacent`** — responsive, but the prompt differed: a news interview that circled the
  subject, a narrower or broader version of the question.
- **`curator-extracted`** — lifted from material not organised as an answer: stump speeches,
  floor remarks, op-eds, platform pages.
- **`excluded`** — hot-mic / secretly-recorded / "gotcha". Banned, not deprioritized.

`adjacent` and `curator-extracted` need a justification note; any WRITTEN source at any level
yields verbatim sentences only. Questioner independence breaks ties *within* a level, never
across them — a partisan-host podcast is never excluded when it is a candidate's only
sourceable speech, and the note says so. Curate against
`.claude/skills/audit-quotes/CHECKS.md` UP FRONT: forward-looking operative clause, answers
the topic's ranking question, honest de-id, no partisan tells, prefer the HOW. Stage as a
publish-quotes `batch.json` per candidate. Goal: >= 2 candidates per topic or the topic
doesn't ship.

**Hunt shared-question moments first.** A question BOTH candidates answered — a debate or
forum question they were each asked, or the same questionnaire item — is worth more than two
separately sourced quotes, because it delivers three things at once: an
`answered-this-question` source, a ready-made ranking question, and a genuine head-to-head.
Comparability is created in the room: the casebook's MI Governor case (one common
multi-candidate debate) yielded twice the rankable questions of AZ-01's two party-segregated
ones, where the general-election candidates were never asked the same question in the same
room. When you find one, record the question, its meeting/segment and its URL, and insert it
as an `essentials.readrank_questions` row with `origin = 'moderator'` and `source_ref` set —
neutralize the moderator's framing and name no candidate first. Questionnaires are the
highest-yield form: same prompt, every ballot-qualified candidate invited, so they fix
comparability and inclusion at once.

**Source for faithfulness only.** Sourcing is judged twice downstream: per quote (faithful,
responsive, direct?) and per set (do a question's answers actually let a citizen choose?).
Never pick a quote because it contrasts. Genuine agreement between candidates is a real
finding and is shown as convergence, so a question whose answers converge is **not** a
sourcing failure. A ranking question may sit off the Compass axis, or carry no Compass topic
at all, when the candidates answered something the taxonomy doesn't cover — declare it via
`readrank_questions.origin` (QUOTE-CURATION-PRINCIPLES §7.3).

**Video-ingest shortlist (always).** Sourcing doubles as ingest scouting: every debate,
candidate forum, news-clip interview, or floor-speech video found — ESPECIALLY ones too
long or untranscribed to mine on the spot — is a discovery candidate. Insert one row per
video into `essentials.discovered_sources` (`discovered_via='agent'`, `status='pending'`,
`route='ingest'`, `source_key` normalized per `src/source_key.py`, `race_id` +
`matched_politician_ids` set for this race, `why` = one sentence citing the evidence).
Ingesting these into on-the-record's transcript pipeline (`meetings.segments`) is how we
pull verbatim, timestamp-deep-linked quotes later; a video covering several candidates on
shared topics is the highest-value ingest. Note in the pipeline row's `notes` that
discovery rows were filed for this race.

### quotes_staged → published
Run the **publish-quotes** skill on each staged batch (dry-run, user OK, --commit). It
inserts drafts and auto-runs **audit-quotes** on the new ids.

### published → audited
Audit findings clean (or fixed via the audit's gated flow) -> `audited`. Findings needing
judgment -> `blocked` + `status_reason`. NOTE: `audited` is machine-done, not voter-visible:
a human still selects the live quote per (candidate, topic) in `/admin/readrank-quotes`;
`rankable_topics` counts only live selections.

## Rules

- Never mark `blocked`/`skipped` without `status_reason`.
- Production DB: additive writes only (inserts, status updates). Never delete/overwrite
  quotes, races, or candidates in a pipeline session.
- All quote sourcing rules live in `essentials/docs/QUOTE-CURATION-PRINCIPLES.md` +
  `.claude/skills/audit-quotes/CHECKS.md` — read both before sourcing.
- MI Aug 4 / WI+MN Aug 11 primaries outrank everything until they pass.

## Discovery queue integration

Before hunting sources for a race, check the discovery triage queue — a human
has already vetted these:

    select url, title, channel_name, why, source_tier_guess
    from essentials.discovered_sources
    where race_id = :race_id and status = 'approved' and route = 'quote_source'
    order by source_tier_guess nulls last;

Use them first (they still get the normal verify-then-cite treatment). After
sourcing quotes from a row, mark it consumed:

    update essentials.discovered_sources set status = 'ingested',
      status_reason = 'quotes sourced' where id = :id;

Video shortlists: rows with `route='ingest'` are handled by the GUI Discovery
tab, not by pipeline sessions — do not ingest them from here. When your own
research finds a NEW source worth ingesting, insert a `discovered_sources` row
(`discovered_via='agent'`, `status='pending'`) instead of writing
`ingest-candidates.json`.
