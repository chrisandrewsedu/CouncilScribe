---
name: audit-quotes
description: Audit curated quotes in essentials.quotes (ev-accounts DB) against the Read & Rank curation principles — mechanical checks, a per-quote judgment pass, a per-set comparability pass, and a portfolio pass — and surface findings with gated, human-confirmed fixes. Use when the user wants to audit quotes, review quotes, check quotes against principles, check whether a race's questions are genuinely rankable, audit the quotes in the DB, or run a quote audit.
---

# Audit Quotes

Audit already-curated quotes in `essentials.quotes` (the **ev-accounts** DB) against
`essentials/docs/QUOTE-CURATION-PRINCIPLES.md`. By default the audit sweeps **all live quotes
across all races** — narrower scopes (a candidate, a topic, explicit ids) are opt-in. It runs a
free mechanical pass, fans out a **per-quote** judgment pass per race, then a **per-set**
comparability pass over that race's questions, runs a portfolio (coverage-skew) pass, and renders
a consolidated report. The two judgment passes are ordered deliberately — the per-set one runs
strictly after, so contrast cannot leak backward into which quote gets chosen.
Any fix is dry-run first and applied only after explicit user
sign-off. Pairs with `publish-quotes` (which sources and inserts new quotes) — this skill reviews
what's already there.

## Workflow

- [ ] **Read principles + catalog first.** `essentials/docs/QUOTE-CURATION-PRINCIPLES.md` (the
      *why*) and this skill's [CHECKS.md](CHECKS.md) (the *mechanics* — findings schema, the
      mechanical checks, the eleven per-quote judgment checks and the two per-set ones, both
      judgment-agent prompt templates, and the portfolio instructions). If the two ever disagree,
      the principles doc wins.
- [ ] **Resolve scope + confirm.** Run `scripts/audit.py` with the user's scope (default: no
      flags, all races). It prints a `SCOPE:` line and `MECHANICAL FINDINGS: N`, and writes
      `.runs/<date>/context/<race>.json` bundles plus `mechanical_findings.json` and
      `mechanical_report.md`. Show the user both printed lines and **confirm before the judgment
      fan-out** — state roughly one subagent per race. The mechanical pass is free and read-only,
      so run it first regardless.
- [ ] **Offer `--include-drafts` when the run decides what to ship.** The per-set pass judges the
      answers a question actually carries. On a default (live-only) run it grades only what already
      shipped; with drafts in scope it judges each candidate's *best available* answer and tells
      you which pairing is worth selecting. Turn it on for any run whose purpose is deciding what
      to ship — that is most runs on a race with a large draft pool and sparse live selections,
      which is the common case. Drafts cost no network I/O, just a larger bundle.
- [ ] **Offer `--verify-written`** (a.k.a. `--verify-sources`). Without it, quotes from candidate
      sites, op-eds and news articles are never compared to their cited source — the gap that let
      the WI-02 clip through (CHECKS.md §2.2). This is the highest-severity blind spot in the
      pipeline: a quote can be fabricated, paraphrased, or cut to mean the opposite, and every
      other check still passes it. It fetches each cited page (cached, rate-limited), so it is off
      by default: **ask before enabling it on a wide scope**, since a full sweep hits hundreds of
      third-party sites. On a single race it's cheap — just turn it on.
      (`source-nested-quotation` is the exception: it runs on video transcripts and on every
      quote's own text without the flag — see CHECKS.md §2.3.)
- [ ] **Read the casebook.** [CASEBOOK.md](CASEBOOK.md) holds the rulings that bind judgment calls —
      situation → decision → principle, seeded from the reference runs. Pass it to every judgment
      subagent alongside the principles doc. It is **precedent, not background reading**: a case
      that matches an entry is ruled the same way.
- [ ] **Judgment fan-out — per-quote.** For each `.runs/<date>/context/<race>.json` bundle, dispatch
      a parallel `Agent`-tool subagent using the **per-quote** judgment-agent prompt template in
      CHECKS.md §4 (fill in `{context_bundle_json}` with the bundle; the agent also needs the
      principles doc and CASEBOOK.md). Each subagent returns a JSON array of findings (empty array
      = clean for that race). Aggregate these with the mechanical findings.
- [ ] **Judgment fan-out — per-set.** Only **after** a race's per-quote findings are back, dispatch
      one more subagent for that race using the **per-set** template in CHECKS.md §4.1 (fill in
      `{questions_json}` with `bundle["questions"]` and `{per_quote_findings_json}` with that
      race's per-quote findings). It returns `set-incommensurable` / `set-undifferentiated`
      findings at `level: "topic"`. **The ordering is a guardrail, not scheduling convenience** —
      running it before or alongside the per-quote pass lets contrast leak backward into which
      quote gets chosen. Never merge the two prompts.
- [ ] **Portfolio pass.** Per race, apply the CHECKS.md §5 coverage-skew instructions over that
      race's bundle (per-candidate topic coverage, compared across candidates). Append any
      `portfolio`-level `coverage-skew` findings to the aggregate.
- [ ] **Render the report.** Merge mechanical + judgment + portfolio findings and write the
      consolidated report with `scripts/report.py`'s `render(findings, scope_label)` to
      `docs/audits/<YYYY-MM-DD>-quote-audit[-<scope>].md`. Summarize inline for the user: total
      counts by severity and the top races by finding count.
- [ ] **Gated fixes, per race.** For each race with mechanical or guided fixes: build a fixes JSON
      (see op shapes below), run `scripts/apply_fixes.py fixes.json` (default dry-run — transaction
      + rollback, prints a before→after diff), show the user the diff, and re-run with `--commit`
      **only** after explicit user OK. For **guided** fixes, draft the replacement text yourself and
      confirm the exact wording with the user before building the fix op. List every
      **decision-required** finding for the user to resolve manually — never auto-apply those.
- [ ] **Append to the casebook.** For every ruling this run made that [CASEBOOK.md](CASEBOOK.md)
      did not already cover — and for every case where following it felt wrong — propose an entry
      in situation → decision → principle form and confirm it with the user before committing.
      A run that surfaces a **contradiction** with an existing entry is the most valuable outcome
      there is: surface both, never quietly overrule. An overturned ruling is information about the
      rubric, which is the point.

## Running the scripts

Run from the skill directory so the module path resolves; the venv lives three levels up.

```bash
cd .claude/skills/audit-quotes
../../../.venv/bin/python -m scripts.audit                              # default: all live quotes, all races
../../../.venv/bin/python -m scripts.audit --candidate "Steve Hilton" --topic housing
../../../.venv/bin/python -m scripts.audit --ids id1,id2 --include-drafts
../../../.venv/bin/python -m scripts.audit --scope-label "CA governor" --out .runs/ca-gov
../../../.venv/bin/python -m scripts.audit --race RACE_ID --verify-written   # also check written sources
```

Flags: `--race RACE_ID` (scope to one race — both candidates; needed for the portfolio pass on a
single race; find race_ids in a default run's report), `--candidate NAME`, `--topic KEY`,
`--ids id1,id2`, `--include-drafts` (drafts are excluded by default), `--out DIR` (default resolves
relative to the skill, cwd-independent), `--scope-label LABEL` (used in the rendered report heading),
`--verify-written` / `--verify-sources` (**network I/O** — fetch each non-video `source_url` and
verify the quote against the live page: is the text there, whose words are they, and was it cut
defensibly; see CHECKS.md §2.2–2.3).

Fixes file for `scripts/apply_fixes.py` (dry-run by default; `--commit` persists):

```json
[
  {"kind": "set_field", "id": "quote-uuid", "field": "editor_note", "value": "New note text."},
  {"kind": "regex_sub", "id": "quote-uuid", "field": "deidentified_text",
   "pattern": "\\.\\.\\.$", "repl": ""},
  {"kind": "set_live", "id": "quote-uuid", "value": false}
]
```

Allowed `field` values for `set_field`/`regex_sub`: `editor_note`, `deidentified_text`,
`quote_text`, `topic_key`. `set_live` toggles `readrank_selected` and takes no `field`.

```bash
../../../.venv/bin/python -m scripts.apply_fixes fixes.json            # dry-run: shows diff, rolls back
../../../.venv/bin/python -m scripts.apply_fixes fixes.json --commit   # writes for real
```

## Non-negotiables

- **Read-only until the gated fix step.** Every write goes through `apply_fixes.py`'s
  dry-run-then-explicit-OK flow — this is a production DB.
- **The audit is DB-only unless the user opts into `--verify-written`.** That flag is the only
  thing in this skill that reaches the public internet. It only ever issues GETs to already-cited
  `source_url`s, but on a wide scope that's hundreds of third-party sites — confirm first.
- **Never auto-apply `decision-required` findings.** Those are for the user to resolve; list them,
  don't act on them.
- **The report is the primary deliverable.** Even a run with zero applied fixes is a success if
  the consolidated report accurately surfaces what's there.
- **Only `trailing-ellipsis` is a truly mechanical auto-fix** (a regex strip). Note, de-id, and
  partisan-tell fixes are **guided**: draft the replacement text, confirm wording with the user,
  then apply — never rewrite and commit in the same step.
- **Never merge the two judgment passes.** Per-set runs strictly after per-quote so contrast
  cannot influence selection. Merging them to save a dispatch silently breaks the guardrail the
  whole rubric rests on (QUOTE-CURATION-PRINCIPLES §4.6).
- **Agreement is information.** An undifferentiated question is *shown as convergence*, not
  dropped and not sharpened. Never hide agreement to make a race look sharper, and never suggest
  sourcing or swapping a quote in order to create contrast.
- **Demotions are policy, not automation.** Neither per-set check flips `readrank_selected`; any
  change goes through `apply_fixes.py`'s dry-run and explicit OK like every other write.
