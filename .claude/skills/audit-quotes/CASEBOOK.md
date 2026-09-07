# Casebook — rulings that bind the audit

Precedent for `audit-quotes`. [CHECKS.md](CHECKS.md) says *what* each check looks for; this says
*how it was actually ruled* in cases the rule alone did not settle. Read it before a judgment pass;
append to it after one.

Every entry is **situation → decision → principle**. The principle is the transferable part; the
situation is what lets you tell whether a new case is really the same one.

**How to use it.** Find the nearest precedent before ruling. If your case matches, rule the same way
and cite the entry. If it differs in a way that matters, say how — and propose a new entry. If a new
case *contradicts* an entry, do not quietly overrule it: surface both to the human. An overturned
ruling is a signal about the rubric, which is the point.

**What belongs here.** Rulings that were genuinely contested, where the rule alone did not settle
it. Not restatements of `CHECKS.md`, and not routine applications.

**Sources.** Seeded from
`on-the-record/docs/superpowers/specs/2026-07-23-readrank-comparability-model.md` §11 (TX Senate 2026,
the first reference run; then AZ-01, MI Gov R, MI Senate Dem, KS Senate Dem) and from the LA Mayor
provenance pass of 2026-08-07.

---

## Responsiveness and set membership

### Off-question vs non-rival are different failures
- **Situation.** Two candidates both speak to a topic, but their answers don't line up.
- **Decision.** Diagnose the cause first. *Off-question* — the quote doesn't answer the question →
  the candidate is **absent**. *Non-rival* — both genuinely answer, but no shared axis exists →
  **split into two questions**; neither quote is at fault.
- **Principle.** Absence and splitting fix different problems. Applying the wrong one either
  silences a candidate who did answer, or manufactures a comparison that was never there.

### Record ≠ position
- **Situation.** Paxton on immigration enforcement: everything on the record is what he *did*, or
  attacks on opponents. No forward statement.
- **Decision.** **Absent** from the question. De-identification may anonymise a blame target
  ("Biden" → "the previous administration") but **cannot manufacture a stance** from a critique.
- **Principle.** A critique is not a position. Laundering record into a pseudo-position is worse
  than an empty cell.

### Evasion is absence
- **Situation.** MI Senate Dem primary: Stevens was on stage and deflected every question.
- **Decision.** **Absent** — no forward position, despite being present and speaking at length.
- **Principle.** Presence in the room is not an answer. Being on stage does not earn a cell.

## Faithfulness

### Misleading-verbatim is rejected
- **Situation.** Paxton on campaign finance: *"$3,500 limit only on the challenger side"* —
  accurately transcribed, but the cap applies to everyone.
- **Decision.** Not ranked, despite being verbatim.
- **Principle.** **Verbatim is a floor, not a defence.** Judge what a citizen takes away from the
  blind card against what the full passage supports. Audited as `misleading-verbatim`.

### A quote with no provenance is not a quote
- **Situation.** LA Mayor: 30 quotes carried `source_url IS NULL`, blank `source_name` and blank
  `editor_note` — a bulk extraction that bypassed `publish-quotes`. The text was real and, as it
  turned out, accurate.
- **Decision.** Unusable until traced. All 30 were traced to one debate and backfilled; anything
  untraceable would have been retired, not guessed.
- **Principle.** The reveal *is* the provenance. Plausible text with no verifiable source cannot
  ship, however good it looks — and a wrong citation is far worse than a missing one.

### When two rows capture the same moment, keep the one that carries the mechanism
- **Situation.** LA Mayor: four rows duplicated another row's moment from the same debate. Two of
  the better versions were the *newer, uncurated* rows; two were the older curated ones.
- **Decision.** Chose on content, not age. Kept Bass's *"Everybody needs to go inside"* opener
  (without it she states only what she opposes) and Raman's *"You don't get an opportunity to say
  no"* (the mechanism distinguishing her from Bass). Merged the dropped row's `editor_note` onto
  the survivor first.
- **Principle.** Deduplicate on which row lets a citizen see what the candidate would *do*. Seniority
  and prior curation effort are not tie-breakers; the mechanism is. Carry the rationale across
  rather than discarding it.

## Comparability

### Commensurable-but-undifferentiated → show agreement, don't rank
- **Situation.** TX Senate tariffs: Talarico wants them rolled back, Brown is free-trade. Same axis,
  no gap.
- **Decision.** Show convergence. Do not rank. Do not drop the question.
- **Principle.** **Agreement is information.** Never hide it to make a race look sharper, and never
  hunt a sharper quote to manufacture a gap.

### Differentiation is observed, never engineered
- **Situation.** The standing temptation, once a set looks flat, to swap in a spicier quote.
- **Decision.** Select each candidate's most faithful answer first; diagnose contrast only after.
- **Principle.** Contrast may never influence selection. This is why the audit's per-set pass runs
  strictly after its per-quote pass — the ordering is the guardrail, not merely the intention.

### Convergence at scale still collapses
- **Situation.** KS Senate Dem primary: 8 candidates live on one healthcare question, nearly all
  "restore ACA / expand Medicaid," several of them mere fragments.
- **Decision.** `set-undifferentiated`. Eight near-identical answers collapse to "these converge,"
  not an 8-way rank.
- **Principle.** `rankable = ≥2` has no upper bound, and a large set is not a rich one. Crowded
  same-party primaries are the worst case: many candidates, high agreement, thin sourcing.

### Same goal, different mechanism IS a real choice
- **Situation.** LA Mayor, 2026-05-06 debate, both candidates answering the same moderator question
  on street camping. Bass: *"Everybody needs to go inside. Making it illegal and arresting people is
  not the way to solve this problem."* Raman: *"Yes, people need to go inside. When they're offered
  shelter, they go inside. You don't get an opportunity to say no."*
- **Decision.** **Rankable.** Commensurable (both on how compulsory the move indoors should be) and
  differentiated (Raman adds compulsion; Bass rejects criminalisation).
- **Principle.** A shared goal is not undifferentiation. When candidates agree on the destination,
  the **mechanism** is the axis — and it is often the most decision-relevant difference available.

### Articulacy is not a differentiation signal
- **Situation.** Two candidates at effectively the same point on an axis, one of them markedly more
  fluent.
- **Decision.** `set-undifferentiated`. The fluency gap is not contrast.
- **Principle.** Ranking that set would measure rhetoric rather than policy. Symmetrically, a blunt,
  plainly-worded genuine difference *is* differentiated. Fluency is never evidence in either
  direction.

## Provenance and inclusion

### Directness of answer, not medium
- **Situation.** The old ladder ranked a Vote411/LWV questionnaire as low-tier because it is written
  and self-published.
- **Decision.** Questionnaires are **level 1**. Identical prompts across candidates, and every
  ballot-qualified candidate is invited.
- **Principle.** Provenance is *how directly this answers this question*, not what medium it arrived
  in. Questioner independence breaks ties **within** a level; it does not set the level.

### Inclusion is protected at origination
- **Situation.** AZ-01: the Libertarian (Alponte) had no substantive spoken media at all — a channel
  with two dog videos.
- **Decision.** Included via campaign-site pledges with a provenance label, **and** allowed to
  originate Libertarian-specific questions the majors then answer.
- **Principle.** Protect third parties by **whose questions get asked**, never by padding their
  answers. Never manufacture a weak answer to fill a cell.

### Salience exception
- **Situation.** Abortion in TX Senate: no spoken policy statements existed, only theological
  sparring — but it is among the race's highest-salience issues.
- **Decision.** Accurate, unambiguous, genuinely opposite *scraped* positions were kept.
- **Principle.** On a race's highest-salience question, a scraped-but-unambiguous pair can beat
  silence. Narrow: requires high salience, accuracy, **and** genuine opposition together.

## Forum structure

### One common multi-candidate forum beats several separate ones
- **Situation.** MI Governor R primary: a single debate with all three candidates answering the same
  moderator questions yielded 4 rankables (two of them 3-way). AZ-01's two party-segregated primary
  debates yielded 2 — the general-election candidates were never asked the same question in the same
  room.
- **Decision.** Prioritise sourcing from common forums; treat cross-debate matching as lossy.
- **Principle.** Comparability is created in the room. A shared prompt is worth more than two good
  separate interviews.

### A primary debate can still be a general-election asset
- **Situation.** LA Mayor: the only debate is a *primary* debate with three participants, one of
  whom (Pratt) is not on the November ballot. No Bass–Raman head-to-head has occurred.
- **Decision.** Use it. Both November candidates answered the same moderator questions in the same
  room; the third participant's answers are simply out of scope.
- **Principle.** What matters is whether **the candidates who remain** answered a shared prompt —
  not whether the event was billed as the right stage. Contrast with AZ-01, where the primary
  debates were *party-segregated*, so the general's candidates never shared a prompt at all. That is
  the distinction: shared room, not shared season.

### Debates are a QUESTION source even when the answers aren't usable
- **Situation.** AZ-01: third parties weren't invited to the debates.
- **Decision.** Harvest the debate's *questions* as the yardstick; source *answers* from the
  questionnaire, which everyone is invited to.
- **Principle.** The inclusive answer source is the questionnaire, never the debate. A debate's
  durable asset is its question bank.

### Mis-curation, not missing material, is often the bottleneck
- **Situation.** MI Governor R: the live quotes were ontheissues.org scrapes while the real spoken
  debate answers sat as drafts under shared question ids. LA Mayor: 30 debate quotes sat unusable
  with no `source_url` while the debate itself was already ingested in `meetings.meetings`.
- **Decision.** Re-select before re-sourcing. MI flipped 4 rankables live in minutes with no new
  sourcing at all.
- **Principle.** Check what you already have before hunting. The cheapest rankable question is one
  whose material is already in the database.
