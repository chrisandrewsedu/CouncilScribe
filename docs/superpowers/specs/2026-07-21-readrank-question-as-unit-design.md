# Read & Rank: the Question as the Unit of Comparison — Design

- **Date:** 2026-07-21
- **Status:** Draft for review
- **Scope:** Foundation only — the data model, question-surfacing rules, and coverage grid that make quotes *comparable by construction*. Two things are explicitly deferred to their own specs (see [Deferred](#deferred-to-follow-on-specs)): the discovery/clustering *automation quality*, and the *compelling/contrast* selection layer.

## Problem

Read & Rank shows citizens blind, de-identified candidate quotes grouped for comparison, and lets them rank the quotes without knowing who said what. Today quotes attach to a **compass topic** (`topic_key`), and rankability is judged at the (race, topic) level. Curating the 2026 U.S. Senate Texas race exposed three distinct pains:

1. **Comparability.** A topic is a *browsing bucket*, not a guarantee that two quotes answer the same thing. Under `economic-development` we had Talarico on legalizing gambling to fund schools, Paxton on low regulation attracting business, and Ted Brown on deregulating business formation — three real positions, none *comparable*, because no one asked them the same question. Topic overlap produced few genuinely rankable comparisons even after days of sourcing.
2. **Scale.** Sourcing was a per-candidate, per-topic hunt across many interviews, hoping answers would overlap. That does not scale to hundreds of races; you can't spend days each, and you can't see how thin a race is until you're already deep in it.
3. **Inclusion.** If comparability is sourced from debates/forums/news interviews, minor-party and independent candidates — who often aren't invited — get structurally silenced. That is the opposite of the mission: informing voters about *all* their options, non-partisan.

The common root: **the unit of comparison is wrong.** It should be the **question**, not the topic.

## Goals

- Make the **question** a first-class entity; define and *measure* comparability at the question level.
- Let questions be **hyper-local** to a race (gambling in TX, an AI data center where one is actually being built) without hand-authoring every race — questions **emerge from what candidates actually said**, with debate/moderator questions folded in when they exist.
- Guarantee **inclusion**: minor-party/independent voices are surfaced and can even *originate* questions; comparability never becomes a filter that silences them.
- Enforce **faithfulness**: a derived question must be fair to the whole field, and no candidate's quote is ever stretched to "answer" a question it doesn't genuinely address.
- Make a race's comparability state **visible up front** (the coverage grid) so human effort flows to the highest-leverage gaps, and so many races can be triaged, not just one at a time.

## Non-goals (this spec)

- The *quality* of the automated discovery/clustering that populates candidate answers — designed here at the interface level, deferred in depth.
- The *compelling/contrast* signal (do the answers genuinely diverge, so ranking is enjoyable) — a later property on rankable questions.
- Results-page redesign — how (if at all) topics appear there is an open UX question, not settled here.

## Core principles

- **The question is the unit.** Read & Rank navigates by question. Comparability = candidates answering *the same question*.
- **Topics are Compass coupling, not an organizing axis.** `topic_key` is retained so a quote can feed the compass card on the politician profile; in Read & Rank it is at most a small chip, never the navigation frame.
- **Rankable vs. surfaced.** A question is *rankable* (enters the ranking game) only with **≥2** candidates answering; a single candidate's answer is still *surfaced* (≥1) on the race/profile page as "here's where they stand." Being the only voice on a question is honest presence, not exclusion.
- **Questions can originate from anyone.** An emergent question raised by an independent is a first-class citizen alongside a debate moderator's question. When a candidate raises something, we go *seek the other candidates' answers to it* — filling the grid in the inclusive direction, not only chasing front-runner questions.
- **Absence beats distortion.** If a candidate did not genuinely address a question, they are marked absent from it — never force-fit to manufacture a comparison.
- **Two faithfulness gates** (see [Question surfacing](#question-surfacing)) bound question-crafting and quote-attachment.

## Data model

Grounded in the existing schema (`essentials.quotes`, `essentials.readrank_race_topic_questions`, `essentials.races`, `essentials.race_candidates`, `inform.compass_topics`).

### New table: `essentials.readrank_questions`

One row per question. **Multiple rows allowed per (race, topic)** — this is what lets several local questions (and the compass default) coexist under one compass topic.

| column | notes |
|---|---|
| `id` | uuid PK |
| `race_id` | uuid → `essentials.races` |
| `topic_key` | parent compass topic (the Compass-coupling backbone); FK to `inform.compass_topics` |
| `question_text` | the ranking question — blind, neutral, on-axis |
| `origin` | `compass` \| `moderator` \| `emergent` |
| `origin_quote_id` | uuid → `essentials.quotes`, nullable — the quote an `emergent` question was derived from (provenance for gate 1) |
| `source_ref` | for `moderator` origin: debate meeting id + timestamp; nullable |
| `status` | `proposed` \| `confirmed` \| `rejected` (curator gate) |
| `created_at`, `updated_at`, `updated_by` | audit |

### `essentials.quotes` change

- Add **`question_id`** (uuid → `essentials.readrank_questions.id`, nullable during migration). **A quote answers exactly one question.**
- `topic_key` **stays** (derivable from the question) for Compass coupling and back-compat.

### Rankability semantics (derived, not stored)

- **Rankable question:** ≥2 *distinct* candidates each have a `readrank_selected = true` quote with this `question_id`.
- **Surfaced question:** ≥1 candidate with a live quote.
- Live selection stays a human pick, now one live quote per **(candidate, question)** (generalizes today's per-(politician, topic)).

### Migration

- `readrank_race_topic_questions` folds in: each existing per-race override becomes a `readrank_questions` row (`status = confirmed`; `origin` = `emergent` or `moderator` as appropriate), preserving `question_text`. The old table is retired once backfilled.
- **Compass floor question:** for a topic with no local question, `inform.compass_topics.question_text` is the implicit default; a concrete `origin = 'compass'` row is materialized the first time a quote attaches to it.
- Existing quotes: `topic_key` unchanged; `question_id` backfilled best-effort (where a race-topic has exactly one question, attach; otherwise left null and completed by a curator via the grid). No quote is silently mis-attached.

## Question surfacing

Questions enter the pool two ways; both land in the same faithfulness-gated review.

- **Emergent (primary engine).** Discover candidate positions from *every* candidate's ingested corpus — including self-produced video, so an independent who was never invited anywhere still generates and answers questions. Cluster positions into candidate themes; where a theme is contested/recurring, it becomes a candidate question.
- **Moderator (folded in).** When a debate/forum exists, its moderator questions enter the pool pre-validated (someone already did the salience work). They *contribute* questions; they never *gate* them. Reuses the meeting-segmentation pipeline's structural segmentation of debates.

### The two faithfulness gates

1. **Question ↔ its origin.** A crafted question must be a *neutral generalization* of what the originating quote engages — never smuggling in that candidate's stance or framing. **Test: could an opponent answer this in the opposite direction without it feeling rigged?** If only the originating framing fits, the question is gerrymandered → rewrite or drop. (This is the existing per-race override discipline — blind, on-axis, the race's real question tightened — applied to emergent questions.)
2. **Question ↔ each attached quote (responsiveness).** Before *any* candidate's quote attaches, it must genuinely answer the question *in context* — the audit's `off-question` check, run at attach-time. A quote that doesn't → the candidate is **absent** from that question, never force-fit.

### Backstops

- **"A moderator could have asked this."** An emergent question must stand on its own as a fair civic question to the whole field. This is also what makes emergent (A) and moderator (B) questions interchangeable — a good emergent question is indistinguishable from a good moderator question.
- **Verifiability.** Every quote keeps its source deep-link and is auditable against the full source passage (the meaning-preservation editing rules in `publish-quotes/EDITORIAL.md`). "Out of context" is checkable by anyone — the ultimate protection for the candidate.
- **Curator confirms** both the question wording (gate 1) and every attachment (gate 2); `status` on `readrank_questions` records the decision.

## Coverage grid

The artifact that makes the model visible and curatable — and the direct fix for "couldn't tell how thin the race was until days in."

**Shape:** a matrix per race. **Rows = questions** (each tagged with its parent compass topic as a small chip). **Columns = candidates** (all of them, equally). Each **cell** is the state of one (candidate × question):

- **live** — a confirmed quote attached
- **draft** — material found and extracted, pending curator confirm
- **not-addressed** — searched this candidate's corpus; they genuinely don't speak to it (a first-class, honest state — "absence beats distortion" made visible)
- **empty** — not yet searched

The `not-addressed` vs `empty` distinction is load-bearing: it separates "no position" from "not looked yet," which is what enables triage rather than guessing.

**Row (question) status derives from the cells:** *rankable* (≥2 live), *near-rankable* (1 live + others draft), *solo* (1 candidate — surfaced, not head-to-head), *proposed* (emergent question awaiting gate-1 confirmation).

**Three jobs:**
1. **Triage to the highest-leverage action** — surface "one candidate away from rankable" and "this candidate's column is mostly empty," so effort converts to rankable comparisons fastest.
2. **Inclusion with teeth** — a minor candidate's sparse column is *visible*; under-coverage becomes a deliberate decision, never a silent drop-out.
3. **Shippability + cross-race triage** — a per-race metric (rankable questions ≥ N) shows which races are near-shippable vs. genuinely too thin, so human time can be spread across hundreds of races by looking at the grid first.

**Where it lives:** a derived view over `readrank_questions × race_candidates × quotes` (not a new source-of-truth table), rendered as the evolution of today's `/admin/readrank-quotes` into a question-first grid. Click a cell → the candidate's found material + source deep-link, to confirm/edit/reject. Click a row → confirm/rewrite the question (gate 1) and review its attachments (gate 2).

## Relationship to existing infrastructure

- **`inform.compass_topics`** — remains the national backbone; supplies the floor/default question per topic and the Compass coupling.
- **Meeting-segmentation pipeline** — its structural segmentation of debates/forums feeds moderator-origin questions and candidate answer-units.
- **`publish-quotes` / `audit-quotes`** — adapt so a quote targets a **question** (not just a topic): publish attaches `question_id`; audit's `off-question` check *is* gate 2, run at attach-time; the existing de-id / faithfulness discipline is unchanged.

## Deferred to follow-on specs

- **Discovery/clustering automation quality** — the algorithm that turns a corpus into candidate answer-units and clustered questions. Designed here at the interface (what it produces: `draft` cells + `proposed` questions); its accuracy is its own spec.
- **Compelling/contrast layer** — a signal on rankable rows for whether answers genuinely diverge (so ranking is enjoyable, not agreeable-goal mush). A later property on the grid.

## Open questions

- **Results page & topics** — whether the results page links to topics at all, or stays purely question-oriented.
- **Shippability threshold** — what `N` rankable questions makes a race worth publishing.
- **Compass default vs. local** — whether a race should always carry the compass floor questions, or only questions that actually lit up locally.
