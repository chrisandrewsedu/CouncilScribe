# Read & Rank — Comparability & Question-Pass Model (shared understanding)

Status: **design/shared-understanding** (seed for a future `read-rank-question-pass` skill; not yet built).
Origin: grill session 2026-07-23, after the TX Senate 2026 quality pass (first reference run).
Companion: `2026-07-21-readrank-question-as-unit-design.md` (the data model this rides on).

This captures *what we agreed*, so the eventual skill and its rubric can be built from it. It is
deliberately a **living** document: the model gets better by running races that surface new edge
cases, not by being finished up front.

---

## 1. Purpose hierarchy (decides everything downstream)

- **B — inform the vote — is primary.** Help a citizen find who genuinely represents them.
- **A — teach the ranked-choice muscle — is secondary.** A welcome side-effect, *never* a reason
  to force a ranking.
- When they conflict, **B wins.** A forced or misleading ranking *feels* informative while
  misleading the citizen — worse than honestly showing an un-rankable spread, and it costs the
  return visit. Teach RCV on the questions that genuinely *are* rankable.

## 2. Read & Rank is one lens, not the whole picture

- **Essentials** = bio / record. **Compass** = positions-as-record. **Read & Rank** = *policy
  philosophy in the candidate's own words.*
- Therefore R&R is **allowed to be honestly incomplete.** Completeness lives in the other tools;
  the results page can hand the citizen off to them ("compare these candidates on other areas").
- R&R is *not* responsible for covering every candidate on every question, nor for forcing RCV
  practice onto non-rankable sets.

## 3. Two states only — rank or surface (no "compare")

- **Rankable** — ≥2 answers that are both **commensurable** and **differentiated** (see §4).
- **Surfaced** — ≥1 lone voice shown for inclusion, not ranked.
- There is **no third "compare" mode.** If two answers are same-topic but genuinely
  incommensurable, that's the signal they're **different questions** → split them; each half then
  becomes rank-or-surface.

## 4. The rankability test (per set)

A set is rankable when its answers **sit at different points on a shared latent dimension, such
that a preference between them is *meaningful*.** Two properties, both required:

- **Commensurable** — a shared axis exists. NOT "same words," NOT "mutually exclusive."
  - Passes: healthcare coverage (Talarico) vs supply (Brown) — both are positions on *how big a
    role government should play*; "I lean smaller-government" is a meaningful preference.
  - Fails (→ split): dimensionless pairs like "cap insulin prices" vs "build more medical schools"
    — no shared axis, no meaningful ordering.
- **Differentiated** — real distance on that axis. Commensurable-but-identical is not rankable.
  - Fails: tariffs (Talarico roll-back vs Brown free-trade — both anti-tariff). Same axis, no gap.
  - Undifferentiated ≠ failure: **agreement is information.** Show "these candidates converge here";
    just don't rank it.

## 5. Inclusion is protected at origination, not by answer-forcing

- Rank whoever genuinely answered (≥2). Mark non-answerers as an explicit, neutral **"hasn't
  answered this yet"** — never a silent gap. (Doubles as a soft incentive for candidates to answer.)
- Balance the **question bank** so independents/third parties **originate** questions — harvest from
  their *own* channels, not just the debates/major interviews that are two-party-shaped. That way the
  independent isn't perpetually the empty cell; they also anchor questions the majors may be absent on.
- **Never manufacture a weak answer to fill a cell.** A record-only or non-responsive candidate is
  *absent* (see casebook: "record ≠ position").

## 6. The pipeline is question-first

1. **Harvest questions** from all media — what moderators *asked* + what candidates *emphasized
   unprompted* — as the best available signal of the live, local questions (gambling in TX, not NV).
   Include independents' own channels. Media is primarily a **question** source, not an answer source.
   Neutralize/blind any moderator framing before a question becomes the ranking question.
2. **Source answers** per candidate across every source. **Provenance hierarchy:**
   `answered-this-question  >  answered-an-adjacent-question  >  curator-extracted`.
   The old "spoken > scraped" instinct is really *this* — directness of answer, not medium.
   - **Questionnaires (Vote411/LWV, Ballotpedia surveys) are gold:** identical questions across
     candidates *and* they invite every ballot-qualified candidate (third parties included). Solves
     comparability + inclusion at once. **Mine existing ones now** (piggyback convening power we don't
     have yet); our own questionnaires get <1% response today — earn that later.
3. **Grade** with the rubric (§7). 4. **Route** to rank / surface / split.

## 7. The rubric — 7 dimensions, two layers

**Per-quote** (judge each answer alone):
- **Responsive** — answers *this* question, not just the topic.
- **Faithful** — verbatim-accurate *and not misleading even when accurate*; de-id preserves the
  position (never launders record/blame into a stance).
- **Substantive** — shows the *how*/mechanism, not just an agreeable goal.
- **Provenance** — where it sits in the §6 hierarchy.

**Per-set** (judged across candidates, *only after* faithful per-quote selection):
- **Commensurable** — shared latent axis (§4).
- **Differentiated** — real distance on it (§4).
- **Inclusive origination** — is the question bank balanced so independents aren't always absent (§5)?

The **layering enforces the differentiation guardrail:** per-set checks run *after* each quote was
picked faithfully on its own, so contrast can never leak backward into selection.

## 8. The differentiation guardrail (observe, never engineer)

- **Contrast is observed, never engineered.** Select each candidate's *most faithful* answer first;
  *then* look at the set and ask "is there real difference?" Never pick quotes *because* they contrast.
- Genuine difference → rank (satisfying *and* honest). Genuine agreement → show convergence, don't rank,
  and **never hide agreement to make a race look sharper.**
- Difference often lives in the **how** — when candidates share a goal, the mechanism is where they
  diverge. So hunting genuine differentiation is often hunting the mechanism-revealing quote (ties to
  *Substantive*).

## 9. Open ideas (noted, not decided)

- **Provenance labels in the ranking UI** (e.g. "debate answer" / "questionnaire" / "campaign site").
  A scraped bullet *feels* different from a real answer; the label tells the citizen *how directly*
  this person answered the question. Revisit as R&R UX.
- **Results-page handoff** to Compass/Essentials for the completeness R&R deliberately doesn't carry.

## 10. Scaling — the bootstrapping strategy

- **Current floor:** the human reads the transcripts. We cannot yet sign off on a rankable set
  without reading the source. That's fine — it's the starting point, not the end state.
- **Goal architecture:** two agents — one **sources**, one **rates** (against the rubric) — with the
  human spot-checking and catching the problematic ones.
- **Trust is earned, not assumed.** The guidelines only get good enough by running rounds that
  surface what breaks. The rubric's job is to **convert "vibes" into teachable rulings**; the residue
  that won't convert is the frontier where the human stays longest — that's a feature.
- **Graduation is per-pattern, not all-at-once**, and it's a rubric + retained qualitative judgment
  (a measurable-ish overturn signal, not a single number). Trust the agent on clean rival questions
  long before salience-exceptions or misleading-verbatim catches.
- **Tuning strategy:** do **not** perfect any single race now. **Mine** races for edge cases, using
  fresh/different races as examples; once the guidelines mature, **re-run all races** with them.
- **Post-mortem ("what went wrong") is a required step each round** — it's the learning engine.

## 11. The casebook (rulings seeded this session)

Store every ruling as **situation → decision → principle**, so the agent reasons by precedent and the
human audits reasoning without re-reading source. Seeded from the TX Senate run:

- **Off-question vs non-rival** — two different failure modes. Off-question → absent. Non-rival
  (responsive but no shared axis) → split into separate questions.
- **Record ≠ position** — a candidate who only speaks in record/attacks has no forward position →
  absent. De-id can anonymize a blame target ("Biden"→"the previous administration") but **cannot
  manufacture a stance** from a critique. (Paxton, immigration enforcement.)
- **Misleading-verbatim rejected** — verbatim ≠ fair. A quote that misleads the reader is not ranked
  even if accurately transcribed. (Paxton, campaign finance: "$3,500 limit only on the challenger
  side" — the cap applies to everyone.)
- **Salience-exception** — on the race's highest-salience issue, accurate/unambiguous/opposite
  *scraped* positions may be kept when no spoken policy exists (only theological sparring). (Abortion.)
- **Provenance hierarchy** — answered-this-question > adjacent > curator-extracted; questionnaires gold.
- **Origination-stage inclusion** — protect third parties by whose questions get asked, not by padding.
- **Observe-don't-engineer differentiation** — faithful selection first, contrast diagnosed after.
- **Commensurable-but-undifferentiated = show agreement, don't rank.** (Tariffs.)

## 12. Empirical status

- **TX Senate 2026** = first reference run. Produced **7 rankable questions** (housing 4-way; legal
  immigration 3-way; redistricting, enforcement, voting, healthcare 2-way; abortion 2-way on the
  salience exception), most rebuilt from real spoken interviews replacing scraped bullets.
- Its value was **discovering the model + seeding the casebook**, not being "finished." Next races
  are for mining more edge cases until the rubric + casebook are trustworthy enough to graduate work
  to the sourcer/rater agents. See memory `tx-senate-quality-pass`.
- **AZ U.S. House District 1 (2026)** = race 2, run under real down-ballot scarcity. Cold start (no
  ingested media, one Feely campaign-site quote). Ingested the two AZ Clean Elections *primary*
  debates (Shah's D debate, Feely's R debate — same moderators). **Yield: only 2 clean rankables**
  (immigration/border; elections). Findings: (a) primary debates are *party-segregated* so the two
  GE candidates never answered the same question in the same room — cross-debate matching is lossy;
  (b) primary framing skews answers to blame/signaling, not mechanism; (c) the Libertarian (Alponte)
  had **zero substantive spoken media** (channel = 2 dog videos) → included via campaign-site pledges
  with a provenance label + let her originate Libertarian-specific questions; (d) **timing**: sourcing
  a race pre-general-debate is premature — the inclusive common forum (GE debate / Vote411 GE
  questionnaire) hasn't happened yet. Lesson: **debates are a QUESTION source even when third parties
  aren't invited** (a serious third-party candidate answers those same questions elsewhere; the
  debate's questions become the yardstick). The inclusive *answer* source is the **questionnaire**,
  never the debate. Harvested AZ-01 question bank is durable — bank it, fill answers at GE time.
- **MI Governor R primary (2026)** = race 3, the inverse of AZ and the cleanest validation. A single
  multi-candidate primary debate (James/Cox/Johnson, YouTube h7ZSBAcF3Ww) was **already ingested**,
  with all candidates answering the *same* moderator questions — but mis-curated: the LIVE quotes
  were ontheissues.org **scrapes** (James had 8 scraped bullets live) while the real spoken debate
  answers sat as drafts under shared question_ids. The pass was **pure re-selection** (no re-sourcing,
  compass question wording already fit): flipped **4 rankables live in minutes** — data-centers
  (3-way), transportation (3-way), taxes (Cox/Johnson 2-way, James's scrape dropped), housing (2-way);
  James's other scrapes left as surfaced solos. **Takeaway: a single common multi-candidate forum
  yields far higher comparability (4, incl. two 3-ways) than separate/absent forums (AZ's 2), and the
  bottleneck is often mis-curation, not missing material.** `data-centers` recurred in both AZ and MI
  → seeds the cross-race question library.
- **MI Senate Dem primary (2026)** = race 4, the *extract* path (debates ingested but quotes never pulled). Sourced from the 05-29 all-three debate. Yielded 2 clean 2-ways (El-Sayed vs McMorrow: healthcare M4A-vs-protect-ACA, taxes wealth-tax-vs-end-corporate-subsidies). New edge cases: **loose panel-debate format** (open discussion + interruptions) is harder to extract cleanly than structured Clean-Elections debates; **candidate evasion → absent** (Stevens deflected every question → no forward position → absent despite being on stage); and **a salient question can be un-publishable for lack of a compass topic** — the standout Israel-aid 2-way couldn't be created because `readrank_questions.topic_key` FKs to `inform.compass_topics` and there's no Israel/foreign-policy topic (only Ukraine). *Implication: the "question is first-class" claim is still gated by the compass-topic spine; either add topics via compass-topic-builder or relax the FK for emergent questions.*
- **Kansas Senate Dem primary (2026)** = race 5, the **crowd stress-test** and a cautionary result. An 11-candidate same-party primary produced **8 candidates live on one healthcare question** (5-way immigration, 4-way abortion/tariffs). All currently sourced from **campaign-site bullets**; there's no single 11-way debate (you can't stage 11 — itself a symptom of the crowd problem), but a **WIBW per-candidate interview series covers 9 of 11** (spoken, structured, cross-matchable) plus KPR + the party convention — so spoken sourcing IS possible, just heavy (WIBW is Gray-TV-hosted → KXAN-style stream extraction ×9). The 8 healthcare answers **largely converge** ("restore ACA / expand Medicaid"), several just fragments ("Expand Medicare and Medicaid"). *Implications for the model: (a) the `rankable = ≥2` rule has **no upper bound** — an 8-way blind ranking is unusable UX; crowded fields need a cap / top-N / different presentation; (b) **differentiation must be enforced at scale** — 8 near-identical answers should collapse to "these converge," not an 8-way rank; (c) crowded same-party primaries are the worst case (many candidates + high agreement + often campaign-site-only sourcing).*
- **⚠️ OPERATIONAL HAZARD (2026-07-23):** running background task sessions on the *same working checkout* as live curation is dangerous — a concurrent task (`ac84471`) git-cleaned the tree and **reverted the uncommitted Plan-2 question-attachment in `insert_quotes.py`**, so the publish script silently stopped setting `question_id` mid-session (quotes went in live-but-unattached; found + fixed via SQL; script restored + committed as `b355e36`). Lesson: commit pipeline changes, and isolate background agents in worktrees.
- **Scale strategy crystallized:** front-load the QUESTION harvest (persistent, reusable across
  cycles/races) so the GE-forum crunch is answer-sourcing only; the two-agent (sourcer + rater)
  architecture is the goal but trust is earned per-pattern via the rubric; graduation ≠ a single
  metric (qualitative residue stays human); mine races for edge cases, don't perfect any one, re-run
  all once mature.
