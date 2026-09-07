# Check catalog for audit-quotes

> **Principles live in** `essentials/docs/QUOTE-CURATION-PRINCIPLES.md` — the canonical *why*
> behind selection, editing, sources, anonymity, the Compass coupling model, and accountability.
> This file is the audit's *mechanics*: what each check looks for, how severe it is, and how it
> gets fixed. If the two disagree, the principles doc wins and this file should be updated.

The audit runs in three passes over `essentials.quotes` (scoped to a race, a candidate, a topic,
or explicit ids):

1. **Mechanical** — `scripts/checks.py`, deterministic, no model in the loop.
2. **Judgment** — a Claude agent per race (or race×topic), reading the context bundle and
   applying the checks in §3 below. It runs in two stages, in this order: **per-quote** (§4), which
   judges each quote alone, then **per-set** (§4.1), which judges each question's answer set. The
   ordering is a guardrail — see §4.1.
3. **Portfolio** — a per-race skew pass (§5) over the combined mechanical + judgment findings.

All three passes emit the same `Finding` shape (`scripts/models.py`).

## 1. Findings schema

Every check — mechanical or judgment — produces zero or more findings. A judgment agent's output
is a **JSON array of finding objects**; an **empty array means clean** (no findings for that
scope). Each finding object has these fields:

| Field | Type | Meaning |
|---|---|---|
| `check_id` | string | Short id, e.g. `"note-missing"`, `"is-attack"`. Matches an id in §2 or §3. |
| `level` | `"quote"` \| `"topic"` \| `"portfolio"` | What the finding is about — a single quote row, a race×topic group, or a whole race's topic coverage. |
| `principle` | string | Short human phrase naming the rule, e.g. `"forward-looking, not retrospective"`. |
| `severity` | `"high"` \| `"medium"` \| `"low"` | How much this undermines the quote/topic if left as-is. |
| `fix_class` | `"mechanical"` \| `"guided"` \| `"decision-required"` | Who can resolve it: a script, a curator following a suggestion, or a curator making a judgment call. |
| `what` | string | What's wrong, in plain language, specific to this quote/topic. |
| `suggested_fix` | string | What to do about it — a proposal, not an instruction to auto-apply. |
| `quote_id` | string \| null | The `essentials.quotes.id` this finding is about (quote-level findings). |
| `topic_key` | string \| null | The topic this finding is about (topic- and quote-level findings). |
| `race_id` | string \| null | The race this finding is about. |
| `candidate` | string \| null | The candidate's full name, where applicable. |

Judgment agents **do not** set `fix_op` — that field is reserved for mechanical checks whose fix
can be applied by a script (`scripts/apply_fixes.py`) without human judgment. Leave it absent (or
`null`) on every finding a judgment agent returns.

## 2. Mechanical checks

Detected deterministically by `scripts/checks.py` — no model in the loop, no ambiguity. Only
`trailing-ellipsis` carries an auto-applied `fix_op` (a regex strip); the rest are flagged for a
human to resolve even though the *detection* is mechanical.

| id | level | principle | severity | fix-class |
|---|---|---|---|---|
| `note-missing` | quote | `editor_note` required | high | guided |
| `note-section-ref` | quote | notes are self-contained (no §-refs/jargon) | medium | guided |
| `note-too-long` | quote | `editor_note` ≤ 3 sentences (2 preferred) | low | guided |
| `deid-missing` | quote | blind text required | high | guided |
| `trailing-ellipsis` | quote | no trailing ellipsis | low | **mechanical** (auto-fix: regex strip) |
| `partisan-tell` | quote | no partisan/side tell on blind card | high | guided |
| `invalid-source` | quote | cite the ORIGINAL source, not an aggregator (ontheissues.org, wikipedia.org) — **re-attribute** | high | decision-required |
| `unquotable-source` | quote | quiz/questionnaire sites (isidewith.com) publish nothing quotable — **delete** | high | decision-required |
| `scorecard-source` | quote | a scorecard publishes votes and ratings, not utterances — **re-source** | high | decision-required |
| `stance-label` | quote | a quote must state a position, not name a topic (≤4 words) | medium | decision-required |
| `multiple-live` | topic | one live quote per candidate per topic | high | decision-required |
| `not-rankable` | topic | ≥2 candidates to be rankable | medium | decision-required |

### 2.1 The three bad-source classes

`invalid-source`, `unquotable-source` and `scorecard-source` are deliberately separate checks
because their remedies differ.

- **`invalid-source` — aggregator; an original exists, so re-attribute.** ontheissues.org and
  wikipedia.org restate or paraphrase something the candidate actually said elsewhere. The quote is
  badly *cited*, not fabricated: follow the aggregator down to the speech/interview/vote/statement
  and re-source it. Deselect from live only until the original is found. (Method and its gotchas:
  `docs/audits/2026-07-25-ballotpedia-triage.md` §"Method notes".)
- **`unquotable-source` — quiz site; no original exists, so delete.** A quiz/questionnaire
  comparison page such as isidewith.com publishes, side by side under one candidate's name: (a)
  canned multiple-choice option text nobody ever said ("No, this is a violation of free speech"),
  (b) third-party aggregates (rows labelled `PARTY'S SUPPORT BASE` — surveyed party-affiliated
  voters), and (c) AI-generated positions (rows labelled `CHATGPT` — 87 of 397 stance rows on one
  candidate's page). So **no row on such a page is quotable regardless of which row it came from**,
  including the candidate-labelled rows: those are still canned option text, not utterances. There
  is nothing to descend to. All 29 isidewith.com rows were hard-deleted on 2026-07-25 — evidence
  and cost in `docs/audits/2026-07-25-isidewith-purge.md`.

- **`scorecard-source` — a rating, not speech; find the real statement.** An advocacy group's
  legislative scorecard (lcv.org, and the same shape at AFL-CIO, Heritage Action, NRA) publishes
  what a member *did* — a score and a vote record — and never what they said. A quote attributed
  to one therefore came from somewhere else, or from nowhere.

  This check exists because the fetch pass proved it rather than because it sounded right: in the
  2026-08-02 full sweep, **all 30 scorecard-sourced live quotes came back `source-unverified` —
  30 of 30, no exceptions.** Catching them by URL costs nothing and needs no network, so they no
  longer have to wait for `--verify-sources` to run.

  **Two neighbouring classes were tested and deliberately left out**, so they don't get
  re-litigated:
  - *Bill and statute pages* (congress.gov, leg.\*/bills/): 8 live quotes, of which **one
    verified clean**. Bill pages sometimes carry sponsor statements and CRS summaries, so
    "structurally cannot carry a quote" is simply false for them.
  - *Roll-call tallies* (clerk.house.gov/Votes): only 2 live quotes, and both were
    `source-unfetchable`, so the sweep never actually tested the premise. Too thin to justify a
    high-severity rule.

  The pattern is path-anchored (`/scorecard/`, `/congressional-scorecard/`) so it does not match
  a news article whose slug merely contains the word. An earlier draft keyed on `/roll-?call`
  and wrongly matched **rollcall.com — CQ Roll Call, a news outlet**, which is a perfectly good
  source; there is a regression test for that.

**ballotpedia.org is in none of these classes and is not flagged.** It reproduces campaign-site text
verbatim under an attribution line with a footnote to the original (re-attributable case by case),
and its Candidate Connection survey answers are candidate-written *for* Ballotpedia — Ballotpedia
**is** the original publisher there, so the citation is already correct. Treating it as
delete-on-sight would have cost 10 races and 55 topics for no sourcing gain; the 266-row triage in
`docs/audits/2026-07-25-ballotpedia-triage.md` resolved it row by row instead. The real Ballotpedia
defects — stale-cycle answers and text not on the cited page — are curation calls a host-list check
can't see.

### 2.1a `stance-label` — a topic name is not a position

101 live quotes are four words or fewer: *"Universal Healthcare"*, *"Abolish ICE"*, *"Medicare For
All"*, *"Protect the unborn"*, *"Tax the rich"*, *"Theft."*. Read & Rank asks a reader to weigh one
candidate's answer against another's, and a row like this offers nothing to weigh — it names the
topic and stops. No mechanism, no direction beyond the topic itself.

**These are not curator inventions.** That was the initial theory and the data refuted it: **26 of
the 49 shortest verify clean against their cited source**, i.e. the text really is on the
candidate's own platform page. What went wrong is that a *bullet was harvested as if it were an
utterance*. That is why severity is medium rather than high — the text is usually genuine, it is
just too thin to rank — and why the suggested fix is to find a real sentence on the same page
rather than to delete.

**Why a word count, of all things.** It is the only signal here with no judgment in it, which is
the price of admission to the mechanical pass. Distinguishing a bare noun phrase from an
imperative bullet needs part-of-speech tagging (the venv has no `nltk`/`spacy`, and this does not
warrant a new dependency), and hand-rolled verb lists fail immediately on an open word class —
a draft list missed *Achieve*, *Enforce*, *Modernize*, and could not see the verb in
*"Billionaires shouldn't exist."* at all.

Four is calibrated, not guessed: every one of the 101 rows at ≤4 words reads as a slogan or a
platform bullet, while the 5–6 word band already contains real mechanism-bearing sentences
(*"Lower Medicare eligibility to Age 55."*, *"Rein in private insurers managing Medicaid."*).
Raising the bar would start flagging good quotes.

**The ambiguous middle is deliberately left to the judgment pass**, which is better equipped for
it: `source-summary` (§3, high) already covers a platform page rendered as a curator-summarized
bullet list, and `non-differentiating-goal` (§3, medium) covers an agreeable goal stated with no
mechanism. `stance-label` is only the mechanical floor beneath them — the cases so short that no
reading rescues them.

Source-verification checks (`scripts/verify_source.py`) — also deterministic. **Video sources**
(YouTube `source_url`) are matched against the ingested transcript in `meetings.segments`; this
runs always and is DB-only. **Written sources** (campaign issue pages, op-eds, news articles) are
matched against the live page, which costs network I/O, so that path is **opt-in** via
`scripts/audit --verify-written` (`--verify-sources` is accepted as a synonym; see §2.2).

| id | level | principle | severity | fix-class | source kind |
|---|---|---|---|---|---|
| `source-unverified` | quote | quote must appear in its cited source | high | decision-required | both |
| `source-speaker-mismatch` | quote | quote must be spoken by the candidate | high | decision-required | video |
| `source-nested-quotation` | quote | a quote must be the candidate's own words, not words they relay | high | decision-required | both (always on) |
| `source-not-ingested` | quote | source must be verifiable | medium | decision-required | video |
| `source-timestamp-drift` | quote | deep-link should point at the quote | low | decision-required | video |
| `source-midsentence-clip` | quote | an excerpt must not misrepresent by where it is cut | medium | decision-required | written |
| `source-unfetchable` | quote | source must be verifiable | medium | decision-required | written |

### 2.2 Written-source verification (`--verify-written`)

Before 2026-08-01, `check_source` returned `None` for every non-YouTube `source_url`, so a quote
from a candidate website or news article passed the source pass **without ever being compared to
its cited source**. The WI-02 audit found the cost of that gap by hand
(`docs/audits/2026-08-01-quote-audit-wi-house-02.md`): three quotes reported zero source findings,
and one of them was a meaning-altering clip.

With `--verify-written`, the cited page is fetched, reduced to visible prose, and run through the
same span-matching machinery the transcript path uses (`verbatim_runs` →
`longest_verbatim_match`, threshold `MIN_RUN_WORDS`). Matching is punctuation-insensitive by
construction — `normalize` drops `…`, `[bracketed]` insertions and every non-alphanumeric
character, so smart vs. straight quotes, em-dashes and stray whitespace can never cause a false
`source-unverified`, and a quote that elides across `…` is checked as its separate verbatim runs
rather than as one impossible contiguous string. Three things can then go wrong:

- **`source-unverified`** — no distinctive contiguous run of the quote appears on the page. In
  practice this catches curator **paraphrase sold as verbatim** ("Reverse Medicaid cuts and
  restore ACA subsidies" vs. the page's "Reverse the cuts to Medicaid and restore the ACA
  subsidies that hold premiums down") and text that is **simply not on the cited page at all**.
- **`source-nested-quotation`** — the words are in the source and are the candidate's *utterance*,
  but not their *position*: they are relaying what someone else says. Not written-only, and not
  gated on the flag — see §2.3.
- **`source-midsentence-clip`** — the run *is* verbatim, but it starts in the middle of a sentence
  with no `…` marking the cut, so the clause before it may carry the candidate's actual position.
  This is the WI-02 defect, and a plain verbatim check cannot see it. It also catches bare
  noun-phrase fragments stored as quotes ("independent redistricting commission").

The three are ordered by severity, and the nested-quotation check runs **before** the clip check.
A relayed quote is almost always also a mid-sentence cut, so without that ordering the TN defect
below would surface as `source-midsentence-clip` at medium — the wrong defect, understated.

The clip check is deliberately conservative and stays silent when it can't be sure: when the
curator marked the cut with a leading `…`, when the run isn't on the page, when the run opens the
page or a block, when it follows `. ! ? … :`, and — importantly — **when the source itself opened
a quotation right before it**. That last exemption is what keeps ordinary journalism (`…adding
that "a government-paying program is the most moral…"`) from flooding the report: there the cut is
the *source's*, faithfully reproduced. A cut that falls *inside* the source's quotation still flags.

**`source-unfetchable` is not a soft `source-unverified`.** A JS-rendered campaign site, a 403
(Ballotpedia blocks the fetcher), a paywall, or a PDF yields no prose to match against. Calling
that "quote not in its source" would be a false accusation, so it is reported separately, at
medium, meaning *go read the page yourself*.

Two rules keep that boundary honest, both added after the first full sweep put 86 nav-only pages
into the high-severity bucket:

- **Readability is judged by prose, not word count, and only after the match fails.**
  `MIN_PAGE_WORDS` counts every word, so a roll-call tally or a link farm clears it on menu
  labels alone. `prose_word_count` instead sums the words living in blocks of at least
  `PROSE_BLOCK_WORDS` (8). That bar is deliberately low: campaign platform pages are legitimately
  written as short bullets and are where most true positives come from, so a stricter test would
  discard real findings. The check runs *after* the verbatim match, never before — a page that
  matched is by definition legible, so a sparse page whose only real prose **is** the quote can
  never be dismissed as unreadable.
- **PDFs are detected, not parsed.** `extract_page_text` over binary returns mojibake from which
  every quote is "absent". `looks_like_pdf` checks both the extension and the `%PDF-` magic
  bytes, since plenty of PDFs are served from extensionless URLs.

Residual limit: a page whose only long block is a cookie/consent notice can still slip past the
prose gate and produce a false `source-unverified` (observed on `gowithmunro.com/promises`, where
the two longest blocks were a cookie banner and a repeated site title). Rare — after the fix, 274
of 294 findings sit on pages with ≥120 words of genuine prose — but not zero.

What these checks **cannot** do: judge whether a verbatim, well-bounded excerpt is the candidate's
*distinctive* position, whether it is a curator summary of a bulleted platform (`source-summary`,
a judgment check), whether it is as direct an answer as this candidate has (`source-not-an-answer`,
also a judgment check), or whether the page is a legitimate source at all (`invalid-source`,
`unquotable-source`, `scorecard-source`). They answer a narrow question: is this text on that page,
whose words are they, and was it cut somewhere defensible.

Operationally: pages are cached under `.runs/.source-cache` (sha1 of the URL, 7-day TTL) so a
sweep fetches each URL once rather than once per quote, and failed fetches are cached too so a
dead host isn't re-hit for every quote citing it. Live requests to the same host are spaced one
second apart. The cache directory is inside the gitignored `.runs/`.

### 2.3 `source-nested-quotation` — words the candidate is relaying

**The defect.** On 2026-08-02 a manual pass on the TN Governor R primary
(race `ea27533a-f24a-4f9e-b804-cd11c34698dd`) found a Marsha Blackburn quote that was **perfectly
verbatim on its source page and still an invention of her position**. The page had her relaying
what voters tell her:

> Blackburn: "People will say, 'Hey, let's make certain our communities are safe … let's pick up
> the pace deporting illegal aliens.'"

The inner sentence was curated as her own pledge. Every other check in this file passes it: the
text *is* on the page, it *is* in her mouth, it *is* on-topic and single-claim. Only the
punctuation and framing *around* the match reveal that the opinion is not hers. That is why this
is high severity — a false `source-unverified` wastes a re-read, but a missed nested quotation
publishes a policy position the candidate never took, under their name, on a ranking page.

**How it is detected.** Three independent signals, any one of which flags (`nested_quotation`):

1. **Self-framing** — the stored quote *itself* opens with a third-party frame and then opens a
   quotation ("People will say, 'Hey, …"). Needs no source at all.
2. **Nesting** — the matched text sits inside a `'…'` span, itself inside the `"…"` span of the
   candidate's own quotation. Quote depth is tracked across straight and curly marks, and is
   scoped to the current block (paragraph on a page, segment in a transcript).
3. **Adjacent frame** — a third-party frame ("People will say", "they tell me", "critics argue")
   sits within 100 characters before the match, in the same sentence and the same block.

**Which sources it runs against.** All of them, and unlike the rest of §2.2 it is **not gated on
`--verify-sources`**:

- **Signal 1 needs no source**, so `check_source` runs it up front for every quote regardless of
  source kind — including the cases nothing else can inspect: an un-ingested video, a 403ing
  page, an aggregator URL the fetcher deliberately skips, or a written source when the caller
  never opted into network I/O. It costs nothing and the defect is real either way.
- **Video sources** run signals 2 and 3 against the candidate's **own transcript segments**,
  joined with newlines. Restricting to their own segments matters: otherwise a moderator asking
  "People will say we should deport everyone — is that your view?" would be read as framing the
  candidate's answer. The newline join then stops a frame reaching across a segment boundary.
- **Written sources** run signals 2 and 3 against the fetched page, so those two need the flag.

The two paths do not carry equal weight. ASR seldom transcribes quotation marks, so on video it
is signal 3 that does the work and signal 2 almost never fires; on a page both are live.

**Why it does not fire on ordinary journalism.** The subject list contains only speakers who are
demonstrably *not* the candidate (people, voters, folks, constituents, they, critics, opponents,
the report/bill/ad …). The candidate's own name and a bare "he/she" are deliberately excluded, so
`Blackburn said, "I will secure the border"` — the single most common shape in the corpus — can
never flag. Apostrophes are not mistaken for quotation marks: a mark only opens a quotation after
whitespace and before a word, so `let's` is inert and a possessive (`workers'`) reads as a *close*,
which can only ever suppress a finding. Signal 1 additionally requires an inner opening mark, so a
candidate's own rhetorical setup ("People say we can't fix this. They're wrong.") stays silent.
A frame in a *previous* sentence is ignored — a `.`/`!`/`?` between the frame and the match means
that sentence closed.

**Measured noise.** The always-on path (signal 1 for all quotes, plus signals 2–3 against the
transcript for the 70 video-sourced ones) was swept over all 3,272 live quotes in
`essentials.quotes`: **zero** hits, with the pre-existing `source-not-ingested` ×4 and
`source-speaker-mismatch` ×1 unchanged. A live `--verify-sources` run over the 11 TN Governor
quotes produced zero nested findings across 8 fetchable pages (verbatim runs of 7–27 words on
pages of 1.1k–12k words).

That sweep is also what calibrated the apostrophe handling. Its first run produced two findings,
**both false positives**, and both from the same cause: a `'` that opens no quotation but looks
like one — `'cause` in one transcript, and Whisper splitting `o'clock` into `o 'clock` in
another. Never being closed, each marked every later match in its transcript as nested. Hence
two rules that matter more than they look: a single mark followed by a known elision
(`'cause`, `'em`, `'til`, `'clock`, …) or by a digit (`the '90s`) does not open a quotation, and
quote depth resets at every block boundary so one misread mark cannot poison a whole document.

**Known limits.** It is a punctuation-and-framing heuristic, not comprehension. It will miss a
relayed quote that the source paraphrased without quotation marks ("Voters tell her the border
must be secured, and she agrees" carries no inner quote to find), and it does not check *whether
the candidate endorsed* what they relayed — a human still has to read the passage. On video it
leans almost entirely on signal 3, so a relayed quote spoken without any framing verb ("the
border is broken, that's what I hear") will pass. The third-party subject list is closed, so an
unusual framing subject is invisible to it.

## 3. Judgment checks

These require reading the quote against its topic question and the candidate's Compass stance —
not pattern-matching. The eleven checks in the table below are **per-quote**: a Claude agent
applies them per race (or race×topic) using the prompt in §4. The two **per-set** checks in §3.1
judge a whole answer set instead, and run in their own later pass (§4.1).

| id | what to look for | severity | fix-class |
|---|---|---|---|
| `not-forward` | The quote's operative clause is record ("I did X") or an attack, not a forward-looking position ("here's how I'd approach X"). Scaffolding by a little record or a glancing opponent mention is fine — judge the *main assertion*. | high | decision-required |
| `is-attack` | The operative clause targets a *person* (character, family, fitness) rather than a policy, law, or institution. Policy/institution critique is allowed even when combative (the carve-out). | high | guided (if it can be trimmed down to the surviving position) or decision-required (if the attack is the whole quote) |
| `off-question` | The quote doesn't genuinely answer the topic's **ranking question** (`stance.question_text` — the per-race override if one exists, else the Compass question) — it touches the subject but engages a different axis, or answers an adjacent question entirely. Comparability is the precondition for a valid ranking; this is a gate, not a preference. | high | decision-required |
| `question-override` | A per-race ranking-question override (`stance.override_active` is true) has drifted from its Compass topic: it shifts the **axis/dimension** away from `stance.compass_question_text` (should be a Compass fix or re-home, not an override), or it names/leaks a candidate (not blind), or it is not derived from the race's actual question. Axis-invariance is what keeps responsiveness and coupling valid (QUOTE-CURATION-PRINCIPLES §7.3). | high | decision-required |
| `deid-dishonest` | `deidentified_text` was produced by paraphrasing/summarizing instead of marking (`…`, `[brackets]`), or it still leaks a self-identifying clause ("as governor," "in my district") or a named third person that should have been depersonalized. | high | guided |
| `note-not-self-contained` | `editor_note` doesn't state how the quote aligns with the candidate's current Compass stance on the topic, or a skeptical reader who hasn't read the principles doc couldn't follow it without outside context. | medium | guided |
| `source-summary` | A written source of ANY kind, at any level of directness (op-ed, platform page, questionnaire answer) is rendered as a curator-summarized bullet list or paraphrase rather than a verbatim sentence actually written by the candidate. | high | decision-required |
| `coupling-in-tension` | The quote pulls against the direction of the candidate's synthesized Compass `value` for this topic (as opposed to reinforcing it or elaborating on a different sub-dimension). This doesn't mean the quote is wrong — it means the tension needs resolving before the quote is surfaced next to the value. | medium | decision-required |
| `non-differentiating-goal` | The quote clears responsiveness but states only an **agreeable goal no candidate in the race would contest** ("who wouldn't want safe streets?") **and names no mechanism/approach/means** — the HOW. Both conditions required: a contested/directional goal without a mechanism is fine and does not trip this. A preference, not a gate. | medium | decision-required |
| `source-not-an-answer` | **Provenance = directness of answer, not medium.** Rank the quote: (1) *answered-this-question* — the candidate was asked this question, or its clear equivalent, and this is their answer; (2) *answered-an-adjacent-question* — responsive, but the prompt they were given differed; (3) *curator-extracted* — the position was lifted from material not organised as an answer (stump speech, platform page, op-ed). Flag at level 3 when a more direct answer plausibly exists, or when nothing establishes the candidate was answering anything. A **questionnaire answer is level 1** even though it is written and self-published — identical prompts across candidates make it the most directly comparable source there is. Not a gate: a level-3 quote may be all a candidate has, and that is honest presence. | medium | decision-required |
| `misleading-verbatim` | The quote is accurately transcribed and genuinely the candidate's words, yet **misleads the reader as presented** — it states as fact something the source context contradicts, or a trim has removed the qualifier that made it true. Verbatim is a floor, not a defence. Judge what a citizen would take away from the blind card against what the full passage supports. | high | decision-required |

### 3.1 Per-set checks — is this question's answer set actually rankable?

Everything above judges **one quote**. These two judge a **question's answer set**: all the live
quotes attached to one question, read across candidates. They are read from the bundle's
`questions` map, **not** its `topics` map. Several questions can share one `topic_key`, and
comparing answers given to *different questions* is precisely the error these checks exist to
catch — so a topic-shaped set is the wrong unit and will manufacture false findings in both
directions.

**The rankability test** (comparability model §4): a set is rankable when its answers **sit at
different points on a shared latent dimension, such that a preference between them is meaningful.**
Two properties, both required — and they fail for different reasons, with **different remedies**:

| id | fails when | remedy | severity | fix-class |
|---|---|---|---|---|
| `set-incommensurable` | **No shared axis.** The answers are each responsive but not *rival* — there is no dimension along which one can be preferred to the other ("cap insulin prices" vs "build more medical schools"). | **Split into two questions.** Each half then becomes rank-or-surface on its own. This is **not** a quality problem with either quote; neither is at fault and neither should be dropped. | high | decision-required |
| `set-undifferentiated` | **Shared axis, no distance.** The answers sit at effectively the same point (both candidates want tariffs rolled back). | **Show convergence, do not rank.** Agreement is information. Never drop the question to hide it, and never hunt a sharper quote to manufacture a gap. | medium | decision-required |

**Procedure.**

1. **Discard everything flagged `off-question` first** (and any draft, since only live quotes are
   in the set a citizen sees). Responsiveness precedes comparability: a quote that doesn't answer
   the question isn't in the set, and grading a set that still contains one grades a comparison
   that was never offered.
2. If fewer than 2 candidates remain, the question is **surfaced, not rankable** — a lone voice
   shown for inclusion. That is already reported as `not-rankable` (§2); **do not double-report it**
   as a per-set finding.
3. Otherwise apply the two properties in order: commensurable first (is there an axis at all?),
   then differentiated (is there distance on it?). A set that fails the first cannot be judged on
   the second.

**Guardrails — all mandatory.**

- **Articulacy is not a differentiation signal, in either direction.** The same position stated
  more fluently is still undifferentiated: ranking that set measures rhetoric, not policy. And a
  blunt, plainly-worded genuine difference **is** differentiated. Fluency is never evidence.
- **Contrast is observed, never engineered.** You may **not** recommend swapping in a different
  quote *because* it contrasts more. Each quote was selected on its own faithfulness; per-set
  judgment reads that selection, it does not revise it.
- **Never manufacture, never hide.** Do not invent a gap that isn't there, and do not suppress
  genuine agreement to make a race look sharper.
- **This is not a substance-policing test.** Judge whether the answers *differ*, never whether a
  position is good, sensible, or well-supported. When you are genuinely unsure, **prefer rankable**
  — a false `set-incommensurable` splits a real comparison apart, and a false
  `set-undifferentiated` silences a real choice.

**Finding fields.** Both emit at `level: "topic"`, with `race_id` set and `topic_key` = the
question's **parent topic**. `quote_id` and `candidate` are **null** — the finding is about the set,
not any one quote or candidate. Name the **question id, the question text, and the set's quote ids
and candidates** in `what`, so a curator can reconstruct the set without re-querying.

**The gate is policy, not automation.** Neither check flips `readrank_selected`. A split or a
demotion is a curation decision; it goes through `apply_fixes.py`'s dry-run + explicit OK like every
other write in this skill.

### 3.2 `source-not-an-answer` replaced `source-tier-4`

The old check matched a **campaign-site URL pattern** and called the source "tier 4", from a ladder
ordered by *medium* and *questioner independence*: spoken-and-probed at the top, candidate-bylined
written at the bottom. That ordering is actively wrong for Read & Rank. It filed a **Vote411/LWV
questionnaire** — written, self-published, therefore low-tier — beneath a stump speech, when the
questionnaire is the **most directly comparable source there is**: every ballot-qualified candidate
is invited, and they all answer *identical prompts*. Comparability is what Read & Rank is for, so
the thing worth grading is **how directly this answers this question**, not what medium it arrived
in. Questioner independence still breaks ties *within* a level; it no longer sets the level.
A URL pattern cannot see directness at all — the same campaign domain hosts both a questionnaire
reprint and a scraped platform bullet — so the check could not stay mechanical, and moved to §3.
(Ruling: CASEBOOK.md, "Directness of answer, not medium". Model: §6–7 of this repo's
`docs/superpowers/specs/2026-07-23-readrank-comparability-model.md`.)

**What did *not* change, and must not be read as relaxed by this:**

- **Hot-mic, private, secretly-recorded and off-the-cuff "gotcha" remarks stay hard-excluded**
  (QUOTE-CURATION-PRINCIPLES §5). That is a consent-and-fairness rule, not a ranking; being a
  direct answer cannot rescue such a quote.
- **Written sources at any level still yield verbatim sentences only.** A platform page or
  questionnaire answer rendered as a curator-summarized bullet list is still `source-summary`
  (high). Level 1 says the candidate was answering the question; it says nothing about whether the
  curator transcribed them honestly.
- **The accuracy floor is untouched** — correct speaker, no fabricated completions, a deep-link
  that actually contains the quote (`source-speaker-mismatch`, `source-unverified`,
  `source-timestamp-drift`).
- **The three bad-source classes are untouched** (§2.1). `invalid-source`, `unquotable-source` and
  `scorecard-source` detect sources that cannot carry a candidate utterance *at all*, which is
  orthogonal to directness: a level-1-looking page is still unusable if nobody said the words on it.

✅ `QUOTE-CURATION-PRINCIPLES.md` §5 now states this hierarchy (2026-08-07). The two documents
agree; the principles doc remains the authority. Its levels are **named, not numbered** —
`answered-this-question` / `adjacent` / `curator-extracted` / `excluded` — deliberately, because
`essentials.discovered_sources.source_tier_guess` runs 1–4 on a *different* scale.

**Source discovery still ranks by questioner independence, and that is intended.** The classifier
guesses from a title and channel name before anything has been read, where directness is not yet
knowable; its tier is a **triage-priority** signal, not a quality claim about a quote. Do not read
a discovery `source_tier_guess` of 4 as "this quote is `curator-extracted`" — they are different
scales measuring different things at different times.

## 4. Judgment-agent prompt template — the **per-quote** pass

The judgment pass runs in **two stages**: this one, which judges each quote on its own, and the
per-set pass in §4.1, which judges each question's answer set **after** this one has returned.
Run them in that order and never merge them — §4.1 explains why the ordering is the guardrail.

The skill sends one instance of this prompt per race (or per race×topic, for large races) to a
Claude agent, filling in `{context_bundle_json}` with the contents of
`.runs/<date>/context/<race_id>.json` (or a single topic's slice of it) written by `scripts/audit.py`.

```
You are auditing candidate quotes for Read & Rank, a tool that shows citizens blind,
de-identified candidate quotes grouped by topic and lets them rank the quotes without
knowing who said what. You are checking a batch of already-curated quotes against the
project's curation principles — you are not curating new quotes, and you must not
propose rewrites of quote_text or deidentified_text yourself. Flag only; do not fix.

## Context

You will receive a JSON object for one race. It has a `topics` map keyed by `topic_key`;
each topic has a `quotes` array. Each quote has:
  - `id`, `topic_key`, `race_id`, `candidate`, `readrank_selected`
  - `quote_text` — the canonical/revealed quote (keeps names, self-identification)
  - `deidentified_text` — the blind-card version (canonical + extra de-identification),
    or null if none exists yet
  - `editor_note` — the curator's justification for the quote and its edits
  - `source_name`, `source_url` — provenance
  - `stance` — `{question_text, compass_question_text, override_active, value, chairs}` for this
    candidate+topic. `question_text` is the **ranking question** you gate responsiveness against
    (the per-race override if `override_active`, else the Compass question). `compass_question_text`
    is the canonical Compass question. `value` is the candidate's numeric Compass value on this
    topic's spectrum (may be null), and `chairs` are the spectrum's labeled anchor points
    (roughly 1-5, from one pole to the other)

## The rules (summarized — the full principles live in QUOTE-CURATION-PRINCIPLES.md)

- **Forward, not record.** A Read & Rank quote is the candidate reasoning about what
  should be done and why — not a recitation of what they've already done. Judge by the
  quote's *operative clause* (what it's mainly asserting): a little record or a glancing
  opponent mention as scaffolding is fine if the main assertion is a forward position.
- **Position, not attack — with a carve-out.** The quote must articulate the candidate's
  own stance, not primarily attack an opponent. Critiquing a *policy, law, or institution*
  (a program, a law, "the current administration's policy") is legitimate position speech,
  even when combative. The line is the target: attacking a *policy or office* stays;
  attacking a *person* (character, family, fitness) does not belong in Read & Rank.
- **Responsiveness — a hard gate, not a preference.** The quote must genuinely answer the
  topic's **ranking question** (`stance.question_text` — the per-race override if
  `stance.override_active`, else the Compass question) — engage the axis/dimension it sets, not
  merely touch the subject. If it answers a different question (even a related one), it is not a
  valid comparison point for this topic, no matter how well-written or distinctive it is.
- **Override must stay on-axis.** When `stance.override_active` is true, the override
  (`stance.question_text`) must engage the *same* axis as `stance.compass_question_text`, be blind
  (name no candidate), and read as the race's real question tightened — not a different question.
  If it shifts the axis, flag `question-override` (that's a Compass fix or re-home, not an override).
- **De-identification honesty.** `deidentified_text` should be the canonical quote with
  identity leaks removed via honest marking (`…` for cut spans, `[brackets]` for inserted
  or substituted words) — never a paraphrase or summary. It must not still contain the
  speaker's self-identification ("as governor," "in my district"), a named third party in
  a policy critique that should have been depersonalized ("Newsom" → "[the current
  administration]"), or a partisan/side tell ("Democrat," "Republican," "my party") that
  would reveal which side is speaking in a two-way race.
- **Notes must be self-contained.** `editor_note` should let a skeptical reader who has
  never seen the principles doc understand why this quote was chosen and how it relates
  to the candidate's current Compass stance on the topic — without citing internal
  section numbers or jargon.
- **Verbatim, not summary.** For written sources of any kind (op-eds, platform pages, questionnaire answers), the
  quote must be an actual sentence the candidate wrote — never a curator-authored summary
  or bullet list dressed up as a quote.
- **Verbatim is a floor, not a defence.** A quote can be accurately transcribed, genuinely the
  candidate's words, and still **mislead the reader as presented** — it states as fact something
  the source context contradicts, or a trim has removed the qualifier that made it true. Judge
  what a citizen would take away from the blind card against what the full passage supports. Flag
  `misleading-verbatim`.
- **Provenance is directness of answer, not medium.** Place the quote on this ladder:
  (1) *answered-this-question* — the candidate was asked this question, or its clear equivalent,
  and this is their answer; (2) *answered-an-adjacent-question* — responsive, but the prompt they
  were given differed; (3) *curator-extracted* — the position was lifted from material not
  organised as an answer (stump speech, platform page, op-ed). A **questionnaire answer is
  level 1** even though it is written and self-published: identical prompts across candidates make
  it the most directly comparable source there is. Never downgrade a source for being written, and
  never upgrade one for being spoken. Flag `source-not-an-answer` at level 3 when a more direct
  answer plausibly exists, or when nothing establishes the candidate was answering anything at
  all. This is **not a gate** — a level-3 quote may be all a candidate has, and that is honest
  presence, not a defect.
- **Coupling to the Compass value.** Among quotes that already pass the responsiveness
  gate, a quote's relationship to the candidate's synthesized Compass `value` for this
  topic is one of: reinforcing (illustrates the numeric stance directly), elaborating
  (answers the question but on a different sub-dimension than the numeric axis — still
  valid), or in tension (pulls against the synthesized value — needs a flag, not a silent
  pass). Use `stance.chairs` to understand what each end of the spectrum means before
  judging reinforcing vs. in-tension.
- **Prefer the HOW.** Among quotes that pass the responsiveness gate, prefer the one that
  shows *how* the candidate would pursue the goal — the mechanism, approach, or means — not
  merely that the goal is desirable. Flag a quote **only** when BOTH hold: (1) it is
  *non-differentiating* — no candidate in this race would plausibly disagree with the goal
  ("who wouldn't want safe, beautiful streets?"), and (2) it is *mechanism-free* — it names no
  approach or means. A contested/directional goal without a mechanism is fine (it is still
  rankable contrast). This is a preference, never a gate; do not use it to reject positions you
  find thin.

## Your task

For every quote in the bundle, apply these judgment checks. Judge **each quote on its own** —
do not compare candidates' answers to each other, and do not let how well a quote contrasts with
another candidate's influence anything you flag. Whether a *set* of answers is rankable is a
separate pass that runs after this one:

- `not-forward` — quote is record/attack, no forward position (severity high, decision-required)
- `is-attack` — attacks a person, not a policy/institution (severity high; guided if it
  can be trimmed to a surviving position, decision-required if the attack is the whole quote)
- `off-question` — doesn't answer the topic's ranking question (override ?? Compass) (severity high, decision-required)
- `question-override` — an active override shifts the axis / isn't blind / isn't the race's real question (severity high, decision-required)
- `deid-dishonest` — blind text paraphrased instead of marked, or still leaks a self-ID
  or named person (severity high, guided)
- `note-not-self-contained` — note doesn't state Compass-stance alignment / needs the
  principles doc to parse (severity medium, guided)
- `source-summary` — a written-source quote is a summarized bullet list, not a verbatim
  sentence (severity high, decision-required)
- `coupling-in-tension` — quote pulls against the candidate's Compass value (severity
  medium, decision-required)
- `non-differentiating-goal` — on-question quote states an agreeable goal no one would
  contest AND names no mechanism/HOW (severity medium, decision-required)
- `source-not-an-answer` — the quote is curator-extracted (level 3) rather than an answer the
  candidate actually gave, and a more direct answer plausibly exists (severity medium,
  decision-required)
- `misleading-verbatim` — accurately transcribed and genuinely theirs, yet misleads as presented
  (severity high, decision-required)

## Output

Return **only** a JSON array of finding objects, one per problem found (a quote can
produce more than one finding; a clean quote produces none). Return `[]` if nothing in
the whole bundle warrants a finding. Each object must have exactly these fields:

  check_id, level, principle, severity, fix_class, what, suggested_fix,
  quote_id, topic_key, race_id, candidate

- `level` is `"quote"` for every check in this list — this pass judges quotes, never sets. The two
  per-set checks (`set-incommensurable`, `set-undifferentiated`) are `level: "topic"` and belong to
  the separate pass that runs after this one; do not emit them here.
- `quote_id`, `topic_key`, `race_id`, `candidate` — copy from the quote you're flagging.
- `what` — one or two sentences, specific to this quote (quote the offending phrase where
  useful).
- `suggested_fix` — a proposal for a human curator to consider, not an instruction you
  are authorized to execute.
- Do **not** include a `fix_op` field.
- Do not rewrite `quote_text` or `deidentified_text` yourself, and do not invent findings
  outside the eleven check ids above.

Context bundle:
{context_bundle_json}
```

## 4.1 Judgment-agent prompt template — the **per-set** pass

Run **after** the per-quote pass has returned for a race, never alongside it. **That ordering is
the guardrail, not a convenience.** Each quote must be selected and judged on its own faithfulness
*before* anyone looks at how the answers compare; run the two together and contrast leaks backward
into which quote gets chosen, which is exactly the failure the comparability model forbids
— §7 (the two-layer rubric) and §8 ("contrast is observed, never engineered") of this repo's
`docs/superpowers/specs/2026-07-23-readrank-comparability-model.md`, and CASEBOOK.md,
"Differentiation is observed, never engineered".

**One dispatch per race.** Its input is the bundle's `questions` map plus the per-quote findings
that pass returned for the same race. Fill `{questions_json}` with `bundle["questions"]` and
`{per_quote_findings_json}` with the JSON array from §4.

```
You are auditing candidate quotes for Read & Rank, a tool that shows citizens blind,
de-identified candidate quotes grouped for comparison and lets them rank the quotes
without knowing who said what. A per-quote pass has already run: every quote below was
judged on its own merits, and its findings are given to you. Your job is different and
narrower — judge whether each QUESTION's set of answers is actually rankable.

You are not selecting quotes and not re-judging them. Flag only; do not fix.

## Context

You will receive two JSON values for one race:

  - `questions` — a map keyed by question_id. Each entry has `question_id`,
    `question_text`, `origin`, `topic_key`, and a `quotes` array (the answers; each quote
    carries at least `id`, `candidate`, `race_id`, `quote_text`, `deidentified_text` and
    `readrank_selected`).
  - `per_quote_findings` — the findings array the per-quote pass returned for this race.

Judge the **`questions` map**, not any topic grouping. Several questions can share one
`topic_key`; comparing answers that were given to *different questions* is exactly the
error you are here to catch, so never merge two questions because their topic matches.

## Procedure

1. **Discard first.** Drop from each set every quote the per-quote pass flagged
   `off-question`. Responsiveness precedes comparability — a quote that doesn't answer the
   question was never in the set.

   **Then decide what the set is, from the run's scope:**
   - **Live-only run** (the default): the set is the live quotes (`readrank_selected` true).
     You are grading what a citizen actually sees.
   - **Drafts in scope** (`--include-drafts`): the set is **each candidate's single best
     available answer** — the one the per-quote pass judged most faithful, live or draft.
     You are answering *"is this question worth shipping, and with which answers?"*, which is
     the reason the run included drafts. Judging only live quotes on such a run tells you
     nothing about a race whose draft pool is large and whose live selections are sparse —
     which is the common case, and the case the run exists to fix.

   This does **not** loosen "contrast is observed, never engineered". Each candidate's best
   answer was chosen on *faithfulness alone*, in the earlier pass, with no knowledge of what
   the other candidate said. You are judging the set that selection produced — you are not
   selecting.
2. **Count what's left.** If fewer than 2 candidates remain, the question is *surfaced,
   not rankable* — a lone voice shown for inclusion. That is already reported as
   `not-rankable` by the mechanical pass; emit nothing for it. Do not double-report.
3. **Test the remaining set.** A set is rankable when its answers sit at different points
   on a shared latent dimension, such that a preference between them is meaningful. Two
   properties, both required, tested in this order:
   - **Commensurable** — a shared axis exists. NOT "same words", NOT "mutually exclusive".
     Coverage-vs-supply in healthcare is commensurable: both are positions on how large a
     role government should play, so "I lean smaller-government" is a meaningful
     preference. "Cap insulin prices" vs "build more medical schools" is not: responsive
     both, but no dimension orders them.
     Fails → `set-incommensurable`. The remedy is to **split the question in two**; each
     half then becomes rank-or-surface. Neither quote is at fault — do not propose
     dropping or demoting either one.
   - **Differentiated** — real distance on that axis. Two candidates who both want tariffs
     rolled back share an axis with no gap.
     Fails → `set-undifferentiated`. The remedy is to **show convergence and not rank it**.
     Agreement is information: never drop the question to hide it, and never suggest
     hunting a sharper quote to manufacture a gap.
   A set that fails commensurability cannot be judged on differentiation — report the
   first failure only.

## Guardrails

- **Articulacy is not a differentiation signal, in either direction.** The same position
  stated more fluently is still undifferentiated; ranking it would measure rhetoric rather
  than policy. A blunt, plainly-worded genuine difference IS differentiated. Fluency is
  never evidence.
- **A shared goal is not undifferentiation.** When candidates agree on the destination, the
  *mechanism* is the axis, and it is often the most decision-relevant difference there is.
- **You may NOT propose a different quote for a candidate.** Not a better one, not a
  sharper one, not a more contrasting one — for any reason. Selection happened before you
  and contrast may never influence it.
- **Never manufacture, never hide.** Do not invent a gap that isn't there; do not suppress
  genuine agreement to make a race look sharper.
- **Not a substance-policing test.** Judge whether the answers *differ*, never whether a
  position is good, sensible, or well-supported. When genuinely unsure, **prefer rankable**.

## Precedent

Consult CASEBOOK.md before ruling and follow the nearest matching entry — it is precedent,
not background. If your case differs in a way that matters, say how, and **propose a new
entry** (situation → decision → principle) for any ruling it does not already cover. If a
case contradicts an entry, surface both rather than overruling it silently.

## Output

Return **only** a JSON array of finding objects. Return `[]` if every set in the race is
rankable (that is the common, good outcome). Use **only** these two check ids:

- `set-incommensurable` — no shared axis; split into two questions (severity high,
  decision-required)
- `set-undifferentiated` — shared axis, no distance; show convergence, don't rank
  (severity medium, decision-required)

Each object must have exactly these fields:

  check_id, level, principle, severity, fix_class, what, suggested_fix,
  quote_id, topic_key, race_id, candidate

- `level` is `"topic"` for both checks.
- `race_id` — copy it from any quote in the set (every quote carries it). `topic_key` — the
  question's parent topic (`questions[qid].topic_key`).
- `principle` — the short rule in human words: `"commensurable — a shared axis exists"` for
  `set-incommensurable`, `"agreement is information"` for `set-undifferentiated`.
- `quote_id` and `candidate` are **null**: the finding is about the set, not any one quote
  or candidate.
- `what` — name the question id, quote the question text, and list the set's quote ids and
  candidates, then say concretely why the set fails. A curator must be able to reconstruct
  the set from this field alone.
- `suggested_fix` — the remedy for that failure (split, or show convergence), framed as a
  proposal for a human. Never a quote swap.
- Do **not** include a `fix_op` field, and do not propose flipping `readrank_selected`
  yourself — demotions and splits are curation decisions made by a human.

Questions:
{questions_json}

Per-quote findings:
{per_quote_findings_json}
```

## 5. Portfolio check

Run once per race, after the mechanical and judgment passes for every topic in that race are in
hand. This is a **skew audit**, not a balancing instruction (principles §8: "process neutrality
with a skew audit" — never engineer outcome balance).

- **Compute per-candidate topic coverage**: for each candidate in the race, the count (and list)
  of topics where they have a live (`readrank_selected`) quote that passed responsiveness, versus
  the total topics in the race.
- **Compare across candidates.** If one candidate is live on most of the race's topics while
  another is live on few or none — i.e. coverage is lopsided rather than roughly comparable —
  that asymmetry is worth surfacing.
- **Emit one finding** at `level: "portfolio"`:
  - `check_id`: `coverage-skew`
  - `severity`: `medium`
  - `fix_class`: `decision-required`
  - `principle`: `"equal curation effort across candidates and topics"`
  - `what`: describe the asymmetry concretely — e.g. "Candidate A is live on 8/9 topics;
    Candidate B is live on 2/9, absent from housing, climate-change, immigration, ..."
  - `suggested_fix`: frame it as **a signal to investigate, not a defect to correct** — the
    skew may be a true reflection of one candidate being more on-record or more articulate
    (which voters should see, per §8), or it may be an effort gap in the curation pass that
    should get a second look. Never suggest sourcing a quote *in order to* balance the
    count; only ever suggest checking effort/coverage was applied evenly.
  - `race_id`: the race id. `topic_key` and `quote_id`/`candidate` are left null — this
    finding is about the race's topic portfolio as a whole, not a single quote or topic.
- If coverage is roughly comparable across candidates, emit no `coverage-skew` finding for
  that race.

## 6. The casebook

[CASEBOOK.md](CASEBOOK.md) carries the rulings that bind judgment calls, as
**situation → decision → principle**. This file says what each check looks for; the casebook says
how hard cases were actually decided — so an agent reasons by precedent, and a human can audit that
reasoning without re-reading the source material.

Consult it before every judgment pass. Propose an entry for any ruling it does not cover. If a new
case contradicts an entry, surface both rather than overruling silently: an overturned ruling is
information about the rubric, which is the point of keeping it.
