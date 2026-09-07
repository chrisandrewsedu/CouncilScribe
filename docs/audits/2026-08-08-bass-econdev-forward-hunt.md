# Karen Bass — forward-looking economic-development quote hunt

**Date:** 2026-08-08
**Race:** Los Angeles Mayor, Nov 3 2026 general/runoff — `9e888818-c50b-4c61-a106-a0839ff2479d`
**Candidates:** Karen Bass (incumbent) vs. Nithya Raman
**Questions in scope:**

| id | question_text |
|---|---|
| `025217d5-8c55-465c-86ae-f206634740d6` | What should Los Angeles do to keep film and television production from leaving? |
| `263728ff-140c-4969-8eb3-f30f7989946a` | How should Los Angeles respond to the decline of its downtown core? |

**Mandate:** find whether Bass has stated a *forward-looking* position on either question, verbatim and
sourceable, so the questions can be rankable rather than single-voice. Read-only. **No database writes were
made.** No migration is proposed here — a human decides what, if anything, gets inserted.

---

## Verdict

**Bass is not record-only on these questions. She has forward-looking, verbatim, well-sourced positions on
both — they were simply never curated.** Five candidate quotes are documented below; three are
recommendable.

| # | Question | Source | Directness | Recommend |
|---|---|---|---|---|
| **D** | downtown `263728ff` | NBC4/Telemundo debate, 2026-05-06 (**already in our DB**) | `answered-this-question` | ★ **primary** — re-trim of existing row `b2d1f06d` |
| **B** | film `025217d5` | TheWrap interview, 2026-05-20 | `answered-this-question` | ★ **primary** |
| **C** | film `025217d5` | The Hollywood Reporter interview, 2026-05-22 | `answered-this-question` | ★ safe alternate |
| **A** | film `025217d5` | Brian Tyler Cohen interview, 2026-05-13 (**already in our DB**) | `adjacent` | conditional — see set-level warning |
| **E** | downtown `263728ff` | Commercial Observer / Connect CRE, 2026-05-28 | `adjacent` | ✗ not as a quote — use as corroboration |

The headline finding repeats the casebook precedent *"Mis-curation, not missing material, is often the
bottleneck."* The single best downtown quote is **the second half of an answer we already store** — the
curator kept the record clause and discarded the forward one. The best film quotes required going to the
web, but one usable film answer has also been sitting in `meetings.segments` since July.

**De-identification is viable for every recommended quote, and in the downtown case de-identification gets
*easier*, not harder** — the incumbency vectors ("We have a strategy that is working", "That's why I did the
adaptive reuse ordinance") all sit in the record half that the forward trim drops.

---

## Why she looked record-only

The premise of the task is correct and worth recording, because it is now confirmed **twice more** than
before. Bass was asked a direct "what is your plan" question about film production on at least three
occasions, and answered with record on two of them:

1. **NBC4/Telemundo debate, 2026-05-06** — *"do we do enough to keep production in Southern California?"* →
   *"Let me just tell you what I have done…"* (seg 395). Pure record. This is the existing row `ce9bb5b9`.
2. **Sherman Oaks Homeowners Association debate, 2026-05-05** — *"What is your plan?"* → *"Sure, in two areas.
   One is advocacy, which we could talk in a minute, but I believe in pulling policies together by putting the
   people who are directly impacted. The industry came together. I established an entertainment industry
   council when I first came in…"* — she names two areas, then spends the entire answer on the second
   (record) and **never returns to advocacy**. The moderator moves on. The forward half of her own answer is
   never delivered.
3. **The Hollywood Reporter, 2026-05-22** — same question, and here she *does* answer forward (**C** below).

So the record-recital pattern is real and it is what a debate transcript captures. It is not the whole
corpus. The forward answers exist in longer-form print interviews, where the follow-up question forces her
past the record.

---

## D — Downtown (recommended primary)

**This is a re-trim of an existing row, not a new quote.** Row `b2d1f06d` (Bass, `economic-development`,
currently attached to legacy question `ded400bd`) stores the **first half** of this answer. The forward
position is the second half.

- **Event:** 2026 Los Angeles mayoral primary debate, NBC4 Los Angeles (KNBC) / Telemundo 52
- **Date:** 2026-05-06
- **Meeting:** `f2cf80ef-a811-4d95-990d-b9c598284eb6`, segment **364**, start `4652`s
- **Deep link:** `https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4652#seg-364`
- **Video:** `https://www.youtube.com/watch?v=8rI3A6alVHM&t=4652s`
- **Directness:** `answered-this-question`

### The prompt she was given

Moderator, seg 357 (this is the segment question `263728ff` was derived from):

> "Downtown Los Angeles seems to be in a state of crisis… What is your plan for downtown? Can we afford to
> let it die?"

and directly to Bass, seg 361: *"Mayor Bass… Can we afford to let downtown LA die?"*

### The full answer (verbatim, seg 364)

> "So let me just tell you that we absolutely cannot. Downtown is the center of our city and it is an
> economic engine that absolutely needs to be attended to. **We have a strategy that is working. We are
> working with the downtown business associations. We are increasing public safety there. That's why I did
> the adaptive reuse ordinance, which allows for the office buildings that are vacant to be converted into
> housing and those conversions are taking place right now.** That is why we have to deal with the street
> homelessness that is there. There needs to be massive intervention there. And then, of course, there is
> the convention center. And the convention center is a long-term investment that we have to make in our
> city because the more people you have downtown, whether it's a convention or people coming downtown for
> concerts, is the way to make the city more safe and downtown more safe. And my number one obligation is to
> keep our city safe. So downtown is absolutely critical."

Bolded is what row `b2d1f06d` currently stores. Everything after it was discarded.

### Proposed quote (canonical / revealed)

> "…we have to deal with the street homelessness that is there. There needs to be massive intervention
> there. And then, of course, there is the convention center. And the convention center is a long-term
> investment that we have to make in our city because the more people you have downtown, whether it's a
> convention or people coming downtown for concerts, is the way to make the city more safe and downtown more
> safe."

**Marks: one leading `…`**, because the run starts partway into the sentence "That is why we have to deal
with…" — "That is why" points back to the record clause and would be misleading if retained. Otherwise it is
a single contiguous run with **no interior elisions at all**. Nothing is reordered. The closing peroration
("And my number one obligation is to keep our city safe") is dropped at a genuine sentence boundary — it adds
no position and it is the one mildly office-holder-flavoured clause in the run.

### Why this is forward-looking

Operative clause: **"the convention center is a long-term investment that we have to make in our city
because the more people you have downtown … is the way to make the city more safe and downtown more safe."**

That is a prescription ("we have to make"), not a report. It also supplies a *mechanism* and a causal theory:
downtown's problem is emptiness, the cure is footfall, and the lever for footfall is conventions and events —
so safety is a *consequence* of activity rather than only a precondition for it. The second forward clause,
"There needs to be massive intervention there" on street homelessness, is likewise prescriptive.

This clears §4.6 comfortably: it is not an agreeable-and-mechanism-free goal statement.

### Proposed `deidentified_text`

**Identical to the canonical text.** No changes needed.

De-identification is not merely possible here, it is *free*: every incumbency vector in the full answer —
"We have a strategy that is working", "That's why **I did** the adaptive reuse ordinance" — lives in the
record half that the trim removes. The retained run contains no self-identification, no named person, and no
partisan tell. A challenger could have said these sentences verbatim.

### Verification

Matched word-for-word against **two independent transcripts** of the same event:

1. `meetings.segments` for `f2cf80ef-…` (VTT-aligned, speaker-diarized; speaker `Karen Bass`)
2. YouTube auto-captions for the NBC4 re-upload `-83WHHCKZDY`, downloaded and de-duplicated

The two agree on every word of the retained run (the only divergence anywhere in the passage is comma
placement in "Downtown is the center of our city, and it is an economic engine", which is not in the retained
span).

### Corroboration of currency (post-primary)

Three weeks later, at a Connect CRE conference (2026-05-28, published 2026-06-01), Bass gave the same
mechanism unprompted — see **E** below. The position is current as of June 2026 and has not been reversed.

---

## B — Film / TV (recommended primary)

- **Publication:** TheWrap — *"Karen Bass Says She's Ready to Cut More Red Tape for Local LA Production:
  'Whatever Is in the Way'"*, Tess Patton
- **Date:** 2026-05-20 (Zoom interview conducted Tuesday 2026-05-19)
- **URL:** `https://www.thewrap.com/media-platforms/politics/karen-bass-reelection-interview-la-film-production/`
- **Medium:** written Q&A, transcript-style; **no video/audio**
- **Directness:** `answered-this-question`

### The prompt she was given

> "We've heard some candidates say that they support this idea of just wiping the slate clean [with regards
> to red tape at the city level], starting from the beginning. Are you willing to go that far, or is it
> something that you want to take a more measured approach to?"

This is a *city-level* question about what LA should do to make filming easier — the exact frame of question
`025217d5`, and a directional one (wholesale vs. measured).

### Proposed quote (canonical / revealed)

> "I'm open to looking at any special condition. There's a lot of stuff in the city that happens because it's
> always happened for no particular reason, or maybe it made sense 25 years ago, and makes no sense right
> now. Unfortunately, those things kind of have to come up, as opposed to there's some magic list somewhere
> that I could just say I'm eliminating all these things. I'm open to eliminating or changing or waiving
> whatever is in the way."

**Marks: none. Verbatim, contiguous, whole answer, zero edits.**

### Why this is forward-looking

Operative clause: **"I'm open to eliminating or changing or waiving whatever is in the way."**

There is not one word of record in this answer. It is entirely about what she would do, and it carries a real
*how*: she is explicitly declining the wipe-the-slate-clean approach in favour of a case-by-case one —
**"as opposed to there's some magic list somewhere that I could just say I'm eliminating all these things."**
That is a contestable, directional stance on method, which is precisely what §4.6 asks for.

### Proposed `deidentified_text`

**Identical to the canonical text.** Fully blind-able as-is — no "on my watch", no "my administration", no
record, no named opponent, no partisan tell.

### Caveats a human must weigh

- The article states *"The following interview has been edited for clarity and length."* This is a
  publisher-edited Q&A, so it is a **high-quality attributed transcript, not a strict verbatim recording**.
  There is no audio to check it against (Zoom interview, not published).
- Directness is `answered-this-question` but the questioner is a trade outlet interviewing an incumbent —
  independent, and probing (this specific question pushes back on her), so questioner-independence does not
  lower it within its level.

---

## C — Film / TV (recommended safe alternate)

- **Publication:** The Hollywood Reporter — *"Karen Bass Says Spencer Pratt's Insurgent Campaign Hasn't
  'Prompted Any Soul-Searching From Me'"*, Katie Kilkenny
- **Date:** 2026-05-22 (interview conducted after the IATSE endorsement event at Sunset Gower Studios)
- **URL:** `https://www.hollywoodreporter.com/news/politics-news/l-a-mayor-karen-bass-interview-runaway-production-1236603948/`
- **Medium:** written Q&A; **no video/audio**
- **Directness:** `answered-this-question`

### The prompt she was given

> "To start with the obvious, runaway production from L.A. has been devastating the city and entertainment
> workers. What is your plan, if elected to a second term, to address that issue?"

This is question `025217d5` almost word for word.

### The full answer (verbatim)

> "Well, to me it's important to address it on all levels. So locally continuing to look for ways to cut the
> costs for production and to make it easier to film, to produce here. And part of that is eliminating the
> red tape. So we've already done a lot of that. Now the question is, is it going to be helpful? So we
> evaluate it and make any additional changes or additions to it."

### Proposed quote (canonical / revealed)

> "To me it's important to address it on all levels. So locally continuing to look for ways to cut the costs
> for production and to make it easier to film, to produce here. And part of that is eliminating the red
> tape."

Leading "Well," is a verbal tic — silent removal, no mark. The trim ends at a genuine sentence boundary. The
dropped tail ("So we've already done a lot of that…") is record scaffolding and would otherwise have to be
de-identified; cutting it is cleaner than editing it.

### Why this is forward-looking

Operative clause: **"it's important to address it on all levels. So locally continuing to look for ways to
cut the costs for production and to make it easier to film, to produce here."**

The question was explicitly "your plan, if elected to a second term." The answer is a plan, not a report.
"On all levels" is her actual differentiator — later in the same interview she adds *"I will also continue
advocating on a state and a federal level."*

### Proposed `deidentified_text`

**Identical to the canonical text.** Fully blind-able.

### Caveats

- Same *"edited for length and clarity"* disclaimer; no audio.
- §4.6 flag: the goal ("cut costs, make it easier to film, cut red tape") is one **no candidate in this race
  disagrees with**, and the mechanism named is thin. Raman's live quote `644bc8e7` states the same goal plus
  a dedicated film office and county coordination. If C is selected, the set risks being a *specificity* gap
  rather than a *position* gap — and the casebook is explicit that **articulacy is not a differentiation
  signal in either direction**. **B is the better pick on principle**, because B states a method she would
  and would not use, not merely a goal.

---

## A — Film / TV, tax-credit level (conditional; already in our DB)

- **Event:** Brian Tyler Cohen interview — *"Karen Bass on Spencer Pratt and the LA mayor's race"*
- **Date:** 2026-05-13
- **Meeting:** `cd884d5c-6c41-4472-bb69-89532dacaa0f`, segment **41**, start `1329`s
- **Deep link:** `https://on-the-record.onrender.com/meetings/cd884d5c-6c41-4472-bb69-89532dacaa0f?t=1329#seg-41`
- **Video:** `https://www.youtube.com/watch?v=Tks6PkKj6cU&t=1329s`
- **Directness:** `adjacent`

### The prompt she was given

Interviewer, seg 40:

> "So what would you like to see on the state side in Sacramento? … For those like myself for whom the
> resurrection of the entertainment industry is a major issue, what would you like them to put forward?"

The enclosing block was framed as LA revitalization (seg 36: *"What would you like to see in terms of a tax
credit to revitalize the industry here in LA?"*), but the prompt she is answering here is explicitly about
**Sacramento**, not the city. Hence `adjacent`, not `answered-this-question`.

### The verbatim run (seg 41)

> "Well, I— well, one continue if not expand the tax credits, no cap. Um, I think that there's other things,
> you know, that they could do statewide. What I really want to see is federal."

Followed ~2 seconds later, after a laugh, by: *"It's federal tax credits."*

### Proposed quote (canonical / revealed)

> "Continue if not expand the tax credits, no cap. I think that there's other things … that they could do
> statewide. What I really want to see is federal [tax credits]."

- "Well, I— well, one" — false start, silent removal. **Judgment call to record:** "one" may be an
  enumeration ("one: continue…") rather than a stutter; dropping it does not change the position either way,
  but a human should decide.
- "Um" and "you know" — verbal tics, silent removal.
- One `…` for the substantive removal.
- `[tax credits]` — bracketed clarification, and it is **her own next words**, not an inference.

### Why this is forward-looking

Operative clause: **"Continue if not expand the tax credits, no cap. … What I really want to see is federal
[tax credits]."** She is asked what she wants put forward; she answers with a policy ask. No record at all in
the retained run (the record sits in seg 39, a different question — see stitching note).

### Proposed `deidentified_text`

**Identical to the canonical text.** Blind-able.

### Caveats — three, and they compound

1. **Single-transcript verification.** The DB text is `vtt_alignment` over the YouTube **auto-captions**;
   there are no human captions for this video. I re-downloaded the captions independently and they match the
   DB word for word — but that is the *same* underlying transcript, not a second witness. There is no
   independent transcript and I could not check audio.
2. **Speaker ambiguity on the follow-on.** The caption stream renders `>> [laughter] >> Yeah. >> It's federal
   tax credits.` The `>>` markers suggest a speaker change; our diarizer assigns all of it to Bass. The
   bracketed `[tax credits]` proposal above relies on that assignment. If a human cannot confirm it from
   audio, drop the bracket and end at "What I really want to see is federal" — or drop this quote.
3. **Set-level warning (the important one).** Raman's live tax-credit quote `79641bcb` is *"We need a tax
   credit that has no cap, that is guaranteed years into the future."* Bass's is *"continue if not expand the
   tax credits, no cap."* These are **the same position**. Variety reported this convergence explicitly
   ("Karen Bass Joins Mayoral Rival Nithya Raman in Supporting Unlimited Film Incentive", 2026-05-02). If A is
   paired against `79641bcb`, the honest verdict is **`set-undifferentiated` — show convergence, do not
   rank** (casebook: *"Agreement is information."*). The only real distance is Bass's added federal ask.
   B or C paired against `644bc8e7` produces a genuinely rankable set; A does not.

### Do not stitch A with seg 39

Seg 39 opens *"we can't do a local tax credit right now. I mean, we just can't. We can't afford to do that."*
— which is a forward-looking *negative* position squarely on question `025217d5` (LA cannot buy its way in;
the lever is Sacramento). It is tempting to join it to seg 41. **It answers a different question** (seg 36 vs
seg 40) and §4.5 forbids stitching across questions. It could stand alone as a quote, but on its own it
states only what she would *not* do and the rest of seg 39 is unbroken record.

---

## E — Downtown, corroboration only (not recommended as a quote)

- **Publication:** Commercial Observer — *"L.A. Mayor Karen Bass On Fast-Tracking Housing and Reviving
  Downtown"*, Greg Cornfield
- **Date:** conversation 2026-05-28 (Connect CRE conference, with Lew Horne of CBRE); published 2026-06-01
- **URL:** `https://commercialobserver.com/2026/06/la-mayor-karen-bass-housing-downtown/`
- **Directness:** `adjacent`

Asked *"What do you see happening over the next 10 years with initiatives that you're implementing today?"*
about the downtown exodus, she said:

> "But I'll tell you what's really going to make L.A.'s downtown safe is that thousands and thousands of
> people don't want to come if they don't perceive it as being safe. So wanting to strategically take
> advantage of the events that are coming our way — obviously, the World Cup won't be played Downtown, but
> there will be massive watch parties — but also getting the convention center expanded, because we were
> losing major conventions."

**Why not to use it:** the first sentence is self-contradictory as printed (it begins as "what will make
downtown safe is X" and lands on "people won't come if it isn't safe"), and the second is a sentence
fragment. The page carries *"This conversation has been edited for length and clarity"* and there is no
recording. Making it readable would require edits beyond the substance cap.

**Why it matters anyway:** it is the same mechanism as **D**, stated three weeks later, unprompted, to a
different audience. It establishes that D's position is **current** — the §4.4 check — and it is the most
recent downtown articulation found.

---

## Not recommended: the press-release and speech material

For completeness, since these are the first things a search surfaces and they look usable until read
closely.

**mayor.lacity.gov, ~2026-05-01** — *"Mayor Bass Calls for 'No Cap' State Film Tax Credit, Federal Film
Incentive…"*, also carried by TheWrap 2026-05-02 and Variety:

> "We are in a global battle for entertainment jobs, and we must hold nothing back in our fight. This is
> about an industry that is essential to our middle class and who we are as a city."

Forward in tone, but **the policy content is not in her quoted words** — "no cap" and "federal credit" appear
only in the press office's framing sentence and the reporters' paraphrase, which we cannot use. What she is
actually quoted saying is a mechanism-free rally line that no candidate would contest (§4.6 flags both
conditions). It is also a press-office statement and therefore of **unconfirmed authorship** (§5). Use it as
evidence that her uncapped/federal position is real and dated — not as the quote.

**mayor.lacity.gov, 2026-05-27** — AB 2319 post-production credit:

> "I will fight with everything we have to make sure that our entertainment industry isn't relegated to
> L.A.'s history, but is a thriving part of our future that puts our next generation to work in good-paying,
> union jobs."

Same problem: forward, aspirational, mechanism-free, staff-drafted press statement.

**State of the City address (mayor.lacity.gov, "As Delivered")** — the convention-center passage restates D's
mechanism (*"I'm tired of those conventions going to other cities. They need to be right here in Los Angeles.
Those jobs, the hotel nights, full restaurants, and activities on our streets, because that's part of feeling
safe, getting people out there"*) but the whole address is saturated with first-person-mayor record and
`curator-extracted` at best. **D says the same thing better, from a moderated debate.** No reason to prefer
this.

---

## A blocker a human must clear before either question ships

Raman's downtown quote `f7625f7b` has a **de-identification failure in the shipped blind text**. Its
`deidentified_text` reads:

> "…Instead, what **Mayor Bass** has done is to dismantle our economic development department…"

The blind card would name the opponent. Per §4.1/§4.5 the named person must be depersonalized on the blind
card only. This does not matter today because `263728ff` has no second voice and cannot be ranked. **The
moment quote D is inserted, `263728ff` becomes rankable and this ships.** It should be fixed in the same
change, or D held back.

(Also worth noting: `f7625f7b`'s companion row `e7f20cf7` was correctly de-identified — "Angelenos" →
"residents" — so the failure is isolated to this row, not systemic.)

Related housekeeping, not blocking: all six existing Bass/Raman `economic-development` rows are still
attached to the legacy question `ded400bd`, and none is `readrank_selected`. Reassignment to `025217d5` /
`263728ff` is a separate decision.

---

## What I searched and found nothing in

So the next person does not repeat it.

**Our own corpus first** (`meetings.meetings` / `meetings.segments`, read-only). Seven meetings contain Bass
speech:

| date | meeting | verdict |
|---|---|---|
| 2026-05-20 | The Story Is with Elex Michaelson (Fox 11), `66aaf01b` | no film, no downtown |
| 2026-05-15 | AirTalk / LAist, `2bf6c5a1` | no film, no downtown. Only "downtown mall" as nostalgia inside a **term-limits** answer. Cross-checked against LAist's own published transcript — same, nothing more |
| 2026-05-15 | Eyewitness News (ABC7), `b31b9945` | 11 Bass segments; nothing on either topic |
| 2026-05-13 | Brian Tyler Cohen, `cd884d5c` | **hit** — quote A, plus the seg-39 "can't afford a local credit" fragment |
| 2026-05-06 | NBC4/Telemundo debate, `f2cf80ef` | **hit** — quote D (downtown). Film answer is record-only |
| 2025-11-25 | The Fifth Column, `306812f6` | "downtown" appears only re: the federal building and National Guard deployment. Nothing econ-dev |
| 2025-10-22 | *MAYOR KAREN BASS: On the Record* (Bonin podcast), `b28c9255` | downtown-mall nostalgia + infrastructure/term-limits; Olympics small-business summits. No downtown or film position |

**Web / video, searched and rejected:**

- **SOHA debate, 2026-05-05** (`youtube.com/watch?v=AkuJMniInXQ`) — the *only* Bass-vs-Raman head-to-head.
  Captions pulled and grepped. Film question asked; **Bass answers record-only.** **No downtown question at
  all.** Worth ingesting for other topics (housing, homelessness, LA28), but it yields nothing here.
- **karenbass.com** — no policy platform exists. `/issues/` and `/priorities/` both 404. The homepage is two
  paragraphs of record framing. Nothing quotable.
- **Vote411 / League of Women Voters** — no LA Mayor 2026 questionnaire located. LWV Greater Los Angeles and
  the Pat Brown Institute voter-guide hub were both checked. If one is published for the November runoff it
  would be the best available source and should be re-checked closer to the election.
- **mayor.lacity.gov press releases** (downtown walk, Oceanwide Plaza graffiti abatement, convention-center
  bookings ×2, convention-center groundbreaking, Executive Directive 11, film tax-credit celebrations, AB
  2319) — all announcement-shaped. Where Bass is quoted, the quote is record, ceremonial, or aspirational;
  the forward policy content is consistently in the press office's own prose. **Nothing usable across the
  whole set.**
- **Variety, 2026-04-28** ("Hollywood Is 'Turning a Corner'"), **Variety, 2026-05-02**, **TheWrap,
  2026-04-21** (candidate plan roundup), **LA Business Journal LA500 2026** — reporter narrative plus
  campaign-*spokesperson* statements (Alex Stack), not Bass's own words. The LA500 entry contains no quote
  at all.
- **General-election material** — as of 2026-08-08 **no Bass–Raman runoff debate or forum has occurred**, and
  no post-primary long-form Bass interview on film or downtown was found. July–August 2026 coverage is
  entirely press releases. The most recent usable articulation on either question is E (2026-05-28,
  downtown); on film it is C (2026-05-22).

## Tooling notes

- `trafilatura` (repo venv) handled every page cleanly, including Variety, THR, TheWrap and Commercial
  Observer. No 403s, no archive.org fallback needed.
- `yt-dlp` for the SOHA debate and the Brian Tyler Cohen video. **Both have auto-captions only** — no human
  captions exist for either. Rolling auto-captions repeat each line up to three times and must be
  de-duplicated before matching.
- Every DB query was a `SELECT`. Nothing was written.
