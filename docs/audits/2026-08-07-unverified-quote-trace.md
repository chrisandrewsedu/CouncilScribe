# Unverified quote trace — CA Governor + LA Mayor

**Date:** 2026-08-07
**Scope:** the 16 quotes flagged `source-unverified` / `source-speaker-mismatch` by the
2026-08-07 comparability audits (`.claude/skills/audit-quotes/.runs/2026-08-07-ca-gov`,
`.../2026-08-07-la-mayor`) that were **not** resolved by the meetings-transcript sweep.
The 12 that the sweep did resolve (a video-swap at import, now carrying `&t=` timestamps)
are excluded.

**Method:** open-web search on distinctive verbatim phrases, plus archived copies of the
cited pages (`web.archive.org`), plus YouTube caption tracks for debate video (`yt-dlp`,
auto-captions, longest-contiguous-run matching).

**Verification bar applied:** a source counts as **TRACED** only when a *distinctive
contiguous run of the quote's own words* actually appears in it. Matching topic, date,
candidate and plausibility is not confirmation. Secondhand reporting *about* a statement is
not the source.

**Result: 10 TRACED, 6 UNRESOLVED.** No database writes were made; everything here is a
document plus the companion draft SQL at
`backend-migrations-draft/1569_trace_unverified_quotes.sql`.

---

## Summary table

| # | id | candidate | topic | live | verdict |
|---|----|-----------|-------|------|---------|
| 1 | `41ac4890` | Karen Bass | public-safety-approach | **LIVE** | **TRACED** |
| 2 | `1a1a9e98` | Karen Bass | homelessness | **LIVE** | **UNRESOLVED** |
| 3 | `8f51a4e3` | Nithya Raman | growth-and-development | **LIVE** | **UNRESOLVED — fabricated quotation** |
| 4 | `17d7f049` | Karen Bass | housing | draft | **TRACED** |
| 5 | `ef8712c2` | Nithya Raman | homelessness | draft | **TRACED** (archived copy) |
| 6 | `8403a778` | Nithya Raman | deportation | draft | **TRACED** |
| 7 | `97459d18` | Nithya Raman | civil-rights | draft | **TRACED** |
| 8 | `ea35df13` | Nithya Raman | housing | draft | **TRACED** (same text as #7) |
| 9 | `9eb66701` | Steve Hilton | fossil-fuels | draft | **TRACED** (cited URL was right; timestamp missing) |
| 10 | `ebb39e53` | Steve Hilton | abortion | draft | **TRACED** — text defect |
| 11 | `9403fcba` | Xavier Becerra | abortion | draft | **TRACED** |
| 12 | `e842e9d2` | Xavier Becerra | tariffs | draft | **TRACED** — text defect |
| 13 | `3975e2c3` | Karen Bass | public-safety-approach | draft | **UNRESOLVED** |
| 14 | `3dbb0df7` | Nithya Raman | immigration | draft | **UNRESOLVED** |
| 15 | `9afd3c42` | Xavier Becerra | abortion | draft | **UNRESOLVED — wrong speaker** |
| 16 | `66088b79` | Xavier Becerra | deportation | draft | **UNRESOLVED — not a quotation** |

---

# The three LIVE quotes

## 1. Karen Bass / public-safety-approach — **TRACED**

- **id:** `41ac4890-8de7-46b3-8e85-b62a99bdf5f5` · **LIVE**
- **Quote:** "The second largest city in the United States cannot have an effective police
  department with 8,300 officers — levels not seen since 1995."
- **Cited source:** `https://lapublicpress.org/2026/04/la-budget-2026-bass/` (LA Public Press)
- **Confirmed source:** <https://lamag.com/news/mayor-karen-bass-pushes-city-council-to-approve-hiring-of-more-lapd-cops/>
  — *Los Angeles Magazine*, "Karen Bass Warns of LAPD Staffing Low", Michele McPhee,
  **11 December 2025**.

**Verbatim runs matched** (both on that one page, both attributed to Bass's letter to the
City Council):

> In the letter, the mayor urged members to "prioritize the safety of Angelenos," by
> allocating $4.4 million for the Los Angeles Police Department, adding: **"The second
> largest city in the United States cannot have an effective police department with 8,300
> officers…"**

> **"The largest city in the United States cannot have an effective police department with
> 8,300 officers – levels not seen since 1995."**

The stored text is a composite of the article's two renderings of the same sentence from the
same letter — "second largest" from the first, "levels not seen since 1995" from the second.
The article itself is internally inconsistent (the second rendering drops "second"); the first
rendering is the one that matches the stored text, and the underlying primary source is a
single December 2025 Bass letter to the City Council.

**On the cited page:** `lapublicpress.org` returns HTTP 403 to automated fetches and has no
Wayback capture for that URL, so it could not be re-checked directly. The audit's finding that
the quote is not on it stands unchallenged, and it is in any case an **April 2026** budget
article, two months adrift of the December 2025 letter.

---

## 2. Karen Bass / homelessness — **UNRESOLVED**

- **id:** `1a1a9e98-f428-43e0-a35d-f341e0f07510` · **LIVE**
- **Quote:** "I will maintain focus on interim housing until the last street encampment is gone."
- **Cited source:** `https://mayor.lacity.gov/InsideSafe`

**No source found. Recommend retirement.**

The cited page is a static description of how the Inside Safe programme operates ("Move-In
Day", "Housing", partner agencies). It contains no first-person Bass quotation of any kind,
live or in any archived capture.

Searches run (all negative for any contiguous run of the quote):

- `"until the last street encampment is gone" Karen Bass`
- `Karen Bass "maintain focus on interim housing"`
- `Karen Bass "interim housing" "street encampment" "is gone" quote`
- `"last street encampment" Bass Los Angeles`
- Live `mayor.lacity.gov/InsideSafe` probed for `encampment is gone`, `interim housing`,
  `last street` — only generic programme prose matched `interim housing`.
- Archived `mayor.lacity.gov/InsideSafe` snapshots `20240113055920`, `20240602042029`,
  `20240821025504` probed for `encampment is gone`, `last street`, `maintain focus` — zero hits.

The nearest genuine Bass statements found are differently worded and differently framed
("I will never stop fighting until we end street homelessness"; on CNN in May 2026, "the
biggest obstacle is the resistance to building interim housing"). Neither shares a
distinctive contiguous run with the stored text, so neither can substitute.

---

## 3. Nithya Raman / growth-and-development — **UNRESOLVED — fabricated quotation**

- **id:** `8f51a4e3-954b-48b9-acb0-71a437bf8b08` · **LIVE**
- **Quote:** "[Measure ULA has] become a major obstacle [to new housing development]."
- **Cited source:** `https://www.nithyaforthecity.com/housing`

**This is not a quotation. Raman never said these words. Retire immediately.**

This one is worth reading in full, because the failure mode is the one the curation
principles exist to prevent: a reporter's paraphrase was laundered into a direct quotation
through two intermediaries.

**The chain, traced to its origin:**

1. **Origin — *Los Angeles Times*, 15 February 2026**, "Bass helped Raman win reelection.
   Now Raman wants to unseat her. Some call it 'a betrayal'"
   (verified via the Wayback capture
   `https://web.archive.org/web/20260217115230/https://www.latimes.com/california/story/2026-02-15/bass-helped-raman-win-reelection-now-raman-wants-to-unseat-her-some-call-it-a-betrayal`).
   The sentence there is, in full:

   > Raman pointed to Measure ULA, the voter-approved tax on property sales of $5.3 million
   > and up, as a catalyst for her mayoral bid. **Although she has been a supporter of the
   > tax, she has also concluded that it is a major obstacle to building new housing.**

   Note: **no quotation marks anywhere.** This is the reporter's own summary of Raman's
   position. "a major obstacle" is the *LA Times'* phrase, not Raman's.

2. **Wikipedia** (`en.wikipedia.org/wiki/Nithya_Raman`) then rewrote that sentence and added
   quotation marks that the source does not have:

   > Raman, who supported the measure's passage in 2022, said that it had become a
   > **"major obstacle"** to building new housing.

   Confirmed against the article wikitext: the citation on that sentence is
   `<ref name="Surprise!" />`, which resolves to the LA Times piece above. Wikipedia
   introduced the quotation marks.

3. **American Kahani** then copied Wikipedia's sentence verbatim into an aggregated article
   (`americankahani.com/california/indian-american-nithya-raman-polls-second-in-crowded-la-mayoral-race-as-bass-faces-unfavorability-crisis/`),
   inside a run of text explicitly framed "According to Wikipedia…".

4. **Our curator** then reconstructed that reported-speech sentence into a first-person
   quotation with bracketed subject and object, producing the stored text.

**What actually survives as Raman's own words: none.** The only material inside quotation
marks anywhere in the chain is the two-word phrase "major obstacle", and even that originates
with the LA Times reporter, not with Raman.

Pages checked and found *not* to contain "obstacle" at all: the cited campaign page
`nithyaforthecity.com/housing` (live, plus every Wayback capture — `20260512231643`,
`20260515185340`, `20260531031106`, `20260609024414`, `20260620020729`, `20260729060043`,
`20260804222956`); the CD4 press release announcing the ULA reform motion
(`cd4.lacity.gov/press-releases/councilmember-raman-introduces-motion-to-reform-ula/`);
CalMatters (both January 2026 ULA pieces); Commercial Observer (28 July 2026); Westside
Current (two ULA articles); The Real Deal; Mar Vista Voice; LAist's mayoral housing guide;
Wikipedia's *Measure ULA* article.

Raman *has* said publicly comparable things in her own words — from the CD4 release:
"A policy that unintentionally stalls housing production ultimately undermines the very goals
voters asked us to achieve." That is a real, sourceable quotation and a curator may wish to
use it. It is **not** this quote, and swapping the text under the existing row would be a
different editorial act; that decision belongs to a human.

---

# TRACED — remaining

## 4. Karen Bass / housing — **TRACED**

- **id:** `17d7f049-984a-4d36-b4dd-dab0dcab0069` · draft
- **Cited source:** `mayor.lacity.gov/news/delivering-results-2024-bass-highlights-unprecedented-green-year-la`
  (probed: contains neither "status quo" nor "stunted" — wrong page)
- **Confirmed source:** <https://mayor.lacity.gov/news/mayor-bass-visits-adaptive-reuse-project-will-create-more-500-units-affordable-housing>
  — Office of the Mayor press release, April 2026 (Adaptive Reuse Ordinance / DTLA World
  Trade Center conversion).

**Verbatim run matched (full quote, exact):**

> "These projects represent the kind of innovation we are applying to break away from the
> status quo that has stunted L.A.'s housing production and driven up rents for decades,"
> said Mayor Bass.

Independently corroborated with the identical wording at ABC7
(`abc7.com/post/mayor-karen-bass-highlights-adaptive-reuse-ordinance-create-more-housing-units/18996632/`),
MyNewsLA, LA Downtown News and the LA Wave. The mayor's own release is cited as the primary.

## 5. Nithya Raman / homelessness — **TRACED** (archived copy)

- **id:** `ef8712c2-f45c-4905-a177-bab2285e6a89` · draft
- **Cited source:** `cityclerk.lacity.org/…&cfnumber=25-0002-S8` — this is council file
  *"Los Angeles County / Local Solutions Fund / Point in Time Count / Measure A"* (2025), a
  legislative index page. Council-file pages list documents and votes; they never carry
  candidate quotations. Wrong citation by construction.
- **Confirmed source:** <http://web.archive.org/web/20220819070334/https://councildistrict4.lacity.org/councilmember-nithya-raman-remarks-todays-la-city-council-meeting-revised-city-ordinance-4118>
  — "Councilmember Nithya Raman remarks from today's LA City Council meeting on revised city
  ordinance 41.18", CD4 press release. **Wayback capture 19 August 2022.**

**Verbatim run matched (24 words, exact):**

> "This creates a district by district arms-race, where people will get pushed around from
> district to district instead of having a citywide strategy that prioritizes intervention in
> encampments by need, by safety, by fire risk…"

This is the benign "the page changed" case: `councildistrict4.lacity.org` is the retired CD4
site (the URL now 301-redirects and the office has moved to `cd4.lacity.gov`). The archived
capture is the citable artefact.

## 6. Nithya Raman / deportation — **TRACED**

- **id:** `8403a778-c94c-4bd1-971d-76ae1abdaf8c` · draft
- **Cited source:** `cityclerk.lacity.org/…&cfnumber=21-0002-S55` — council file
  *"The U.S. Citizenship Act of 2021"*, moved by **Nury Martinez**, seconded by Gilbert
  Cedillo. Raman is not even a mover on this file.
- **Confirmed source:** <https://www.nithyaforthecity.com/immigrants> ("Immigrants In, ICE Out",
  campaign platform).

**Verbatim run matched (9 words, exact — the entire quote):**

> "LAPD's job is keeping Angelenos safe, not assisting ICE."

Context on the page: "…Many residents do not trust that LAPD is not cooperating with ICE. We
will conduct a transparent audit of department practices… **LAPD's job is keeping Angelenos
safe, not assisting ICE.** Any engagement with federal law enforcement must serve that
mission…"

*Note for the curator:* this is published campaign-platform prose, not speech. It is the
candidate's own published words and sourceable, but a reviewer may want to confirm that
platform text is in scope for this topic under the curation principles.

## 7 & 8. Nithya Raman / civil-rights **and** / housing — **TRACED**

- **ids:** `97459d18-b3dd-44d4-8648-8358c18969dd` (civil-rights),
  `ea35df13-e121-41e5-9a0d-fc63ffdfd06f` (housing) · both draft · identical quote text
- **Cited source:** `cityclerk.lacity.org/…&cfnumber=21-0972` — council file *"Affordable
  Housing Overlay Zone / Development Incentive Programs"* (movers Harris-Dawson and Raman,
  31 Aug 2021). Again a legislative index page, not a quotation source — though this time it
  is at least the correct underlying legislation.
- **Confirmed source:** <https://la.streetsblog.org/2021/10/28/council-approves-raman-harris-dawson-motion-to-foster-affordable-development-in-high-resource-areas>
  — Streetsblog LA, **28 October 2021**.

**Verbatim run matched (10 words, exact):**

> "**L.A. is one of the most segregated cities in America**, and our affordability crisis is
> making it even worse," said Councilmember Raman, "Before this was the outcome of an
> intentional regime of laws and intimidation, but now it is reinforced by more subtle forms
> of exclusion, including through our zoning codes."

Streetsblog attributes it to a Raman press statement issued with the AHOZ motion. The stored
text truncates at "America." and closes the sentence with a full stop where the original has a
comma and continues; the meaning of the retained clause is unaffected, but a curator should
confirm this elision is acceptable and, if the row keeps the truncation, note it in
`editor_note`.

## 9. Steve Hilton / fossil-fuels — **TRACED** (cited URL was correct)

- **id:** `9eb66701-bd1f-4479-8d6b-c4ec1a484ffa` · draft
- **Cited source:** `https://www.youtube.com/watch?v=UUOsiG5tkDU` — NBCLA, "Full NBC4
  broadcast: Watch 2026 California governor candidates discuss key issues"
- **Confirmed source:** the same video, **at 00:20:26 (`&t=1226`)**.

The audit flagged this as unverified against the *ingested* transcript. YouTube's own caption
track for the cited video does contain it. The citation was right all along; only the
timestamp was missing.

**Verbatim run matched (13 words, exact):**

> "i'll get it done directly as governor by instructing the california department of"

Full caption context (auto-caption, ASR): "…*would you advocate for an increase in oil
production in the state* … *mr hilton* — **i'll get it done directly as governor by
instructing the california department of geologic and energy management to open that as a
production** — *this is the problem with democrats endless legislation instead of just getting
the job done*". Speaker confirmed by the moderator's roll-call immediately before.

The tail diverges from the stored text ("geologic … open that as a production" vs "Geological
… open up oil production"); given the agency's real name is the California **Geologic** Energy
Management Division (CalGEM) and the question was explicitly about oil production, the ASR is
the garbled party here, not the stored text. The 13-word run is decisive either way.

## 10. Steve Hilton / abortion — **TRACED**, with a text defect

- **id:** `ebb39e53-3e7f-4ecd-b982-a45db47242e7` · draft
- **Quote:** "I don't want Louisiana dictating our laws. We shouldn't be dictating Louisiana's laws."
- **Cited source:** `https://www.youtube.com/watch?v=qRNZ0kuA49k` — KTLA 5, "California
  Governor's Debate - Full Broadcast". **This video has no caption track at all** (verified
  with `yt-dlp --list-subs`: "has no automatic captions / has no subtitles"), which is why the
  audit could not verify it and why the transcript sweep found nothing.
- **Confirmed source:** <https://www.youtube.com/watch?v=xFNkHY_m_eE> — Face the Nation, "Top
  candidates for California governor face off in debate | full video", uploaded 15 May 2026 —
  **at 01:14:11 (`&t=4451`)**. This is the CBS News California / San Francisco Examiner
  gubernatorial debate.

**Verbatim run matched (11 words, exact):**

> "don't want Louisiana dictating our laws. We shouldn't be dictating Louisiana's"

Caption context: *moderator:* "Why do you want to interfere in another state's laws in that
way?" → *Hilton:* "Why are they interfering with our… **We don't want Louisiana dictating our
laws. We shouldn't be dictating Louisiana's**…" → *moderator:* "Mr. Hilton, you've made
yourself clear. Mr. Steyer." Speaker is Hilton — he is the one being addressed before and
after, and the preceding turn at 01:13:35 is his ("This is not about abortion rights. This is
about one state trying to undermine another state's laws").

> **TEXT DEFECT — must be fixed before this ships.** The stored text opens "**I** don't want
> Louisiana dictating our laws." The audio says "**We** don't want…". One word, but it is the
> first word of the quote and it changes the register from personal to institutional. A human
> should correct the text at the same time as the source.

## 11. Xavier Becerra / abortion — **TRACED**

- **id:** `9403fcba-bb26-425f-bf27-997a4667b05d` · draft
- **Cited source:** `https://www.youtube.com/watch?v=qRNZ0kuA49k` (KTLA — no captions, see #10)
- **Confirmed source:** <https://www.youtube.com/watch?v=xFNkHY_m_eE> — **at 01:13:29 (`&t=4409`)**

**Verbatim run matched (15 words — the entire quote, exact):**

> "absolutely no and when i was ag i protected with reproductive rights here in california"

Speaker confirmed: the moderator calls the turn immediately before — "Thank you, Mr.
[Villaraigosa]. **Mr. Becerra.**" — and "when I was AG" is self-identifying (Becerra was
California Attorney General 2017–2021; no other candidate on that stage was). Corroborated in
the second upload of the same debate (`E77UzwNExAA`, CBS News Sacramento) at 01:36:05.

## 12. Xavier Becerra / tariffs — **TRACED**, with a text defect

- **id:** `e842e9d2-abcb-4fb1-8131-c8f4bd4cbbe5` · draft
- **Cited source:** `https://www.youtube.com/watch?v=qRNZ0kuA49k` (KTLA — no captions)
- **Confirmed source:** <https://www.youtube.com/watch?v=xFNkHY_m_eE> — **at 00:21:04–00:21:12
  (`&t=1264`)**

**Verbatim run matched (strictly contiguous and exact): 6 words** —
"we're going to fight against Trump".

The enclosing span matches near-identically across 13 words. Caption text:

> "…And whether it's the Trump uh tax because he's re-increased the fight the price by going
> to war in Iran **or whether it's a tariffs that are tax. We're going to fight against
> Trump.**" → *moderator:* "**Thank you, Mr. Becerra.** On the issue of health care…"

Speaker is Becerra — named by the moderator immediately after, and the preceding 40 seconds
are his ("…the way I had to do over 120 times when I was attorney general").

> **I am reporting this as TRACED on the strength of the timestamp, the speaker
> identification and the 13-word near-identical span — not on the 6-word exact run alone,
> which is not distinctive by itself.**
>
> **TEXT DEFECT — must be fixed before this ships.** The stored text reads "the tariffs that
> are **attacks**". Both independent caption tracks render it "tariffs that are **tax**"
> (`xFNkHY_m_eE`) / "the tariffs that are tax" (`E77UzwNExAA`). "attacks" is an ASR homophone
> error for "a tax", and the stored text reproduces the error — which is itself evidence that
> the quote was lifted from a low-quality machine transcript rather than heard. As written the
> sentence is not grammatical English and should not go in front of a voter. A human should
> re-transcribe from `&t=1264` or retire the row.

---

# UNRESOLVED — remaining

## 13. Karen Bass / public-safety-approach (World Cup) — **UNRESOLVED**

- **id:** `3975e2c3-0080-496b-9c9c-82c65696c454` · draft
- **Quote:** "We are days away from the World Cup and other international events. L.A., we
  know, is under-policed and that's not an option."
- **Cited source:** `https://nbclosangeles.com/news/local/mayor-bass-lapd-funding-recruitment/3814358/`

**No source found. Recommend retirement.**

The cited NBC4 page was fetched successfully and **does** cover Bass, LAPD hiring and the
World Cup — but contains neither "under-policed" nor "not an option" nor "days away". It is
the December 2025 $4.4M-letter story; "days away from the World Cup" would place the statement
around June 2026, six months adrift.

Searches run (all negative):

- `"L.A., we know, is under-policed"` (exact phrase) — no political results at all
- `Karen Bass "under-policed" "World Cup" LAPD "not an option"`
- `Bass "we know is under-policed" Los Angeles LAPD June 2026`
- `"days away from the World Cup" Bass LAPD officers police hiring`
- `"under-policed" Los Angeles Bass 2026 police department mayor said`
- YouTube search for Bass World Cup / LAPD press conferences — nothing relevant returned

Pages probed for `under-policed` / `underpoliced` / `not an option` / `days away`, all zero
hits: the cited NBC4 page; `mayor.lacity.gov/news/mayor-bass-public-safety-most-important-service-city-los-angeles-can-provide`;
ABC7 (`…find-44m-lapd-police-hiring/18277318/`); CBS LA; LAist (`laist.com/news/mayor-bass-lapd-budget`);
FOX 11; Patch. `lapublicpress.org/2026/04/la-budget-2026-bass/` returns HTTP 403 and has no
Wayback capture, so it could not be checked — that is the one stone left unturned, and a human
with browser access may want to open it.

## 14. Nithya Raman / immigration — **UNRESOLVED**

- **id:** `3dbb0df7-1bc1-49d1-9989-7c6e7cb890c7` · draft
- **Quote:** "At a time when the rights of immigrants in this country are under daily attack,
  Los Angeles has the opportunity to define what a better future for our nation could look like."
- **Cited source:** `cityclerk.lacity.org/…&cfnumber=21-0002-S55` — council file *"The U.S.
  Citizenship Act of 2021"*, **moved by Nury Martinez, seconded by Gilbert Cedillo**. Raman is
  not a mover, and a council-file page carries no quotations regardless.

**No source found. Recommend retirement.**

Searches run (all negative for any contiguous run):

- `"rights of immigrants" "under daily attack" "Los Angeles has the opportunity to define"`
- `Raman "Los Angeles has the opportunity to define what a better future"`
- `"under daily attack" Raman Los Angeles immigrants sanctuary statement`
- `Raman "U.S. Citizenship Act" 2021 Los Angeles resolution statement immigrants "daily attack"`

Pages probed for `under daily attack` / `daily attack` / `better future for our nation` /
`opportunity to define`, all zero hits:

- `nithyaforthecity.com/immigrants` — live, plus Wayback captures `20260515185407`,
  `20260529041001`, `20260609024453`
- `cd4.lacity.gov/immigrant-rights-resources/`
- `cd4.lacity.gov/press-releases/city-council-votes-to-establish-los-angeles-as-a-sanctuary-city/`
- `cd4.lacity.gov/newsletter/establishing-los-angeles-as-a-sanctuary-city/`
- **the entire archived retired CD4 site** — all 193 distinct archived pages of
  `councildistrict4.lacity.org` were fetched from the Wayback Machine and full-text searched
  for `segregated cities`, `most segregated`, `under daily attack`, `better future for our
  nation`, `rights of immigrants`, `assisting ICE`. Zero hits for this quote. (That same sweep
  is what surfaced quote #5, so the method demonstrably works.)

Note also the date mismatch: council file 21-0002-S55 is February 2021, when the federal
posture toward immigrants was the newly-introduced *U.S. Citizenship Act* — "under daily
attack" does not describe that moment. The phrasing fits 2025–26 far better, which suggests
the citation and the text were never connected.

## 15. Xavier Becerra / abortion (67%) — **UNRESOLVED — wrong speaker**

- **id:** `9afd3c42-1aeb-45e6-87ac-5d98386543a1` · draft
- **Quote:** "We protect the women of the state 67% voted for a woman's right to choose."
- **Cited source:** `https://www.youtube.com/watch?v=qRNZ0kuA49k`

**The words were spoken at that debate — but not by Becerra. Recommend retirement.**

The material is in the CBS News California / SF Examiner debate of 15 May 2026, at
**01:13:19–01:13:26**, and the speaker is **Antonio Villaraigosa**. Becerra is demonstrably
the *next* speaker.

Turn-by-turn from the caption track (`xFNkHY_m_eE`), with the moderator's roll-call intact:

```
01:12:42  MODERATOR  "...Would you, as governor, extradite this physician to
                      Louisiana for prosecution, yes or no?"
01:13:00  MODERATOR  "Mr. Bianco."
01:13:01  BIANCO     "Absolutely, yes."
01:13:03  MODERATOR  "Mr. [Villa]raigosa."
01:13:03  SPEAKER A  "No. Louisiana has a ban on abortion without exception for rape
                      and incest. And Mr. Hilton said he would and now Mr. Bianco said
                      they would extradite this woman."
01:13:17  MODERATOR  "Thank you, Mr. [Villa]raigosa."          <- trying to cut him off
01:13:19  SPEAKER A  "This is a state that where we protect the women right to choose.
                      67% of voters voted for a woman's right to choose."   <- THE QUOTE
01:13:26  MODERATOR  "Thank you, Mr. [Villa]raigosa.  Mr. Becerra."
01:13:29  BECERRA    "Absolutely no and when I was AG I protected with reproductive
                      rights here in California."                    <- this is quote #11
```

Speaker A is called by name twice by the moderator before the line and once after; he refers
to Bianco in the third person, so he is not Bianco; and Becerra is explicitly handed the floor
*after* he finishes. The name is ASR-rendered "Biargosa" in this caption track and "Bianco
sir" in the second upload (`E77UzwNExAA` @ 01:35:59) — machine caption tracks are unreliable
on proper nouns — but **both tracks agree on the turn structure**, and that structure is what
matters: whoever said the 67% line, the moderator then thanks him and calls Becerra, who
answers separately. "Biargosa" is a close phonetic rendering of Villaraigosa, and Villaraigosa
was on that stage.

So the certain finding is **the speaker is not Becerra**; the strong finding is that it is
Villaraigosa. This is a classic off-by-one attribution error — the line immediately preceding
the target candidate's turn was captured along with it.

Two further reasons not to salvage this row by re-attributing it: Villaraigosa is not a
candidate in either race in scope; and the stored text is not a faithful contiguous rendering
of what was said in the first place ("We protect the women **of the state** 67% voted…" vs
"we protect the women['s] right to choose. 67% of voters voted…"), with a longest exact run of
only 7 words.

## 16. Xavier Becerra / deportation — **UNRESOLVED — not a quotation**

- **id:** `66088b79-e0dc-4ea7-84b2-d2d0f5c6d676` · draft
- **Quote:** "Support DACA, oppose Muslim ban and family separation"
- **Cited source:** `https://www.xavierbecerra2026.com/immigration`

**Not a quotation, and the cited page has never existed. Recommend retirement.**

Two independent findings:

1. **The cited URL is fictitious.** `https://www.xavierbecerra2026.com/immigration` returns
   **HTTP 404** live (the domain root returns 200, so the site is up — the subpage is not).
   The Wayback CDX index for the whole `xavierbecerra2026.com` domain, `matchType=domain`,
   returns captures of the **root page only** across 2025–2026. No `/immigration` subpage has
   ever been archived. This is not a page that changed; it is a page that was never there.

2. **The text is an aggregator's editorial heading.** It appears verbatim as a section
   *heading* on OnTheIssues:
   `http://web.archive.org/web/20260510015151/https://www.ontheissues.org/Governor/Xavier_Becerra_Immigration.htm`
   (and on the Cabinet-era page,
   `http://web.archive.org/web/20260306010736/https://www.ontheissues.org/Cabinet/Xavier_Becerra_Immigration.htm`):

   > **Support DACA, oppose Muslim ban and family separation**
   > Attorney General Becerra has taken several actions to defend the rights of immigrants,
   > including: Defended Dreamers and challenged the repeal of DACA. …
   > *Source: California Attorney General website: Press Rel…*

   OnTheIssues writes these headings itself to label a position; the body beneath is a summary
   of press releases. They are not Becerra's words in any register.

This is exactly the aggregator pattern already purged from the corpus (all 1,173
ontheissues.org and 273 en.wikipedia.org quotes were hard-deleted on 2026-07-25). This row
survived that purge only because it carries a plausible-looking campaign URL instead of the
aggregator URL it actually came from.

---

# The Bass / Mike Bonin diarization question — determination

The brief asked which of three readings applies to the 6-word run found in
`whatsnextlosangeles.buzzsprout.com/1414123/episodes/18052144-mayor-karen-bass-on-the-record`
that diarization attributes to Mike Bonin: (a) the host quoting/paraphrasing Bass, (b) the
diarization is wrong, or (c) Bass never said it and the quote is misattributed.

**None of the three. The match is a coincidental n-gram collision on a generic phrase, in a
passage about an unrelated subject. The diarization is correct, and the podcast is simply not
the source.**

Evidence:

**1. What the matched segment actually says.** Read-only query against
`meetings.segments` for meeting `b28c9255-a1bf-4cc9-b356-d54baa38b3a6` returned exactly one
matching segment:

```
[segment_index 22, start_time 194.465, speaker_name "Mike Bonin"]
"civic activist to legislative leader to now an executive, the second largest
 city in the country. I'm wondering what the adjustment was for you. Going from
 legislative"
```

The 6-word run is **"the second largest city in the"**. Bonin is asking Bass how she adjusted
from legislator to executive. There is no police officer, no staffing number, no 1995, no
budget — the subject is her career transition. The phrase "the second largest city in the"
is a stock descriptor of Los Angeles and will collide with anything else that uses it.

**2. Chronology rules out (a) outright.** The podcast episode was recorded 16 October 2025 and
published **22 October 2025** (`meetings.meetings.date = 2025-10-22`). The quote originates in
a Bass letter to the City Council reported on **11 December 2025** — seven weeks *later*. Bonin
could not have been quoting or paraphrasing a document that did not yet exist.

**3. (b) and (c) are both unsupported.** The diarization is right: this is the host, asking a
question, in the show's opening minutes. And the quote is not misattributed to Bass — it is
genuinely hers, from her own letter, quoted inside quotation marks in LA Magazine (see #1
above), and the row is TRACED on that basis.

**Practical implication for the audit tooling:** a 6-word threshold on contiguous n-grams is
below the noise floor for stock civic phrases ("the second largest city in the", "the city of
Los Angeles", "we're going to fight against Trump"). This match cost real investigation time
and pointed the wrong way. Worth considering a longer floor, or a stop-phrase list, for
`source-speaker-mismatch` specifically.

---

# Concerns for the human reviewer

1. **Three rows should not ship as written even though they are TRACED.** #10 (Hilton "I" vs
   "We"), #12 (Becerra "attacks" vs "a tax") carry text defects; #12's defect is an ASR
   artefact reproduced verbatim, which is a smell worth chasing across the rest of the
   debate-sourced corpus. #7/#8 truncate mid-sentence and need an `editor_note`.

2. **`editor_note` is deliberately not set anywhere in the draft SQL.** House style plus human
   sign-off, consistent with migration 1566.

3. **The KTLA debate video (`qRNZ0kuA49k`) has no caption track and, on this evidence, was
   never a viable source for any quote.** Four rows cite it. Three are re-sourced here to the
   CBS/Face-the-Nation debate; one (#15) turns out to be a different speaker. Whatever import
   step attached `qRNZ0kuA49k` to these quotes should be treated as suspect for any *other*
   quote citing it.

4. **Two of the six UNRESOLVED rows are fabrications rather than mis-citations** (#3, #16).
   They were not "sourced to the wrong page"; the words were never said. #3 in particular is
   LIVE and in front of voters now. That is a different severity from a stale URL and is
   flagged as the top priority in the draft SQL.

5. **`lapublicpress.org` is unreachable to automated fetching (HTTP 403, no Wayback captures
   for the URLs in question).** It is cited by #1 and is the one page relevant to #13 that
   could not be checked. A human with a browser should confirm.

6. **#6 is campaign-platform prose, not speech.** Sourced and verbatim, but a reviewer should
   confirm platform text is in scope.

7. **`probe.py` initially failed to decompress gzip responses**, producing one false negative
   (the CD4 41.18 page, which on re-check contained quote #5 in full). The tool was fixed and
   every negative result reported above was produced after the fix. Named here so the next
   person knows the negatives were re-run rather than inherited.
