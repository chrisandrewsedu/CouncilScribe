# Orphan quote provenance — Los Angeles Mayor (Bass / Raman)

**Date:** 2026-08-07  
**Scope:** the 30 `essentials.quotes` rows for Karen Bass and Nithya Raman with `source_url IS NULL`, blank `source_name`, blank `editor_note`, and `created_at IS NULL`.  
**Race:** Los Angeles Mayor — `9e888818-c50b-4c61-a106-a0839ff2479d` (Nov 3 2026 general/runoff). Most of this content was built against the Jun 2 2026 primary row `24bc3631-22cf-41ab-a731-672481502214`.  
**Outcome:** **30 of 30 CONFIRMED** to a single event. **0 UNRESOLVED.**

> **APPLIED 2026-08-07.** The research below was read-only; the backfill was then reviewed, dry-run and committed to production as ev-accounts migration **`1566_backfill_la_mayor_debate_provenance.sql`**. All 30 rows now carry `source_url` + `source_name`, and **0** Bass/Raman quotes remain unsourced. The four lowest-similarity matches were independently re-verified against `meetings.segments` before applying — longest contiguous runs 35w / 56w / 10w / 87w, word coverage 0.97 / 1.00 / 0.88 / 1.00. The sub-1.00 scores are interior elisions across a segment boundary, as the report states.
>
> **Follow-up applied:** ev-accounts migration **`1567_dedupe_la_mayor_debate_quotes.sql`** resolved the duplicate pairs flagged in the Concerns section — see [Dedupe outcome](#dedupe-outcome-applied). Bass + Raman now hold **77** quotes, down from 81.
>
> **Still open:** `editor_note` is blank on **56** Bass/Raman quotes. Notes need house style and human sign-off; the elision annotations below are the raw material.

---

## 1. The event

All 30 quotes come from **one** event:

| | |
|---|---|
| Event | 2026 Los Angeles mayoral primary debate |
| Date | **2026-05-06** |
| Hosts | NBC4 Los Angeles (KNBC) and Telemundo 52 |
| Participants | Karen Bass, Nithya Raman, Spencer Pratt |
| Primary video | `https://www.youtube.com/watch?v=8rI3A6alVHM` (NBCLA, 6,340 s) |
| Re-upload | `https://www.youtube.com/watch?v=-83WHHCKZDY` — "Full NBC4 broadcast", 3,474 s, uploaded 2026-05-07 |
| **Already ingested as** | `meetings.meetings` **`f2cf80ef-a811-4d95-990d-b9c598284eb6`** — `2026-05-06-la-mayoral-debate-(nbcla)`, `event_kind='debate'`, `status='published'`, 379 segments |

This is a **primary-season** debate (three candidates on stage), not a runoff debate — which explains Raman's `I'm the only person on the stage` phrasing. No Bass–Raman runoff debate had occurred as of this audit.

### The source was already in our own database

The decisive finding is that this debate **is already ingested** as meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, and four other Bass/Raman quotes already cite it using the house convention:

```
source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-…?t=<segment start>#seg-<segment_index>'
```

The 30 orphans are almost certainly a bulk extraction from **this meeting's segment transcript** that bypassed `publish-quotes`. The strongest evidence is that where the two available transcripts disagree, the stored quote text follows the ingested one: quote `01c51a86` reads `I'm the only person on **the** stage`, which is what `meetings.segments` seg-186 says, while the YouTube auto-caption says `on **this** stage`.

---

## 2. How each quote was verified

A quote is treated as confirmed only where a **distinctive contiguous run of its own words appears in the source**. Topic, date and participant agreement were not accepted as confirmation.

Each quote was matched independently against **two transcripts of the same event**:

1. **`meetings.segments`** for meeting `f2cf80ef-…` (VTT-aligned, speaker-diarized) — read-only query.
2. **YouTube auto-captions** for `-83WHHCKZDY`, downloaded with `yt-dlp` and de-duplicated (rolling captions repeat each line up to three times, which defeats naive matching).

Three independent signals had to agree for each quote:

- an exact contiguous word run (shortest observed: **12 words**; median **45.5**; longest **100**);
- the same run present in **both** transcripts;
- **speaker diarization** in `meetings.segments` naming the expected candidate. This held for **30/30** — every Bass quote landed in a `Karen Bass` segment and every Raman quote in a `Nithya Raman` segment. Attribution is therefore independently corroborated, not assumed.

Where the whole-quote similarity is below 1.00 it is because of **interior elisions** — the curator dropped moderator interruptions, crosstalk, or an attack line from the middle of a single continuous answer. In every such case both halves were located in the same segment and the same speaker turn; these are noted per row below.

---

## 3. Confirmed quotes (30)

`t` is the segment start used in the URL (matching the existing convention). `t≈` is the interpolated start of the quote's own first word within `8rI3A6alVHM`, for tighter deep-linking if wanted.

### Karen Ruth Bass (14)

#### `2638f09b-b695-4c5b-a29c-a3c7cb739a10` — city-sanitation

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=5293#seg-453>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-453)
- **Timestamp:** `t=5293` s (segment start); quote begins ≈ `5293` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=5293s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (32 contiguous words, present in both transcripts):**

  > we need cooperation from the governor for example we need the governor to support us in cleaning
  > up our highways doing the landscaping it looks a mess we need the graffiti removed

#### `ce9bb5b9-ac51-4eec-8765-2ebe783e316d` — economic-development

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4882#seg-395>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-395)
- **Timestamp:** `t=4882` s (segment start); quote begins ≈ `4885` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4882s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (54 contiguous words, present in both transcripts):**

  > in working with the industry we have expedited permits i established one person who is a czar to
  > provide concierge services for the industry we also lowered the cost to do filming in los
  > angeles so for example the observatory we reduced those amounts by 70 we are beginning to
  > bring the industry back

#### `e294ef7e-5f85-42d6-9b89-cf018e817001` — growth-and-development

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4806#seg-391>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-391)
- **Timestamp:** `t=4806` s (segment start); quote begins ≈ `4824` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4806s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (53 contiguous words, present in both transcripts):**

  > did you know that we are the only major city that does not have a comprehensive infrastructure
  > plan so it's been haphazard up until now left to the wishes of every single council member i
  > have instituted and started a comprehensive infrastructure plan and moving right away into
  > replacing 60 000 solar lights

#### `8418a331-2eeb-418d-8473-fd36d5671276` — homelessness

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2074#seg-126>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-126)
- **Timestamp:** `t=2074` s (segment start); quote begins ≈ `2116` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2074s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.961
- **Verbatim run matched (38 contiguous words, present in both transcripts):**

  > homelessness was going up year after year and under my watch it is the first time we've had a
  > decrease in street homelessness while it went up in the country 18 it came down in los
  > angeles 17

  **Note:** Source renders the figure as `17 .5%`; stored text spells it `17 and a half percent`.

#### `dd5bdce9-667c-4891-955a-98ec6c49d44e` — homelessness

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3066#seg-207>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-207)
- **Timestamp:** `t=3066` s (segment start); quote begins ≈ `3066` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3066s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (19 contiguous words, present in both transcripts):**

  > everybody needs to go inside making it illegal and arresting people is not the way to solve this
  > problem

  **Note:** Near-duplicate of already-sourced `5230bec6` — see Concerns.

#### `4c389a82-e480-4e6e-b4f4-fe8576ced385` — homelessness-response

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3159#seg-238>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-238)
- **Timestamp:** `t=3159` s (segment start); quote begins ≈ `3159` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3159s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.994
- **Verbatim run matched (41 contiguous words, present in both transcripts):**

  > strategy and homelessness was going up year after year we need to have an entirely new system
  > that is frankly independent in the city of los angeles because we need to build out services
  > once we get people off the street

#### `e0ec0298-540a-4ca7-be15-d8c07d4f1412` — homelessness-response

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3551#seg-273>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-273)
- **Timestamp:** `t=3551` s (segment start); quote begins ≈ `3564` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3551s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (88 contiguous words, present in both transcripts):**

  > for the first time we've had a reduction of homelessness two years in a row because of policies
  > that i have put in place i also believe we need to have an overhaul of the system but i can
  > tell you that streets that we have cleared crime is down firefighters don't have to go out
  > firefighters spend 30 of their time putting out fires that are related to homelessness
  > businesses are able to have customers kids are able to walk to school and parents without
  > navigating tents

#### `e606a5f8-ff2f-4150-8d1b-60b81a4e8652` — homelessness-response

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3880#seg-301>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-301)
- **Timestamp:** `t=3880` s (segment start); quote begins ≈ `3881` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3880s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.995
- **Verbatim run matched (65 contiguous words, present in both transcripts):**

  > years in a row we absolutely need a system a system that is able to provide services to people
  > it has been woefully unacceptable what has been happening so far i feel we're moving in the
  > right direction and under my watch we will continue to do that because we need to make
  > comprehensive changes to what has been going on for so many years

#### `0a1aacc9-3044-4b79-9e7a-a6095cc05824` — housing

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3940#seg-305>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-305)
- **Timestamp:** `t=3940` s (segment start); quote begins ≈ `3949` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3940s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (76 contiguous words, present in both transcripts):**

  > 42 000 units are being fast tracked for affordable housing we have another 43 000 units that are
  > potential with our adaptive reuse which means you can change office buildings into housing
  > we also changed the zoning codes so we have the potential for another half million units i
  > know that one of the primary drivers for the lack of affordability in our city is housing
  > and making sure that people can afford to live here

#### `78ba1ed3-e7b3-471c-911b-4e4d49d87cde` — housing

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4082#seg-321>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-321)
- **Timestamp:** `t=4082` s (segment start); quote begins ≈ `4088` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4082s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (79 contiguous words, present in both transcripts):**

  > we have cut red tape so you can go through much quicker than that the 42 000 units of housing
  > that are being fast tracked 6 000 of those units are actively under construction but there
  > are definitely other factors that have weighed into that and some of the factors are the
  > price of construction materials just the general economy and we are doing everything we can
  > to make sure that we are able to fast track that housing

#### `368cfab9-97e5-4ad6-afbf-477b4633e1cc` — public-safety-approach

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2540#seg-162>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-162)
- **Timestamp:** `t=2540` s (segment start); quote begins ≈ `2570` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2540s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.741
- **Verbatim run matched (35 contiguous words, present in both transcripts):**

  > los angeles is understaffed in terms of lapd for the nation's second largest city and so i have
  > been fighting to hire more officers unfortunately i have not had the cooperation from the
  > city council

  **Note:** Interior elision: the source continues `...including unfortunately my colleague next to me who has voted repeatedly against hiring` before `What is in my current budget now is 512 officers...`. Both halves verified in seg-162.

#### `132093b7-7acd-437f-a2fd-cee5af8f2704` — public-safety-approach

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2989#seg-197>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-197)
- **Timestamp:** `t=2989` s (segment start); quote begins ≈ `3005` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2989s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.995
- **Verbatim run matched (57 contiguous words, present in both transcripts):**

  > had to give them a raise had to expand recruitment because officers were going to other cities
  > we were not competitive we have to make a decision and frankly my job as mayor my number one
  > job is to keep la safe and we can't keep la safe with the size of the department we have now

  **Note:** Interior elision across a moderator interruption (`Bass, thank you, your time is up`).

#### `8edf4cb0-ed77-4e93-8ba0-a8dbdc04a09e` — rent-regulation

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3940#seg-305>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-305)
- **Timestamp:** `t=3940` s (segment start); quote begins ≈ `3994` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3940s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (26 contiguous words, present in both transcripts):**

  > every single thing we can do rent stabilization reducing the price of rent eviction prevention i
  > started the mayor's fund that allows us to do that

#### `71044f3e-380e-4887-9169-95fbd7926f08` — residential-zoning

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4277#seg-352>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-352)
- **Timestamp:** `t=4277` s (segment start); quote begins ≈ `4294` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4277s>
- **Diarized speaker:** Karen Bass
- **Whole-quote similarity:** 0.741
- **Verbatim run matched (31 contiguous words, present in both transcripts):**

  > there's ways to do it we are on our road to do it we did not need sacramento to tell us what to
  > do and to mandate what we were doing

  **Note:** Interior elisions (a named-councilmember aside and a Sherman Oaks reference removed). Opening sentence verified verbatim at seg-352. Near-duplicate of already-sourced `5c090c4e` — see Concerns.

### Nithya Raman (16)

#### `01c51a86-040c-414a-bdc4-c978d8b76c8e` — campaign-finance

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2861#seg-186>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-186)
- **Timestamp:** `t=2861` s (segment start); quote begins ≈ `2880` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2861s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (55 contiguous words, present in both transcripts):**

  > the police union is the most powerful force in la city politics and when you give contracts to
  > people because they are going to fund your campaigns and by the way i'm the only person on
  > the stage that is being spent against because i had the courage to speak out about bad
  > budget decisions

  **Note:** Stored text matches the ingested transcript exactly, including `on the stage` (the YouTube auto-caption renders this as `on this stage`).

#### `a6bb4672-eaf0-4f93-991c-040440c1394c` — city-sanitation

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4521#seg-358>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-358)
- **Timestamp:** `t=4521` s (segment start); quote begins ≈ `4561` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4521s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.864
- **Verbatim run matched (12 contiguous words, present in both transcripts):**

  > it needs regular cleanups it needs real maintenance it needs a strategy

  **Note:** Condensation: source reads `Downtown LA needs attention and it needs real care`. Overlaps `f7625f7b` (same answer, same segment) — see Concerns.

#### `f7625f7b-634f-49d4-9155-ab38455ba400` — economic-development

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4521#seg-358>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-358)
- **Timestamp:** `t=4521` s (segment start); quote begins ≈ `4549` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4521s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (89 contiguous words, present in both transcripts):**

  > downtown la needs attention and it needs real care it needs more public safety officials on the
  > streets it needs work with businesses to ensure that businesses aren't just fleeing downtown
  > la that they're actually staying there it needs regular cleanups it needs real maintenance
  > it needs a strategy instead what mayor bass has done is to dismantle our economic
  > development department we don't have a strategy to keep businesses here in los angeles and
  > we're watching as they walk away from this city instead of investing in it

  **Note:** Same answer as `a6bb4672` (city-sanitation) — see Concerns.

#### `644bc8e7-4545-4f50-b40e-0c2b6ae0a055` — economic-development

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4996#seg-400>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-400)
- **Timestamp:** `t=4996` s (segment start); quote begins ≈ `5023` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4996s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (75 contiguous words, present in both transcripts):**

  > we need to make sure that we're reducing red tape in city hall to make sure that productions
  > have no bar to being able to film here i also would create a real film office here we don't
  > have enough people at the city to make sure that filming can happen quickly and efficiently
  > we need people who know the industry and know the city and can make sure that we're working
  > across county jurisdictions

#### `9ea38e55-6e78-4ec1-b41c-eb74fa6a5217` — growth-and-development

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-354)
- **Timestamp:** `t=4338` s (segment start); quote begins ≈ `4380` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4338s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.997
- **Verbatim run matched (44 contiguous words, present in both transcripts):**

  > we can do it if we plan for it and as mayor that's exactly what i would do i would ensure that i
  > would use the executive authority in the mayoralty to make sure that every department was
  > planning for density in ways that

  **Note:** Overlaps `84f56b9d` by 31 contiguous words (same answer, same segment) — see Concerns.

#### `7d2e49cd-81bc-411b-bef6-498552588a1e` — homelessness

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3072#seg-209>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-209)
- **Timestamp:** `t=3072` s (segment start); quote begins ≈ `3072` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3072s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.966
- **Verbatim run matched (12 contiguous words, present in both transcripts):**

  > people need to go inside when they're offered shelter they go inside

  **Note:** Near-duplicate of already-sourced `27f705ef` — see Concerns.

#### `ee3427dd-f91a-459f-8712-98117e3e428c` — homelessness

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3092#seg-213>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-213)
- **Timestamp:** `t=3092` s (segment start); quote begins ≈ `3092` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3092s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.923
- **Verbatim run matched (20 contiguous words, present in both transcripts):**

  > i support keeping our streets safe i did vote against the structure of this particular ordinance
  > and it is because

  **Note:** Interior elision across a moderator interruption (`is it yes or no`). Closing clause `it does not keep our children safe` verified in seg-213.

#### `c31d035f-deb2-4472-8b18-51e42d841244` — homelessness-response

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3240#seg-241>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-241)
- **Timestamp:** `t=3240` s (segment start); quote begins ≈ `3269` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3240s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.993
- **Verbatim run matched (34 contiguous words, present in both transcripts):**

  > 100 000 a year motel rooms for a year or more per person this system is not fiscally sustainable
  > and we must work to end this crisis with urgency and with accountability right now

#### `b9139e1d-824f-4670-b46c-caa624b2a83d` — homelessness-response

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3636#seg-276>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-276)
- **Timestamp:** `t=3636` s (segment start); quote begins ≈ `3642` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=3636s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.997
- **Verbatim run matched (73 contiguous words, present in both transcripts):**

  > no one even as we're spending hundreds of millions of dollars every year there is no
  > accountability in the city there's no staff at the city that are making sure that every
  > single dollar that you are spending your tax dollars are going to the issue of homelessness
  > people are not watching to make sure those are going towards outcomes they're not watching
  > to make sure that every dollar is being spent appropriately

#### `baa16620-68a5-465e-85b9-71c9b4f5b6f0` — housing

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4006#seg-307>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-307)
- **Timestamp:** `t=4006` s (segment start); quote begins ≈ `4012` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4006s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.777
- **Verbatim run matched (34 contiguous words, present in both transcripts):**

  > the cost of housing is driving young families out of the city it's driving young people out of
  > the city it is making this into a city that is no longer one of opportunity

  **Note:** Interior elision: source continues `...and it is not central to anyone else's agenda on this stage, except for mine. The mayor has not had a deputy mayor of housing during a housing crisis for the last two years.` before `The ED-1 program has entitled 42,000 units...`. Both halves verified in seg-307.

#### `bdc80d45-40c2-4f1d-a0ef-e5619aab2d74` — housing

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4006#seg-307>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-307)
- **Timestamp:** `t=4006` s (segment start); quote begins ≈ `4044` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4006s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.999
- **Verbatim run matched (73 contiguous words, present in both transcripts):**

  > housing apartments exactly the kind of housing that we desperately need in order to bring prices
  > down why is the city standing in the way as mayor i will take my executive authority over
  > the departments and ensure that they respond to new apartment applications within 60 days if
  > they are zoning compliant so that we can build exactly the kind of housing that will make
  > this into a city of opportunity again

#### `eaaa2cdb-c56d-4be3-a798-b4e4e85c4836` — local-environment

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2412#seg-157>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-157)
- **Timestamp:** `t=2412` s (segment start); quote begins ≈ `2428` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2412s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 1.000
- **Verbatim run matched (100 contiguous words, present in both transcripts):**

  > in my role as the council member for a hillside area we have to do a lot of work preparing for
  > really extreme weather before rainy seasons actually in the past mudslides have been the
  > biggest issue in my district entire homes have gone off of their foundations and we go in
  > advance and make sure that traps are cleared that areas are available for flood and clearing
  > we do brush clearance and make sure that all of the departments and all of the inter
  > jurisdictional areas that are supposed to be doing brush clearance are actually doing that
  > work

#### `6c92a9fc-0082-4612-8f22-7d5f97097c09` — public-safety-approach

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2861#seg-186>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-186)
- **Timestamp:** `t=2861` s (segment start); quote begins ≈ `2862` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2861s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.994
- **Verbatim run matched (38 contiguous words, present in both transcripts):**

  > they and all of our city employees deserve a living wage they deserve to be paid what i am
  > arguing against is bad fiscal management which is what has gotten the city of los angeles
  > into this moment

#### `d7490082-efcf-4820-b571-8dc9a642f782` — public-safety-approach

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2918#seg-193>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-193)
- **Timestamp:** `t=2918` s (segment start); quote begins ≈ `2921` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=2918s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.989
- **Verbatim run matched (47 contiguous words, present in both transcripts):**

  > the city's most important response is public safety it is so important to me i'm the mother of
  > young children the safety of people in los angeles is absolutely essential but that's why i
  > think we have to start getting honest with people about how we're actually

  **Note:** Source renders `picks up the phone and 911`; stored text reads `at 911`.

#### `d8f9bf5d-0a2b-45e4-b98f-d186f401c73c` — residential-zoning

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-354)
- **Timestamp:** `t=4338` s (segment start); quote begins ≈ `4342` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4338s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.940
- **Verbatim run matched (56 contiguous words, present in both transcripts):**

  > they come in and impose these mandates on us here's what i would do instead let's actually plan
  > for the housing that we need let's build out the neighborhoods that we want to build let's
  > make them beautiful let's actually solve for the kind of density and affordability that the
  > state is asking us to do

  **Note:** Interior elision: source includes `...including from this mayor who has pushed back against these state mandates` before `they come in and impose these mandates on us`.

#### `84f56b9d-ca40-4268-b02a-81d4ac116ce0` — residential-zoning

- **Source:** <https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354>
- **Event:** NBC4 / Telemundo 52 LA mayoral debate, 2026-05-06 (meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6`, seg-354)
- **Timestamp:** `t=4338` s (segment start); quote begins ≈ `4387` s
- **Video:** <https://www.youtube.com/watch?v=8rI3A6alVHM&t=4338s>
- **Diarized speaker:** Nithya Raman
- **Whole-quote similarity:** 0.995
- **Verbatim run matched (22 contiguous words, present in both transcripts):**

  > i would use the executive authority in the mayoralty to make sure that every department was
  > planning for density in ways that

  **Note:** Overlaps `9ea38e55` by 31 contiguous words (same answer, same segment) — see Concerns.

---

## 4. Concerns for the human reviewer

These do not affect the provenance finding, but they should be settled before the backfill is applied, and certainly before any of these 30 are flipped to `readrank_selected = true`.

### 4.1 Three orphans duplicate quotes that already cite this same debate

Three already-sourced quotes point at the raw YouTube URL `.../watch?v=8rI3A6alVHM` for the same utterances. Backfilling the orphans as-is would leave two rows per moment on the same candidate+topic:

| Orphan | Already-sourced twin | Topic | Shared contiguous words |
|---|---|---|---|
| `71044f3e` | `5c090c4e` | residential-zoning | 20 |
| `dd5bdce9` | `5230bec6` | homelessness | 14 |
| `7d2e49cd` | `27f705ef` | homelessness | 12 |

All six rows are currently `readrank_selected = false`, so nothing user-visible is affected today. **Recommendation:** keep one row per utterance — the orphan versions are generally the fuller passage — and retire the other. That is an editorial call, so the draft SQL backfills all 30 and does not delete anything.

Separately, those three existing rows cite the bare YouTube URL rather than the meeting convention used by the other four quotes from this event. Worth normalising at the same time.

### 4.2 Two pairs of orphans overlap each other

| Pair | Topics | Shared contiguous words | Segment |
|---|---|---|---|
| `9ea38e55` / `84f56b9d` | growth-and-development / residential-zoning | 31 | seg-354 |
| `a6bb4672` / `f7625f7b` | city-sanitation / economic-development | 12 | seg-358 |

Each pair is one answer split across two topics. This may be legitimate under the question-as-unit model, but `84f56b9d` is a strict substring of `9ea38e55`, which is harder to defend — a voter could meet the same sentence twice in one race.

### 4.3 Two rows carry small wording drift from the source

- `8418a331` — source reads `17 .5%`; stored text spells it `17 and a half percent`.
- `d7490082` — source reads `picks up the phone and 911`; stored text reads `at 911`.

Both are ASR-level artifacts rather than meaning changes, but the reveal shows the quote as the source of truth, so a human should decide whether to normalise the stored text.

### 4.4 `editor_note` is still blank on all 30

Per the task constraint, the draft SQL sets only `source_url` and `source_name`. All 30 rows still need house-style `editor_note` text — justifying the elisions listed above and stating Compass-stance alignment — before they can be selected. The elision notes in §3 are the raw material for that, not a substitute for it.

---

## 5. Search record

Kept so this ground is not re-covered. Nothing here ended UNRESOLVED, but the path is worth recording: web search identified the event, while the citable source turned out to be already in our own database.

**Web searches run**

- `"Karen Bass" "Nithya Raman" mayoral debate 2026 transcript`
- `Los Angeles mayor debate 2026 Bass Raman "SB 79" Sacramento zoning`
- `Bass Raman LA mayor runoff debate July 2026 OR August 2026`

**Dead ends and corrections**

- Secondhand reporting on the debate (The Hill, LAist, The Real Deal, NBC News) was found early and was **not** accepted as a source — it reports *about* the debate. The Real Deal did quote Bass's `I don't support Sacramento saying that this is what we need to do here in Los Angeles` verbatim, which raised confidence, but confirmation waited for the transcript.
- Search summaries disagreed on the venue and date — one said *Sherman Oaks, May 5*, another *Skirball Cultural Center, May 6*. The May 5 / Sherman Oaks reading appears to be an artifact of Bass saying `I always talk about Sherman Oaks` during the debate. The NBCLA video description states **Wednesday, May 6, 2026**, and the ingested meeting row independently carries `date = 2026-05-06`.
- The re-upload `-83WHHCKZDY` (3,474 s) was found first and used for the initial match. It is **not** the canonical asset: the longer original `8rI3A6alVHM` (6,340 s) is what the meeting was ingested from, and the two run at different offsets. Timestamps in §3 are all against `8rI3A6alVHM` / the meeting.
- A runoff (Bass v. Raman head-to-head) debate was searched for and none was found, consistent with the three-candidate `on the stage` phrasing.

**Database queries run (all read-only)**

- `essentials.quotes` — the 30 orphans; source conventions for YouTube-sourced quotes; all Bass/Raman quotes for duplicate detection.
- `essentials.discovered_sources` — schema and existing rows; confirmed **no** row exists for either video.
- `meetings.meetings` / `meetings.segments` — meeting `f2cf80ef-…`, 379 segments, used as the primary verification transcript.

**Not needed**

- Ballotpedia and Vote411 were not consulted. The source was confirmed directly against two transcripts, so a debate-listing page would have added nothing.


---

## Dedupe outcome (applied)

Four rows repeated another row's moment from this same debate. Kept exactly one per moment,
chosen on **content, not age** — so two of the drops are pre-existing curated rows.

| Question / candidate | Kept | Dropped | Why |
|---|---|---|---|
| residential-zoning / Bass | `5c090c4e` | `71044f3e` | Not strict duplicates — two sentences from one answer. The keeper carries the tension ("We need absolutely more housing built, **but** SB 79…"); the drop opened with flat opposition, and its blind text rewrote "I don't support" into "[The candidate] doesn't support" instead of marking a cut. |
| homelessness / Bass | `dd5bdce9` | `5230bec6` | The keeper is the drop **plus** its opening line, "Everybody needs to go inside." Without it Bass states only what she opposes and carries no forward position. The dropped row's `editor_note` was merged onto the keeper, adapted to cover the fuller quote. |
| homelessness / Raman | `27f705ef` | `7d2e49cd` | The keeper retains "You don't get an opportunity to say no" — the mechanism that distinguishes her from Bass on this question. |
| growth-and-development ∕ residential-zoning / Raman | `9ea38e55` | `84f56b9d` | `84f56b9d` was a strict substring filed under a *different* question, so the same sentence could appear twice in one race. Raman keeps a stronger residential-zoning answer (`d8f9bf5d`, exact 106-word match). |

All four drops were drafts; none was a question's `origin_quote_id`. The migration guards abort if
any had gone live or if a keeper were missing.

### What this pairing produced

Dropping `5230bec6` and `7d2e49cd` leaves the cleanest head-to-head in the race, both candidates
answering the same moderator question:

> **Bass:** "Everybody needs to go inside. Making it illegal and arresting people is not the way to
> solve this problem."
>
> **Raman:** "Yes, people need to go inside. When they're offered shelter, they go inside. You don't
> get an opportunity to say no."

Same goal, genuinely different mechanism — Raman adds compulsion, Bass rejects criminalisation.
Under the comparability rubric that is *commensurable* **and** *differentiated*: a real choice, and
a good seed entry for the casebook.

## Note for the contrast work

This is a **primary** debate — Bass, Raman and Spencer Pratt — not a runoff debate. No Bass–Raman
head-to-head has occurred. That is the **MI Governor** pattern from the comparability casebook, not
the AZ-01 one: both November candidates answered the *same moderator questions in the same room*,
which is the configuration that yields the highest comparability. The shared questions are
recoverable from `meetings.segments` for meeting `f2cf80ef-a811-4d95-990d-b9c598284eb6` and should
seed `readrank_questions` rows with `origin = 'moderator'`.

Pratt's answers are in the same transcript. He is not on the November ballot, so he is out of scope
for this race — but the *questions* he was asked are the same ones, and they are the durable asset.
