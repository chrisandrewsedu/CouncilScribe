# Mis-merge scan — one label holding two people

**Date:** 2026-09-08 · **Tool:** `mismerge_scan.py` (`src/mismerge.py`) · **Corpus:** 178 local
meeting directories, 172 with a reviewed `transcript_named.json`

## Why this scan exists

`review.duplicate_named_speakers` and `review.ambiguous_speaker_surnames` are both
name-based: they catch a rename that puts one name onto two diarized labels. The
mirror-image mistake — a review-time **merge** that folded two different people into one
label — leaves ONE label carrying ONE name, so both detectors return `{}` and a clean
scan is false reassurance. PR #162's merge-time cosine guard prevents new mis-merges; it
does nothing about the ones already in the corpus.

`embeddings.json` cannot answer the question: it stores one centroid per label, and a
merge already averaged both voices into that vector, so the intra-label spread that would
betray two people is destroyed before it is persisted. The scan therefore re-embeds from
the audio, per turn.

## Method

**Stage 1 — provenance (cheap, exact, no audio).** `transcript_raw.json` keeps the
ORIGINAL diarized labels, so a raw label whose turns now sit inside a *different* named
label is a merge, exactly. Grouping is done at raw-turn granularity because a merge lets
the adjacent-same-speaker pass stitch a source turn and a target turn into ONE named turn,
which then straddles both voices.

Measured: **36 merges in 26 of 172 meetings**. That is ~10 minutes of embedding, against
the ~4 hours a blind re-embed of all 128 hours of corpus speech would cost.

**Stage 2 — acoustic (per-turn re-embedding).** Both sides of each merge are re-embedded
with `pyannote/wespeaker-voxceleb-resnet34-LM` and their centroids compared, banded with
`review.merge_voice_verdict`'s existing calibration (`MERGE_SIM_MISMATCH` 0.42 /
`MERGE_SIM_CONFIDENT` 0.60) rather than a second scale. Each slice is truncated to 20s so
the budget buys many turns instead of two monologues; that change left the mismatch set
byte-identical while making every `match` more confident, which is the robustness check.

Coverage: **all 172 reviewed meetings, none skipped.** The 6 directories without a
`transcript_named.json` were never identified or reviewed, so there is nothing to
mis-merge in them.

## Results — 36 merges judged

| verdict | count | reading |
|---|---|---|
| `mismatch` (cos ≤ 0.42) | 11 | the two sides are different voices |
| `uncertain` (0.42–0.60) | 5 | real ambiguity band |
| `unknown` | 1 | too little embeddable audio to judge |
| `match` (cos ≥ 0.60) | 19 | one voice — a split cluster stitched back |

A `mismatch` is **not** a diagnosis. Case (a) a split diarization cluster and case (b) two
real people both look like this, and per this project's rule you cannot tell which without
reading the transcript. Every mismatch below therefore carries its text evidence.

A `match` is not a clean bill of health either. The 0.42/0.60 bands were calibrated on
pairs already known to share a NAME, and the same-day triage of the five corpus-wide
match-band pairs found 4 of 5 were case (b) — two different people scoring above 0.60. A
`match` here says only "these two sides share a voice"; it does not say the surviving
label holds exactly one person. `2026-04-03-lwv-brown-county-candidate-forum-auditor`
proves the difference: its merge scores +0.629 (`match`, and the merge WAS fine), yet raw
SPEAKER_09 carries three separate self-introductions and is a three-person label from
original diarization. Provenance cannot see that, which is what the second pass below is
for.

### Confirmed real errors — a named candidate's label holding a broadcaster

**1. `2026-07-08-debate-mi-us-senate-dem-primary` — SPEAKER_05 "Haley Stevens" ← SPEAKER_03,
262s over 65 turns, cos +0.107.** The absorbed block is the WOOD TV post-debate spin room
and the host names himself in it: *"Hi everybody, I'm Brian Sterling. We're now in the
debate spin room."* 262 seconds of a television presenter are filed under a US Senate
candidate. This is the largest finding in the corpus and matches the recorded case-(b)
pattern "broadcast news packages inside debate recordings".

**2. `2026-07-14-interview-mi-governor-candidates-affordability` — SPEAKER_05 "Ty Steele"
← SPEAKER_03, 24s over 9 turns, cos +0.253.** The host side is plainly reporter narration
(*"Republican candidate for governor … Mike Cox says the best way to save Michiganers
money is…"*); the absorbed side is **Perry Johnson speaking in the first person** (*"This
is actually the entire theme on which I've been running"*, *"I suggested the mega audit,
the Michigan efficiency government audit. I know Gretchen doesn't want to audit
anything."*). A candidate's own words sit under a reporter's name — the quote-integrity
failure this bug class was expected to cause.

**3. `2026-05-15-governor-debate-(cbs-and-sf-examiner)` — SPEAKER_10 "Kaylee" ← SPEAKER_07,
21s, cos +0.135.** Same shape, inverted: the host side is the reporter's narration about a
teacher (*"Zach Kaley's journey into the classroom…"*, *"Kaley says stagnant salaries…"*)
and the absorbed side is Kaley himself. One label, reporter plus subject, named for the
subject.

**4. `2026-05-12-interview` — SPEAKER_01 "Annie Rose" ← SPEAKER_02 (21s, cos +0.047) and
← SPEAKER_00 (13s, cos +0.106).** Two separate absorbed voices in one label. The
give-away is in the text: *"All right, Annie Rose, thank you very much"* cannot be Annie
Rose. Two other journalists' turns are under her name.

### Confirmed real errors — a candidate's answer under an unidentified label

**5. `2026-07-09-debate-mi-governor-gop-primary` — SPEAKER_03 "Unidentified Speaker",
host SPEAKER_04 71s / absorbed SPEAKER_03 22s, cos +0.383, 1 straddling named turn.**
Perry Johnson's own 71-second answer (*"as many of you know, I grew up in a 600 foot home.
I had to work my way through college…"*) is published under the audience questioner's
unidentified label. Note the shape: this is NOT a label merge — `diarization.json` still
labels those turns SPEAKER_04. ONE named segment spans 3006.8–3106.0s and swallowed the
answer, so no label-level operation can fix it; the segment has to be cut first.

**6. `2026-05-19-candidate-forum-mi-governor-dem-primary` — SPEAKER_16 "Unidentified
Speaker", host SPEAKER_08 40s / absorbed SPEAKER_16 21s, cos +0.121, 1 straddling turn.**
Same swallow shape. Chris Swanson's answer (*"I spent my whole life protecting people's
freedoms and rights…"*) is under the questioner's unidentified label.

**7. `2026-05-19-candidate-forum-mi-governor-dem-primary` — SPEAKER_00 "Unidentified
Speaker" ← SPEAKER_01, 36s, cos +0.242, 0 straddling turns.** Two different audience
questioners pooled into one "Unidentified Speaker". Real (one label, two people) but low
harm — neither is named or published as a person.

### Weaker / likely benign mismatches

**8. `2025-10-22-podcast-mike-bonin-la-mayor-karen-bass` — SPEAKER_02 "Karen Ruth Bass"
← SPEAKER_00, 26s over 30 turns, cos +0.415, 15 straddling turns.** The absorbed turns are
backchannels and empties (*"Mm-hmm."*, *"Yeah. No,"*, blank), almost all sub-second — the
interviewer's interjections pooled into a fragment cluster. cos sits right at the mismatch
boundary and only 8 turns were embeddable. Low harm, but worth a listen because 15 named
turns straddle both.

**9. `2026-03-30-lwv-candidate-forum---county-clerk-and-prosecutor` — SPEAKER_02
"Moderator" ← SPEAKER_00, 4s over 2 turns, cos −0.021, judged on a single 3.4s slice.**
Probably a **false positive**: the absorbed text (*"behalf of the forum - sponsors, remember
to"*) is a mid-sentence continuation of the moderator's own closing. This is the known
Tanner Branham junk cluster, whose fragments were deliberately reassigned to their true
owners on 2026-08-21. It shows the floor of the method: at `MIN_SIDE_SECONDS` = 3.0 a
verdict can rest on one short slice. The floor is kept low on purpose (suppressing is worse
than reporting), and every line prints how much audio it was judged on so thin evidence can
be discounted.

**Also in finding 1's meeting:** a second absorbed side, SPEAKER_05 ← SPEAKER_08 (14s,
cos +0.129), is a **journalist questioning Haley Stevens** — *"Rep Stevens, I just asked
this to um Abdul about the regulation of AI"* — again under her own label. Same meeting,
same error, separate merge.

That accounts for all 11 mismatches: findings 1 (×2 absorbed sides), 2, 3, 4 (×2), 5, 6, 7,
8 and 9.

### `uncertain` band (5) — all look like correct merges with unembeddable evidence

`2026-02-04-council` SPEAKER_21 "Isak Nti Asare" (+0.573), `2026-02-25-council` SPEAKER_02
"Isak Nti Asare" (+0.596), `2026-07-21-mn-senate-minnesota-podcast` SPEAKER_05 (+0.533):
every absorbed side is a handful of sub-second turn-boundary fragments plus one longer
mid-sentence continuation that is obviously the same speaker — e.g. *"Seeing none, we'll
move to council schedule"* is unmistakably the presiding officer. Case (a); the cosine is
weak because there is barely any voice to embed, not because the voices differ.

`2026-05-06-la-mayoral-debate-(nbcla)` SPEAKER_21 "Spencer Pratt" (+0.577, one 5.6s turn)
and SPEAKER_15 "Karen Bass" (+0.542, one 3.7s turn) are genuinely ambiguous single turns.
Both are substantive sentences, so these two are worth 30 seconds of listening each.

### `unknown` (1)

`2026-07-24-mo-representative-district-1-candidate-forum` SPEAKER_03 "Jason Rosenbaum" ←
SPEAKER_02: 1.7s of embeddable speech, below the 3.0s floor. This is the merge applied by
hand earlier on 2026-09-08 (a station ID cut across two labels), so `unknown` is the
correct and honest answer here rather than a finding.

## Second pass — provenance-free bimodality (`--all-labels`)

Provenance is blind to conflation that **diarization itself** created: raw and named agree,
because the two people were one label before review began. `--all-labels` asks the question
directly — re-embed every substantial label's turns and split them in two — for **555 of
1033 labels, 13 minutes** of embedding.

**Result: 53 in the mismatch band, 29 uncertain, 473 match. Treat these as leads only.**

What it does well: it independently re-found five of the provenance scan's findings by a
completely different route — `2026-05-12` SPEAKER_01 (+0.058), `2026-05-19` SPEAKER_16
(+0.121) and SPEAKER_00 (+0.242), `2026-05-15` SPEAKER_10 (+0.135), `2026-07-14`
SPEAKER_05 (+0.253). Two methods agreeing on the same labels is real corroboration.

### 🔴 The measured blind spot — do not re-derive these three

The corpus's one **confirmed** multi-person label is
`2026-04-03-lwv-brown-county-candidate-forum-auditor` SPEAKER_09, established by hand
earlier today: 359 turns / 2304s carrying three self-introductions (moderator Madison
Miller plus both auditor candidates). **This pass scores it +0.676 — match band, rank 105
of 555. It misses it completely.**

The cause is structural, not a tuning problem: a balanced 2-way split cannot find a
*minority* speaker inside a label one voice dominates, because the moderator's own
variation is a bigger axis than moderator-vs-candidate. Two alternatives were measured on
the same case and also failed:

| approach | result on SPEAKER_09 |
|---|---|
| longest-first 2-means (shipped) | **+0.676** — match band |
| time-stratified 2-means (40 turns) | **+0.655** — so it is not a sampling artefact |
| leave-one-out per-turn outlier scoring | the two candidate turns rank 7th and 10th lowest, **below four genuine moderator turns** — duration confounds it (every low scorer is a 2–4s turn) |

Also: 30 of the 53 mismatch-band hits have a side of only two turns, i.e. a boundary
fragment pair rather than a person, so 53 is not a finding count. Filtering to both sides
≥ 4 turns leaves 11, which is a more plausible lead set.

**Conclusion: detecting a minority intruder inside a dominated label is unsolved here.** The
shipped, validated detector is the provenance-gated one. A clean bimodality scan is NOT
evidence a label holds one person.

## Live status — checked, not assumed

Per the standing rule (never infer "not live" from work looking local-only),
`gui.publish_api.live_published_slugs()` was queried: **all 12 meetings carrying a
`mismatch` or `uncertain` finding are LIVE** on otr.empowered.vote. Only the two
`unknown`/`match`-band meetings named above (`2026-07-24-mo-…-candidate-forum`,
`2026-04-03-lwv-brown-county-…`) are not.

Three of the findings were then verified directly against `meetings.segments` — the error
is on the public page, not only in the local artifact:

| meeting | live segment | attributed to | actual speaker |
|---|---|---|---|
| `2026-07-08-debate-mi-us-senate-dem-primary` | 5263.2s | **Haley Stevens** | *"Hi everybody, I'm Brian Sterling"* (WOOD TV) |
| — same, in aggregate | t ≥ 5255s | **Haley Stevens**, 19 segs / 698s | ~262s of that is the presenter and a questioning journalist |
| `2026-07-14-interview-mi-governor-…-affordability` | 96.3–165.2s | **Ty Steele** | Perry Johnson, first person: *"I suggested the mega audit"* |
| `2026-07-09-debate-mi-governor-gop-primary` | 3006.8–3106.0s | **Unidentified Speaker** | questioner **plus** Perry Johnson's whole answer |

Note the prod segmentation can be COARSER than the local artifact (the 2026-07-14 meeting
publishes 18 segments where the local file has far more), so a live segment can carry both
voices in one block even where the local labels look separable. Any fix has to be checked
against what prod actually holds, and each of these meetings needs a republish after
repair.

## How to reproduce

```bash
.venv/bin/python mismerge_scan.py --gate-only          # cheap stage, 4.5s, no model
.venv/bin/python mismerge_scan.py                      # full merge scan, ~40s of embedding
.venv/bin/python mismerge_scan.py <meeting_id>         # one meeting
.venv/bin/python mismerge_scan.py --json findings.json
.venv/bin/python mismerge_scan.py --all-labels         # second pass, ~13 min (leads only)
```

Nothing is mutated. There is no `--apply`: the remedy differs per case (re-split, per-segment
reassignment, re-identify, or a segment cut before any label work), and choosing needs a
human reading the two spans.
