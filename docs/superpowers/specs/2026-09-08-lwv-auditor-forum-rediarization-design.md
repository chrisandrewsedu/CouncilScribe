# Repairing the LWV Brown County auditor forum diarization

**Date:** 2026-09-08
**Meeting:** `2026-04-03-lwv-brown-county-candidate-forum-auditor` (61 min, `event_kind='forum'`)
**Status:** design, approved. Not live — nothing public is affected, no republish pressure.

---

## Problem

One diarized label, `SPEAKER_09`, holds 2295s of the meeting's 3049s of speech
(75%) and contains three different people. In `transcript_named.json` that label
is named "Madison Miller (Moderator)" with `human_review` provenance, so both
auditor candidates' entire forum answers publish under the moderator's name.

The label cannot be repaired by merges or splits at review level: `src/review.py`
offers `merge_speakers`, `rename_speaker` and linking, but no split. A label that
holds three people can only be fixed upstream of review.

### Two corrections to the original diagnosis

**1. This meeting was not diarized by single-pass OSS pyannote.**
`transcript_named.json` records `processing_metadata.diarization_model =
"pyannote/ai-precision-2"`. It ran through `src/diarize_api.py`, the pyannote.ai
Precision-2 REST path. Local OSS pyannote 3.1 is therefore a genuinely different
model that has never run on this audio, not a repeat of what already failed.

**2. The segmentation is correct. Only the labels are wrong.**
The briefing reports segments containing two people's speech. That is true of
`transcript_named.json` but false of `transcript_raw.json`, and the difference is
the adjacent-same-speaker merge, not diarization. Precision-2 placed a clean
boundary at every question-to-answer transition:

| raw turn | span | text |
| --- | --- | --- |
| 12 | 195.50–197.60 | "Okay, are we ready, timer? All right." (moderator) |
| 13 | 198.12–221.66 | "Hi, I'm Andy Vasquez Bond…" (candidate) |
| 63 | 463.64–470.69 | "Ms. Bond, same question…" (moderator) |
| 64 | 471.58–488.55 | "So I think my experience with the Treasurer's office…" (candidate) |

Named segment 2 (195.50–296.87) and named segment 10 (463.64–560.02) are merges
across those boundaries. So this is a clustering failure over good segmentation,
and the fix does not have to re-run segmentation.

### Scope of the damage

`SPEAKER_09` spans 8.5s–3670.5s and holds 619s of speech after t=2000, inside the
meet-and-greet section. Two further labels span suspiciously wide ranges:
`SPEAKER_03` (731→3327) and `SPEAKER_04` (377→3260). The damage is not confined
to the forum Q&A.

---

## Goal

Publishable quality: the two auditor candidates' answers correctly attributed, to
a standard that supports sourcing quotes for the Brown County Auditor race.

Human review is **label-level only** — the operator arrives at a label set where
each label is one person, and does the normal name-and-link pass in the GUI.
Correct clustering must therefore be delivered and *proved* before review, without
anyone listening to the audio.

---

## The acceptance gate

Because the result must be proved without listening, the first deliverable is a
measurement, not a diarization.

### The oracle

The moderator names who speaks next before nearly every answer ("Ms. Bond, same
question", "All right, Teresa, you have your opening statement"). Each named
handoff opens a window belonging to that candidate until the moderator takes the
floor back. This yields a per-turn reference built from text alone, independent
of any voice model.

Prototype measurement over the incumbent turns:

| | |
| --- | --- |
| Handoff anchors | 29 |
| Reference coverage | 1907s of 2239s forum speech (85%) |
| Self-introduction checks passed | 6 of 7 |

### Two known defects in the reference, to fix before gating on it

1. **Split anchors are missed.** Turn 331 fails because the closing-remarks
   handoff spans two turns — the cue ("closing remarks") is in turn 329, the name
   ("Kobian") in turn 330. The anchor matcher must read each cue turn together
   with its successor.
2. **Answer windows over-run.** The prototype assigns Bond 1088s against Kobian's
   622s. In a two-candidate forum with equal timed answers a near-2:1 split is
   wrong; some windows continue past the point where the moderator resumed. The
   moderator-resumption cue list is too thin.

Both are correctness bugs in the gate, not refinements. The reference is not
trustworthy until they are fixed.

### The gate

Applied to any candidate turn set:

- Every reference person maps to a distinct label. No label holds two of the three.
- For each label, minority share against the reference is at most **5%** of its speech.
- **Fragmentation is reported but does not fail the gate.** An extra unnamed
  speaker is fixable at label level in seconds. Conflation is not, and conflation
  is what filed both candidates under the moderator's name.

### Baseline

The incumbent Precision-2 clustering fails maximally: `SPEAKER_09` holds all three
reference people across 2295s of 3049s. Both numbers get published side by side so
the improvement is a measurement rather than a claim.

### Guard against tuning on the test set

Any threshold is calibrated on **odd-numbered anchors** and scored on
**even-numbered anchors**. Tuning and reporting on the same 29 anchors would
prove nothing.

---

## Approach

Two experiments, both on Modal (`--compute modal`; `~/.modal.toml` present, client
1.5.0). Both run standalone and score against the gate. Neither touches the
pipeline or the meeting directory. A pipeline run happens only after one passes.

Shared step: upload `audio.wav` once via `src.modal_compute.upload_audio`.

### Experiment A — OSS pyannote 3.1, single-pass, L4

Call `src.modal_compute.run_diarization(wav_path, meeting_id, use_merge=False,
diarizer="oss", chunk_minutes=0)` directly rather than through `run_local.py`. It
dispatches `bench/modal_app.py:1692 pipeline_diarize_and_embed` on one L4 and
returns `(segments, centroids)`. Score the segments and stop.

`chunk_minutes=0` is the single-pass path, which is also what the kind gate would
choose: `forum` is not in `DIARIZE_CHUNK_EVENT_KINDS`, and a 61-minute meeting is
one window regardless.

The honest prior is that this fails. `src/config.py:248` records that pyannote's
own clustering merges speakers when many voices each hold little speech and turns
are short, and that this mechanism — not the kind label — is why `forum` is
excluded from chunking. This meeting is exactly that shape. It runs anyway
because it costs one call and no new code, so measuring beats assuming.

`--num-speakers` is **not** used. The forum has three voices, but the true count
across the meet-and-greet is unknown, and forcing a wrong K trades one conflation
for another.

### Experiment B — re-cluster Precision-2's turns (expected winner)

`bench/modal_app.py:1305` `pipeline_extract_embeddings(meeting_id, segments_json)`
accepts arbitrary segments and returns one wespeaker centroid per `speaker_label`.
Passing the 479 turns with a **unique label each** makes the returned per-speaker
centroid the per-turn embedding. Experiment B therefore needs no new Modal code
and no local GPU work — one L4 call.

Locally: agglomerative clustering at **average** linkage over those vectors.
Linkage is not a free choice — `src/config.py:329` records that `complete` merges
almost nothing (a real person's worst turn pair is often anti-correlated) and that
`centroid` conflated two real people at the most conservative threshold tested.

Three rules fixed by design:

1. **Unembeddable turns go to one bucket label.** 32 turns fall under the worker's
   0.3s floor, carrying 2.6s of speech in total. Assigning them by adjacency would
   guess at exactly the question-to-answer boundaries that matter most; 32
   singleton labels would wreck a label-level review. One bucket label, left
   unnamed for review.
2. **Calibrate on odd anchors, score on even anchors** (above).
3. **`src/global_identity.py` is read, not reused.** Its atom is a window-local
   speaker it trusts and never splits, and its 0.50 threshold was calibrated on
   multi-turn nodes, not single turns. Borrowing that number into a different
   regime is the exact error its own config comment warns against. New code, same
   method, its own calibration.

Code location: a script under `scripts/`, alongside `sweep_chunk_thresholds.py` —
not a new module in `src/`. This is calibration for one meeting and should not
become a production import until something has measured it more than once.

### Rejected: label the turns from the handoffs directly

This would spend the oracle. If the handoffs assign the labels, nothing
independent remains to verify with, which defeats a label-level-only review.

---

## Landing the winner

**Backups before any write:** `transcript_named.json`, `transcript_raw.json`,
`diarization.json`, `embeddings.json`, `summary.json` into
`backups/transcript-repair-<timestamp>/`, matching the three already present. A
merge has no undo.

**If A wins:**

```
run_local.py --resume 2026-04-03-lwv-brown-county-candidate-forum-auditor \
  --redo diarize --diarizer oss --compute modal
```

**If B wins:** stage 2 reloads from disk when diarization is already complete
(`run_local.py:1071`). Write the re-labelled turns to `diarization.json` and the
new per-label centroids to `embeddings.json`, then `--redo transcribe`. That
re-aligns `captions.vtt` onto the new segments, then runs identify, summary and
export. `transcript_raw.json` is regenerated by the transcribe stage
(`run_local.py:1334`), so the raw-versus-named provenance the mis-merge detector
reads describes the diarization that actually exists.

**Provenance defect to fix, not inherit.** `processing_metadata.diarization_model`
is set from the `--diarizer` flag (`run_local.py:1176`). On path B that records a
lie: segmentation is still Precision-2 and only clustering changed. It needs its
own value — `pyannote/ai-precision-2+recluster` — so the corpus does not later
claim this meeting was diarized by a model that never touched it.

**Naming needs no help.** Layer-3 speaker-ID is on for `forum` (`src/config.py:87`,
`src/event_kinds.py:128`) and its prompt anchors on the same handoff cues this
meeting is full of. No bespoke naming code.

---

## What this will not prove

The oracle covers the forum only — 85% of forum speech, ending near t=2650. The
meet-and-greet after it, roughly 800s and six or more voices, has no handoffs and
therefore no reference. `SPEAKER_03` and `SPEAKER_04` currently span from the
forum into that section and are the labels most likely still wrong when this work
finishes. They will be **reported as unverified**, not counted as fixed.

If quote sourcing needs only the auditor Q&A, that gap costs nothing. If the
meet-and-greet is also needed, that is separate work.

---

## Out of scope

- **No production default changes.** One meeting is not a calibration. If B works
  well that is a finding worth a follow-up, not a default flip.
- **The named-segment-98 stray is dropped.** Re-running regenerates segments from
  scratch, so that artefact ceases to exist and the one-off remerge script planned
  for it is unnecessary. A different stray may appear; label review will show it.
- **Publishing.** The meeting is not live. Work stops at the quality gate and
  hands over the label set.

---

## Operational notes

- Use `~/Documents/GitHub/on-the-record/.venv/bin/python`.
- Standalone scripts do not auto-load `.env.local`; run `set -a; . ./.env.local; set +a` first.
  (`run_local.py:32` and `gui/asgi.py` have their own loader, so grepping for
  `load_dotenv` finds nothing and misleads.)
- `transcript_raw.json` is the trustworthy record of original diarized labels.
  `diarization.json` was rewritten with post-review segments by
  `gui.review_api._persist_after_review`.
