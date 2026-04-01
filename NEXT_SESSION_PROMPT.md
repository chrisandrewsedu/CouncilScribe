# CouncilScribe — Next Session Prompt

Copy everything below the line into a new Claude Code session:

---

We're working on CouncilScribe (`/Users/chrisandrews/Documents/GitHub/CouncilScribe/`), an automated city council meeting transcription system. Read the memory file at `/Users/chrisandrews/.claude/projects/-Users-chrisandrews-Documents-GitHub/memory/councilscribe.md` for full project context, then read these key source files to understand the current architecture:

- `src/config.py` — Thresholds, model config, paths
- `src/identify.py` — 3-layer speaker ID cascade + soft matching
- `src/enroll.py` — Voice profile management (StoredProfile, ProfileDB, merge, rename)
- `src/diarize.py` — Pyannote diarization + embedding extraction
- `src/roster.py` — Council roster fuzzy name correction
- `src/export.py` — Markdown/JSON/SRT export
- `run_local.py` — CLI entry point with full pipeline + utility commands
- `src/download.py` — CATS TV archive scraping (has VTT URL support)

Data directory is `~/CouncilScribe/` with `meetings/`, `profiles/`, `config/`. Three meetings processed: `2026-02-25-council`, `2026-02-18-council`, `2026-02-04-council`. The venv is at `.venv/` (Python 3.13) — use `python3` to run outside the venv (scipy won't be available outside it, structure imports to avoid loading it for utility commands).

## Build these 6 features in priority order:

### 1. Auto-merge fragmented speakers (after diarization, before identification)
Pyannote frequently splits one person into multiple SPEAKER labels. For example, Feb 25 has SPEAKER_02, _03, _04 all being the same person (Council President Asare) and SPEAKER_05, _06, _07 all being City Clerk Bolden.

**Implementation:**
- In `src/diarize.py` or a new `src/merge.py`, add `merge_similar_speakers(segments, speaker_embeddings, threshold=0.80)` that:
  - Computes pairwise cosine similarity between all speaker embedding centroids
  - Groups speakers whose similarity exceeds the threshold using union-find / connected components
  - Merges segments: relabels all speakers in a group to the label with the most speech time
  - Recomputes merged embedding centroids
  - Returns merged segments, merged embeddings, and a merge log (e.g., "SPEAKER_03 + SPEAKER_04 merged into SPEAKER_02")
- Add `SPEAKER_MERGE_THRESHOLD = 0.80` to `config.py`
- Call this in `run_local.py` between Stage 2 and the pre-identification step
- Also update embeddings.json on disk after merging
- Add `--no-merge` flag to skip if needed

### 2. Profile confidence escalation for returning speakers
Profiles seen in multiple meetings should auto-match at a lower threshold.

**Implementation:**
- In `src/identify.py`, modify `match_voice_profiles()` to accept the full `ProfileDB` (not just centroids) so it can check `meetings_seen` count
- For profiles with `len(meetings_seen) >= 3`: use threshold 0.70 instead of 0.85
- For profiles with `len(meetings_seen) >= 2`: use threshold 0.78
- Add `RETURNING_SPEAKER_THRESHOLD_3 = 0.70` and `RETURNING_SPEAKER_THRESHOLD_2 = 0.78` to config
- Log when a returning speaker is matched at a lower threshold: "Matched SPEAKER_02 -> Councilmember Rowley (0.74, returning speaker)"

### 3. Batch processing mode
Run GPU-intensive stages unattended on multiple meetings.

**Implementation:**
- Add `--batch FILE_OR_DIR` argument: accepts either a text file with one input per line (path or URL + date), or a directory of video files
- Runs Stages 1-3 (ingest, diarize, transcribe) + automated Stage 4 (no interactive review) for each meeting
- Skips `--pre-identify` and human review in batch mode
- Prints a summary at the end: which meetings completed, which need review
- Add `--batch-resume` to continue a batch that was interrupted

### 4. VTT alignment from CATS TV (skip Whisper transcription)
CATS TV provides VTT subtitle files alongside videos. Using these instead of Whisper would cut processing from ~2hrs to ~15min per meeting.

**Implementation:**
- In `src/download.py`, check how `fetch_catstv_meetings()` already works — the meeting dicts likely have a `vtt_url` or similar field. If not, scrape it.
- Create `src/vtt_align.py` with `align_vtt_to_segments(vtt_path, diarized_segments) -> list[Segment]`:
  - Parse VTT cues (timestamp + text)
  - For each diarized segment, find overlapping VTT cues by timestamp
  - Assign the VTT text to the segment (proportional split if a cue spans multiple segments)
  - This replaces Whisper — segments get text from VTT instead
- Add `--use-vtt` flag (or auto-detect: if VTT was downloaded alongside video, offer to use it)
- In the pipeline, if VTT alignment is used, skip Stage 3 (Whisper) entirely and mark TRANSCRIBED
- VTT files should be saved as `captions.vtt` in the meeting directory

### 5. Roster auto-learning
When `--fix-transcripts` or roster correction changes a name, auto-add the original as an alias.

**Implementation:**
- In `src/roster.py`, add `add_alias(roster_path, canonical_name, new_alias)` that loads the JSON, adds the alias if not already present, and saves
- In `_fix_transcripts()` in `run_local.py`, after corrections are applied, collect all original→corrected pairs and call `add_alias` for each
- Also in `_interactive_speaker_review()`, if the user types a roster member name for a speaker that had a different (wrong) auto-detected name, offer to add the wrong name as an alias
- Guard against adding nonsense aliases (skip if original name is null, too short, or a generic like "SPEAKER_00")

### 6. Post-identification segment merging
After all speakers are identified, merge adjacent segments from the same person.

**Implementation:**
- In `src/export.py` or a new function in `src/identify.py`, add `merge_adjacent_segments(segments, gap_threshold=2.0)`:
  - Walk segments chronologically
  - If consecutive segments have the same `speaker_name` and gap < threshold, merge them (combine text, extend time range, concatenate words)
  - This produces cleaner transcripts with fewer fragmented entries
- Apply this before export (Stage 7), after all identification is finalized
- Add to config: `SEGMENT_MERGE_GAP = 2.0` (seconds)
- The merged segments should preserve word-level timestamps

## General notes:
- Keep imports lazy in `run_local.py` — scipy/numpy/torch only load when needed, not for utility commands
- Test each feature with the existing 3 meetings
- Commit after each feature is complete and tested
