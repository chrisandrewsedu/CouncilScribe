"""Wave 0 test stubs for Phase 109 per-meeting body tagging.

Covers CSMEETING-01, CSMEETING-02, and CSMEETING-03 (D-01..D-13).
All tests start as failing stubs — implementations land in Plans 02 and 03.

Mirror Phase 108 test_roster_load.py conventions:
- tmp_path for isolated meeting dirs
- subprocess.run for CLI-level tests
- direct imports for unit-level tests
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1 (109-01) — Wave 0 stubs
# ---------------------------------------------------------------------------


def test_first_run_persists_body_slug(tmp_path, tmp_config_dir, tmp_meetings_dir, fake_roster_cache):
    """D-01: First invocation with --body X writes body_slug=X into pipeline_state.json.

    Decision D-01: "On first invocation, --body X is required to write
    body_slug=X into per-meeting metadata."
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")


def test_reinvocation_reads_persisted_slug(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir, fake_roster_cache):
    """D-01 + D-06: Re-invoking a tagged meeting without --body reads slug silently and prints 'Body: <slug>'.

    Decision D-01: "On every subsequent invocation against the same meeting,
    omitting --body reads the persisted slug back silently."
    Decision D-06: "No flag, yes persisted slug — read from metadata silently;
    print Body: <slug> info line."
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")


def test_mismatched_body_hard_error(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir, fake_roster_cache):
    """D-02: --body Y against persisted X exits non-zero with 'already tagged as' language.

    Decision D-02: "If the meeting already persists body_slug=X and the operator
    passes --body Y, run_local.py exits non-zero with a clear message: meeting
    already tagged as X, pass --body X or use --force-retag to change it."
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")


def test_force_retag_rewinds_and_clears(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir, fake_roster_cache):
    """D-03 + D-04 + D-11: --body Y --force-retag overwrites slug, rewinds completed_stage to TRANSCRIBED (3), removes pre_identifications.json.

    Decision D-03: "--force-retag is the escape hatch to overwrite persisted body_slug."
    Decision D-04: "--force-retag invalidates downstream stages — rewinds completed_stage
    to TRANSCRIBED (stage 3)."
    Decision D-11: "Deletes pre_identifications.json (stale against old roster)."
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")


def test_force_retag_requires_body(tmp_path, tmp_meetings_dir):
    """D-12: --force-retag without --body exits 2 (argparse usage error).

    Decision D-12: "--force-retag without --body is a hard argparse error (exit 2)."
    D-12 is enforced immediately after args = parser.parse_args() so argparse
    convention exit code 2 applies.
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")


def test_missing_roster_fails_fast(tmp_path, tmp_config_dir, tmp_meetings_dir):
    """D-07 + D-08 + D-13: Tagged meeting with no cached roster prints D-08 stderr error and exits 2 BEFORE Stage 1.

    Decision D-07: "Fail fast BEFORE Stage 1 — the cached-roster existence check
    runs after argparse and after metadata load/merge, but before any
    ingestion/diarization/transcription work."
    Decision D-08: "Error shape — print to stderr + sys.exit(2). Line 1:
    ERROR: Body \\"<slug>\\" has no cached roster at ~/CouncilScribe/config/rosters/<slug>.json
    Line 2: Run: python refresh_roster.py --body <slug>"
    Decision D-13: "Exit before any Stage 1 work begins."
    """
    pytest.fail("not yet implemented — Plan 02, Task 2")


def test_resume_after_cache_delete_fails_fast(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir):
    """D-10: Persisted slug + deleted cache fails fast identically to D-08.

    Decision D-10: "If a prior run persisted body_slug=X, completed through
    Stage 3, and then ~/CouncilScribe/config/rosters/X.json was deleted,
    a subsequent resume invocation fails fast at the same pre-Stage-1 check."
    """
    pytest.fail("not yet implemented — Plan 02, Task 2")


def test_stale_cache_is_non_blocking(tmp_path, tmp_config_dir, fake_roster_cache):
    """D-09: Cached roster >30 days old still loads — staleness warning is non-blocking.

    Decision D-09: "Stale cache does NOT fail fast. A cached roster file that
    exists but is older than 30 days (Phase 108's staleness threshold) still
    loads — the non-blocking WARNING from load_roster(body_slug=...) fires as
    today. Fail-fast only triggers on truly missing files."
    """
    pytest.fail("not yet implemented — Plan 02, Task 2")


def test_stage4_uses_body_roster(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir, fake_roster_cache):
    """CSMEETING-03: When body_slug set, Stage 4 calls load_roster(body_slug=...) and identify_speakers receives body-specific Roster.

    Requirement CSMEETING-03: "Stage 4 loads body-specific roster for
    correct_speaker_name, pattern matching, and LLM prompt; no legacy fallback
    when body is tagged."
    D-05: "No code path falls back to the legacy global roster when a body_slug
    is present."
    """
    pytest.fail("not yet implemented — Plan 03, Task 1")


def test_legacy_fallback_intact(tmp_path, tmp_config_dir, tmp_meetings_dir):
    """D-05: No --body, no persisted slug → bare load_roster() called (legacy fallback intact).

    Decision D-05: "No flag, no persisted slug → legacy fallback. A brand-new
    meeting run with no --body flag at all continues to work exactly as today —
    Stage 4 calls bare load_roster(), which resolves to
    ~/CouncilScribe/config/council_roster.json."
    """
    pytest.fail("not yet implemented — Plan 03, Task 1")


def test_batch_propagates_body(tmp_path, tmp_config_dir, tmp_meetings_dir, fake_roster_cache):
    """CSMEETING-01 batch propagation: _run_batch builds batch_args with body= and force_retag= attributes.

    Requirement CSMEETING-01: "Meeting metadata accepts body_slug field."
    Batch mode must propagate --body and --force-retag into per-entry
    batch_args Namespace so per-row run_pipeline calls see them.
    """
    pytest.fail("not yet implemented — Plan 01, Task 3")
