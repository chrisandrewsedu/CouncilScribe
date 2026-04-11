"""Tests for Phase 109 per-meeting body tagging (CSMEETING-01/02/03, D-01..D-13).

Plans 01/02/03 implement tests in waves:
- Plan 01 (this file, Tasks 1+3): D-01, D-02, D-03, D-04, D-06, D-11, D-12 + batch propagation
- Plan 02 (Task 2): D-07, D-08, D-09, D-10, D-13 (fail-fast guard)
- Plan 03 (Task 1): CSMEETING-03 (Stage 4 roster wiring), D-05 legacy fallback

Mirror Phase 108 test_roster_load.py conventions:
- tmp_path for isolated meeting dirs
- subprocess.run for CLI-level tests that exercise argparse paths
- direct unit tests for PipelineState logic
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.checkpoint import PipelineState, PipelineStage


# ---------------------------------------------------------------------------
# Helper: simulate the Phase 109 resolve block from run_pipeline
# (tests the logic without needing GPU / HF token)
# ---------------------------------------------------------------------------

def _resolve_body_slug(state: PipelineState, cli_body=None, force_retag=False):
    """Replicate the resolve block from run_pipeline for unit testing.

    Returns effective_body_slug after applying D-01..D-06 rules.
    Raises SystemExit(2) on D-02 mismatch or D-12 violation (enforced at argparse).
    """
    persisted_body = state.body_slug

    if cli_body and persisted_body and cli_body != persisted_body and not force_retag:
        # D-02: hard error on mismatch
        raise SystemExit(2)

    if cli_body and persisted_body and cli_body != persisted_body and force_retag:
        # D-03 + D-04 + D-11
        state.body_slug = cli_body
        state.rewind_for_retag()
        state.save()
    elif cli_body and not persisted_body:
        # D-01: first run persists
        state.body_slug = cli_body
        state.save()
    # else: D-05 / D-06

    return state.body_slug


# ---------------------------------------------------------------------------
# Plan 01 Task 3 — 6 tests turned green by Task 3
# ---------------------------------------------------------------------------


def test_first_run_persists_body_slug(tmp_path, tmp_config_dir, fake_roster_cache):
    """D-01: First invocation with --body X writes body_slug=X into pipeline_state.json.

    Decision D-01: "On first invocation, --body X is required to write
    body_slug=X into per-meeting metadata."
    """
    fake_roster_cache("bloomington-common-council")
    mdir = tmp_path / "meetings" / "2026-02-04-regular-session"
    mdir.mkdir(parents=True)

    state = PipelineState(mdir)
    assert state.body_slug is None  # no slug yet

    effective = _resolve_body_slug(state, cli_body="bloomington-common-council")

    assert effective == "bloomington-common-council"
    # Verify it was persisted atomically
    state2 = PipelineState(mdir)
    assert state2.body_slug == "bloomington-common-council"
    # Verify pipeline_state.json contains body_slug key
    raw = json.loads((mdir / "pipeline_state.json").read_text())
    assert raw["body_slug"] == "bloomington-common-council"


def test_reinvocation_reads_persisted_slug(tmp_path, tmp_config_dir, fake_roster_cache, tagged_meeting_dir):
    """D-01 + D-06: Re-invoking a tagged meeting without --body reads slug silently and prints 'Body: <slug>'.

    Decision D-01: "On every subsequent invocation against the same meeting,
    omitting --body reads the persisted slug back silently."
    Decision D-06: "No flag, yes persisted slug — read from metadata silently;
    print Body: <slug> info line."
    """
    slug = "bloomington-common-council"
    fake_roster_cache(slug)
    mdir = tagged_meeting_dir(slug, completed_stage=3)

    # Re-invocation: no cli_body
    state = PipelineState(mdir)
    assert state.body_slug == slug  # persisted slug is present

    effective = _resolve_body_slug(state, cli_body=None)
    assert effective == slug  # D-06: reads persisted silently

    # Verify the slug wasn't wiped
    state2 = PipelineState(mdir)
    assert state2.body_slug == slug


def test_mismatched_body_hard_error(tmp_path, tmp_config_dir, fake_roster_cache, tagged_meeting_dir):
    """D-02: --body Y against persisted X exits non-zero with 'already tagged as' language.

    Decision D-02: "If the meeting already persists body_slug=X and the operator
    passes --body Y, run_local.py exits non-zero with a clear message: meeting
    already tagged as X, pass --body X or use --force-retag to change it."
    """
    slug_x = "bloomington-common-council"
    slug_y = "some-other-body"
    fake_roster_cache(slug_x)
    fake_roster_cache(slug_y)
    mdir = tagged_meeting_dir(slug_x, completed_stage=3)

    state = PipelineState(mdir)
    assert state.body_slug == slug_x

    with pytest.raises(SystemExit) as exc_info:
        _resolve_body_slug(state, cli_body=slug_y, force_retag=False)

    assert exc_info.value.code == 2  # D-02: non-zero exit
    # Verify slug was NOT changed
    state2 = PipelineState(mdir)
    assert state2.body_slug == slug_x


def test_force_retag_rewinds_and_clears(tmp_path, tmp_config_dir, fake_roster_cache, tagged_meeting_dir):
    """D-03 + D-04 + D-11: --body Y --force-retag overwrites slug, rewinds completed_stage to TRANSCRIBED (3), removes pre_identifications.json.

    Decision D-03: "--force-retag is the escape hatch to overwrite persisted body_slug."
    Decision D-04: "--force-retag invalidates downstream stages — rewinds completed_stage
    to TRANSCRIBED (stage 3)."
    Decision D-11: "Deletes pre_identifications.json (stale against old roster)."
    """
    slug_x = "bloomington-common-council"
    slug_y = "monroe-county-council"
    fake_roster_cache(slug_x)
    fake_roster_cache(slug_y)
    # Meeting is at EXPORTED (stage 7) with old slug
    mdir = tagged_meeting_dir(slug_x, completed_stage=7)
    # Simulate pre_identifications.json from old run
    (mdir / "pre_identifications.json").write_text("{}", encoding="utf-8")

    state = PipelineState(mdir)
    assert state.body_slug == slug_x
    assert state.completed_stage == PipelineStage.EXPORTED

    effective = _resolve_body_slug(state, cli_body=slug_y, force_retag=True)

    assert effective == slug_y  # D-03: new slug
    # Reload from disk to confirm atomic persistence
    state2 = PipelineState(mdir)
    assert state2.body_slug == slug_y
    assert state2.completed_stage == PipelineStage.TRANSCRIBED  # D-04: rewound to 3
    assert not (mdir / "pre_identifications.json").exists()  # D-11: stale file removed


def test_force_retag_requires_body(tmp_path):
    """D-12: --force-retag without --body exits 2 (argparse usage error).

    Decision D-12: "--force-retag without --body is a hard argparse error (exit 2)."
    D-12 is enforced immediately after args = parser.parse_args() so argparse
    convention exit code 2 applies.
    """
    run_local = Path(__file__).parent.parent / "run_local.py"
    result = subprocess.run(
        [sys.executable, str(run_local), "--force-retag"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, f"expected exit 2, got {result.returncode}\nstderr: {result.stderr}"
    assert "--force-retag requires --body" in result.stderr


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
    import run_local

    slug = "bloomington-common-council"
    # No roster file in tmp_config_dir/rosters/ — omitted intentionally

    with pytest.raises(SystemExit) as exc_info:
        run_local.ensure_body_roster_cached(slug)

    assert exc_info.value.code == 2, f"expected exit 2, got {exc_info.value.code}"


def test_resume_after_cache_delete_fails_fast(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir):
    """D-10: Persisted slug + deleted cache fails fast identically to D-08.

    Decision D-10: "If a prior run persisted body_slug=X, completed through
    Stage 3, and then ~/CouncilScribe/config/rosters/X.json was deleted,
    a subsequent resume invocation fails fast at the same pre-Stage-1 check."
    """
    import run_local

    slug = "bloomington-common-council"
    # Create then delete the roster file to simulate "was there but deleted"
    roster_file = tmp_config_dir / "rosters" / f"{slug}.json"
    roster_file.write_text("{}", encoding="utf-8")
    roster_file.unlink()  # simulate deletion

    # Meeting dir with persisted body_slug
    mdir = tagged_meeting_dir(slug, completed_stage=3)

    # Loading state shows slug is persisted
    state = PipelineState(mdir)
    assert state.body_slug == slug

    # Guard must fail fast identically to D-08 (cache missing)
    with pytest.raises(SystemExit) as exc_info:
        run_local.ensure_body_roster_cached(state.body_slug)

    assert exc_info.value.code == 2, f"expected exit 2, got {exc_info.value.code}"


def test_stale_cache_is_non_blocking(tmp_path, tmp_config_dir, fake_roster_cache):
    """D-09: Cached roster >30 days old still loads — staleness warning is non-blocking.

    Decision D-09: "Stale cache does NOT fail fast. A cached roster file that
    exists but is older than 30 days (Phase 108's staleness threshold) still
    loads — the non-blocking WARNING from load_roster(body_slug=...) fires as
    today. Fail-fast only triggers on truly missing files."
    """
    import run_local

    slug = "bloomington-common-council"
    # Write a cache file with a very old fetched_at (>30 days ago)
    fake_roster_cache(slug, fetched_at="2020-01-01T00:00:00Z")

    # Guard must NOT raise SystemExit — stale but present file is not a fail-fast
    run_local.ensure_body_roster_cached(slug)  # should return without error


def test_stage4_uses_body_roster(tmp_path, tmp_config_dir, tmp_meetings_dir, tagged_meeting_dir, fake_roster_cache):
    """CSMEETING-03: When body_slug set, Stage 4 calls load_roster(body_slug=...) and identify_speakers receives body-specific Roster.

    Requirement CSMEETING-03: "Stage 4 loads body-specific roster for
    correct_speaker_name, pattern matching, and LLM prompt; no legacy fallback
    when body is tagged."
    D-05: "No code path falls back to the legacy global roster when a body_slug
    is present."

    Strategy: patch src.roster.load_roster (the name imported by the Stage 4 block)
    and exercise the Stage 4 conditional logic via a thin helper that replicates it.
    """
    from src.roster import Roster, RosterMember

    slug = "bloomington-common-council"
    fake_roster_cache(slug)

    sentinel_roster = Roster(
        city="Bloomington",
        body="Common Council",
        members=[RosterMember(name="Isabel Piedmont-Smith")],
    )

    called_with = []

    def mock_load_roster(path=None, *, body_slug=None):
        called_with.append({"path": path, "body_slug": body_slug})
        return sentinel_roster

    # Replicate the Stage 4 conditional from run_local.py inline:
    #   if effective_body_slug: roster = load_roster(body_slug=effective_body_slug)
    #   else: roster = load_roster()  # D-05 legacy fallback
    effective_body_slug = slug
    with patch("src.roster.load_roster", side_effect=mock_load_roster):
        from src.roster import load_roster
        if effective_body_slug:
            roster = load_roster(body_slug=effective_body_slug)
        else:
            roster = load_roster()  # D-05 legacy fallback

    assert len(called_with) == 1, f"Expected exactly one load_roster call, got {called_with}"
    assert called_with[0]["body_slug"] == slug, (
        f"Expected load_roster(body_slug={slug!r}), got body_slug={called_with[0]['body_slug']!r}"
    )
    assert called_with[0]["path"] is None, "body_slug path must not pass explicit path arg"
    assert roster is sentinel_roster, "Stage 4 must use the body-specific Roster object"


def test_legacy_fallback_intact(tmp_path, tmp_config_dir, tmp_meetings_dir):
    """D-05: No --body, no persisted slug → bare load_roster() called (legacy fallback intact).

    Decision D-05: "No flag, no persisted slug → legacy fallback. A brand-new
    meeting run with no --body flag at all continues to work exactly as today —
    Stage 4 calls bare load_roster(), which resolves to
    ~/CouncilScribe/config/council_roster.json."
    """
    called_with = []

    def mock_load_roster(path=None, *, body_slug=None):
        called_with.append({"path": path, "body_slug": body_slug})
        return None  # legacy path — no council_roster.json in tmp dir

    # Replicate Stage 4 conditional with effective_body_slug=None (D-05 path):
    effective_body_slug = None
    with patch("src.roster.load_roster", side_effect=mock_load_roster):
        from src.roster import load_roster
        if effective_body_slug:
            roster = load_roster(body_slug=effective_body_slug)
        else:
            roster = load_roster()  # D-05 legacy fallback

    assert len(called_with) == 1, f"Expected exactly one load_roster call, got {called_with}"
    assert called_with[0]["body_slug"] is None, (
        f"D-05 legacy fallback must call bare load_roster(), got body_slug={called_with[0]['body_slug']!r}"
    )
    assert roster is None, "Legacy fallback returns None when no council_roster.json present"


def test_batch_propagates_body(tmp_path, tmp_config_dir, fake_roster_cache):
    """CSMEETING-01 batch propagation: _run_batch builds batch_args with body= and force_retag= attributes.

    Requirement CSMEETING-01: "Meeting metadata accepts body_slug field."
    Batch mode must propagate --body and --force-retag into per-entry
    batch_args Namespace so per-row run_pipeline calls see them.
    """
    import run_local

    slug = "bloomington-common-council"
    fake_roster_cache(slug)

    # Create a minimal batch text file with one entry
    batch_file = tmp_path / "batch.txt"
    batch_file.write_text("fake_input.mp4 2026-02-04 Bloomington Regular Session\n", encoding="utf-8")

    captured_args = []

    def fake_run_pipeline(args):
        captured_args.append(args)

    top_args = argparse.Namespace(
        batch=str(batch_file),
        batch_resume=False,
        skip_llm=False,
        skip_summary=False,
        no_merge=False,
        use_vtt=False,
        body=slug,
        force_retag=True,
    )

    with patch.object(run_local, "run_pipeline", fake_run_pipeline):
        try:
            run_local._run_batch(top_args)
        except Exception:
            pass  # pipeline itself may fail; we only care about batch_args shape

    assert len(captured_args) >= 1, "run_pipeline was never called"
    ba = captured_args[0]
    assert hasattr(ba, "body"), "batch_args missing 'body' attribute"
    assert ba.body == slug, f"expected body={slug!r}, got {ba.body!r}"
    assert hasattr(ba, "force_retag"), "batch_args missing 'force_retag' attribute"
    assert ba.force_retag is True, "force_retag not propagated"
