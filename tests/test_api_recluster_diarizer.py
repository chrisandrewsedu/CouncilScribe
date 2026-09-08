"""`--diarizer api-recluster` is provenance-only: it stamps
`diarization_model = "pyannote/ai-precision-2+recluster"` without ever running
Precision-2, on the assumption that diarization.json/embeddings.json were
already hand-replaced by an offline re-clustering. Two ways that assumption
can go unenforced:

1. Run it on a meeting with no existing diarization on disk: Stage 2's
   dispatch has no `api-recluster` branch, so it falls through to plain OSS
   pyannote 3.1 and then stamps the recluster provenance anyway — a false
   claim about how the labels were produced.
2. Run it with `--merge`: Stage 2.5's centroid merge would run the same
   merge-by-similarity over a hand-repaired clustering that produced the
   original three-people-one-label defect this backend exists to carry a fix
   for, silently re-merging them while keeping the "+recluster" stamp.

Both are guarded pre-pipeline: (1) in `_validate_diarizer_compute`, (2) via
`_merge_stage_skipped`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import run_local
from src import config


def test_api_recluster_is_accepted_when_diarization_already_exists(monkeypatch, tmp_path):
    meetings_dir = tmp_path / "meetings"
    meeting_dir = meetings_dir / "meeting-1"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "diarization.json").write_text("[]")
    (meeting_dir / "embeddings.json").write_text("{}")
    monkeypatch.setattr(config, "MEETINGS_DIR", meetings_dir)

    run_local._validate_diarizer_compute(
        SimpleNamespace(diarizer="api-recluster", compute="local", resume="meeting-1")
    )


def test_api_recluster_rejects_a_meeting_with_no_diarization_on_disk(monkeypatch, tmp_path):
    """The bug this guards: falling through to OSS diarization and then
    stamping the recluster provenance string as if it never happened."""
    meetings_dir = tmp_path / "meetings"
    meeting_dir = meetings_dir / "meeting-1"
    meeting_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "MEETINGS_DIR", meetings_dir)

    with pytest.raises(ValueError, match="diarization.json"):
        run_local._validate_diarizer_compute(
            SimpleNamespace(diarizer="api-recluster", compute="local", resume="meeting-1")
        )


def test_api_recluster_rejects_a_meeting_with_only_one_of_the_two_files(monkeypatch, tmp_path):
    meetings_dir = tmp_path / "meetings"
    meeting_dir = meetings_dir / "meeting-1"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "diarization.json").write_text("[]")
    monkeypatch.setattr(config, "MEETINGS_DIR", meetings_dir)

    with pytest.raises(ValueError, match="embeddings.json"):
        run_local._validate_diarizer_compute(
            SimpleNamespace(diarizer="api-recluster", compute="local", resume="meeting-1")
        )


def test_api_recluster_rejects_when_no_meeting_can_be_identified(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MEETINGS_DIR", tmp_path / "meetings")

    with pytest.raises(ValueError, match="api-recluster"):
        run_local._validate_diarizer_compute(
            SimpleNamespace(
                diarizer="api-recluster", compute="local",
                resume=None, meeting_id=None, date=None, meeting_type=None,
            )
        )


def test_api_recluster_resolves_a_meeting_id_from_date_and_meeting_type(monkeypatch, tmp_path):
    meetings_dir = tmp_path / "meetings"
    meeting_dir = meetings_dir / "2026-01-01-regular-session"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "diarization.json").write_text("[]")
    (meeting_dir / "embeddings.json").write_text("{}")
    monkeypatch.setattr(config, "MEETINGS_DIR", meetings_dir)

    run_local._validate_diarizer_compute(
        SimpleNamespace(
            diarizer="api-recluster", compute="local",
            resume=None, meeting_id=None,
            date="2026-01-01", meeting_type="Regular Session",
        )
    )


@pytest.mark.parametrize("diarizer", ["api", "vibevoice", "api-recluster"])
def test_merge_stage_is_skipped_for_backends_with_their_own_clustering(diarizer):
    assert run_local._merge_stage_skipped(diarizer) is True


def test_merge_stage_runs_for_oss():
    assert run_local._merge_stage_skipped("oss") is False
