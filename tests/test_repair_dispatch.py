"""Dispatch and output behavior for standalone transcript repair."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import run_local


@pytest.mark.parametrize(
    "meeting_id",
    [
        "../outside",
        ".",
        "..",
    ],
)
def test_repair_transcript_handler_rejects_non_child_meeting_ids(
    monkeypatch,
    tmp_path,
    capsys,
    meeting_id,
):
    called = []
    monkeypatch.setattr("src.config.MEETINGS_DIR", tmp_path / "meetings")
    monkeypatch.setattr("src.repair.repair_transcript", called.append)

    with pytest.raises(SystemExit) as exc_info:
        run_local._repair_transcript_standalone(meeting_id)

    assert exc_info.value.code == 1
    assert called == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Transcript repair failed:" in captured.err


def test_repair_transcript_handler_rejects_absolute_meeting_id(
    monkeypatch,
    tmp_path,
    capsys,
):
    called = []
    monkeypatch.setattr("src.config.MEETINGS_DIR", tmp_path / "meetings")
    monkeypatch.setattr("src.repair.repair_transcript", called.append)

    with pytest.raises(SystemExit) as exc_info:
        run_local._repair_transcript_standalone(str(tmp_path / "outside"))

    assert exc_info.value.code == 1
    assert called == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Transcript repair failed:" in captured.err


def test_repair_transcript_handler_rejects_symlink_escape(
    monkeypatch,
    tmp_path,
    capsys,
):
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    symlink = meetings_dir / "linked-meeting"
    try:
        symlink.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("platform cannot create directory symlinks")

    called = []
    monkeypatch.setattr("src.config.MEETINGS_DIR", meetings_dir)
    monkeypatch.setattr("src.repair.repair_transcript", called.append)

    with pytest.raises(SystemExit) as exc_info:
        run_local._repair_transcript_standalone(symlink.name)

    assert exc_info.value.code == 1
    assert called == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Transcript repair failed:" in captured.err


def test_repair_transcript_handler_accepts_resolved_direct_child(
    monkeypatch,
    tmp_path,
    capsys,
):
    meetings_dir = tmp_path / "meetings"
    meeting_dir = meetings_dir / "meeting-1"
    meeting_dir.mkdir(parents=True)
    called = []
    monkeypatch.setattr("src.config.MEETINGS_DIR", meetings_dir)

    def fake_repair_transcript(resolved_meeting_dir):
        called.append(resolved_meeting_dir)
        return SimpleNamespace(
            meeting_id="meeting-1",
            segment_count=1,
            backup_dir=meeting_dir / "backups" / "repair",
            exports={},
        )

    monkeypatch.setattr("src.repair.repair_transcript", fake_repair_transcript)

    run_local._repair_transcript_standalone("meeting-1")

    assert called == [meeting_dir.resolve()]
    assert "Transcript repair complete:" in capsys.readouterr().out


def test_repair_transcript_dispatches_without_running_pipeline(monkeypatch):
    called = {}
    monkeypatch.setattr(
        run_local,
        "_repair_transcript_standalone",
        lambda meeting_id: called.setdefault("meeting_id", meeting_id),
        raising=False,
    )
    monkeypatch.setattr(
        run_local,
        "run_pipeline",
        lambda args: pytest.fail("run_pipeline must not run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local.py", "--repair-transcript", "meeting-1"],
    )

    run_local.main()

    assert called == {"meeting_id": "meeting-1"}


@pytest.mark.parametrize(
    "abbreviated_args",
    [
        ["--comp", "local"],
        ["--num-sp", "0"],
    ],
)
def test_repair_transcript_rejects_abbreviated_long_options(
    monkeypatch,
    abbreviated_args,
):
    called = []
    monkeypatch.setattr(
        run_local,
        "_repair_transcript_standalone",
        called.append,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local.py", "--repair-transcript", "repair-me"]
        + abbreviated_args,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_local.main()

    assert exc_info.value.code == 2
    assert called == []


@pytest.mark.parametrize(
    ("conflicting_args", "conflict_flags"),
    [
        (["--input", "meeting.mp4"], ["--input"]),
        (["--input=meeting.mp4"], ["--input"]),
        (["-imeeting.mp4"], ["--input"]),
        (["--browse-catstv"], ["--browse-catstv"]),
        (["--resume", "meeting-1"], ["--resume"]),
        (
            ["--redo", "transcribe", "--resume", "meeting-1"],
            ["--redo", "--resume"],
        ),
        (["--batch", "meetings.txt"], ["--batch"]),
        (["--review", "meeting-1"], ["--review"]),
        (["--review-meeting", "meeting-1"], ["--review-meeting"]),
        (["--identify-speakers", "meeting-1"], ["--identify-speakers"]),
        (["--fix-transcripts"], ["--fix-transcripts"]),
        (["--publish-meeting", "meeting-1"], ["--publish-meeting"]),
        (["--list-profiles"], ["--list-profiles"]),
        (["--fix-profiles"], ["--fix-profiles"]),
        (["--show-roster"], ["--show-roster"]),
        (
            ["--merge-profiles", "source", "destination"],
            ["--merge-profiles"],
        ),
        (["--body", "council"], ["--body"]),
        (
            ["--force-retag", "--body", "council"],
            ["--force-retag", "--body"],
        ),
        (["--city", "Bloomington"], ["--city"]),
        (["--date", "2026-01-01"], ["--date"]),
        (["--meeting-type", "Regular"], ["--meeting-type"]),
        (["--meeting-id", "custom-id"], ["--meeting-id"]),
        (["--noise-reduce"], ["--noise-reduce"]),
        (["--cookies", "cookies.txt"], ["--cookies"]),
        (["--skip-llm"], ["--skip-llm"]),
        (["--skip-summary"], ["--skip-summary"]),
        (["--confirm-enroll"], ["--confirm-enroll"]),
        (["--merge"], ["--merge"]),
        (["--use-vtt"], ["--use-vtt"]),
        (["--compute", "modal"], ["--compute"]),
        (["--compute", "local"], ["--compute"]),
        (["--diarizer", "api"], ["--diarizer"]),
        (["--diarizer", "oss"], ["--diarizer"]),
        (["--diarizer", "vibevoice"], ["--diarizer"]),
        (["--diarizer", "api-recluster"], ["--diarizer"]),
        (["--num-speakers", "2"], ["--num-speakers"]),
        (["--num-speakers", "0"], ["--num-speakers"]),
        (["--date", ""], ["--date"]),
        (["--meeting-id", ""], ["--meeting-id"]),
        (["--default"], ["--default"]),
        (["--publish"], ["--publish"]),
        (["--no-review"], ["--no-review"]),
        (["--pre-identify"], ["--pre-identify"]),
        (["--batch-resume"], ["--batch-resume"]),
    ],
)
def test_repair_transcript_rejects_conflicting_commands(
    monkeypatch,
    capsys,
    conflicting_args,
    conflict_flags,
):
    called = []
    monkeypatch.setattr(
        run_local,
        "_repair_transcript_standalone",
        called.append,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local.py", "--repair-transcript", "repair-me"] + conflicting_args,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_local.main()

    assert exc_info.value.code == 2
    assert called == []
    error = capsys.readouterr().err
    assert "--repair-transcript" in error
    for flag in conflict_flags:
        assert flag in error


def test_repair_transcript_handler_prints_result(monkeypatch, tmp_path, capsys):
    meeting_id = "meeting-1"
    backup_dir = tmp_path / meeting_id / "backups" / "transcript-repair-fixed"
    exports = {
        "markdown": tmp_path / meeting_id / "exports" / "transcript.md",
        "json": tmp_path / meeting_id / "exports" / "transcript.json",
        "srt": tmp_path / meeting_id / "exports" / "transcript.srt",
    }
    called = []

    monkeypatch.setattr("src.config.MEETINGS_DIR", tmp_path)

    def fake_repair_transcript(meeting_dir):
        called.append(meeting_dir)
        return SimpleNamespace(
            meeting_id=meeting_id,
            segment_count=7,
            backup_dir=backup_dir,
            exports=exports,
        )

    monkeypatch.setattr("src.repair.repair_transcript", fake_repair_transcript)

    run_local._repair_transcript_standalone(meeting_id)

    assert called == [tmp_path / meeting_id]
    output = capsys.readouterr().out
    assert meeting_id in output
    assert "7" in output
    assert str(backup_dir) in output
    for export_name, export_path in exports.items():
        assert export_name in output
        assert str(export_path) in output


def test_repair_transcript_handler_exits_on_repair_error(
    monkeypatch,
    capsys,
):
    from src.repair import RepairError

    def fail_repair(meeting_dir):
        raise RepairError("captions are unavailable")

    monkeypatch.setattr("src.repair.repair_transcript", fail_repair)

    with pytest.raises(SystemExit) as exc_info:
        run_local._repair_transcript_standalone("meeting-1")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Transcript repair failed:" in captured.err
    assert "captions are unavailable" in captured.err
