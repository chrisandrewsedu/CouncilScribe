"""The meeting-kind gate on chunked diarization, as wired into run_local.

Chunking splits a meeting into independently-diarized windows. That is a pure
speed win on single-room civic meetings, but on dense many-voice audio pyannote
merges speakers INSIDE a window and no cross-window threshold repairs it
(measured: a 106-minute mayoral debate gave 29 labels for 33 real people at
every threshold tried). So unvalidated kinds take the single-pass path, and an
explicit CLI override announces itself.
"""
from types import SimpleNamespace

import pytest

from run_local import _resolve_chunk_minutes
from src import config


def _args(requested=None):
    return SimpleNamespace(diarize_chunk_minutes=requested)


@pytest.fixture
def chunking_on(monkeypatch):
    """The gate only has anything to do when chunking is configured on. Pin that
    here rather than depending on the shipped default, which moves as the feature
    is validated and un-validated."""
    monkeypatch.setattr(config, "DIARIZE_CHUNK_MINUTES", 60)
    return 60


def test_a_validated_kind_chunks_at_the_configured_size(chunking_on):
    assert _resolve_chunk_minutes(_args(), "council") == chunking_on


def test_an_unvalidated_kind_falls_back_to_single_pass(chunking_on, capsys):
    assert _resolve_chunk_minutes(_args(), "debate") == 0
    assert "not a validated kind" in capsys.readouterr().out


def test_a_missing_kind_falls_back_to_single_pass(chunking_on):
    assert _resolve_chunk_minutes(_args(), None) == 0


def test_an_explicit_flag_overrides_the_gate_but_warns(capsys):
    assert _resolve_chunk_minutes(_args(45), "debate") == 45
    out = capsys.readouterr().out
    assert "overrides the meeting-kind gate" in out
    assert "NOT a validated kind" in out


def test_an_explicit_flag_on_a_validated_kind_is_silent(capsys):
    assert _resolve_chunk_minutes(_args(45), "council") == 45
    assert capsys.readouterr().out == ""


def test_an_explicit_zero_disables_chunking_without_a_gate_warning(capsys):
    assert _resolve_chunk_minutes(_args(0), "debate") == 0
    assert capsys.readouterr().out == ""


def test_the_gate_is_inert_when_chunking_is_configured_off(monkeypatch, capsys):
    monkeypatch.setattr(config, "DIARIZE_CHUNK_MINUTES", 0)
    assert _resolve_chunk_minutes(_args(), "council") == 0
    assert _resolve_chunk_minutes(_args(), "debate") == 0
    assert capsys.readouterr().out == ""


def test_reclustered_precision_2_has_its_own_provenance_string():
    """Path B keeps Precision-2's boundaries and replaces only its clustering.
    Recording it as plain 'pyannote/ai-precision-2' would claim the shipped
    labels came from a model that never produced them."""
    import run_local

    assert run_local._diarization_model_name("api") == "pyannote/ai-precision-2"
    assert run_local._diarization_model_name("api-recluster") == (
        "pyannote/ai-precision-2+recluster"
    )
