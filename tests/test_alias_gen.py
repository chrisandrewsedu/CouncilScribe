"""Unit tests for the alias generator (D-01..D-07).

Rules come from .planning/phases/108-councilscribe-roster-client-cli/108-CONTEXT.md.
The generator is a pure function: dict -> list[str], no IO.
"""
from __future__ import annotations

import pytest

from src.alias_gen import generate_aliases


# ---------------------------------------------------------------------------
# D-01 base variants
# ---------------------------------------------------------------------------

def test_d01_base_variants(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert "Isabel Piedmont-Smith" in aliases
    assert "Piedmont-Smith" in aliases
    assert "Councilmember Piedmont-Smith" in aliases


def test_d01_preferred_name_skipped_when_equal(sample_politician_piedmont_smith):
    # preferred_name="Isabel" equals first token of full_name, so
    # `preferred_name + " " + last_name` ("Isabel Piedmont-Smith") dedupes
    # with the full_name variant. Result: only one "Isabel Piedmont-Smith".
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert aliases.count("Isabel Piedmont-Smith") == 1


def test_d01_preferred_name_emitted_when_different():
    pol = {"full_name": "Robert Smith", "preferred_name": "Bob", "title": "Councilmember"}
    aliases = generate_aliases(pol)
    assert "Bob Smith" in aliases
    assert "Robert Smith" in aliases


# ---------------------------------------------------------------------------
# D-02 hyphen-space expansion
# ---------------------------------------------------------------------------

def test_d02_hyphen_space_expansion(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert "Piedmont Smith" in aliases
    assert "Isabel Piedmont Smith" in aliases
    assert "Councilmember Piedmont Smith" in aliases


def test_d02_no_half_splits(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert "Piedmont" not in aliases
    assert "Smith" not in aliases


# ---------------------------------------------------------------------------
# D-03 ASCII folding
# ---------------------------------------------------------------------------

def test_d03_ascii_folding_garcia(sample_politician_garcia):
    aliases = generate_aliases(sample_politician_garcia)
    assert "María García" in aliases
    assert "Maria Garcia" in aliases


def test_d03_ascii_folding_munoz():
    pol = {"full_name": "Ana Muñoz", "preferred_name": "Ana", "title": "Councilmember"}
    aliases = generate_aliases(pol)
    assert "Muñoz" in aliases
    assert "Munoz" in aliases


# ---------------------------------------------------------------------------
# D-04 leadership-only title stripping
# ---------------------------------------------------------------------------

def test_d04_title_stripping_positive(sample_politician_asare):
    aliases = generate_aliases(sample_politician_asare)
    assert "Council President Asare" in aliases
    assert "President Asare" in aliases


def test_d04_title_stripping_negative(sample_politician_bolden):
    aliases = generate_aliases(sample_politician_bolden)
    assert "City Clerk Bolden" in aliases
    assert "Clerk Bolden" not in aliases
    assert "City Bolden" not in aliases


def test_d04_vice_chair_positive():
    pol = {"full_name": "Jane Doe", "preferred_name": "Jane", "title": "Vice Chair"}
    aliases = generate_aliases(pol)
    assert "Vice Chair Doe" in aliases
    assert "Chair Doe" in aliases


def test_d04_deputy_mayor_positive():
    pol = {"full_name": "Sam Lee", "preferred_name": "Sam", "title": "Deputy Mayor"}
    aliases = generate_aliases(pol)
    assert "Deputy Mayor Lee" in aliases
    assert "Mayor Lee" in aliases


# ---------------------------------------------------------------------------
# D-05 case-only dedup
# ---------------------------------------------------------------------------

def test_d05_dedup_case_only():
    # Two strings that differ only in case collapse; strings differing in
    # punctuation/whitespace are kept.
    from src.alias_gen import _case_dedup
    out = _case_dedup(["Piedmont-Smith", "piedmont-smith"])
    assert len(out) == 1
    out2 = _case_dedup(["Piedmont-Smith", "Piedmont Smith"])
    assert len(out2) == 2


def test_d05_dedup_preserves_punctuation(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert "Piedmont-Smith" in aliases
    assert "Piedmont Smith" in aliases


# ---------------------------------------------------------------------------
# D-06 original case preserved
# ---------------------------------------------------------------------------

def test_d06_original_case_preserved(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert "Councilmember Piedmont-Smith" in aliases
    # No lowercase-only variant is emitted deliberately.
    assert "councilmember piedmont-smith" not in aliases


# ---------------------------------------------------------------------------
# D-07 determinism
# ---------------------------------------------------------------------------

def test_d07_determinism(sample_politician_piedmont_smith):
    a = generate_aliases(sample_politician_piedmont_smith)
    b = generate_aliases(sample_politician_piedmont_smith)
    assert a == b


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------

def test_returns_list_of_str(sample_politician_piedmont_smith):
    aliases = generate_aliases(sample_politician_piedmont_smith)
    assert isinstance(aliases, list)
    assert all(isinstance(a, str) for a in aliases)
