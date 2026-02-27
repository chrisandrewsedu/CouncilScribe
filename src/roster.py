"""Council roster management for speaker name correction.

Loads a council roster JSON file and provides fuzzy matching to correct
common transcription errors (e.g. "Sasseberg" -> "President Asare").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class RosterMember:
    name: str  # canonical name, e.g. "President Asare"
    aliases: list[str] = field(default_factory=list)


@dataclass
class Roster:
    city: str = ""
    body: str = ""
    members: list[RosterMember] = field(default_factory=list)


def load_roster(path: Optional[Path] = None) -> Optional[Roster]:
    """Load council roster from config directory.

    Returns None if roster file doesn't exist.
    """
    if path is None:
        path = config.CONFIG_DIR / "council_roster.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    members = [
        RosterMember(name=m["name"], aliases=m.get("aliases", []))
        for m in data.get("members", [])
    ]
    return Roster(
        city=data.get("city", ""),
        body=data.get("body", ""),
        members=members,
    )


def _extract_surname(name: str) -> str:
    """Extract the likely surname from a display name, stripping titles."""
    titles = {
        "councilmember", "councilwoman", "councilman", "alderman",
        "alderwoman", "commissioner", "mayor", "vice-mayor",
        "president", "vice-president", "clerk", "secretary",
        "treasurer", "supervisor", "representative", "city",
        "council", "member",
    }
    words = name.strip().split()
    filtered = [w for w in words if w.lower() not in titles]
    return filtered[-1] if filtered else name


def _similarity(a: str, b: str) -> float:
    """Case-insensitive similarity ratio using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def correct_speaker_name(name: str, roster: Roster, threshold: float = 0.80) -> str:
    """Check if a name matches any roster member and return the canonical name.

    Matching strategy (in order of priority):
    1. Exact match on canonical name (case-insensitive)
    2. Exact match on any alias (case-insensitive)
    3. Alias appears as a word/substring in the name
    4. Fuzzy match on the surname portion of the name against aliases

    Args:
        name: The speaker name to check.
        roster: Loaded council roster.
        threshold: Minimum similarity ratio for fuzzy matching.

    Returns:
        Canonical name if a match is found, otherwise the original name unchanged.
    """
    if not name or not roster or not roster.members:
        return name

    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    # 1. Exact match on canonical name
    for member in roster.members:
        if name_lower == member.name.lower():
            return member.name

    # 2. Exact match on alias
    for member in roster.members:
        for alias in member.aliases:
            if name_lower == alias.lower():
                return member.name

    # 3. Alias appears as word in the name (e.g. "Council Member Sasseberg" matches alias "Sasseberg")
    name_words = [w.lower() for w in name_stripped.split()]
    for member in roster.members:
        for alias in member.aliases:
            alias_lower = alias.lower()
            # Single-word alias: check if it's one of the words
            if " " not in alias and alias_lower in name_words:
                return member.name
            # Multi-word alias: check if it's a substring
            if " " in alias and alias_lower in name_lower:
                return member.name

    # 4. Fuzzy match: extract surname from input, compare against aliases
    surname = _extract_surname(name_stripped)
    best_score = 0.0
    best_member = None

    for member in roster.members:
        for alias in member.aliases:
            alias_surname = _extract_surname(alias)
            score = _similarity(surname, alias_surname)
            if score > best_score:
                best_score = score
                best_member = member

    if best_member and best_score >= threshold:
        return best_member.name

    return name_stripped


def correct_mappings(
    mappings: dict,
    roster: Roster,
) -> dict:
    """Apply roster corrections to all speaker mappings.

    Modifies mappings in place and returns the dict.
    """
    for label, mapping in mappings.items():
        if mapping.speaker_name:
            corrected = correct_speaker_name(mapping.speaker_name, roster)
            if corrected != mapping.speaker_name:
                mapping.speaker_name = corrected

    return mappings


def roster_names_for_prompt(roster: Roster) -> str:
    """Format roster members for inclusion in an LLM prompt.

    Returns a string like:
      Known council members: President Asare, Councilmember Piedmont-Smith, ...
    """
    if not roster or not roster.members:
        return ""

    names = [m.name for m in roster.members]
    return "Known council members for this body: " + ", ".join(names)
