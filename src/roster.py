"""Council roster management for speaker name correction.

Loads a council roster JSON file and provides fuzzy matching to correct
common transcription errors (e.g. "Sasseberg" -> "President Asare").
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class RosterMember:
    name: str  # canonical name, e.g. "President Asare"
    aliases: list[str] = field(default_factory=list)
    politician_slug: Optional[str] = None
    politician_id: Optional[str] = None


@dataclass
class Roster:
    city: str = ""
    body: str = ""
    members: list[RosterMember] = field(default_factory=list)


def load_roster(
    path: Optional[Path] = None,
    *,
    body_slug: Optional[str] = None,
) -> Optional[Roster]:
    """Load council roster from config directory.

    Two paths:
    - Legacy: ``load_roster()`` or ``load_roster(path=...)`` reads the
      existing ``council_roster.json`` format unchanged.
    - Per-body (Phase 108): ``load_roster(body_slug="...")`` reads the
      per-body cache at ``CONFIG_DIR/rosters/{body_slug}.json`` written
      by ``refresh_roster.py`` and returns a ``Roster`` whose
      ``members[].name`` is constructed as ``"{title} {last_name}"`` to
      match the legacy format consumed by ``correct_speaker_name`` and
      ``roster_names_for_prompt``.

    Returns None if the resolved file doesn't exist. The 30-day staleness
    warning only fires on the slug path (CSROSTER-05).
    """
    # Legacy path — unchanged behavior.
    if body_slug is None:
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

    # Slug path — per-body cache written by refresh_roster.py.
    slug_path = config.CONFIG_DIR / "rosters" / f"{body_slug}.json"
    if not slug_path.exists():
        return None
    with open(slug_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Staleness check — CSROSTER-05, non-blocking.
    fetched_at_str = data.get("fetched_at", "")
    if fetched_at_str:
        try:
            # Normalize 'Z' suffix for datetime.fromisoformat compatibility.
            normalized = fetched_at_str.replace("Z", "+00:00")
            fetched_at = datetime.fromisoformat(normalized)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fetched_at).days
            if age_days > 30:
                msg = (
                    f"Roster '{body_slug}' is {age_days} days old "
                    f"(fetched {fetched_at_str[:10]}). "
                    f"Re-run: python refresh_roster.py --body {body_slug}"
                )
                logging.warning(msg)
                print(f"WARNING: {msg}", file=sys.stderr)
        except ValueError:
            logging.warning(
                "Roster '%s' has unparseable fetched_at: %r",
                body_slug,
                fetched_at_str,
            )

    # Build Roster from per-body cache. Canonical RosterMember.name =
    # "{title} {last_name}" where last_name is the last whitespace token
    # of full_name (Phase 107 API does not expose last_name directly).
    slug_members: list[RosterMember] = []
    for pol in data.get("politicians", []):
        full_name = pol.get("full_name", "")
        title = pol.get("title", "")
        last_name = full_name.split()[-1] if full_name else ""
        if title and last_name:
            canonical = f"{title} {last_name}"
        else:
            canonical = full_name
        slug_members.append(
            RosterMember(
                name=canonical,
                aliases=list(pol.get("aliases", [])),
                politician_slug=pol.get("politician_slug"),
                politician_id=pol.get("politician_id"),
            )
        )
    return Roster(
        city="",
        body=data.get("body_key", ""),
        members=slug_members,
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


def add_alias(
    roster_path: Optional[Path],
    canonical_name: str,
    new_alias: str,
) -> bool:
    """Add a new alias to a roster member's alias list.

    Loads the roster JSON, adds the alias if not already present, and saves.
    Guards against nonsense aliases (too short, generic labels, etc.).

    Args:
        roster_path: Path to roster JSON file. Uses default if None.
        canonical_name: The canonical member name to add the alias to.
        new_alias: The alias to add.

    Returns:
        True if alias was added, False otherwise.
    """
    if roster_path is None:
        roster_path = config.CONFIG_DIR / "council_roster.json"
    if not roster_path.exists():
        return False

    # Guard: reject nonsense aliases
    if not new_alias or len(new_alias.strip()) < 3:
        return False
    alias_stripped = new_alias.strip()
    # Skip generic/placeholder names
    _SKIP = {"speaker", "unknown", "unidentified", "none", "n/a"}
    if alias_stripped.lower() in _SKIP:
        return False
    # Skip SPEAKER_XX labels
    if alias_stripped.startswith("SPEAKER_"):
        return False

    with open(roster_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for member in data.get("members", []):
        if member["name"].lower() != canonical_name.lower():
            continue

        existing = [a.lower() for a in member.get("aliases", [])]
        if alias_stripped.lower() in existing:
            return False  # already present
        if alias_stripped.lower() == member["name"].lower():
            return False  # same as canonical

        if "aliases" not in member:
            member["aliases"] = []
        member["aliases"].append(alias_stripped)

        with open(roster_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True

    return False  # member not found


def roster_names_for_prompt(roster: Roster) -> str:
    """Format roster members for inclusion in an LLM prompt.

    Returns a string like:
      Known council members: President Asare, Councilmember Piedmont-Smith, ...
    """
    if not roster or not roster.members:
        return ""

    names = [m.name for m in roster.members]
    return "Known council members for this body: " + ", ".join(names)
