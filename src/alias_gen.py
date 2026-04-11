"""Pure-function alias generator for CouncilScribe roster members.

Generates ordered, deduplicated alias lists per Phase 108 decisions
D-01 through D-08. See 108-CONTEXT.md for rule definitions.

This module has zero network and zero filesystem dependencies so it
can be unit-tested in isolation and reused by Phase 109/111.
"""
from __future__ import annotations

import unicodedata

# D-04: leadership-only title stripping.
# Strip a leading "Council ", "Vice ", or "Deputy " prefix when the
# NEXT token is one of: President, Chair, Mayor, Speaker, Clerk.
_STRIP_PREFIXES = ("Council", "Vice", "Deputy")
_KEEP_AFTER_PREFIX = {"President", "Chair", "Mayor", "Speaker", "Clerk"}


def _ascii_fold(s: str) -> str:
    """NFKD-normalize and drop non-ASCII code points (D-03)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if ch.isascii())


def _expand_accent(s: str) -> list[str]:
    """Return [s] or [s, ascii_folded] when ascii_folded differs."""
    folded = _ascii_fold(s)
    if folded and folded != s:
        return [s, folded]
    return [s]


def _expand_hyphen_space(s: str) -> list[str]:
    """D-02: for strings with '-', also emit the hyphen->space form."""
    if "-" in s:
        return [s, s.replace("-", " ")]
    return [s]


def _strip_leadership_title(title: str) -> str | None:
    """D-04: return title with a leadership prefix stripped, or None.

    Example: 'Council President' -> 'President'; 'City Clerk' -> None;
    'Vice Chair' -> 'Chair'; 'Deputy Mayor' -> 'Mayor'.
    """
    parts = title.split()
    if len(parts) < 2:
        return None
    if parts[0] in _STRIP_PREFIXES and parts[1] in _KEEP_AFTER_PREFIX:
        stripped = " ".join(parts[1:])
        if stripped != title:
            return stripped
    return None


def _case_dedup(items: list[str]) -> list[str]:
    """D-05: collapse pure case duplicates while preserving order.

    Strings that differ in punctuation or whitespace are kept distinct.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def generate_aliases(politician: dict) -> list[str]:
    """Generate an ordered, deduplicated alias list for one politician.

    Input keys used: full_name, preferred_name, title.
    Output: list[str] in deterministic order (D-07).
    """
    full_name: str = politician.get("full_name", "").strip()
    preferred_name: str = politician.get("preferred_name", "").strip()
    title: str = politician.get("title", "").strip()

    if not full_name:
        return []

    tokens = full_name.split()
    first_name = tokens[0] if tokens else ""
    last_name = tokens[-1] if tokens else ""

    raw: list[str] = []

    def push(s: str) -> None:
        # Each push goes through accent expansion then hyphen-space expansion
        # so every base alias produces up to four variants in a stable order.
        for accented in _expand_accent(s):
            for variant in _expand_hyphen_space(accented):
                raw.append(variant)

    # D-01 variant 1: full_name
    push(full_name)

    # D-01 variant 2: surname alone
    if last_name:
        push(last_name)

    # D-01 variant 3: first + last
    if first_name and last_name and first_name != last_name:
        push(f"{first_name} {last_name}")

    # D-01 variant 4: preferred + last (only when preferred set AND differs from first)
    if preferred_name and preferred_name != first_name and last_name:
        push(f"{preferred_name} {last_name}")

    # D-01 variant 5: "{title} {last_name}" verbatim
    if title and last_name:
        push(f"{title} {last_name}")

        # D-01 variant 6 / D-04: title-stripped form (only when stripping applies)
        stripped = _strip_leadership_title(title)
        if stripped is not None:
            push(f"{stripped} {last_name}")

    # D-05: case-only dedup, D-07: stable order
    return _case_dedup(raw)
