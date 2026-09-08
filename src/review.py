"""Pure speaker-review operations shared by the CLI and (later) the GUI.

No prompts, no printing, no file writes — these functions transform in-memory
data (segments, mappings, embeddings) so they are directly unit-testable and
reusable. Persistence and interaction live in the callers (run_local.py).
"""
from __future__ import annotations

import copy as _copy
import re as _re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def snapshot_mapping(mappings, segments, label):
    """Capture a speaker's mapping + its segments' names, for one-step undo."""
    m = mappings.get(label)
    return {
        "label": label,
        "mapping": _copy.deepcopy(m) if m is not None else None,
        "seg_names": {i: s.speaker_name for i, s in enumerate(segments)
                      if s.speaker_label == label},
    }


def restore_mapping(mappings, segments, label, snap):
    """Revert to a snapshot taken by snapshot_mapping."""
    if snap["mapping"] is None:
        mappings.pop(label, None)
    else:
        mappings[label] = _copy.deepcopy(snap["mapping"])
    for i, s in enumerate(segments):
        if i in snap["seg_names"]:
            s.speaker_name = snap["seg_names"][i]


def make_unidentified_slug(meeting_id: str, label: str) -> str:
    """Unique, deterministic handle for an unidentified speaker.

    Keyed by (meeting, diarization label) so two different unknowns never share a
    slug (no merge), while re-running review on the same meeting is idempotent.
    """
    base = _re.sub(r"[^a-z0-9]+", "-", f"{meeting_id}-{label}".lower()).strip("-")
    base = base or _re.sub(r"[^a-z0-9]+", "-", (label or "speaker").lower()).strip("-") or "speaker"
    return f"unidentified-{base}"[:100]


# A site-local person's slug. Mirrors ev-accounts SLUG_REGEX; was an inline
# literal in run_local.py's terminal wizard before the GUI needed it too.
LOCAL_SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,99}$"
LOCAL_SLUG_RE = _re.compile(LOCAL_SLUG_PATTERN)


def default_local_slug(name, label) -> str:
    """Kebab-case slug for a new local person, from the name or the diarized label.

    Always returns a value matching LOCAL_SLUG_RE so the caller can offer it as a
    prefilled default without validating first. Falls through name -> label ->
    'speaker', mirroring make_unidentified_slug's fallback chain.
    """
    for source in ((name or "").strip(), (label or "").strip()):
        slug = _re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")[:100].strip("-")
        if LOCAL_SLUG_RE.fullmatch(slug):
            return slug
    return "speaker"


def identity_label(mapping) -> str:
    """One-word resolved identity for the review table."""
    if mapping is None:
        return "unlinked"
    if mapping.speaker_status == "non_speaker":
        return "non-speaker"
    if mapping.speaker_status == "unidentified":
        return "unidentified"
    if mapping.politician_id:
        # Key on the stable UUID first: politician_slug is NULL for ~99.4% of
        # essentials.politicians. Mirrors resolve_mapping_enrollment (enroll.py).
        return f"essentials:{mapping.politician_id}"
    if mapping.politician_slug:
        return f"essentials:{mapping.politician_slug}"
    if mapping.local_slug:
        return f"local:{mapping.local_slug}"
    return "unlinked"


@dataclass
class SpeakerView:
    label: str
    current_name: Optional[str]
    current_confidence: float
    current_method: Optional[str]
    seg_count: int
    total_speech_seconds: float
    clip_start: Optional[float]
    clip_candidates: list[float] = field(default_factory=list)
    sample_text: Optional[str] = None
    soft_hints: list[tuple[str, float, str]] = field(default_factory=list)
    needs_review: bool = False


def build_review_state(segments, mappings, embeddings, profile_db, *, show_text: bool) -> list[SpeakerView]:
    """Build one SpeakerView per speaker label, sorted by speech time desc.

    soft_hints come from voice-profile soft matching when embeddings + profiles
    are available; otherwise empty.
    """
    by_label: dict[str, list] = {}
    for seg in segments:
        by_label.setdefault(seg.speaker_label, []).append(seg)

    hints: dict[str, list[tuple[str, float, str]]] = {}
    if embeddings and getattr(profile_db, "profiles", None):
        from src.enroll import get_stored_centroids
        from src.identify import soft_match_voice_profiles

        centroids = get_stored_centroids(profile_db)
        if centroids:
            display_names = {pid: p.display_name for pid, p in profile_db.profiles.items()}
            hints = soft_match_voice_profiles(embeddings, centroids, display_names)

    views: list[SpeakerView] = []
    for label, segs in by_label.items():
        total = sum(s.end_time - s.start_time for s in segs)
        # Candidates: this speaker's segments by duration desc (longest turn is
        # the most identifying), capped at 8. Turns much longer than the ~40s
        # playback window also contribute in-turn start points (every 60s while
        # at least 30s of the turn remains) so cycling clips can sample beyond
        # the opening of a long monologue. The default clip + sample come from
        # the longest turn.
        ordered = sorted(segs, key=lambda s: s.end_time - s.start_time, reverse=True)
        clip_candidates: list[float] = []
        for s in ordered:
            if len(clip_candidates) >= 8:
                break
            clip_candidates.append(s.start_time)
            offset = 60.0
            while len(clip_candidates) < 8 and (s.end_time - s.start_time) - offset >= 30.0:
                clip_candidates.append(s.start_time + offset)
                offset += 60.0
        longest = ordered[0] if ordered else None
        mapping = mappings.get(label)
        sample_text = None
        if show_text and longest is not None and getattr(longest, "text", None) and longest.text.strip():
            sample_text = longest.text
        views.append(SpeakerView(
            label=label,
            current_name=getattr(mapping, "speaker_name", None) if mapping else None,
            current_confidence=getattr(mapping, "confidence", 0.0) if mapping else 0.0,
            current_method=getattr(mapping, "id_method", None) if mapping else None,
            seg_count=len(segs),
            total_speech_seconds=total,
            clip_start=longest.start_time if longest is not None else None,
            clip_candidates=clip_candidates,
            sample_text=sample_text,
            soft_hints=hints.get(label, []),
            needs_review=getattr(mapping, "needs_review", False) if mapping else False,
        ))

    views.sort(key=lambda v: v.total_speech_seconds, reverse=True)
    return views


@dataclass
class RenameResult:
    label: str
    old_name: Optional[str]
    new_name: str
    alias_suggestion: Optional[str]


def rename_speaker(mappings, segments, label: str, new_name: str, *, roster=None) -> RenameResult:
    """Assign new_name to a speaker label across its mapping and segments.

    If roster is given, the name is normalized via correct_speaker_name. Returns
    a RenameResult; alias_suggestion is the prior (wrong) name, to offer as an
    alias, or None when there was no prior name or it equals the new name.
    """
    from src.models import SpeakerMapping

    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    old_name = mapping.speaker_name

    final_name = new_name
    if roster is not None:
        from src.roster import correct_speaker_name
        final_name = correct_speaker_name(new_name, roster)

    mapping.speaker_name = final_name
    mapping.confidence = 1.0
    mapping.id_method = "human_review"
    mapping.needs_review = False

    # A human-assigned name is authoritative. Any prior identity link belonged to
    # the OLD name (e.g. a voice-profile collision that was then corrected by
    # hand), so it must not survive a name change — otherwise this voice enrolls
    # under the wrong person, since resolve_mapping_enrollment keys on
    # politician_slug ahead of the name. Re-derive the link from the new name when
    # a roster is available; otherwise drop it. A no-op rename leaves the (already
    # correct, possibly manually-pasted) link untouched.
    if final_name != old_name:
        mapping.local_slug = None
        mapping.local_role = None
        if roster is not None:
            from src.enroll import resolve_enrollment_key
            _key, pol_slug, pol_id = resolve_enrollment_key(final_name, roster)
            mapping.politician_slug = pol_slug
            mapping.politician_id = pol_id
        else:
            mapping.politician_slug = None
            mapping.politician_id = None

    mappings[label] = mapping

    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = final_name

    alias = old_name if (old_name and old_name != final_name) else None
    return RenameResult(label=label, old_name=old_name, new_name=final_name, alias_suggestion=alias)


@dataclass
class MergeResult:
    source_label: str
    target_label: str
    moved_segments: int
    combined_name: Optional[str]


def merge_speakers(segments, embeddings, mappings, source_label: str, target_label: str) -> MergeResult:
    """Full merge: fold source_label into target_label.

    - Relabels every source segment to the target.
    - Combines centroids weighted by each label's pre-merge speech time and
      recomputes the target centroid (if both embeddings present). If only one
      side has an embedding, the surviving embedding is carried to the target.
    - Drops the source from embeddings and mappings.
    - If the target has no name but the source does, the target adopts it.
    - All segments now labeled the target carry the merged speaker's name.

    Raises ValueError if labels are equal or the source has no segments/mapping.
    """
    if source_label == target_label:
        raise ValueError("Cannot merge a speaker into itself.")
    if source_label not in mappings and not any(s.speaker_label == source_label for s in segments):
        raise ValueError(f"Unknown source speaker: {source_label}")

    speech: dict[str, float] = {}
    for s in segments:
        speech[s.speaker_label] = speech.get(s.speaker_label, 0.0) + (s.end_time - s.start_time)

    moved = 0
    for s in segments:
        if s.speaker_label == source_label:
            s.speaker_label = target_label
            moved += 1

    if source_label in embeddings and target_label in embeddings:
        w_src = speech.get(source_label, 0.0)
        w_tgt = speech.get(target_label, 0.0)
        total = w_src + w_tgt
        if total > 0:
            embeddings[target_label] = (
                w_tgt * np.asarray(embeddings[target_label]) + w_src * np.asarray(embeddings[source_label])
            ) / total
        else:
            embeddings[target_label] = np.mean(
                [np.asarray(embeddings[target_label]), np.asarray(embeddings[source_label])], axis=0
            )
    elif source_label in embeddings and target_label not in embeddings:
        # Only the source has an embedding — carry it over so the merged
        # speaker keeps usable voice data instead of losing it.
        embeddings[target_label] = np.asarray(embeddings[source_label])
    embeddings.pop(source_label, None)

    src_map = mappings.pop(source_label, None)
    tgt_map = mappings.get(target_label)
    if tgt_map is not None and not getattr(tgt_map, "speaker_name", None) and src_map is not None and getattr(src_map, "speaker_name", None):
        tgt_map.speaker_name = src_map.speaker_name
        tgt_map.confidence = max(getattr(tgt_map, "confidence", 0.0), getattr(src_map, "confidence", 0.0))
        tgt_map.id_method = src_map.id_method
        tgt_map.needs_review = False

    combined_name = getattr(tgt_map, "speaker_name", None) if tgt_map is not None else None

    # Keep segment names consistent with the merged speaker.
    for s in segments:
        if s.speaker_label == target_label:
            s.speaker_name = combined_name

    return MergeResult(source_label=source_label, target_label=target_label, moved_segments=moved, combined_name=combined_name)


# Voice-similarity bands for a proposed merge. Calibrated on the 25-case
# duplicate-name triage: every pair confirmed to be one person split across two
# labels scored >=0.6 (with mutual nearest-neighbour), and every pair confirmed to
# be two different people scored <=0.42. Between them is a real ambiguity band, so
# it warns rather than deciding.
MERGE_SIM_MISMATCH = 0.42   # at or below: the two voices are different people
MERGE_SIM_CONFIDENT = 0.60  # at or above: the two voices are the same person


def voice_similarity(embeddings, label_a: str, label_b: str) -> Optional[float]:
    """Cosine similarity between two labels' voice centroids, or None when it
    cannot be measured.

    Unmeasurable means a missing, NaN, or zero-norm vector — all of which occur in
    the corpus (NaN especially). None is a distinct answer from "dissimilar": a
    caller must never treat "we could not tell" as evidence of a mis-merge.
    """
    a, b = embeddings.get(label_a) if embeddings else None, embeddings.get(label_b) if embeddings else None
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or np.isnan(a).any() or np.isnan(b).any():
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def merge_voice_verdict(similarity: Optional[float]) -> str:
    """Band a voice_similarity into 'unknown' | 'mismatch' | 'uncertain' | 'match'.

    'unknown' (unmeasurable) deliberately reads as permissive — a merge must never
    be blocked because we lacked data to judge it.
    """
    if similarity is None:
        return "unknown"
    if similarity <= MERGE_SIM_MISMATCH:
        return "mismatch"
    if similarity < MERGE_SIM_CONFIDENT:
        return "uncertain"
    return "match"


def speakers_needing_review(mappings) -> list[str]:
    """Labels whose mapping is flagged needs_review."""
    return [label for label, m in mappings.items() if getattr(m, "needs_review", False)]


def link_speaker(mappings, label, politician_slug, politician_id):
    """Set (or clear, when both are None) the politician identity on a mapping.

    Setting an essentials link (either field non-None) supersedes any local person:
    migration 623's invariant is one identity per speaker, and assign_local_person
    enforces the mirror of this rule. Clearing the link (both None — this is also
    the UNLINK path) leaves an existing local person alone.

    Mutates `mappings` in place; returns the updated SpeakerMapping. Creates a
    bare mapping if the label has none yet.
    """
    from src.models import SpeakerMapping

    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    mapping.politician_slug = politician_slug
    mapping.politician_id = politician_id
    if politician_slug or politician_id:
        # One identity per speaker (migration 623): an essentials link supersedes a
        # local person. assign_local_person enforces the mirror of this rule. Guarded
        # because link_speaker(None, None) is also the UNLINK path — clearing
        # unconditionally would destroy a local person on unlink.
        mapping.local_slug = None
        mapping.local_role = None
    mappings[label] = mapping
    return mapping


def assign_local_person(mappings, label, slug, role):
    """Make `label` a site-local person with `slug` and `role`. Mutates in place.

    Clears any essentials identity: migration 623's invariant is one identity per
    speaker, and a local person is not a roster politician. Enforcing it here means
    publish never has to suppress a contradiction it should not have received.

    Raises ValueError when `slug` fails LOCAL_SLUG_RE, or when a DIFFERENT label in
    this meeting already holds it — two diarized labels cannot be the same person.
    """
    from src.models import SpeakerMapping

    slug = (slug or "").strip()
    if not LOCAL_SLUG_RE.fullmatch(slug):
        raise ValueError(f"invalid local slug {slug!r}; must match {LOCAL_SLUG_PATTERN}")
    for other_label, other in mappings.items():
        if other_label != label and getattr(other, "local_slug", None) == slug:
            raise ValueError(f"local slug {slug!r} already used by label {other_label!r}")

    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    mapping.local_slug = slug
    mapping.local_role = role
    mapping.politician_slug = None
    mapping.politician_id = None
    mappings[label] = mapping
    return mapping


def clear_local_person(mappings, label):
    """Drop a speaker's local-person identity. Returns None (no mutation) if the
    label has no mapping, OR if the mapping is an unidentified handle.

    local_slug is overloaded: a reviewer sets it for a genuine local person, but
    mark_unidentified / link_to_unidentified_handle ALSO set it — to the synthetic
    unidentified-<meeting>-<label> handle whose entire purpose is keeping two
    distinct unknown speakers from sharing one voice-profile enrollment key
    (make_unidentified_slug). Clearing it would drop speaker_name back to
    resolve_enrollment_key('Unidentified Speaker') -> the single shared key
    'unidentified_speaker', silently merging unrelated strangers' voice
    embeddings. So the unidentified case is refused outright rather than
    cleared; a real local person (speaker_status is None) is unaffected.
    """
    mapping = mappings.get(label)
    if mapping is None:
        return None
    if getattr(mapping, "speaker_status", None) == "unidentified":
        return None
    mapping.local_slug = None
    mapping.local_role = None
    return mapping


def clear_speaker_status(mappings, segments, label):
    """Drop 'unidentified' / 'non_speaker' so a label can hold a real identity again.

    Returns None (no mutation) when the label is unknown or its status is already
    clear. A no-op is not success: the GUI route maps None to 404, so an Undo
    button on a speaker that was never marked cannot report that it acted.

    mark_unidentified and mark_non_speaker are otherwise one-way doors — nothing
    else in src/ or gui/ ever clears speaker_status — which left a mis-clicked
    "Not a speaker" unrecoverable and permanently hid the local-person path.

    Three groups of fields go with the mark and must not outlive it:

    - `local_slug` after an 'unidentified' mark is the synthetic
      unidentified-<meeting>-<label> handle from make_unidentified_slug, whose
      only job is keeping two distinct unknowns out of one voice-profile
      enrollment key. It is not a site-local person and must not be presented as
      one. A 'non_speaker' mark clears local_slug outright, so clearing it again
      is a harmless no-op — hence no branch on the status value.
    - `speaker_name` is 'Unidentified Speaker', 'Non-speaker', or a reviewer's
      display_label FOR THE MARK. It names the status, not a person.
    - confidence 1.0 / id_method 'human_review' asserted human certainty about
      the mark. With the mark gone the speaker has no identity, so it returns to
      needs-review rather than staying falsely confirmed.

    A politician link is deliberately left alone: clearing a stale mark is not an
    unlink.
    """
    mapping = mappings.get(label)
    if mapping is None or getattr(mapping, "speaker_status", None) is None:
        return None

    mapping.speaker_status = None
    mapping.local_slug = None
    mapping.local_role = None
    mapping.speaker_name = None
    mapping.confidence = 0.0
    mapping.id_method = None
    mapping.needs_review = True

    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = None

    return mapping


def mark_unidentified(mappings, segments, label, meeting_id, display_label=None):
    """Mark a speaker as a distinct-but-unnamed person: unique handle, enrolled."""
    from src.models import SpeakerMapping
    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    name = (display_label or "").strip() or "Unidentified Speaker"
    mapping.speaker_name = name
    mapping.local_slug = make_unidentified_slug(meeting_id, label)
    mapping.local_role = None
    mapping.politician_slug = None
    mapping.politician_id = None
    mapping.speaker_status = "unidentified"
    mapping.id_method = "human_review"
    mapping.confidence = 1.0
    mapping.needs_review = False
    mappings[label] = mapping
    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = name


def mark_non_speaker(mappings, segments, label, display_label=None):
    """Mark a label as not-a-person (music/pledge/station ID); never enrolled."""
    from src.models import SpeakerMapping
    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    name = (display_label or "").strip() or "Non-speaker"
    mapping.speaker_name = name
    mapping.speaker_status = "non_speaker"
    mapping.politician_slug = None
    mapping.politician_id = None
    mapping.local_slug = None
    mapping.local_role = None
    mapping.id_method = "human_review"
    mapping.confidence = 1.0
    mapping.needs_review = False
    mappings[label] = mapping
    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = name


def link_to_unidentified_handle(mappings, segments, label, handle_key, display_name):
    """Link a speaker to an EXISTING unidentified handle (a returning unknown).

    handle_key is the stored profile key, e.g. 'local:unidentified-<m>-<lbl>'.
    Reuses that handle's slug so the recurring speaker enrolls into the same
    profile. Confirm-only — never called without reviewer action.
    """
    from src.models import SpeakerMapping
    slug = handle_key[len("local:"):] if handle_key.startswith("local:") else handle_key
    mapping = mappings.get(label) or SpeakerMapping(speaker_label=label)
    mapping.speaker_name = display_name or "Unidentified Speaker"
    mapping.local_slug = slug
    mapping.local_role = None
    mapping.politician_slug = None
    mapping.politician_id = None
    mapping.speaker_status = "unidentified"
    mapping.id_method = "human_confirmed"
    mapping.confidence = 1.0
    mapping.needs_review = False
    mappings[label] = mapping
    for seg in segments:
        if seg.speaker_label == label:
            seg.speaker_name = mapping.speaker_name


def parse_link_selection(token, n_matches):
    """Parse the reviewer's link-prompt input.

    Returns (action, index): action in {'pick','skip','search','none','invalid'}.
    'pick' carries a 0-based index into the match list.
    """
    t = (token or "").strip().lower()
    if t in ("", "s", "skip"):
        return ("skip", None)
    if t in ("m", "search"):
        return ("search", None)
    if t in ("n", "none"):
        return ("none", None)
    if t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < n_matches:
            return ("pick", idx)
        return ("invalid", None)
    return ("invalid", None)


def format_match_line(match, index):
    """One-line rendering of a search_politicians() result for the link menu.

    No affiliation detail — the pipeline never surfaces it (antipartisan
    rule, tests/test_antipartisan.py).
    """
    tag = "incumbent" if match.get("is_incumbent") else "candidate"
    detail = []
    if match.get("office_title"):
        loc = match.get("government_name") or match.get("district_label") or ""
        detail.append(f"{match['office_title']}{', ' + loc if loc else ''}")
    elif match.get("district_label"):
        detail.append(match["district_label"])
    suffix = f" · {' · '.join(detail)}" if detail else ""
    name = match.get("full_name") or "(unknown)"
    return f"  {index + 1}. {name}{suffix} [{tag}]"


def _ew_name_tokens(s):
    stop = {"councilmember", "council", "president", "vice", "mayor", "clerk",
            "the", "of", "common", "city", "member", "district", "association",
            "office", "at", "large"}
    return set(_re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()) - stop


def _ew_slug_tokens(slug):
    return set(_re.sub(r"[^a-z0-9]+", " ", (slug or "").lower()).split()) - {"h", "j", "s"}


def duplicate_named_speakers(mappings) -> dict[str, list[str]]:
    """{normalized name: sorted [labels]} for names claimed by 2+ labels.

    The invariant this detects — two distinct diarized labels can't be the same
    person — is enforced at identify time but can be re-created by a reviewer
    rename, so every reader of a reviewed meeting (enrollment, GUI, publish)
    checks through here. Excludes non_speaker and unidentified mappings: their
    names are placeholders, not identities."""
    by_name: dict[str, list[str]] = {}
    for label, m in mappings.items():
        name = getattr(m, "speaker_name", None)
        if name and getattr(m, "speaker_status", None) not in ("non_speaker", "unidentified"):
            by_name.setdefault(name.strip().lower(), []).append(label)
    return {nm: sorted(labels) for nm, labels in by_name.items() if len(labels) > 1}


# What a memo member last name can look like: letters, plus the punctuation real
# surnames carry (Piedmont-Smith, O'Brien, initials). Deliberately excludes role
# annotations like "(Moderator)"/"(Video)", which a clerk memo can never name — so
# grouping on them is noise, and noisy warnings get ignored.
_SURNAME_TOKEN = _re.compile(r"[a-z][a-z.'\-]*")


def ambiguous_speaker_surnames(mappings) -> dict[str, list[str]]:
    """{last name: sorted [labels]} for a surname shared by 2+ labels under
    DIFFERENT full names.

    The collision duplicate_named_speakers cannot see. memo_reconcile.match_speaker
    resolves a memo last name T by taking every speaker whose display_name equals T
    or ends with " " + T, so two speakers are mutually ambiguous for some T exactly
    when they share this final word — "Isak Nti Asare" and "Council President Asare"
    are two names, one surname, and that member's vote record is silently skipped.

    Groups whose labels all carry one identical name are left out: those are
    duplicate_named_speakers' job, and reporting them here too would double-warn
    about a single problem. Placeholders (non_speaker/unidentified) are excluded
    for the same reason as there — they aren't identities.

    Unlike an exact duplicate this is NOT necessarily wrong: two different people
    can share a surname. It is a warning to resolve by hand, never a hard gate.
    """
    by_surname: dict[str, list[tuple[str, str]]] = {}
    for label, m in mappings.items():
        name = getattr(m, "speaker_name", None)
        if not name or getattr(m, "speaker_status", None) in ("non_speaker", "unidentified"):
            continue
        words = name.strip().lower().split()
        if words and _SURNAME_TOKEN.fullmatch(words[-1]):
            by_surname.setdefault(words[-1], []).append((label, " ".join(words)))
    return {
        surname: sorted(label for label, _ in pairs)
        for surname, pairs in by_surname.items()
        if len(pairs) > 1 and len({name for _, name in pairs}) > 1
    }


def enrollment_warnings(mappings, roster=None) -> list[dict]:
    """Flag suspicious states before enrollment. Returns [{kind, label, detail}];
    duplicate_name and ambiguous_surname entries also carry labels (the list form
    of the joined label). kinds: name_slug_mismatch, duplicate_name,
    ambiguous_surname, unlinked_roster_match."""
    warns: list[dict] = []
    # name/slug mismatch (linked slug shares no token with the name)
    for label, m in mappings.items():
        if m.politician_slug and m.speaker_name:
            nt, st = _ew_name_tokens(m.speaker_name), _ew_slug_tokens(m.politician_slug)
            if nt and st and not (nt & st):
                warns.append({"kind": "name_slug_mismatch", "label": label,
                              "detail": f"{m.speaker_name!r} linked to {m.politician_slug!r}"})
    # duplicate name across labels (excluding non-speakers)
    for nm, labels in duplicate_named_speakers(mappings).items():
        warns.append({"kind": "duplicate_name", "label": ",".join(labels), "labels": labels,
                      "detail": f"{len(labels)} labels named {nm!r} (merge?)"})
    # different names sharing a last name — ambiguous to memo_reconcile.match_speaker,
    # which silently skips that member's vote record rather than guessing
    for surname, labels in ambiguous_speaker_surnames(mappings).items():
        names = sorted({mappings[l].speaker_name.strip() for l in labels})
        warns.append({"kind": "ambiguous_surname", "label": ",".join(labels), "labels": labels,
                      "detail": f"{len(labels)} labels share the last name {surname.title()!r} "
                                f"({', '.join(names)}) — memo vote records for that member "
                                f"will be skipped as ambiguous"})
    # named but unlinked, yet matches a roster member
    if roster is not None:
        from src.roster import correct_speaker_name
        for label, m in mappings.items():
            if (m.speaker_name and not m.politician_slug and not m.local_slug
                    and m.speaker_status not in ("non_speaker", "unidentified")):
                corrected = correct_speaker_name(m.speaker_name, roster)
                if any(corrected == mem.name and mem.politician_slug for mem in roster.members):
                    warns.append({"kind": "unlinked_roster_match", "label": label,
                                  "detail": f"{m.speaker_name!r} matches a roster member but isn't linked"})
    return warns
